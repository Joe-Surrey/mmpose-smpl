from torch import nn
import torch
import smplx
from IK.utils.SMPL_H_specs import hand_loss_indexes, hand_joint_indexes

class IKModel(nn.Module):
    """
    Base class for IK modules
    """
    def __init__(self):
        super(IKModel, self).__init__()

    def head_loss(self):
        return 0

    def hand_loss(self):
        return 0


class SMPL_IK(IKModel):
    def __init__(self, batch_size=1, fit_body=True, pca_hands=False):
        #TODO implement batch size and pca hands
        super(SMPL_IK, self).__init__()
        self.name = "SMPL"
        num_betas = 10
        self.fit_body = fit_body

        self.model = smplx.create(model_path="models/",
                                  model_type="smpl",
                                  gender="neutral",
                                  use_face_contour=False,
                                  num_betas=num_betas,
                                  num_expression_coeffs=10,
                                  ext='pkl',
                                  use_hands=False,
                                  use_feet_keypoints=False,
                                  use_face=False
                                  )
        self.faces = self.model.faces
        self.global_orient = torch.zeros((1, 3))  # Not trained

        # Parameters
        if fit_body:
            self.betas = nn.Parameter(torch.zeros([1, num_betas]))

        self.upper_body_pose = nn.Parameter(torch.zeros((1, 12 * 3)))
        self.transl = nn.Parameter(torch.tensor([[0., 0., -1.]]))

    def forward(self, return_verts=False):
        body_pose = torch.zeros((1, self.model.NUM_JOINTS * 3))

        body_pose[:, 11*3:] = self.upper_body_pose
        return self.model(betas=self.betas if self.fit_body else torch.zeros([1, self.model.num_betas]),
                          body_pose=body_pose,
                          global_orient=self.global_orient,
                          transl=self.transl,
                          return_verts=return_verts,
                          return_full_pose=True,
                          pose2rot=True,)


class SMPLH_IK(IKModel):
    def __init__(self, batch_size=1, fit_body=True, pca_hands=False):
        super(SMPLH_IK, self).__init__()
        self.name = "SMPLH"
        self.batch_size = batch_size
        num_betas = 10
        self.fit_body = fit_body
        self.pca_hands = pca_hands

        self.model = smplx.create(model_path="models/",
                                  model_type="smplh",
                                  gender="female",
                                  use_face_contour=False,
                                  num_betas=num_betas,
                                  ext='pkl',
                                  pitch=0.
                                  )
        self.faces = self.model.faces
        self.global_orient = torch.zeros((batch_size, 3))  # Not trained

        # Parameters
        if fit_body:
            self.betas = nn.Parameter(torch.zeros([batch_size, self.model.num_betas]))

        self.upper_body_pose = nn.Parameter(torch.zeros((batch_size, 10 * 3)))
        with torch.no_grad():
            # arm
            self.upper_body_pose[:, 37 - (11*3)] = -1
            self.upper_body_pose[:, 38 - (11*3)] = -0.75
            # elbow
            self.upper_body_pose[:, 52 - (11*3)] = -1.5
            self.upper_body_pose[:, 53 - (11*3)] = 0.5

            # arm
            self.upper_body_pose[:, 40 - (11*3)] = 1
            self.upper_body_pose[:, 41 - (11*3)] = 0.75
            # elbow
            self.upper_body_pose[:, 55 - (11*3)] = 1.5
            self.upper_body_pose[:, 56 - (11*3)] = -0.5

            # wrist
            self.upper_body_pose[:, 58 - (11*3)] = -1
            self.upper_body_pose[:, 61 - (11*3)] = 1

        num_hand_joints = 6 if pca_hands else len(hand_joint_indexes)
        self.left_hand_pose = nn.Parameter(torch.zeros((batch_size, num_hand_joints)))
        self.right_hand_pose = nn.Parameter(torch.zeros((batch_size, num_hand_joints)))

        self.transl = nn.Parameter(torch.tensor([[0., 0., -1.] for _ in range(batch_size)]))
        self.lower_spine_pose = nn.Parameter(torch.zeros((batch_size, 3)))
        self.middle_spine_pose = nn.Parameter(torch.zeros((batch_size, 3)))
        self.upper_spine_pose = nn.Parameter(torch.zeros((batch_size, 3)))

    def forward(self, return_verts=False):
        body_pose = torch.zeros((self.batch_size, self.model.NUM_BODY_JOINTS * 3))
        body_pose[:, 11*3:] = self.upper_body_pose
        #body_pose[:, 6:9] = self.lower_spine_pose
        #body_pose[:, 15:18] = self.middle_spine_pose
        body_pose[:, 24:27] = self.upper_spine_pose

        if self.pca_hands:
            right_hand_pose = self.right_hand_pose
            left_hand_pose = self.left_hand_pose
        else:
            right_hand_pose = torch.zeros((self.batch_size, self.model.NUM_HAND_JOINTS * 3))
            right_hand_pose[:, hand_joint_indexes] = self.right_hand_pose
            left_hand_pose = torch.zeros((self.batch_size, self.model.NUM_HAND_JOINTS * 3))
            left_hand_pose[:, hand_joint_indexes] = self.left_hand_pose

        expression = torch.zeros((self.batch_size, self.model.num_expression_coeffs))
        jaw_pose = torch.zeros((self.batch_size, 3))
        leye_pose = torch.zeros((self.batch_size, 3))
        reye_pose = torch.zeros((self.batch_size, 3))

        return self.model(betas=self.betas if self.fit_body else torch.zeros([self.batch_size, self.model.num_betas]),
                          body_pose=body_pose,
                          left_hand_pose=left_hand_pose,
                          right_hand_pose=right_hand_pose,
                          expression=expression,
                          jaw_pose=jaw_pose,
                          leye_pose=leye_pose,
                          reye_pose=reye_pose,
                          global_orient=self.global_orient,
                          transl=self.transl,
                          return_verts=return_verts,
                          return_full_pose=True,
                          pose2rot=True,)

    def head_loss(self):
        body_pose = torch.zeros((self.batch_size, self.model.NUM_BODY_JOINTS * 3))
        body_pose[:, 11 * 3:] = self.upper_body_pose

        return torch.sum(torch.abs(body_pose[:, [33, 34, 35, 43, 44]]))

    def hand_loss(self):
        if self.pca_hands:
            return 0

        right_hand_pose = torch.zeros((self.batch_size, self.model.NUM_HAND_JOINTS * 3))
        right_hand_pose[:, hand_joint_indexes] = self.right_hand_pose
        left_hand_pose = torch.zeros((self.batch_size, self.model.NUM_HAND_JOINTS * 3))
        left_hand_pose[:, hand_joint_indexes] = self.left_hand_pose

        # Penalise left hand joints greater than 0 and right hand joints less than 0
        left_hand_min_loss = torch.sum(left_hand_pose[:, hand_loss_indexes].clamp(min=0))
        right_hand_min_loss = torch.sum((-right_hand_pose[:, hand_loss_indexes]).clamp(min=0))

        #Penalise left hand joints less than -3 and right hand joints greater than 3
        left_hand_max_loss = torch.sum(left_hand_pose[:, hand_loss_indexes].clamp(max=-3))
        right_hand_max_loss = torch.sum((-right_hand_pose[:, hand_loss_indexes]).clamp(max=-3))

        return left_hand_min_loss + right_hand_min_loss + left_hand_max_loss + right_hand_max_loss


class VPOSER_IK(IKModel):
    def __init__(self, batch_size=1, fit_body=True, pca_hands=False):
        super(VPOSER_IK, self).__init__()
        from human_body_prior.body_model.body_model_vposer import BodyModelWithPoser
        from smplx.vertex_ids import vertex_ids as VERTEX_IDS
        from smplx.vertex_joint_selector import VertexJointSelector
        self.vertex_joint_selector = VertexJointSelector(#TODO Look at if necessary
            vertex_ids=VERTEX_IDS["smplh"], )
        expr_dir = 'vposer_v1_0'
        bm_path = 'models/smplh/SMPLH_FEMALE.pkl'  # 'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
        mano_dir = 'models/mano/'
        self.num_betas = 10
        self.model = BodyModelWithPoser(bm_path=bm_path,
                                        smpl_exp_dir=expr_dir,
                                        mano_exp_dir=mano_dir if pca_hands else None,
                                        num_betas=self.num_betas)
        self.batch_size = batch_size
        self.name = "SMPLH"
        self.fit_body = fit_body
        self.faces = None
        self.pca_hands = pca_hands
        self.sig = nn.Sigmoid()
        #Parameters
        if fit_body:
            self.betas = nn.Parameter(torch.zeros([batch_size, self.num_betas]))
        import numpy as np
        self.latent_body_pose = nn.Parameter(torch.tensor(np.random.normal(0., 1., size=(1, self.model.poser_body_ps.latentD)), dtype=torch.float32))#)torch.zeros((batch_size, self.model.poser_body_ps.latentD)))
        if self.pca_hands:
            self.latent_handl_pose = nn.Parameter(torch.zeros([self.batch_size, self.model.poser_handL_ps.latentD]))
            self.latent_handr_pose = nn.Parameter(torch.zeros([self.batch_size, self.model.poser_handL_ps.latentD]))
        self.transl = nn.Parameter(torch.tensor([[0., 0., -1.] for _ in range(batch_size)]))
        self.global_orient = torch.zeros((batch_size, 3))  # Not trained

    def forward(self, return_verts=False):
        result = self.model(betas=self.betas if self.fit_body else torch.zeros([self.batch_size, self.num_betas]),
                            poZ_body=self.latent_body_pose,#self.sig(self.latent_body_pose),
                            poZ_handL=self.latent_handl_pose if self.pca_hands else None,
                            poZ_handR=self.latent_handr_pose if self.pca_hands else None,
                            root_orient=self.global_orient,
                            trans=self.transl,
                            return_verts=return_verts,
                            return_full_pose=True,
                            pose2rot=True,)

        result.joints = self.vertex_joint_selector(result.v, result.Jtr)
        result.vertices = result.v
        if self.faces is None:
            self.faces = result.f
        if return_verts:
            result.transl = self.transl.detach()
        return result

    def head_loss(self):
        #return 0.
        #return self.latent_body_pose.pow(2).sum()
        std, mean = torch.std_mean(self.latent_body_pose)
        return torch.abs(mean) + torch.abs(1 - std)

        #return torch.sum(torch.abs(self.latent_body_pose))


class MANO_IK(IKModel):
    def __init__(self, body_pose=None, transl=None,  batch_size=1, fit_body=True, pca_hands=False):
        super(MANO_IK, self).__init__()
        self.name = "MANO"
        self.batch_size = batch_size
        num_betas = 10
        self.fit_body = fit_body
        self.pca_hands = pca_hands

        self.model = smplx.create(model_path="models/",
                                  model_type="smplh",
                                  gender="female",
                                  use_face_contour=False,
                                  num_betas=num_betas,
                                  ext='pkl',
                                  pitch=0.,
                                  use_pca=pca_hands
                                  )
        self.faces = self.model.faces
        self.global_orient = torch.zeros((batch_size, 3))  # Not trained

        # Parameters
        if fit_body:
            self.betas = nn.Parameter(torch.zeros([batch_size, self.model.num_betas]))

        self.body_pose = torch.zeros((batch_size, self.model.NUM_BODY_JOINTS * 3)) if body_pose is None else body_pose

        num_hand_joints = 6 if pca_hands else len(hand_joint_indexes)
        self.left_hand_pose = nn.Parameter(torch.zeros((batch_size, num_hand_joints)))
        self.right_hand_pose = nn.Parameter(torch.zeros((batch_size, num_hand_joints)))

        self.transl = torch.tensor([[0., 0., -1.] for _ in range(batch_size)]) if transl is None else transl

    def forward(self, return_verts=False):

        if self.pca_hands:
            right_hand_pose = self.right_hand_pose
            left_hand_pose = self.left_hand_pose
        else:
            right_hand_pose = torch.zeros((self.batch_size, self.model.NUM_HAND_JOINTS * 3))
            right_hand_pose[:, hand_joint_indexes] = self.right_hand_pose
            left_hand_pose = torch.zeros((self.batch_size, self.model.NUM_HAND_JOINTS * 3))
            left_hand_pose[:, hand_joint_indexes] = self.left_hand_pose

        expression = torch.zeros((self.batch_size, self.model.num_expression_coeffs))
        jaw_pose = torch.zeros((self.batch_size, 3))
        leye_pose = torch.zeros((self.batch_size, 3))
        reye_pose = torch.zeros((self.batch_size, 3))

        return self.model(betas=self.betas if self.fit_body else torch.zeros([self.batch_size, self.model.num_betas]),
                          body_pose=self.body_pose,
                          left_hand_pose=left_hand_pose,
                          right_hand_pose=right_hand_pose,
                          expression=expression,
                          jaw_pose=jaw_pose,
                          leye_pose=leye_pose,
                          reye_pose=reye_pose,
                          global_orient=self.global_orient,
                          transl=self.transl,
                          return_verts=return_verts,
                          return_full_pose=True,
                          pose2rot=True,)

    def head_loss(self):
        return 0

    def hand_loss(self):
        if self.pca_hands:
            return 0

        right_hand_pose = torch.zeros((self.batch_size, self.model.NUM_HAND_JOINTS * 3))
        right_hand_pose[:, hand_joint_indexes] = self.right_hand_pose
        left_hand_pose = torch.zeros((self.batch_size, self.model.NUM_HAND_JOINTS * 3))
        left_hand_pose[:, hand_joint_indexes] = self.left_hand_pose

        # Penalise left hand joints greater than 0 and right hand joints less than 0
        left_hand_min_loss = torch.sum(left_hand_pose[:, hand_loss_indexes].clamp(min=0))
        right_hand_min_loss = torch.sum((-right_hand_pose[:, hand_loss_indexes]).clamp(min=0))

        #Penalise left hand joints less than -3 and right hand joints greater than 3
        left_hand_max_loss = torch.sum(left_hand_pose[:, hand_loss_indexes].clamp(max=-3))
        right_hand_max_loss = torch.sum((-right_hand_pose[:, hand_loss_indexes]).clamp(max=-3))

        return left_hand_min_loss + right_hand_min_loss + left_hand_max_loss + right_hand_max_loss