import torch
import torch.nn as nn
import smplx
from IK.utils.SMPL_H_specs import hand_joint_indexes
from human_body_prior.body_model import body_model_vposer
from human_body_prior.body_model.body_model_vposer import BodyModelWithPoser
class Reprojection_Model(nn.Module):
    def __init__(self, device='cpu'):
        super(Reprojection_Model, self).__init__()
        self.device = device


class SMPLH_Reprojection(Reprojection_Model):
    def __init__(self, batch_size=1,
                 num_body_joints=21,
                 num_hand_joints=15,
                 num_expression_coeffs=0,
                 num_betas=10,
                 learn_betas=False,
                 fixed=False,
                 device='cpu'):
        super(SMPLH_Reprojection, self).__init__()
        self.model = smplx.create(model_path="models/",
                                               model_type="smplh",
                                               gender="female",
                                               use_face_contour=False,
                                               num_betas=num_betas,
                                               ext='pkl',
                                               )
        self.learn_betas = learn_betas
        self.fixed = fixed
        self.plot_mesh = False
        self.batch_size = batch_size
        self.num_betas = num_betas
        self.num_body_joints = num_body_joints
        self.num_hand_joints = num_hand_joints
        self.num_expression_coeffs = num_expression_coeffs
        self.faces = self.model.faces

        self.input_size = 76 + (10 if learn_betas else 0)

    def forward(self, input, return_verts=False):# target, target_indexes, confidences):
        # Make arrays
        betas = torch.zeros((self.batch_size, self.num_betas))
        body_pose = torch.zeros((self.batch_size, self.num_body_joints * 3))
        right_hand_pose = torch.zeros((self.batch_size, self.num_hand_joints * 3))
        left_hand_pose = torch.zeros((self.batch_size, self.num_hand_joints * 3))
        expression = torch.zeros((self.batch_size, self.num_expression_coeffs))
        jaw_pose = torch.zeros((self.batch_size, 3))
        leye_pose = torch.zeros((self.batch_size, 3))
        reye_pose = torch.zeros((self.batch_size, 3))
        global_orient = torch.zeros((self.batch_size, 3))
        transl = torch.zeros((self.batch_size, 3))

        # Plug in relevant values
        if self.fixed:
            transl = torch.tensor([-0.5415,  0.2051, -0.8414]*self.batch_size)
        else:
            transl[:, :2] = input[:, :2]
            transl[:, 2] = -torch.abs(input[:, 2])
        body_pose[:, 33:] = input[:, 3:33]  # Upper body
        body_pose[:, 24:27] = input[:, 33:36]  # Upper spine
        right_hand_pose[:, hand_joint_indexes] = input[:, 36:56]
        left_hand_pose[:, hand_joint_indexes] = input[:, 56:76]
        if self.learn_betas:
            betas = input[:, 76:86]

        # Put through model
        joints_3d = self.model(betas=betas,
                               body_pose=body_pose,
                               left_hand_pose=left_hand_pose,
                               right_hand_pose=right_hand_pose,
                               expression=expression,
                               jaw_pose=jaw_pose,
                               leye_pose=leye_pose,
                               reye_pose=reye_pose,
                               global_orient=global_orient,
                               transl=transl,
                               return_verts=return_verts,
                               return_full_pose=True,
                               pose2rot=True, )
        return joints_3d


class VPOSER_Reprojection(Reprojection_Model):
    def __init__(self, batch_size=1,
                 num_body_joints=21,
                 num_hand_joints=15,
                 num_expression_coeffs=0,
                 num_betas=10,
                 fixed=False,
                 learn_betas=False,
                 device='cpu'
                 ):
        super(VPOSER_Reprojection, self).__init__(device)

        expr_dir = 'vposer_v1_0'
        bm_path = 'models/smplh/SMPLH_FEMALE.pkl'  # 'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads
        self.num_betas = 10
        self.model = BodyModelWithPoser(bm_path=bm_path,
                                        smpl_exp_dir=expr_dir,
                                        num_betas=self.num_betas,
                                        batch_size=batch_size)
        from smplx.vertex_ids import vertex_ids as VERTEX_IDS
        from smplx.vertex_joint_selector import VertexJointSelector
        self.vertex_joint_selector = VertexJointSelector(  # TODO Look at if necessary
            vertex_ids=VERTEX_IDS["smplh"], )
        self.fixed = fixed
        self.plot_mesh = False
        self.batch_size = batch_size
        self.num_betas = num_betas
        self.num_body_joints = num_body_joints
        self.num_hand_joints = num_hand_joints
        self.num_expression_coeffs = num_expression_coeffs
        self.faces = None

        self.input_size = self.model.poser_body_ps.latentD + 3  # 3 for transl

    def forward(self, input, return_verts=False):
        # Make arrays
        betas = torch.zeros((self.batch_size, self.num_betas)).to(self.device)
        global_orient = torch.zeros((self.batch_size, 3)).to(self.device)
        transl = torch.zeros((self.batch_size, 3)).to(self.device)
        # Plug in relevant values
        if self.fixed:
            transl[:, :] = torch.tensor([-0.5415, 0.2051, -0.8414] * self.batch_size).to(self.device)
        else:
            transl[:, :2] = input[:, :2]
            transl[:, 2] = -torch.abs(input[:, 2])
        latent_body_pose = input[:, 3:3+self.model.poser_body_ps.latentD]

        result = self.model(betas=betas,
                            poZ_body=latent_body_pose,  # self.sig(self.latent_body_pose),
                            root_orient=global_orient,
                            trans=transl,
                            return_verts=return_verts,
                            return_full_pose=True,
                            pose2rot=True, )

        result.joints = self.vertex_joint_selector(result.v, result.Jtr)
        result.vertices = result.v
        if self.faces is None:
            self.faces = result.f
        return result


class MANO_Reprojection(Reprojection_Model):
    def __init__(self,
                 batch_size=1,
                 num_body_joints=21,
                 num_hand_joints=15,
                 num_expression_coeffs=0,
                 num_betas=10,
                 learn_betas=False,
                 fixed=False,
                 hand='left',
                 device='cpu'):
        super(MANO_Reprojection, self).__init__()
        self.model = smplx.create(model_path="models/",
                                               model_type="smplh",
                                               gender="female",
                                               use_face_contour=False,
                                               num_betas=num_betas,
                                               ext='pkl',
                                               )

        self.batch_size = batch_size
        self.hand = hand
        self.learn_betas = learn_betas
        self.fixed = fixed
        self.plot_mesh = False
        self.num_betas = num_betas
        self.num_body_joints = num_body_joints
        self.global_orient = torch.zeros((batch_size, 3))  # Not trained
        self.num_hand_joints = num_hand_joints
        self.num_expression_coeffs = num_expression_coeffs
        self.faces = self.model.faces

        self.input_size = 6

    def forward(self, input, transl, body_pose, betas=None, return_verts=False):# target, target_indexes, confidences):
        # Make arrays
        left_hand_pose = input if self.hand == 'left' else torch.zeros((self.batch_size, 6))
        right_hand_pose = input if self.hand == 'right' else torch.zeros((self.batch_size, 6))

        if betas is None:
            betas = torch.zeros([self.batch_size, self.model.num_betas])
        expression = torch.zeros((self.batch_size, self.model.num_expression_coeffs))
        jaw_pose = torch.zeros((self.batch_size, 3))
        leye_pose = torch.zeros((self.batch_size, 3))
        reye_pose = torch.zeros((self.batch_size, 3))

        return self.model(betas=betas,
                          body_pose=body_pose,
                          left_hand_pose=left_hand_pose,
                          right_hand_pose=right_hand_pose,
                          expression=expression,
                          jaw_pose=jaw_pose,
                          leye_pose=leye_pose,
                          reye_pose=reye_pose,
                          global_orient=self.global_orient,
                          transl=transl,
                          return_verts=return_verts,
                          return_full_pose=True,
                          pose2rot=True,)