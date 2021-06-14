import torch
import torch.nn as nn
import smplx
from IK.reprojection_models import SMPLH_Reprojection
from einops import rearrange


class reprojection_loss(nn.Module):

    def __init__(self, camera, batch_size=1, model=None, num_body_joints=21, num_hand_joints=15,
                 num_expression_coeffs=0, num_betas=10, learn_camera=False):
        super(reprojection_loss, self).__init__()
        if model is None:
            self.Model = SMPLH_Reprojection(batch_size, num_body_joints, num_hand_joints, num_expression_coeffs, num_betas)
        else:
            self.Model = model
        self.camera = camera
        self.learn_camera = learn_camera

    def forward(self, input, target, target_indexes, confidences, ):
        joints_3d = self.Model(input[:, :-1] if self.learn_camera else input)
        # project 3d - 2d
        projection = self.camera.project(joints_3d.joints,
                                         input_f=torch.abs(input[:, -1]) if self.learn_camera else None)
        # take euclidiean distance
        distance = euclidean_distance(projection, target, target_indexes, confs=confidences)
        return distance

class hand_reprojection_loss(nn.Module):

    def __init__(self, camera, batch_size=1, model=None, num_body_joints=21, num_hand_joints=15,
                 num_expression_coeffs=0, num_betas=10, learn_camera=False):
        super(hand_reprojection_loss, self).__init__()
        if model is None:
            self.Model = SMPLH_Reprojection(batch_size, num_body_joints, num_hand_joints, num_expression_coeffs,
                                            num_betas)
        else:
            self.Model = model
        self.camera = camera
        self.learn_camera = learn_camera

    def forward(self, input, target, target_indexes, confidences, transl, body_pose, f=None, betas=None):
        joints_3d = self.Model(input,
                               body_pose=body_pose, transl=transl, betas=betas)
        # project 3d - 2d
        projection = self.camera.project(joints_3d.joints, input_f=torch.abs(f if self.learn_camera else None))
        # take euclidiean distance
        distance = euclidean_distance(projection, target, target_indexes, confs=confidences)
        return distance


def euclidean_distance(input, target, target_indexes, confs=None):
    """
    Euclidean distance between each projected keypoint of its interest and its target
    """
    target = target[:, :, target_indexes]
    effectors = rearrange(input[:, target_indexes, :], 'b points xyz -> b xyz points')
    n = torch.nn.functional.pairwise_distance(effectors, target, keepdim=True)
    if confs is not None:
        n = n * confs
    return torch.sum(n)


def layer_loss(input, weight=10):
    """
    Loss based on the assumption that the elbows should be further forward than the shoulders and the hands further than
    the elbows
    """
    #get shoulders, elbows and hands
    left_shoulder = input[0, 16]
    left_elbow = input[0, 18]
    left_wrist = input[0, 20]
    right_shoulder = input[0, 17]
    right_elbow = input[0, 19]
    right_wrist = input[0, 21]

    losses = torch.zeros(4, dtype=torch.float32)

    losses[0] = left_shoulder[2] - left_elbow[2]
    losses[1] = left_elbow[2] - left_wrist[2]
    losses[2] = right_shoulder[2] - right_elbow[2]
    losses[3] = right_elbow[2] - right_wrist[2]

    loss = torch.sum(torch.clamp(losses, min=0))
    return loss * weight


