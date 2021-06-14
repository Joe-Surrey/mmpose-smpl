from torch.utils.data import Dataset
import torch
import json as json
import lmdb
import pickle
import cv2
from IK.utils.specs import POINTS, OUTPUT_SIZES
from IK.utils.coco_wholebody_specs import joints as keypoint_joints
from IK.utils.SMPL_H_specs import joints as effector_joints
from IK.reprojection_models import SMPLH_Reprojection
from IK.camera import PerspectiveCamera
from IK.model import Model

from einops import rearrange, repeat
import numpy as np


class Shard(Dataset):
    def __init__(self,
                 model_type="SMPLH",
                 path="/vol/research/SignRecognition/swisstxt/lmdb_results/video_hrnet.json",
                 base_path="",
                 repeat=False,
                 sort_confidences=True,
                 index=None,
                 transforms=None,
                 device='cpu'):
        self.index = index
        self.device=device
        self.index = index
        self.repeat = repeat
        self.out_size = OUTPUT_SIZES["SMPLH"]

        self.transforms = transforms

        with open(base_path + path) as f:
            #self.json_file = sorted(json.loads(f.read()), key=lambda x: int(x['image_paths'][0].split("%")[-1]))
            self.json_file = json.loads(f.read())
            if sort_confidences:
                self.json_file = sorted(self.json_file,
                                        key=lambda x: (sum([item[2] for item in x["preds"][0][91:]])),
                                        reverse=True)
            else:
                self.json_file = sorted(self.json_file, key=lambda x: int(x['image_paths'][0].split("%")[-1]))
            if self.repeat:
                self.json_file = self.json_file[:1000]

        image_file = self.json_file[0]['image_paths'][0]
        shard_name, w_or_n, video_file, frame = image_file.split("%")
        lmdb_path = base_path + shard_name + ".lmdb"

        self.db = lmdb.open(
            path=lmdb_path,
            readonly=True,
            readahead=False,
            max_spare_txns=128,
            lock=False,
        )

        self.effector_indexes = [effector_joints[joint_name] for joint_name in POINTS[model_type]]

        self.kp_indexes = [keypoint_joints[joint_name] for joint_name in POINTS[model_type]]

    def __getitem__(self, index):
        if self.index is not None:
            index = self.index
        annotation = self.json_file[index]
        image = self.load_image(annotation)

        height, width = image.shape[:2]

        image = torch.tensor(self.transforms(image), dtype=torch.float32)

        keypoints, all_confidences = self.load_target(annotation, height, width)
        target = torch.zeros((self.out_size, 3))
        confidences = torch.zeros((self.out_size,))
        for kp_index, effector_index in zip(self.kp_indexes, self.effector_indexes):
            target[effector_index, :] = keypoints[kp_index].reshape(3)
            confidences[effector_index] = all_confidences[kp_index]
        return (rearrange(image, 'w h c -> c w h').to(self.device),
                rearrange(target, 'points xyz -> xyz points').to(self.device),
                self.effector_indexes,
                repeat(confidences[self.effector_indexes], 'conf -> () conf').to(self.device))

    def __len__(self):
        return len(self.json_file)

    def load_image(self, annotation):
        image_file = annotation['image_paths'][0]
        key = pickle.dumps("shard_" + image_file.split("shard_")[-1], protocol=3)

        with self.db.begin() as txn:
            value = txn.get(key=key)
        return cv2.cvtColor(cv2.imdecode((pickle.loads(value)[1]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    @staticmethod
    def load_target(annotation, height, width):
        frame_keypoints = annotation['preds'][0]

        confs = torch.tensor([item[2] for item in frame_keypoints])

        keypoints = torch.tensor(frame_keypoints)
        keypoints[:, 1] = height - keypoints[:, 1]

        dimensions = torch.tensor([[0, 0, 0], [width, height, 0]], dtype=torch.float32)

        # rescale so width is 1
        keypoints /= width
        dimensions /= width

        dimensions[:, 2] = 0
        keypoints[:, 2] = 0

        camera_centre = torch.tensor([0.5, (height/width) / 2, 0], dtype=torch.float32)

        keypoints -= camera_centre
        return keypoints, confs


class HandShard(Shard):
    def __init__(self,
                 reprojection_model,
                 model=None,
                 camera=None,
                 hand="left",
                 model_type="MANO",
                 path="/vol/research/SignRecognition/swisstxt/lmdb_results/video_hrnet.json",
                 base_path="",
                 repeat=False,
                 index=None,
                 train_camera=False,
                 transforms=None):
        super(HandShard, self).__init__(model_type, path, base_path, repeat, index, transforms)
        self.index = index
        self.hand = hand
        self.train_camera = train_camera
        if isinstance(camera, str):
            self.camera = PerspectiveCamera()
            if self.train_camera:
                self.camera.load_state_dict(torch.load(camera))
        else:
            self.camera = camera

        self.reprojection_model = reprojection_model
        if isinstance(model, str):
            self.m = Model(output_features=reprojection_model.input_size + (1 if train_camera else 0))
            self.m.load_state_dict(torch.load(model))
        else:
            self.m = model

        self.all_effector_indexes = [effector_joints[joint_name] for joint_name in POINTS["SMPLH"]]
        self.target_effector_indexes = [effector_joints[joint_name] for joint_name in POINTS[model_type]]

        self.kp_indexes = [keypoint_joints[joint_name] for joint_name in POINTS["SMPLH"]]

    def __getitem__(self, index):
        # Get wrist coordinates
        if self.index is not None:
            index = self.index

        annotation = self.json_file[index]
        orig_image = self.load_image(annotation)  # h w c

        height, width = orig_image.shape[:2]


        image = torch.tensor(self.transforms(orig_image), dtype=torch.float32)

        keypoints, all_confidences = self.load_target(annotation, height, width)

        confidences = torch.zeros((self.out_size,))
        target = torch.zeros((self.out_size, 3))
        for kp_index, effector_index in zip(self.kp_indexes, self.all_effector_indexes):
            target[effector_index, :] = keypoints[kp_index].reshape(3)
            confidences[effector_index] = all_confidences[kp_index]

        image = repeat(image, 'h w c -> b c h w', b=self.reprojection_model.batch_size)

        angles = self.m(image)
        points_3d = self.reprojection_model(angles[:, :-1] if self.train_camera else angles)
        points_2d = self.camera.project(points_3d.joints, input_f=angles[:, -1].detach() if self.train_camera else None)
        wrist = points_2d[0, effector_joints[f"{self.hand} wrist"]].clone()

        #target = target - wrist

        wrist[1] = -wrist[1]
        wrist = (wrist * width) + torch.tensor([width/2, height/2, 0])
        cropped_image = torch.zeros((3, 224, 224))
        # crop image
        x = torch.round(wrist[0]).long()
        if x < 0 or x >= width:
            x = width//2
        y = torch.round(wrist[1]).long()
        if y < 0 or y >= height:
            y = height//2
        orig_image = rearrange(orig_image, 'h w c -> c h w')
        cropped = torch.tensor(orig_image[:, max(0, y-112):min(height, y+112), max(0, x-112):min(width, 112+x)])
        cropped_image[:, max(0, -(y - 112)):224 + min(0, height - (y + 112)),
        max(0, -(x - 112)):224 + min(0, width - (112 + x))] = cropped

        return (
            cropped_image.to(self.device),
            rearrange(target, 'points xyz -> xyz points').to(self.device),
            self.target_effector_indexes,
            repeat(confidences[self.target_effector_indexes], 'conf -> () conf').to(self.device),
            angles[0, :3].detach(),
            points_3d.pose_body[0].detach(),
            angles[0, -1].detach()
        )


def collate(data):
    images, targets, target_indexes, confidences = list(zip(*data))
    images = torch.stack(images)
    targets = torch.stack(targets)
    target_indexes = target_indexes[0]
    confidences = torch.stack(confidences)
    return images, targets, target_indexes, confidences


def collate_hand(data):
    images, targets, target_indexes, confidences, transl, body_pose, f = list(zip(*data))
    images = torch.stack(images)
    targets = torch.stack(targets)
    target_indexes = target_indexes[0]
    confidences = torch.stack(confidences)
    transl = torch.stack(transl)
    body_pose = torch.stack(body_pose)
    f = torch.stack(f)
    return images, targets, target_indexes, confidences, transl, body_pose, f


