import json
import numpy as np
import torch

import lmdb
import pickle
import cv2


from .specs import POINTS, OUTPUT_SIZES
from .coco_wholebody_specs import joints as keypoint_joints
from .SMPL_H_specs import joints as effector_joints
KEYPOINT_INDEXES = {
    "SMPL": (0, 10, 9, 8, 7, 6, 5),
    "SMPLH": (0, 10, 9, 8, 7, 6, 5,  # Body
              95, 99, 103, 107, 111,  # Left hand ends
              116, 120, 124, 128, 132,  # Right hand ends
              108, 109, 110, 104, 105, 106, 100, 101, 102, 96, 97, 98, 92, 93, 94,  # Left hand joints
              129, 130, 131, 125, 126, 127, 121, 122, 123, 117, 118, 119, 113, 114, 115,  # Right hand ends
              19, 16),  # ears
    "MANO": (0,
             95, 99, 103, 107, 111,  # Left hand ends
             116, 120, 124, 128, 132,  # Right hand ends
             108, 109, 110, 104, 105, 106, 100, 101, 102, 96, 97, 98, 92, 93, 94,  # Left hand joints
             129, 130, 131, 125, 126, 127, 121, 122, 123, 117, 118, 119, 113, 114, 115,)  # Right hand ends
}
EFFECTOR_INDEXES = {
    "SMPL": (12, 21, 20, 19, 18, 17, 16),  # 15 head 12 neck
    "SMPLH": (52, 21, 20, 19, 18, 17, 16,  # Body
              63, 64, 65, 66, 67,  # Left hand ends
              68, 69, 70, 71, 72,  # Right hand ends
              28, 29, 30, 31, 32, 33, 25, 26, 27, 22, 23, 24, 34, 35, 36,  # Left hand joints
              43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51,  # Right hand ends
              55, 56),  # ears
    "MANO": (12,
             63, 64, 65, 66, 67,  # Left hand ends
             68, 69, 70, 71, 72,  # Right hand ends
             28, 29, 30, 31, 32, 33, 25, 26, 27, 22, 23, 24, 34, 35, 36,  # Left hand joints
             43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51),  # Right hand ends
}

def load_target(file_name, index, model_type="SMPL", device="cpu", do_load_image=False,  base_path="", db=None):
    """
    Load COCO-wholebody coordinates from json file
    """
    with torch.no_grad():
        keypoints, confs, dimensions, image, frame_keypoints = read_json(file_name, index, do_load_image=do_load_image, base_path=base_path, db=db)

        effector_indexes = [effector_joints[joint_name] for joint_name in POINTS[model_type]]

        kp_indexes = [keypoint_joints[joint_name] for joint_name in POINTS[model_type]]

        unprojected_keypoints = torch.zeros((73, 3))
        target_indexes = []
        target = torch.zeros((1, 73, 3))  # torch.zeros((1, OUTPUT_SIZES[model_type], 3))
        confidences = torch.zeros((1, 1, 73,))  # torch.zeros((1, 1, OUTPUT_SIZES[model_type],))

        for kp_index, effector_index in zip(kp_indexes, effector_indexes):
            if confs[kp_index] is not None:  # > 0
                target_indexes.append(effector_index)
                target[0, effector_index, :] = keypoints[kp_index].reshape(3)
                unprojected_keypoints[effector_index, :] = frame_keypoints[kp_index].reshape(3)
                confidences[0, 0, effector_index] = confs[kp_index]

        return target_indexes, target.to(device), keypoints, image, unprojected_keypoints, confidences[:, :, target_indexes], dimensions


def read_json(file_name, index, do_load_image,  base_path="", db=None):
    # read json
    if isinstance(file_name, str):
        with open(file_name) as f:
            json_file = sorted(json.loads(f.read()), key=lambda x: int(x['image_paths'][0].split("%")[-1]))
    else:
        json_file = file_name
    frame = json_file[index]

    height, width = 720, 1280

    frame_keypoints = frame['preds'][0]


    confs = [item[2] for item in frame_keypoints]

    keypoints = torch.tensor(frame_keypoints)
    keypoints[:, 1] = height - keypoints[:, 1]

    dimensions = torch.tensor([[0, 0, 0], [width, height, 0]], dtype=torch.float32)

    #rescale so width is 1
    keypoints /= dimensions[1, 0]
    dimensions /= dimensions[1, 0].clone()

    dimensions[:, 2] = 0
    keypoints[:, 2] = 0

    camera_centre = torch.tensor([(dimensions[1][0] + dimensions[0][0]) / 2,
                                  (dimensions[1][1] + dimensions[0][1]) / 2,
                                  0], dtype=torch.float32)

    keypoints -= camera_centre  # keypoint_shoulder_centre
    dimensions -= camera_centre  # keypoint_shoulder_centre

    image = load_image(json_file, index,  base_path=base_path, db=db) if do_load_image else None
    return keypoints, confs, dimensions, image, torch.tensor(frame_keypoints)


def load_image(json_file, index, base_path="", db=None):
    image_file = json_file[index]['image_paths'][0]
    lmdb_path = base_path + image_file.split("%")[0] + ".lmdb"
    if db is None:
        db = lmdb.open(
                    path=lmdb_path,
                    readonly=True,
                    readahead=False,
                    max_spare_txns=128,
                    lock=False,
                )

    key = pickle.dumps("shard_" + image_file.split("shard_")[-1], protocol=3)

    with db.begin() as txn:
        value = txn.get(key=key)
        canvas = cv2.cvtColor(cv2.imdecode((pickle.loads(value)[1]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    return canvas
