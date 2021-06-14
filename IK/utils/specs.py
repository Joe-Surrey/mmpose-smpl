ears = (
    "right ear",
    "left ear",
)
upper_body = (
    "head",
    "jaw",
    "right wrist",
    "left wrist",
    "right elbow",
    "left elbow",
    "right shoulder",
    "left shoulder",
) + ears
left_hand = (
    "left thumb end",
    "left index end",
    "left middle end",
    "left ring end",
    "left little end",

    "left little knuckle",
    "left little joint 1",
    "left little joint 2",
    "left ring knuckle",
    "left ring joint 1",
    "left ring joint 2",
    "left middle knuckle",
    "left middle joint 1",
    "left middle joint 2",
    "left index knuckle",
    "left index joint 1",
    "left index joint 2",
    "left thumb knuckle",
    "left thumb joint 1",
    "left thumb joint 2",
)
right_hand = (
    "right thumb end",
    "right index end",
    "right middle end",
    "right ring end",
    "right little end",

    "right little knuckle",
    "right little joint 1",
    "right little joint 2",
    "right ring knuckle",
    "right ring joint 1",
    "right ring joint 2",
    "right middle knuckle",
    "right middle joint 1",
    "right middle joint 2",
    "right index knuckle",
    "right index joint 1",
    "right index joint 2",
    "right thumb knuckle",
    "right thumb joint 1",
    "right thumb joint 2",
)


POINTS = {
    "SMPL": upper_body,# + ears,
    "SMPLH": upper_body + left_hand + right_hand,# + ears,
    "MANO":  left_hand + right_hand, #("head",) +
    "smpl": upper_body,
    "smplh": upper_body + left_hand + right_hand,# + ears,
    "mano": left_hand + right_hand,#("head",) +
    "ears": ("left ear"),
}

OUTPUT_SIZES = {
    "SMPL": 24,
    "SMPLH": 73,
    "MANO": 144
}

right_arm_chain = ["right ear",
    "head", "jaw",
] + ["right " + name for name in ("shoulder", "elbow", "wrist")]
left_arm_chain = ["left ear",
    "head", "jaw",
] + ["left " + name for name in ("shoulder", "elbow", "wrist")]
upper_body_chains = [right_arm_chain, left_arm_chain]

finger_chain = (
    "knuckle",
    "joint 1",
    "joint 2",
    "end"
)
fingers = (
    " thumb ",
    " index ",
    " middle ",
    " ring ",
    " little "
)

right_hand_chains = [["right wrist"] + ["right" + finger + name for name in finger_chain] for finger in fingers]
left_hand_chains = [["left wrist"] + ["left" + finger + name for name in finger_chain] for finger in fingers]
