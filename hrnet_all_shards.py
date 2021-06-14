log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth'
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1), ('val', 1),
('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1), ('val', 1),
('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1), ('val', 1),
('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1), ('val', 1),
('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1), ('val', 1),
('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1), ('val', 1),
('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1), ('val', 1),
('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1), ('val', 1),
('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1), ('val', 1),
('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1), ('val', 1),
('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1), ('val', 1),
('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1),('train', 1), ('val', 1)]
checkpoint_config = dict(interval=6)#After how many epochs to save
evaluation = dict(interval=12, metric='mAP', key_indicator='AP')#after how many epochs to evaluate

optimizer = dict(
    type='Adam',
    lr=1e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    #warmup=None,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10, 15])
total_epochs = 148
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=133,
    dataset_joints=133,
    dataset_channel=[
        list(range(133)),
    ],
    inference_channel=list(range(133)))

# model settings
model = dict(
    type='TopDown',
    pretrained='https://download.openmmlab.com/mmpose/'
    'pretrain_models/hrnet_w48-8ef0771d.pth',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
    ),
    keypoint_head=dict(
        type='TopDownSimpleHead',
        in_channels=48,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[288, 384],
    heatmap_size=[72, 96],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='/vol/research/SignRecognition/swisstxt/weather/2020-03-01.json',
)

train_pipeline = [
    dict(type='LoadImageFromlmdb'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=3),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromlmdb'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline
json_root = "/vol/research/SignTranslation/data/SWISSTXT/mmpose/annotations"
video_root = '/vol/research/SignTranslation/data/SWISSTXT/mmpose/lmdbs'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train_dataloader=dict(
        shuffle=False
    ),
    train=dict(
        type='TopDownCocoWholeBodyLazyDataset',
        ann_file=[f'{json_root}/shard_0.json', f'{json_root}/shard_1.json', f'{json_root}/shard_10.json', f'{json_root}/shard_100.json', f'{json_root}/shard_101.json', f'{json_root}/shard_102.json', f'{json_root}/shard_103.json', f'{json_root}/shard_104.json', f'{json_root}/shard_105.json', f'{json_root}/shard_106.json', f'{json_root}/shard_107.json', f'{json_root}/shard_108.json', f'{json_root}/shard_109.json', f'{json_root}/shard_11.json', f'{json_root}/shard_110.json', f'{json_root}/shard_111.json', f'{json_root}/shard_112.json', f'{json_root}/shard_113.json', f'{json_root}/shard_114.json', f'{json_root}/shard_115.json', f'{json_root}/shard_116.json', f'{json_root}/shard_117.json', f'{json_root}/shard_118.json', f'{json_root}/shard_119.json', f'{json_root}/shard_12.json', f'{json_root}/shard_120.json', f'{json_root}/shard_121.json', f'{json_root}/shard_122.json', f'{json_root}/shard_123.json', f'{json_root}/shard_124.json', f'{json_root}/shard_125.json', f'{json_root}/shard_126.json', f'{json_root}/shard_127.json', f'{json_root}/shard_128.json', f'{json_root}/shard_129.json', f'{json_root}/shard_13.json', f'{json_root}/shard_130.json', f'{json_root}/shard_131.json', f'{json_root}/shard_132.json', f'{json_root}/shard_133.json', f'{json_root}/shard_134.json', f'{json_root}/shard_135.json', f'{json_root}/shard_136.json', f'{json_root}/shard_137.json', f'{json_root}/shard_138.json', f'{json_root}/shard_139.json', f'{json_root}/shard_14.json', f'{json_root}/shard_140.json', f'{json_root}/shard_15.json', f'{json_root}/shard_16.json', f'{json_root}/shard_17.json', f'{json_root}/shard_18.json', f'{json_root}/shard_19.json', f'{json_root}/shard_2.json', f'{json_root}/shard_20.json', f'{json_root}/shard_21.json', f'{json_root}/shard_22.json', f'{json_root}/shard_23.json', f'{json_root}/shard_24.json', f'{json_root}/shard_25.json', f'{json_root}/shard_26.json', f'{json_root}/shard_27.json', f'{json_root}/shard_28.json', f'{json_root}/shard_29.json', f'{json_root}/shard_3.json', f'{json_root}/shard_30.json', f'{json_root}/shard_31.json', f'{json_root}/shard_32.json', f'{json_root}/shard_33.json', f'{json_root}/shard_34.json', f'{json_root}/shard_35.json', f'{json_root}/shard_36.json', f'{json_root}/shard_37.json', f'{json_root}/shard_38.json', f'{json_root}/shard_39.json', f'{json_root}/shard_4.json', f'{json_root}/shard_40.json', f'{json_root}/shard_41.json', f'{json_root}/shard_42.json', f'{json_root}/shard_43.json', f'{json_root}/shard_44.json', f'{json_root}/shard_45.json', f'{json_root}/shard_46.json', f'{json_root}/shard_47.json', f'{json_root}/shard_48.json', f'{json_root}/shard_49.json', f'{json_root}/shard_5.json', f'{json_root}/shard_50.json', f'{json_root}/shard_51.json', f'{json_root}/shard_52.json', f'{json_root}/shard_53.json', f'{json_root}/shard_54.json', f'{json_root}/shard_55.json', f'{json_root}/shard_56.json', f'{json_root}/shard_57.json', f'{json_root}/shard_58.json', f'{json_root}/shard_59.json', f'{json_root}/shard_6.json', f'{json_root}/shard_60.json', f'{json_root}/shard_61.json', f'{json_root}/shard_62.json', f'{json_root}/shard_63.json', f'{json_root}/shard_64.json', f'{json_root}/shard_65.json', f'{json_root}/shard_66.json', f'{json_root}/shard_67.json', f'{json_root}/shard_68.json', f'{json_root}/shard_69.json', f'{json_root}/shard_7.json', f'{json_root}/shard_70.json', f'{json_root}/shard_71.json', f'{json_root}/shard_72.json', f'{json_root}/shard_73.json', f'{json_root}/shard_74.json', f'{json_root}/shard_75.json', f'{json_root}/shard_76.json', f'{json_root}/shard_77.json', f'{json_root}/shard_78.json', f'{json_root}/shard_79.json', f'{json_root}/shard_8.json', f'{json_root}/shard_80.json', f'{json_root}/shard_81.json', f'{json_root}/shard_82.json', f'{json_root}/shard_83.json', f'{json_root}/shard_84.json', f'{json_root}/shard_85.json', f'{json_root}/shard_86.json', f'{json_root}/shard_87.json', f'{json_root}/shard_88.json', f'{json_root}/shard_89.json', f'{json_root}/shard_9.json', f'{json_root}/shard_90.json', f'{json_root}/shard_91.json', f'{json_root}/shard_92.json'],
        img_prefix=f'{video_root}/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='TopDownCocoWholeBodyLazyDataset',
        ann_file=[f'{json_root}/shard_93.json', f'{json_root}/shard_94.json', f'{json_root}/shard_95.json', f'{json_root}/shard_96.json', f'{json_root}/shard_97.json', f'{json_root}/shard_98.json', f'{json_root}/shard_99.json'],
        img_prefix=f'{video_root}/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='TopDownCocoWholeBodyLazyDataset',
        ann_file=f'{json_root}/shard_116.json',#shard_93
        img_prefix=f'{video_root}/',
        data_cfg=data_cfg,
        pipeline=test_pipeline),
)
