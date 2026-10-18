# MAGneT-3D, ResNet-50, trained on Lyft (front camera).
# Trained end to end in one stage: tools/train.py <this config>
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/magnet3d_r50.py',
    '../_base_/eval/benchmark_eval.py',
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# class order of the model outputs
class_names = ['bicycle', 'vehicle', 'pedestrian']

dataset_type = 'MonoVideoDataset'
data_root = './data'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp',
                'ego_pose', 'ego_pose_inv', 'lidar2img_inv']

# input resolutions are cycled frame by frame within each training sequence
ida_aug_configs = [
    dict(resize_lim=(0.38, 0.55), final_dim=(320, 512), bot_pct_lim=(0.0, 0.05),
         rot_lim=(0.0, 0.0), H=-1, W=-1, rand_flip=False),
    dict(resize_lim=(0.55, 0.77), final_dim=(512, 768), bot_pct_lim=(0.0, 0.0),
         rot_lim=(0.0, 0.0), H=-1, W=-1, rand_flip=False),
    dict(resize_lim=(0.38, 0.55), final_dim=(288, 608), bot_pct_lim=(0.0, 0.0),
         rot_lim=(0.0, 0.0), H=-1, W=-1, rand_flip=False),
]

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_bbox=True, with_label=True, with_bbox_depth=True),
    dict(type='AlignBoxes'),
    dict(type='RangeFilter', point_cloud_range=point_cloud_range),
    dict(type='MultiResizeCropFlipRotImage', data_aug_configs=ida_aug_configs, training=True),
    dict(type='BranchOutGTBoxes', prefix='monocular'),
    dict(type='InverseLidarToImage'),
    dict(type='PhotoMetricDistortion3D', brightness_delta=32, contrast_range=(0.8, 1.2),
         saturation_range=(0.8, 1.2), hue_delta=18),
    dict(type='MultiCamRandomGamma', gamma_range=(0.85, 1.15)),
    dict(type='MultiCamGaussianNoise', noise_std=0.01),
    dict(type='GlobalRotScaleTransImage', rot_range=[-0.3925, 0.3925], translation_std=[0, 0, 0],
         scale_ratio_range=[1, 1], reverse_angle=True, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='ObjectCameraFrustumFilter', use_center=False),
    dict(type='FilterUnmatchedBboxes', gt_prefix='monocular'),
    dict(type='PETRFormatBundle3D', collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D',
         keys=['monocular_gt_bboxes_3d', 'monocular_gt_labels_3d', 'gt_bboxes_3d', 'gt_labels_3d',
               'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists'] + collect_keys,
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token',
                    'monocular_gt_bboxes_3d', 'monocular_gt_labels_3d', 'gt_bboxes_3d',
                    'gt_labels_3d', 'scene_token', 'lidar2cam', 'GlobalRotScaleTrans',
                    'mixup_seed')),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        annotation_file='lyft_filtered_train.json',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        sequence_length=20,
        limit_sequence_length=True,
        branched_gt_prefix='monocular'),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

# 2 GPUs x 8 samples per GPU
num_gpus = 2
num_iters_per_epoch = 15498 // (num_gpus * 8)
num_epochs = 43
optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.25))),
    weight_decay=0.01)
optimizer_config = dict(
    type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
evaluation = dict(interval=num_epochs * num_iters_per_epoch)
