# Monocular StreamPETR baseline, ResNet-50, trained on nuScenes (front camera).
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/streampetr_r50.py',
    '../_base_/eval/streampetr_eval.py',
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# class order of the model outputs
class_names = ['pedestrian', 'vehicle', 'bicycle']

model = dict(use_grid_mask=True, pts_bbox_head=dict(bbox_coder=dict(max_num=300)))

dataset_type = 'MonoVideoDataset'
data_root = './data'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp',
                'ego_pose', 'ego_pose_inv']
ida_aug_conf = dict(resize_lim=(0.38, 0.55), final_dim=(256, 704), bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0), H=900, W=1600, rand_flip=False)
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_bbox=True, with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ParameterFreeObjectNameFilter'),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d',
               'depths', 'prev_exists'] + collect_keys,
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token',
                    'gt_bboxes_3d', 'gt_labels_3d')),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        annotation_file='joint/nuscenes_train.json',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        sequence_length=20),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

# 4 GPUs x 4 samples per GPU
num_gpus = 4
num_iters_per_epoch = 28130 // (num_gpus * 4)
num_epochs = 24
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
