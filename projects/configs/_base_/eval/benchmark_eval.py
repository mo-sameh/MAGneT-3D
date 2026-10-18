# Cross-dataset test sets (nuScenes, Waymo, Lyft, ONCE val) for tools/eval_cross_dataset.py.
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
annotation_files = dict(
    nuscenes='joint/nuscenes_val.json',
    waymo='joint/waymo_val.json',
    lyft='lyft_filtered_val.json',
    once='joint/ONCE_val.json')


def test_set(name, final_dim, bot_pct_lim=(0.0, 0.0), **kwargs):
    """Test dataset config of one benchmark dataset at test resolution ``final_dim``;
    ``kwargs`` are passed to the dataset."""
    collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp',
                    'ego_pose', 'ego_pose_inv', 'lidar2img_inv']
    pipeline = [
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(type='ResizeCropFlipRotImage',
             data_aug_conf=dict(final_dim=final_dim, bot_pct_lim=bot_pct_lim, rot_lim=(0.0, 0.0),
                                H=-1, W=-1, rand_flip=False),
             training=False),
        dict(type='NormalizeMultiviewImage', **img_norm_cfg),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(type='InverseLidarToImage'),
        dict(type='MultiScaleFlipAug3D', img_scale=(1333, 800), pts_scale_ratio=1, flip=False,
             transforms=[
                 dict(type='BranchOutGTBoxes', training=False, prefix='monocular'),
                 dict(type='PETRFormatBundle3D', collect_keys=collect_keys, with_label=False),
                 dict(type='Collect3D', keys=['img'] + collect_keys,
                      meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                                 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                                 'img_norm_cfg', 'scene_token', 'GlobalRotScaleTrans')),
             ]),
    ]
    return dict(type='MonoVideoDataset', data_root='./data', annotation_file=annotation_files[name],
                pipeline=pipeline, test_mode=True, **kwargs)


# final_dim: a float rescales the full image; a tuple resizes and takes a bottom-aligned crop.
eval_datasets = dict(
    nuscenes=test_set('nuscenes', 0.66),
    waymo=test_set('waymo', (320, 864)),
    lyft=test_set('lyft', 0.66),
    once=test_set('once', 0.55))
