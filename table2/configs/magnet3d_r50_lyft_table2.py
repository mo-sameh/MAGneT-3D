# Reproduces the Table 2 results of MAGneT-3D trained on Lyft.
_base_ = ['../../projects/configs/magnet3d/magnet3d_r50_lyft.py']
from projects.configs._base_.eval.benchmark_eval import test_set

model = dict(pts_bbox_head=dict(bbox_coder=dict(
    dedup_cfg=dict(mode='connected', sim_threshold=0.8, dist_threshold=3.0, dist_weight=0.5))))
eval_datasets = dict(
    nuscenes=test_set('nuscenes', (320, 864), tp_error_averaging='all_tp'),
    waymo=test_set('waymo', (320, 864), tp_error_averaging='all_tp'),
    lyft=test_set('lyft', 0.66, tp_error_averaging='all_tp'),
    once=test_set('once', 0.55, tp_error_averaging='all_tp'))
