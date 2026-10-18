# Reproduces the Table 2 results of MAGneT-3D trained on nuScenes.
_base_ = ['../../projects/configs/magnet3d/magnet3d_r50_nuscenes.py']
from projects.configs._base_.eval.benchmark_eval import test_set

eval_datasets = dict(
    nuscenes=test_set('nuscenes', 0.66, tp_error_averaging='all_tp'),
    waymo=test_set('waymo', 0.66, tp_error_averaging='all_tp'),
    lyft=test_set('lyft', 0.66, tp_error_averaging='all_tp'),
    once=test_set('once', 0.66, tp_error_averaging='all_tp'))
