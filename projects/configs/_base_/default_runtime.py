plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
dist_params = dict(backend='nccl')
log_level = 'INFO'
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
find_unused_parameters = False
