# MAGneT-3D with a ResNet-50 backbone (front camera only).
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
num_classes = 3

model = dict(
    type='MAGneT3D',
    num_frame_head_grads=1,
    num_frame_backbone_grads=1,
    num_frame_losses=1,
    position_level=1,
    use_grid_mask=False,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CPFPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=3),
    # auxiliary 2D head (training only)
    img_roi_head=dict(
        type='FocalHead',
        num_classes=num_classes,
        in_channels=256,
        loss_cls2d=dict(
            type='QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=2.0),
        loss_centerness=dict(
            type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
        loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
        loss_centers2d=dict(type='L1Loss', loss_weight=10.0),
        train_cfg=dict(
            assigner2d=dict(
                type='HungarianAssigner2D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0)))),
    # DRAG anchor generator; regress ranges and center radius are focal-normalised (px / 1266)
    drag_head=dict(
        type='DRAGHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        append_ray_map=True,
        normalized_regress_ranges=True,
        use_virtual_depth=True,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        pred_attrs=False,
        pred_velo=True,
        dir_offset=0.7854,
        strides=[8, 16, 32],
        group_reg_dims=(2, 1, 3, 1, 2),
        cls_branch=(256, ),
        reg_branch=((256, ), (256, ), (256, ), (256, ), ()),
        regress_ranges=((-1, 48 / 1266), (48 / 1266, 96 / 1266), (96 / 1266, 1e8)),
        center_sample_radius=1.5 / 1266,
        dir_branch=(256, ),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        norm_on_bbox=True,
        centerness_on_reg=True,
        center_sampling=True,
        conv_bias=True,
        dcn_on_last_conv=True,
        train_cfg=dict(
            allowed_border=0,
            code_weight=[1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05],
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.8,
            score_thr=0.05,
            min_bbox_size=0,
            max_per_img=200)),
    # temporal refiner with TRIM; all queries come from DRAG
    pts_bbox_head=dict(
        type='MAGneT3DHead',
        num_classes=num_classes,
        in_channels=256,
        num_query=0,
        memory_len=256,
        topk_proposals=64,
        num_propagated=64,
        with_ego_pos=True,
        match_with_velo=False,
        scalar=10,
        noise_scale=1.0,
        dn_weight=1.0,
        split=0.75,
        LID=True,
        with_position=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PETRTemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTemporalDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1),
                        dict(type='PETRMultiheadAttention', embed_dims=256, num_heads=8,
                             dropout=0.1, fp16=True)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=100,
            voxel_size=voxel_size,
            num_classes=num_classes,
            dedup_cfg=dict(sim_threshold=0.8, dist_threshold=3.0)),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_reid=dict(type='SupConLoss', temperature=0.15, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='SoftTopKAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                pc_range=point_cloud_range,
                top_k=5,
                radius=5.0,
                topk_decay=True))))
