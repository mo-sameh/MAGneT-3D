import numpy as np
import torch
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet3d.core.bbox.coders import FCOS3DBBoxCoder
from mmdet3d.core.bbox.structures import limit_period


@BBOX_CODERS.register_module()
class DRAGBBoxCoder(FCOS3DBBoxCoder):
    """FCOS3D box coder for DRAG.

    DRAG predicts boxes in the (pseudo-)LiDAR frame, so the local-to-global yaw
    conversion uses ``atan2(fx, u - cx)`` instead of ``atan2(u - cx, fx)``.
    """

    @staticmethod
    def decode_yaw(bbox, centers2d, dir_cls, dir_offset, cam2img):
        if bbox.shape[0] > 0:
            dir_rot = limit_period(bbox[..., 6] - dir_offset, 0, np.pi)
            bbox[..., 6] = \
                dir_rot + dir_offset + np.pi * dir_cls.to(bbox.dtype)

        bbox[:, 6] = torch.atan2(cam2img[0, 0],
                                 centers2d[:, 0] - cam2img[0, 2]) + bbox[:, 6]
        return bbox
