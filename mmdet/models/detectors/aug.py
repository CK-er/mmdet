from .two_stage_new import NewTwoStageDetector
from ..builder import DETECTORS


@DETECTORS.register_module
class Aug_FasterRCNN(NewTwoStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None):
        super(Aug_FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)