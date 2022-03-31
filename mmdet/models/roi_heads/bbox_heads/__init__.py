from .bbox_head import BBoxHead
from .convfc_bbox_head import (ConvFCBBoxHead, Shared2FCBBoxHead,
                               Shared4Conv1FCBBoxHead)
from .double_bbox_head import DoubleConvFCBBoxHead
from .convfc_bbox_head_auxiliary import AuxiliaryBBoxHead, AuxiliaryConvFCBBoxHead, AuxiliarySharedFCBBoxHead

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead',
    'AuxiliaryBBoxHead', 'AuxiliaryConvFCBBoxHead', 'AuxiliarySharedFCBBoxHead'
]
