import mmcv
from mmcv.utils import Registry, build_from_cfg
from torch import nn

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
HEADS = Registry('head')
LOSSES = Registry('loss')
DETECTORS = Registry('detector')

# def _build_module(cfg, registry, default_args):
#     assert isinstance(cfg, dict) and 'type' in cfg
#     assert isinstance(default_args, dict) or default_args is None
#     args = cfg.copy()
#     obj_type = args.pop('type')
#     if mmcv.is_str(obj_type):
#         if obj_type not in registry.module_dict:
#             raise KeyError('{} is not in the {} registry'.format(
#                 obj_type, registry.name))
#         obj_type = registry.module_dict[obj_type]
#     elif not isinstance(obj_type, type):
#         raise TypeError('type must be a str or valid type, but got {}'.format(
#             type(obj_type)))
#     if default_args is not None:
#         for name, value in default_args.items():
#             args.setdefault(name, value)
#     return obj_type(**args)

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    return build(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg):
    return build(cfg, SHARED_HEADS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
