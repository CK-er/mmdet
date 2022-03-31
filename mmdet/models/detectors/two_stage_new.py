import torch
import torch.nn as nn

from .base import BaseDetector
from ..roi_heads.test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..builder import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
import numpy as np


@DETECTORS.register_module
class NewTwoStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(NewTwoStageDetector, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
            ######################################  改（start）  ################################################################
            if 'use_consistent_supervision' in self.train_cfg.rcnn:
                self.use_consistent_supervision = self.train_cfg.rcnn.use_consistent_supervision
            else:
                self.use_consistent_supervision = False

            if self.use_consistent_supervision:
                bbox_roi_extractor['type'] = 'AuxAllLevelRoIExtractor'
                self.auxiliary_bbox_roi_extractor = builder.build_roi_extractor(
                    bbox_roi_extractor)
                bbox_head['type'] = 'AuxiliarySharedFCBBoxHead'
                self.auxiliary_bbox_head = builder.build_head(bbox_head)
        ######################################  改（end）  ################################################################
        if mask_head is not None:
            self.mask_roi_extractor = builder.build_roi_extractor(
                mask_roi_extractor)
            self.mask_head = builder.build_head(mask_head)
            ######################################  改(start)  ################################################################
            if self.use_consistent_supervision:
                mask_roi_extractor['type'] = 'AuxAllLevelRoIExtractor'
                self.auxiliary_mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.auxiliary_mask_head = builder.build_head(mask_head)
        ######################################  改(end)  ################################################################
        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(NewTwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_roi_extractor.init_weights()
            self.mask_head.init_weights()

    def extract_feat(self, img):

        x = self.backbone(img)
        if self.with_neck:
            ######################################  改  ################################################################
            if self.use_consistent_supervision:
                x, y = self.neck(x)
                return x, y
            else:
                x = self.neck(x)
                return x

    ######################################  改  ################################################################

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        ######################################  改  ################################################################
        if self.use_consistent_supervision:
            x, y = self.extract_feat(img)
        else:
            x = self.extract_feat(img)

        losses = dict()
        ######################################  改  ################################################################
        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)
            ######################################  改  ################################################################
            # loss of consistent supervison
            if self.use_consistent_supervision:
                bbox_feats_raw = self.auxiliary_bbox_roi_extractor(y[:self.bbox_roi_extractor.num_inputs], rois)
                cls_score_auxiliary, bbox_pred_auxiliary = self.auxiliary_bbox_head(bbox_feats_raw)

                loss_bbox_auxiliary = self.auxiliary_bbox_head.loss(cls_score_auxiliary, bbox_pred_auxiliary,
                                                                    *bbox_targets, alpha=self.train_cfg.rcnn.alpha)
                ######################################  改  ################################################################

                losses.update(loss_bbox_auxiliary)
        # mask head forward and loss
        if self.with_mask:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], pos_rois)
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)

            ######################################  改  ################################################################
            # loss of consistent supervision in mask rcnn. It bring marginal improvement of mask.
            if self.use_consistent_supervision and self.train_cfg.rcnn.mask_auxiliary:
                mask_feats_raw = self.auxiliary_mask_roi_extractor(y[:self.mask_roi_extractor.num_inputs], pos_rois)
                mask_pred_auxiliary = self.auxiliary_mask_head(mask_feats_raw)

                loss_mask_auxiliary = self.auxiliary_mask_head.loss_aux(mask_pred_auxiliary, mask_targets,
                                                                        pos_labels, alpha=self.train_cfg.rcnn.alpha)
                losses.update(loss_mask_auxiliary)

            losses.update(loss_mask)
        ######################################  改  ################################################################
        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        ######################################  改  ################################################################
        if self.use_consistent_supervision:
            x, y = self.extract_feat(img)
        else:
            x = self.extract_feat(img)
        ######################################  改  ################################################################
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

