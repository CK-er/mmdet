import torch
import sys
import torch.nn as nn
import numpy as np

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms, multiclass_nms1)
from ..builder import HEADS, build_head, build_roi_extractor, build_loss
from .base_roi_head import BaseRoIHead
from .standard_roi_head import StandardRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
import torch.nn.functional as F

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit

@HEADS.register_module()
class CascadeD2DetRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 reg_roi_extractor=None,
                 d2det_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None):
        assert d2det_head is not None                  ## 改
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super(CascadeD2DetRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            # reg_roi_extractor=reg_roi_extractor,
            # d2det_head=d2det_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
        ##  改
        if reg_roi_extractor is not None:
            self.reg_roi_extractor = build_roi_extractor(reg_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.reg_roi_extractor = self.bbox_roi_extractor
        self.D2Det_head = build_head(d2det_head)

        self.loss_roi_reg = build_loss(dict(type='IoULoss', loss_weight=1.0))
        self.loss_roi_mask = build_loss(dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
        self.MASK_ON = d2det_head.MASK_ON
        self.num_classes = d2det_head.num_classes
        if self.MASK_ON:
            self.loss_roi_instance = build_loss(dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
            self.loss_iou = build_loss(dict(type='MSELoss', loss_weight=0.5))

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = nn.ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_assigner_sampler(self):
        # build assigner and smapler for each stage
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for rcnn_train_cfg in self.train_cfg:
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.bbox_sampler.append(build_sampler(rcnn_train_cfg.sampler))
###########改
    def init_weights(self, pretrained):
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        else:
            self.reg_roi_extractor.init_weights()
        self.D2Det_head.init_weights()
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_mask:
                if not self.share_roi_extractor:
                    self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()
 ############ 改
    def _random_jitter(self, sampling_results, img_metas, amplitude=0.15):
        """Ramdom jitter positive proposals for training."""
        for sampling_result, img_meta in zip(sampling_results, img_metas):
            bboxes = sampling_result.pos_bboxes
            random_offsets = bboxes.new_empty(bboxes.shape[0], 4).uniform_(
                -amplitude, amplitude)
            # before jittering
            cxcy = (bboxes[:, 2:4] + bboxes[:, :2]) / 2
            wh = (bboxes[:, 2:4] - bboxes[:, :2]).abs()
            # after jittering
            new_cxcy = cxcy + wh * random_offsets[:, :2]
            new_wh = wh * (1 + random_offsets[:, 2:])
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bboxes = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            max_shape = img_meta['img_shape']
            if max_shape is not None:
                new_bboxes[:, 0::2].clamp_(min=0, max=max_shape[1] - 1)
                new_bboxes[:, 1::2].clamp_(min=0, max=max_shape[0] - 1)

            sampling_result.pos_bboxes = new_bboxes
        return sampling_results

    def forward_dummy(self, x, proposals):
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                outs = outs + (mask_results['mask_pred'], )
        return outs

    def _bbox_forward(self, stage, x, rois):
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_first_forward_train(self, stage, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, gt_masks, rcnn_train_cfg):

        rois = bbox2roi([res.bboxes for res in sampling_results])     # deal with proposals bbox to roi
        bbox_results = self._bbox_forward(stage, x, rois)             # 调用_bbox_forward前向函数
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)   # 获得 gt 框？？？
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)

        # bbox_results = super(CascadeD2DetRoIHead,
        #                      self)._bbox_first_forward_train(x, sampling_results,        #   应该是继承了父类 standard_roi_head 中的 bbox_forward_train 前向函数
        #                                                gt_bboxes, gt_labels,
        #                                                img_metas, rcnn_train_cfg)

        #####dense local regression head ################################
        sampling_results = self._random_jitter(sampling_results, img_metas)
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])

        # print(self.reg_roi_extractor.num_inputs,gt_labels)
        reg_feats = self.reg_roi_extractor(
            x[:self.reg_roi_extractor.num_inputs], pos_rois)

        if self.with_shared_head:
            reg_feats = self.shared_head(reg_feats)
        # Accelerate training
        max_sample_num_reg = rcnn_train_cfg.get('max_num_reg', 192)                        # train_cfg
        sample_idx = torch.randperm(
            reg_feats.shape[0])[:min(reg_feats.shape[0], max_sample_num_reg)]
        reg_feats = reg_feats[sample_idx]
        pos_gt_labels = torch.cat([
            res.pos_gt_labels for res in sampling_results
        ])
        pos_gt_labels = pos_gt_labels[sample_idx]

        if self.MASK_ON == False:
            #####################instance segmentation############################
            if reg_feats.shape[0] == 0:
                bbox_results['loss_bbox'].update(dict(loss_reg=reg_feats.sum() * 0, loss_mask=reg_feats.sum() * 0))
            else:
                reg_pred, reg_masks_pred = self.D2Det_head(reg_feats)
                reg_points, reg_targets, reg_masks = self.D2Det_head.get_target(sampling_results)
                reg_targets = reg_targets[sample_idx]
                reg_points = reg_points[sample_idx]
                reg_masks = reg_masks[sample_idx]
                x1 = reg_points[:, 0, :, :] - reg_pred[:, 0, :, :] * reg_points[:, 2, :, :]
                x2 = reg_points[:, 0, :, :] + reg_pred[:, 1, :, :] * reg_points[:, 2, :, :]
                y1 = reg_points[:, 1, :, :] - reg_pred[:, 2, :, :] * reg_points[:, 3, :, :]
                y2 = reg_points[:, 1, :, :] + reg_pred[:, 3, :, :] * reg_points[:, 3, :, :]

                pos_decoded_bbox_preds = torch.stack([x1, y1, x2, y2], dim=1)
                # print(pos_decoded_bbox_preds.shape[0])
                pos_decoded_bbox_preds_ = pos_decoded_bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4)
                red_masks_ = reg_masks.reshape(-1)
                red_masks_index = np.nonzero(red_masks_)
                bbox_preds_ = pos_decoded_bbox_preds_[red_masks_index]
                # print(red_masks_)
                # print(red_masks_index)
                # print(bbox_preds_)
                # print(bbox_preds_.shape[0])


                x1_1 = reg_points[:, 0, :, :] - reg_targets[:, 0, :, :]
                x2_1 = reg_points[:, 0, :, :] + reg_targets[:, 1, :, :]
                y1_1 = reg_points[:, 1, :, :] - reg_targets[:, 2, :, :]
                y2_1 = reg_points[:, 1, :, :] + reg_targets[:, 3, :, :]

                pos_decoded_target_preds = torch.stack([x1_1, y1_1, x2_1, y2_1], dim=1)
                loss_reg = self.loss_roi_reg(
                    pos_decoded_bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4),
                    pos_decoded_target_preds.permute(0, 2, 3, 1).reshape(-1, 4),
                    weight=reg_masks.reshape(-1))

                loss_mask = self.loss_roi_mask(
                    reg_masks_pred.reshape(-1, reg_masks.shape[2] * reg_masks.shape[3]),
                    reg_masks.reshape(-1, reg_masks.shape[2] * reg_masks.shape[3]))
                bbox_results.update(bbox_pred=bbox_preds_)
                bbox_results['loss_bbox'].update(dict(loss_reg=loss_reg, loss_mask=loss_mask))
                #############################################
        else:
            #####################object detection############################
            reg_pred, reg_masks_pred, reg_instances_pred, reg_iou = self.D2Det_head(reg_feats, pos_gt_labels)           # d2det_head
            reg_points, reg_targets, reg_masks, reg_instances = self.D2Det_head.get_target_mask(sampling_results,
                                                                                                gt_masks,
                                                                                                rcnn_train_cfg)         # train_cfg

            reg_targets = reg_targets[sample_idx]
            reg_points = reg_points[sample_idx]
            reg_masks = reg_masks[sample_idx]
            reg_instances = reg_instances[sample_idx]

            x1 = reg_points[:, 0, :, :] - reg_pred[:, 0, :, :] * reg_points[:, 2, :, :]
            x2 = reg_points[:, 0, :, :] + reg_pred[:, 1, :, :] * reg_points[:, 2, :, :]
            y1 = reg_points[:, 1, :, :] - reg_pred[:, 2, :, :] * reg_points[:, 3, :, :]
            y2 = reg_points[:, 1, :, :] + reg_pred[:, 3, :, :] * reg_points[:, 3, :, :]

            pos_decoded_bbox_preds = torch.stack([x1, y1, x2, y2], dim=1)              # bbox prediction

            x1_1 = reg_points[:, 0, :, :] - reg_targets[:, 0, :, :]
            x2_1 = reg_points[:, 0, :, :] + reg_targets[:, 1, :, :]
            y1_1 = reg_points[:, 1, :, :] - reg_targets[:, 2, :, :]
            y2_1 = reg_points[:, 1, :, :] + reg_targets[:, 3, :, :]

            pos_decoded_target_preds = torch.stack([x1_1, y1_1, x2_1, y2_1], dim=1)    # gt
            loss_reg = self.loss_roi_reg(
                pos_decoded_bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4),             # 重新排列 x1 y1 x2 y2
                pos_decoded_target_preds.permute(0, 2, 3, 1).reshape(-1, 4),
                weight=reg_masks.reshape(-1))
            loss_mask = self.loss_roi_mask(
                reg_masks_pred.reshape(-1, reg_masks.shape[1] * reg_masks.shape[2]),
                reg_masks.reshape(-1, reg_masks.shape[1] * reg_masks.shape[2]))

            loss_instance = self.loss_roi_instance(reg_instances_pred, reg_instances, pos_gt_labels)
            reg_iou_targets = self.D2Det_head.get_target_maskiou(sampling_results, gt_masks,
                                                                 reg_instances_pred[pos_gt_labels >= 0, pos_gt_labels],
                                                                 reg_instances, sample_idx)
            reg_iou_weights = ((reg_iou_targets > 0.1) & (reg_iou_targets <= 1.0)).float()
            loss_iou = self.loss_iou(
                reg_iou[pos_gt_labels >= 0, pos_gt_labels],
                reg_iou_targets,
                weight=reg_iou_weights.reshape(-1))

            bbox_results['loss_bbox'].update(dict(loss_reg=loss_reg, loss_mask=loss_mask, loss_instance=loss_instance, loss_iou=loss_iou))
            #############################################
        return bbox_results

    def _bbox_second_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward(self, stage, x, rois):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_pred = mask_head(mask_feats)

        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            bbox_feats=None):
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        if len(pos_rois) == 0:
            # If there are no predicted and/or truth boxes, then we cannot
            # compute head / mask losses
            return dict(loss_mask=None)
        mask_results = self._mask_forward(stage, x, pos_rois)

        mask_targets = self.mask_head[stage].get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask)
        return mask_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        losses = dict()
        for i in range(self.num_stages):
            if i == 0:
                self.current_stage = i
                rcnn_train_cfg = self.train_cfg[i]
                lw = self.stage_loss_weights[i]

                # assign gts and sample proposals
                sampling_results = []
                if self.with_bbox or self.with_mask:
                    bbox_assigner = self.bbox_assigner[i]
                    bbox_sampler = self.bbox_sampler[i]
                    num_imgs = len(img_metas)
                    if gt_bboxes_ignore is None:
                        gt_bboxes_ignore = [None for _ in range(num_imgs)]

                    for j in range(num_imgs):
                        assign_result = bbox_assigner.assign(
                            proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                            gt_labels[j])
                        sampling_result = bbox_sampler.sample(
                            assign_result,
                            proposal_list[j],
                            gt_bboxes[j],
                            gt_labels[j],
                            feats=[lvl_feat[j][None] for lvl_feat in x])
                        sampling_results.append(sampling_result)

                # bbox head forward and loss
                bbox_results = self._bbox_first_forward_train(i, x, sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        img_metas, gt_masks, rcnn_train_cfg)
                for name, value in bbox_results['loss_bbox'].items():
                    losses[f's{i}.{name}'] = (
                            value * lw if 'loss' in name else value)
                # losses.update(bbox_results['loss_bbox'])

            # refine bboxes
            # if i < self.num_stages - 1:
            #     for key in bbox_results.items():
            #         print(key)
            #     print(bbox_results['bbox_pred'].size())

                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    # proposal_list = self.bbox_head[i].refine_bboxes(
                    #     bbox_results['rois'], roi_labels,
                    #     bbox_results['bbox_pred'], pos_is_gts, img_metas)
                    proposal_list = bbox_results['bbox_pred']


            else:
                self.current_stage = i
                rcnn_train_cfg = self.train_cfg[i]
                lw = self.stage_loss_weights[i]

                # assign gts and sample proposals
                sampling_results = []
                if self.with_bbox or self.with_mask:
                    bbox_assigner = self.bbox_assigner[i]
                    bbox_sampler = self.bbox_sampler[i]
                    num_imgs = len(img_metas)
                    if gt_bboxes_ignore is None:
                        gt_bboxes_ignore = [None for _ in range(num_imgs)]

                    for j in range(num_imgs):
                        assign_result = bbox_assigner.assign(
                            proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                            gt_labels[j])
                        sampling_result = bbox_sampler.sample(
                            assign_result,
                            proposal_list[j],
                            gt_bboxes[j],
                            gt_labels[j],
                            feats=[lvl_feat[j][None] for lvl_feat in x])
                        sampling_results.append(sampling_result)


                bbox_results = self._bbox_second_forward_train(i, x, sampling_results,
                                                            gt_bboxes, gt_labels,
                                                            rcnn_train_cfg)
                for name, value in bbox_results['loss_bbox'].items():
                    losses[f's{i}.{name}'] = (
                            value * lw if 'loss' in name else value)
        return losses
                # # refine bboxes
                # if i < self.num_stages - 1:
                #     pos_is_gts = [res.pos_is_gt for res in sampling_results]
                #     # bbox_targets is a tuple
                #     roi_labels = bbox_results['bbox_targets'][0]
                #     with torch.no_grad():
                #         proposal_list = self.bbox_head[i].refine_bboxes(
                #             bbox_results['rois'], roi_labels,
                #             bbox_results['bbox_pred'], pos_is_gts, img_metas)

        # losses = dict()
        # for i in range(self.num_stages):
        #     self.current_stage = i
        #     rcnn_train_cfg = self.train_cfg[i]
        #     lw = self.stage_loss_weights[i]
        #
        #     # assign gts and sample proposals
        #     sampling_results = []
        #     if self.with_bbox or self.with_mask:
        #         bbox_assigner = self.bbox_assigner[i]
        #         bbox_sampler = self.bbox_sampler[i]
        #         num_imgs = len(img_metas)
        #         if gt_bboxes_ignore is None:
        #             gt_bboxes_ignore = [None for _ in range(num_imgs)]
        #
        #         for j in range(num_imgs):
        #             assign_result = bbox_assigner.assign(
        #                 proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
        #                 gt_labels[j])
        #             sampling_result = bbox_sampler.sample(
        #                 assign_result,
        #                 proposal_list[j],
        #                 gt_bboxes[j],
        #                 gt_labels[j],
        #                 feats=[lvl_feat[j][None] for lvl_feat in x])
        #             sampling_results.append(sampling_result)
        #
        #     # bbox head forward and loss
        #     bbox_results = self._bbox_forward_train(i, x, sampling_results,
        #                                             gt_bboxes, gt_labels,
        #                                             rcnn_train_cfg)
        #
        #     for name, value in bbox_results['loss_bbox'].items():
        #         losses[f's{i}.{name}'] = (
        #             value * lw if 'loss' in name else value)
        #
        #     # mask head forward and loss
        #     if self.with_mask:
        #         mask_results = self._mask_forward_train(
        #             i, x, sampling_results, gt_masks, rcnn_train_cfg,
        #             bbox_results['bbox_feats'])
        #         # TODO: Support empty tensor input. #2280
        #         if mask_results['loss_mask'] is not None:
        #             for name, value in mask_results['loss_mask'].items():
        #                 losses[f's{i}.{name}'] = (
        #                     value * lw if 'loss' in name else value)
        #
        #     # refine bboxes
        #     if i < self.num_stages - 1:
        #         pos_is_gts = [res.pos_is_gt for res in sampling_results]
        #         # bbox_targets is a tuple
        #         roi_labels = bbox_results['bbox_targets'][0]
        #         with torch.no_grad():
        #             proposal_list = self.bbox_head[i].refine_bboxes(
        #                 bbox_results['rois'], roi_labels,
        #                 bbox_results['bbox_pred'], pos_is_gts, img_metas)
        #
        # return losses


    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        # det_bboxes, det_labels = self.simple_test_bboxes(
        #     x, img_metas, proposal_list, self.test_cfg, rescale=False)



     #  stage = 1
        #####  simple_test_bboxes ############
        rois = bbox2roi(proposal_list)
        # proposal = torch.Tensor(proposal_list)
        # print('proposal', proposal_list)
        # print('rois', rois)
        # print('proposal', proposal.size)
        # print('rois', rois.shape)
        bbox_results = self._bbox_forward(0, x, rois)
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        rcnn_test_cfg = self.test_cfg
        det_bboxes, det_labels = self.bbox_head[0].get_bboxes(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        # return det_bboxes, det_labels


        # print(torch.sum(det_labels==0))
        if det_bboxes.shape[0] != 0:
            reg_rois = bbox2roi([det_bboxes[:, :4]])
            reg_feats = self.reg_roi_extractor(
                x[:len(self.reg_roi_extractor.featmap_strides)], reg_rois)
            self.D2Det_head.test_mode = True

            ####dense local regression predection############
            reg_pred, reg_pred_mask = self.D2Det_head(reg_feats)

            num_rois = det_bboxes.shape[0]
            map_size = 7
            points = torch.zeros((num_rois, 4, map_size, map_size), dtype=torch.float)
            for j in range(map_size):
                y = det_bboxes[:, 1] + (det_bboxes[:, 3] - det_bboxes[:, 1]) / map_size * (j + 0.5)

                # dy = (det_bboxes[:, 3] - det_bboxes[:, 1]) / (map_size - 1)
                for i in range(map_size):
                    x = det_bboxes[:, 0] + (det_bboxes[:, 2] - det_bboxes[:, 0]) / map_size * (i + 0.5)

                    # dx = (det_bboxes[:, 2] - det_bboxes[:, 0]) / (map_size - 1)

                points[:, 0, j, i] = x
                points[:, 1, j, i] = y
                points[:, 2, j, i] = det_bboxes[:, 2] - det_bboxes[:, 0]
                points[:, 3, j, i] = det_bboxes[:, 3] - det_bboxes[:, 1]  # points = (x,y,w,h)   四维张量（num_rois, 4, 7, 7）

            points = points.cuda()
            reg_points = points
            # reg_points = self.D2Det_head.get_target(det_bboxes[:, :4])

            x1 = reg_points[:, 0, :, :] - reg_pred[:, 0, :, :] * reg_points[:, 2, :, :]
            x2 = reg_points[:, 0, :, :] + reg_pred[:, 1, :, :] * reg_points[:, 2, :, :]
            y1 = reg_points[:, 1, :, :] - reg_pred[:, 2, :, :] * reg_points[:, 3, :, :]
            y2 = reg_points[:, 1, :, :] + reg_pred[:, 3, :, :] * reg_points[:, 3, :, :]

            pos_decoded_bbox_preds = torch.stack([x1, y1, x2, y2], dim=1)
            pos_decoded_bbox_preds_ = pos_decoded_bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4)
            print(pos_decoded_bbox_preds_)
            # det_bboxes = self.D2Det_head.get_bboxes_avg(det_bboxes,
            #                                             reg_pred,
            #                                             reg_pred_mask,
            #                                             img_metas)


            # det_bboxes, det_labels = multiclass_nms1(det_bboxes[:, :4], det_bboxes[:, 4], det_labels,
            #                                              self.num_classes, dict(type='soft_nms', iou_thr=0.5), 300)

            if rescale:
                scale_factor = det_bboxes.new_tensor(img_metas[0]['scale_factor'])
                det_bboxes[:, :4] /= scale_factor
        else:
            det_bboxes = torch.Tensor([])
            # segm_result = [[] for _ in range(self.num_classes)]
            # mask_scores = [[] for _ in range(self.num_classes)]


    # stage = 1
        proposal_list = list()
        proposal_list.append(pos_decoded_bbox_preds_)#.cpu().numpy().tolist()
        # proposal_list[:, [0, 1, 2, 3, 4]] = proposal_list[:, [4, 0, 1, 2, 3]]
        # print('a ', proposal_list.shape)
        print(proposal_list)
        rois = bbox2roi(proposal_list)
        print('b ', rois.shape)
        bbox_results = self._bbox_forward(1, x, rois)
        det_bboxes, det_labels = self.bbox_head[-1].get_bboxes(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)

        bbox_results = bbox2result(det_bboxes, det_labels, self.bbox_head[-1].num_classes)
        print(bbox_results)
        # if self.MASK_ON:
        #     return bbox_results, (segm_result, mask_scores)
        # for key in bbox_results.items():
        #     print(key)
        return bbox_results





        # rois = bbox2roi(proposal_list)
        # for i in range(self.num_stages):
        #     bbox_results = self._bbox_forward(i, x, rois)
        #     ms_scores.append(bbox_results['cls_score'])
        #
        #     if i < self.num_stages - 1:
        #         bbox_label = bbox_results['cls_score'].argmax(dim=1)
        #         rois = self.bbox_head[i].regress_by_class(
        #             rois, bbox_label, bbox_results['bbox_pred'], img_metas[0])
        #
        # cls_score = sum(ms_scores) / self.num_stages
        # det_bboxes, det_labels = self.bbox_head[-1].get_bboxes(
        #     rois,
        #     cls_score,
        #     bbox_results['bbox_pred'],
        #     img_shape,
        #     scale_factor,
        #     rescale=rescale,
        #     cfg=rcnn_test_cfg)
        # bbox_result = bbox2result(det_bboxes, det_labels,
        #                           self.bbox_head[-1].num_classes)
        # ms_bbox_result['ensemble'] = bbox_result
        #
        # if self.with_mask:
        #     if det_bboxes.shape[0] == 0:
        #         mask_classes = self.mask_head[-1].num_classes
        #         segm_result = [[] for _ in range(mask_classes)]
        #     else:
        #         _bboxes = (
        #             det_bboxes[:, :4] * det_bboxes.new_tensor(scale_factor)
        #             if rescale else det_bboxes)
        #
        #         mask_rois = bbox2roi([_bboxes])
        #         aug_masks = []
        #         for i in range(self.num_stages):
        #             mask_results = self._mask_forward(i, x, mask_rois)
        #             aug_masks.append(
        #                 mask_results['mask_pred'].sigmoid().cpu().numpy())
        #         merged_masks = merge_aug_masks(aug_masks,
        #                                        [img_metas] * self.num_stages,
        #                                        self.test_cfg)
        #         segm_result = self.mask_head[-1].get_seg_masks(
        #             merged_masks, _bboxes, det_labels, rcnn_test_cfg,
        #             ori_shape, scale_factor, rescale)
        #     ms_segm_result['ensemble'] = segm_result
        #
        # if self.with_mask:
        #     results = (ms_bbox_result['ensemble'], ms_segm_result['ensemble'])
        # else:
        #     results = ms_bbox_result['ensemble']
        #
        # return results


    # def aug_test(self, x, proposal_list, img_metas, rescale=False):
    #     """Test with augmentations.
    #
    #     If rescale is False, then returned bboxes and masks will fit the scale
    #     of imgs[0].
    #     """
    #     # recompute feats to save memory
    #     det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
    #                                                   proposal_list,
    #                                                   self.test_cfg)
    #
    #     if rescale:
    #         _det_bboxes = det_bboxes
    #     else:
    #         _det_bboxes = det_bboxes.clone()
    #         _det_bboxes[:, :4] *= det_bboxes.new_tensor(
    #             img_metas[0][0]['scale_factor'])
    #     bbox_results = bbox2result(_det_bboxes, det_labels,
    #                                self.bbox_head.num_classes)
    #
    #     # det_bboxes always keep the original scale
    #     if self.with_mask:
    #         segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
    #                                           det_labels)
    #         return bbox_results, segm_results
    #     else:
    #         return bbox_results


    # def aug_test(self, features, proposal_list, img_metas, rescale=False):
    #     """Test with augmentations.
    #
    #     If rescale is False, then returned bboxes and masks will fit the scale
    #     of imgs[0].
    #     """
    #     rcnn_test_cfg = self.test_cfg
    #     aug_bboxes = []
    #     aug_scores = []
    #     for x, img_meta in zip(features, img_metas):
    #         # only one image in the batch
    #         img_shape = img_meta[0]['img_shape']
    #         scale_factor = img_meta[0]['scale_factor']
    #         flip = img_meta[0]['flip']
    #         flip_direction = img_meta[0]['flip_direction']
    #
    #         proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
    #                                  scale_factor, flip, flip_direction)
    #         # "ms" in variable names means multi-stage
    #         ms_scores = []
    #
    #         rois = bbox2roi([proposals])
    #         for i in range(self.num_stages):
    #             bbox_results = self._bbox_forward(i, x, rois)
    #             ms_scores.append(bbox_results['cls_score'])
    #
    #             if i < self.num_stages - 1:
    #                 bbox_label = bbox_results['cls_score'].argmax(dim=1)
    #                 rois = self.bbox_head[i].regress_by_class(
    #                     rois, bbox_label, bbox_results['bbox_pred'],
    #                     img_meta[0])
    #
    #         cls_score = sum(ms_scores) / float(len(ms_scores))
    #         bboxes, scores = self.bbox_head[-1].get_bboxes(
    #             rois,
    #             cls_score,
    #             bbox_results['bbox_pred'],
    #             img_shape,
    #             scale_factor,
    #             rescale=False,
    #             cfg=None)
    #         aug_bboxes.append(bboxes)
    #         aug_scores.append(scores)
    #
    #     # after merging, bboxes will be rescaled to the original image size
    #     merged_bboxes, merged_scores = merge_aug_bboxes(
    #         aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
    #     det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
    #                                             rcnn_test_cfg.score_thr,
    #                                             rcnn_test_cfg.nms,
    #                                             rcnn_test_cfg.max_per_img)
    #
    #     bbox_result = bbox2result(det_bboxes, det_labels,
    #                               self.bbox_head[-1].num_classes)
    #
    #     if self.with_mask:
    #         if det_bboxes.shape[0] == 0:
    #             segm_result = [[]
    #                            for _ in range(self.mask_head[-1].num_classes)]
    #         else:
    #             aug_masks = []
    #             aug_img_metas = []
    #             for x, img_meta in zip(features, img_metas):
    #                 img_shape = img_meta[0]['img_shape']
    #                 scale_factor = img_meta[0]['scale_factor']
    #                 flip = img_meta[0]['flip']
    #                 flip_direction = img_meta[0]['flip_direction']
    #                 _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
    #                                        scale_factor, flip, flip_direction)
    #                 mask_rois = bbox2roi([_bboxes])
    #                 for i in range(self.num_stages):
    #                     mask_results = self._mask_forward(i, x, mask_rois)
    #                     aug_masks.append(
    #                         mask_results['mask_pred'].sigmoid().cpu().numpy())
    #                     aug_img_metas.append(img_meta)
    #             merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
    #                                            self.test_cfg)
    #
    #             ori_shape = img_metas[0][0]['ori_shape']
    #             segm_result = self.mask_head[-1].get_seg_masks(
    #                 merged_masks,
    #                 det_bboxes,
    #                 det_labels,
    #                 rcnn_test_cfg,
    #                 ori_shape,
    #                 scale_factor=1.0,
    #                 rescale=False)
    #         return bbox_result, segm_result
    #     else:
    #         return bbox_result

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid()
        else:
            mask_pred = det_bboxes.new_tensor(mask_pred)

        device = mask_pred.device
        cls_segms = [[] for _ in range(self.num_classes)
                     ]  # BG is not included in num_classes
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        if not isinstance(scale_factor, (float, torch.Tensor)):
            scale_factor = bboxes.new_tensor(scale_factor)
        bboxes = bboxes / scale_factor

        N = len(mask_pred)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == 'cpu':
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            num_chunks = int(
                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
            assert (num_chunks <=
                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = 0.5
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8)

        mask_pred = mask_pred[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_pred[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == 'cpu')

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds,) + spatial_inds] = masks_chunk

        for i in range(N):
            cls_segms[labels[i]].append(im_mask[i].cpu().numpy())
        return cls_segms

    def get_mask_scores(self, mask_iou_pred, det_bboxes, det_labels):
        """Get the mask scores.

        mask_score = bbox_score * mask_iou
        """
        inds = range(det_labels.size(0))
        mask_scores = 0.3 * mask_iou_pred[inds, det_labels] + det_bboxes[inds, -1]
        mask_scores = mask_scores.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        return [
            mask_scores[det_labels == i] for i in range(self.num_classes)
        ]

def _do_paste_mask(masks, boxes, img_h, img_w, skip_empty=True):
    """Paste instance masks acoording to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(
        y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(
        x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    if torch.isinf(img_x).any():
        inds = torch.where(torch.isinf(img_x))
        img_x[inds] = 0
    if torch.isinf(img_y).any():
        inds = torch.where(torch.isinf(img_y))
        img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()
