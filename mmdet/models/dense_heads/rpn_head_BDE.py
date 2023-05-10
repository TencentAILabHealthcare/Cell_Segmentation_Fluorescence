# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmdet.core import images_to_levels, multi_apply
from mmcv.runner import force_fp32

from ..builder import HEADS
from .anchor_head import AnchorHead

from sklearn.cluster import KMeans

@HEADS.register_module()
class RPNHeadBDE(AnchorHead):
    """RPN head with BDE 
    (https://link.springer.com/chapter/10.1007/978-3-030-59722-1_31)

    Args:
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        num_convs (int): Number of convolution layers in the head. Default 1.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                 num_convs=1,
                 **kwargs):
        self.num_convs = num_convs
        super(RPNHeadBDE, self).__init__(
            1, in_channels, init_cfg=init_cfg, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(
                self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4,
                                 1)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=False)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred
    
    def split_group(self, bboxes_preds, group_num):
        """ 
            Kmeans based bboxes group split
            bboxes_pred: (batch_size, N, 4)
            group_num: int 
        """
        b, n, _ = bboxes_preds.shape
        device = bboxes_preds.device
        group_labels = torch.zeros((b, n), device=device)
        for i in range(b):
            kmeans = KMeans(n_clusters=group_num, random_state=0).fit(bboxes_preds.detach().cpu().numpy())
            group_labels[i] = torch.from_numpy(kmeans.labels_).to(device=device)
        return group_labels

    @torch.no_grad()
    def boxes_density_energy(self, bbox_preds, anchors):
        """boxes density energy"""
        bs, n, h, w = bbox_preds.shape
        bbox_preds = bbox_preds.permute(0, 2, 3, 1).reshape(-1, 4)
        anchors = anchors.reshape(-1, 4)
        bbox_preds = self.bbox_coder.decode(anchors, bbox_preds)
        bbox_preds = bbox_preds.reshape(bs, -1, 4)

        # box energy estimation
        bbox_center_x = (bbox_preds[:, :, 0] + bbox_preds[:, :, 1]) / 2 
        bbox_center_y = (bbox_preds[:, :, 2] + bbox_preds[:, :, 3]) / 2 
        bbox_center = torch.stack([bbox_center_x, bbox_center_y], dim=-1)
        
        # use cpu to avoid OOM
        # device = bbox_center.device
        # bbox_center_cpu = bbox_center.cpu()
        # dist_mat = torch.cdist(bbox_center_cpu, bbox_center_cpu, p=1)
        # bboxes_density = dist_mat.mean(dim=-1).to(device=device)
        # max_bboxes_dist = dist_mat.reshape(bs, -1).max(dim=1, keepdim=True)[0].to(device=device)
        
        dist_mat = torch.cdist(bbox_center, bbox_center, p=1)
        bboxes_density = dist_mat.mean(dim=-1)
        max_bboxes_dist = dist_mat.reshape(bs, -1).max(dim=1, keepdim=True)[0]
        
        bboxes_energy = bboxes_density / max_bboxes_dist

        bboxes_energy = torch.stack([bboxes_energy, bboxes_energy, bboxes_energy, bboxes_energy], dim=-1)

        return (bboxes_energy, )

    @torch.no_grad()
    def fast_boxes_density_energy(self, bbox_preds, anchors):
        """boxes density energy"""
        bs, n, h, w = bbox_preds.shape
        bbox_preds = bbox_preds.clone().permute(0, 2, 3, 1).reshape(-1, 4)
        anchors = anchors.reshape(-1, 4)
        bbox_preds = self.bbox_coder.decode(anchors, bbox_preds)
        bbox_preds = bbox_preds.reshape(bs, -1, 4)

        bs, n, _ = bbox_preds.shape
        bboxes_density = torch.zeros_like(bbox_preds).to(device=bbox_preds.device)
        for b in range(bs):
            bbox_pred = bbox_preds[b].int()
            min_h, _, min_w, _ = bbox_pred.min(dim=0)[0]
            _, max_h, _, max_w = bbox_pred.max(dim=0)[0]
            len_h = max_h - min_h
            len_w = max_w - min_w
            bbox_pred[:, 0:2] -= min_h
            bbox_pred[:, 2:4] -= min_w
            counter = torch.zeros(len_h, len_w).to(device=bbox_preds.device)
            # import pdb;pdb.set_trace()
            for i in range(n):
                counter[bbox_pred[i, 0]:bbox_pred[i, 1], bbox_pred[i, 2]:bbox_pred[i, 3]] += 1
            for i in range(n):
                bboxes_density[b, i, :] = counter[bbox_pred[i, 0]:bbox_pred[i, 1], bbox_pred[i, 2]:bbox_pred[i, 3]].mean()

        bboxes_energy = bboxes_density / bboxes_density.max(dim=1, keepdim=True)[0]
        # # box energy estimation
        # bbox_center_x = (bbox_preds[:, :, 0] + bbox_preds[:, :, 1]) / 2 
        # bbox_center_y = (bbox_preds[:, :, 2] + bbox_preds[:, :, 3]) / 2 
        # bbox_center = torch.stack([bbox_center_x, bbox_center_y], dim=-1)
        
        # dist_mat = torch.cdist(bbox_center, bbox_center, p=1)
        # bboxes_density = dist_mat.mean(dim=-1)
        # max_bboxes_dist = dist_mat.reshape(bs, -1).max(dim=1, keepdim=True)[0]
        
        # bboxes_energy = bboxes_density / max_bboxes_dist

        # bboxes_energy = torch.stack([bboxes_energy, bboxes_energy, bboxes_energy, bboxes_energy], dim=-1)

        return (bboxes_energy, )

    def show_boxes(self, cls_scores, bbox_preds, gt_bboxes, anchors, img_metas):
        import cv2
        import numpy as np
        idx = 0
        gt = np.zeros((*img_metas[idx]['img_shape'][:2], 3), dtype=np.uint8)
        pred_bbox = np.zeros((*img_metas[idx]['img_shape'][:2], 3), dtype=np.uint8)
        pred_center = np.zeros((*img_metas[idx]['img_shape'][:2], 3), dtype=np.uint8)
        # decode bbox
        bbox_preds = bbox_preds[0][idx].permute(1, 2, 0).reshape(-1, 4)
        anchors = anchors[0][idx].reshape(-1, 4)
        bbox_preds = self.bbox_coder.decode(anchors, bbox_preds)
        cls_scores = cls_scores[0][idx].permute(1, 2, 0).flatten()
        bbox_preds = bbox_preds[torch.argsort(cls_scores, descending=True)]

        # group bbox
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=len(gt_bboxes[0])*2, random_state=0).fit(bbox_preds.detach().cpu().numpy())
        import matplotlib.colors as mcolors
        colors = (np.array([mcolors.to_rgb(color) for color in mcolors.CSS4_COLORS]) * 255).astype(int)
        bbox_group_color = colors[kmeans.labels_]

        bbox_center_x = (bbox_preds[:, 0] + bbox_preds[:, 1]) / 2 
        bbox_center_y = (bbox_preds[:, 2] + bbox_preds[:, 3]) / 2 
        bbox_center = torch.stack([bbox_center_x, bbox_center_y], dim=-1)
        kmeans = KMeans(n_clusters=len(gt_bboxes[0])*2, random_state=0).fit(bbox_center.detach().cpu().numpy())
        center_group_color = colors[kmeans.labels_]

        for i in range(len(bbox_preds[:1000])):
            bbox = bbox_preds[:1000][i]
            bbox = torch.round(bbox).int().cpu().tolist()
            cv2.rectangle(pred_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_group_color[i].tolist(), 1)
            cv2.rectangle(pred_center, (bbox[0], bbox[1]), (bbox[2], bbox[3]), center_group_color[i].tolist(), 1)

        # gt
        for bbox in gt_bboxes[0]:
            bbox = torch.round(bbox).int().cpu().tolist()
            cv2.rectangle(gt, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)
        cv2.imwrite(
            "/mnt/zihanwu/wentaopan/cell_instance_segmentation/cell_instance_segmentation/test.png", 
            np.concatenate((gt, pred_bbox, pred_center), axis=1))

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=None,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        # import time
        # start = time.time()
        results = multi_apply(
            self.boxes_density_energy,
            bbox_preds,
            all_anchor_list)
        energy_list = results[0]

        # re_start = time.time()
        # multi_apply(
        #     self.fast_boxes_density_energy,
        #     bbox_preds,
        #     all_anchor_list)
        # energy_list = results[0]
        # print("energy use time: {}/{}".format(re_start-start, time.time()-re_start))

        # calibrate cls loss in background
        for scale in range(len(energy_list)):
            # Foreground is the first class since v2.5.0
            bg_inds = labels_list[scale] == 1
            label_weights_list[scale][bg_inds] = energy_list[scale][:, :, 0][bg_inds]

        # print("all use time: {}".format(time.time()-start))
        # self.show_boxes(cls_scores, bbox_preds, gt_bboxes, all_anchor_list, img_metas)



        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

        # losses = super(RPNHeadBDE, self).loss(
        #     cls_scores,
        #     bbox_preds,
        #     gt_bboxes,
        #     None,
        #     img_metas,
        #     gt_bboxes_ignore=gt_bboxes_ignore)
        # return dict(
        #     loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_anchors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has
                shape (num_anchors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RPN head does not need this value.
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_anchors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']

        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            anchors = mlvl_anchors[level_idx]
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))

        return self._bbox_post_process(mlvl_scores, mlvl_bbox_preds,
                                       mlvl_valid_anchors, level_ids, cfg,
                                       img_shape)

    def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors,
                           level_ids, cfg, img_shape, **kwargs):
        """bbox post-processing method.

        Do the nms operation for bboxes in same level.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            mlvl_valid_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_bboxes, 4).
            level_ids (list[Tensor]): Indexes from all scale levels of a
                single image, each item has shape (num_bboxes, ).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, `self.test_cfg` would be used.
            img_shape (tuple(int)): The shape of model's input image.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        if proposals.numel() > 0:
            dets, _ = batched_nms(proposals, scores, ids, cfg.nms)
        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg.max_per_img]

    def onnx_export(self, x, img_metas):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.
        Returns:
            Tensor: dets of shape [N, num_det, 5].
        """
        cls_scores, bbox_preds = self(x)

        assert len(cls_scores) == len(bbox_preds)

        batch_bboxes, batch_scores = super(RPNHeadBDE, self).onnx_export(
            cls_scores, bbox_preds, img_metas=img_metas, with_nms=False)
        # Use ONNX::NonMaxSuppression in deployment
        from mmdet.core.export import add_dummy_nms_for_onnx
        cfg = copy.deepcopy(self.test_cfg)
        score_threshold = cfg.nms.get('score_thr', 0.0)
        nms_pre = cfg.get('deploy_nms_pre', -1)
        # Different from the normal forward doing NMS level by level,
        # we do NMS across all levels when exporting ONNX.
        dets, _ = add_dummy_nms_for_onnx(batch_bboxes, batch_scores,
                                         cfg.max_per_img,
                                         cfg.nms.iou_threshold,
                                         score_threshold, nms_pre,
                                         cfg.max_per_img)
        return dets
