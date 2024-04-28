import torch
from torch import nn
from .backbone import build_backbone
from .fpn import build_fpn
from .pan import PAN
from .segment_predictor import SegmentPredictor
from .divide_predictor import HeadBodyDividePredictor
from .cells_extractor import CellsExtractor
from .decoder import Decoder
from .utils import extend_segments, spatial_att_to_spans

from .merger import Merger

from .fpn_text import build_fpn_text


class Model(nn.Module):
    def __init__(self, cfg, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.backbone = build_backbone(cfg.arch, cfg.pretrained_backbone, norm_layer=norm_layer)
        self.fpn = build_fpn(cfg.backbone_out_channels, cfg.fpn_out_channels)
        self.pan = PAN(cfg.pan_num_levels, cfg.pan_in_dim, cfg.pan_out_dim)
        self.row_segment_predictor = SegmentPredictor(cfg.fpn_out_channels, scale=cfg.rs_scale, type='row')
        self.col_segment_predictor = SegmentPredictor(cfg.fpn_out_channels, scale=cfg.cs_scale, type='col')
        self.divide_predictor = HeadBodyDividePredictor(cfg.fpn_out_channels, cfg.dp_head_nums, scale=cfg.dp_scale)
        self.cells_extractor = CellsExtractor(cfg.fpn_out_channels, cfg.ce_dim, cfg.ce_heads, cfg.ce_head_nums, cfg.ce_pool_size, cfg.ce_scale)

        self.merger = Merger(cfg.merger_dim, cfg.ce_dim * 3)
        # self.merger = Merger(cfg.merger_dim, cfg.ce_dim) # without AFA

        # text_mask
        self.fpn1 = build_fpn_text(cfg.backbone_out_channels, cfg.fpn_out_channels)
        self.backbone1 = build_backbone(cfg.arch, cfg.pretrained_backbone, norm_layer=norm_layer)

    def forward(self, images, images_size, cls_labels=None, labels_mask=None, layouts=None, rows_fg_spans=None,
        rows_bg_spans=None, cols_fg_spans=None, cols_bg_spans=None, cells_spans=None, divide_labels=None, merge_targets=None, text_masks=None, image_paths=None):

        # TQE
        text_feats = self.fpn1(self.backbone1(text_masks))
        feats = self.fpn(self.backbone(images), text_feats) 

        # feats = self.fpn(self.backbone(images))

        row_feats = torch.mean(feats[0], dim=3)
        result_info = dict()
        ext_info = dict()

        row_segments, rs_result_info, rs_ext_info = self.row_segment_predictor(feats[0], images_size, rows_fg_spans, rows_bg_spans)

        # row_segments, rs_result_info, rs_ext_info = self.row_segment_predictor(feats[3], images_size, rows_fg_spans, rows_bg_spans, text_feats[3])
        rs_result_info = {'row_%s' % key: val for key, val in rs_result_info.items()}
        rs_ext_info = {'row_%s' % key: val for key, val in rs_ext_info.items()}
        result_info.update(rs_result_info)
        ext_info.update(rs_ext_info)

        col_segments, cs_result_info, cs_ext_info = self.col_segment_predictor(feats[0], images_size, cols_fg_spans, cols_bg_spans)

        # col_segments, cs_result_info, cs_ext_info = self.col_segment_predictor(feats[3], images_size, cols_fg_spans, cols_bg_spans, text_feats[3])
        cs_result_info = {'col_%s' % key: val for key, val in cs_result_info.items()}
        cs_ext_info = {'col_%s' % key: val for key, val in cs_ext_info.items()}
        result_info.update(cs_result_info)
        ext_info.update(cs_ext_info)

        if self.training:
            row_segments, col_segments, cells_spans, layouts, divide_labels, merge_targets = extend_segments(row_segments, rs_ext_info['row_ext_segments'],
                col_segments, cs_ext_info['col_ext_segments'], cells_spans, layouts, divide_labels, merge_targets)

        divide_preds, dp_result_info, dp_ext_info = self.divide_predictor(row_feats, row_segments, divide_labels=divide_labels)
        result_info.update(dp_result_info)
        ext_info.update(dp_ext_info)

        # b,512,h,w  b,h,w 
        # feat_maps, feats_masks = self.cells_extractor(self.pan(feats), row_segments, col_segments, images_size, image_paths)
        feat_maps, feats_masks = self.cells_extractor(self.pan(feats), row_segments, col_segments, images_size)
        if self.training:
            assert feat_maps.shape[-2:] == layouts.shape[-2:], print('feat_maps is not the same with layouts') 

        # baseline
        spanses, merge_result_info = self.merger(feat_maps, feats_masks, merge_targets)
        result_info.update(merge_result_info)

        if not self.training:
            return (row_segments, col_segments, divide_preds, spanses), result_info
        else:
            return (row_segments, col_segments, divide_preds), result_info
