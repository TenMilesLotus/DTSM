import torch
from torch import nn
from torch.nn import functional as F

class Merger(nn.Module):
    def __init__(self, merger_dim, ce_dim):
        super().__init__()
        self.merger_dim = merger_dim
        self.ce_dim = ce_dim
        self.fc1 = nn.Linear(ce_dim, merger_dim)
        self.fc2 = nn.Linear(merger_dim, 3)

    # feat_maps: b, c, h, w
    def forward(self, feat_maps, feats_masks, merge_targets):
        b, c, h, w = feat_maps.shape
        merge_result_info = dict()
        feat_maps = feat_maps * feats_masks
        feat_maps = feat_maps.view(b, c, -1).permute(0, 2, 1)  # b, h*w, c
        feat_maps = F.relu(self.fc1(feat_maps))
        feat_maps = self.fc2(feat_maps).view(-1, 3)  # b, h*w, 3
        if self.training:
            merge_targets = torch.tensor(merge_targets, device=feat_maps.device)
            # merge_targets = merge_targets.view(b, -1)  # b, h*w
            merge_targets = merge_targets.view(-1)  # b, h*w, 1
            merge_loss = F.cross_entropy(feat_maps, merge_targets, ignore_index=-1)
            merge_result_info.update(merge_loss=merge_loss)
        merge_maps = feat_maps.view(b, h, w, 3)  # b, h, w, 3
        probs = torch.softmax(merge_maps, dim=-1)
        _, merge_maps = torch.max(probs, dim=-1)  # b, h, w(012)
        spanses = []
        for merge_map in merge_maps:
            merge_map[0][0] = 0
            spans = []
            for i in range(len(merge_map)):  # row
                for j in range(len(merge_map[i])):  # col
                    if i == 0 and merge_map[0][j] == 2: merge_map[0][j] = 0
                    if j == 0 and merge_map[i][0] == 1: merge_map[i][0] = 0
                    if merge_map[i][j] == 0:
                        rowspan = 0
                        colspan = 0
                        for c in range(j+1, len(merge_map[i])):
                            if merge_map[i][c] == 1: colspan += 1
                            else: break
                        for r in range(i+1, len(merge_map)):
                            if merge_map[r][j] == 2: rowspan += 1
                            else: break
                        spans.append([i, j, i+rowspan, j+colspan])
            spanses.append(spans)
            
        return spanses, merge_result_info