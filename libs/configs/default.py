import os
import torch
from libs.utils.vocab import Vocab

device = torch.device('cuda')

train_lrcs_path = [
    "/home/cxh/cxh/Datasets/DenseTab/train/dtsm_label/table.lrc"
]
train_data_dir = '/home/cxh/cxh/Datasets/DenseTab/train/img'
train_max_pixel_nums = 4000 * 4000
train_bucket_seps = (50, 50, 50)
train_max_batch_size = 1
train_num_workers = 0

valid_lrc_path = '/home/cxh/cxh/Datasets/DenseTab/test_A/dtsm_label/table.lrc'
valid_data_dir = '/home/cxh/cxh/Datasets/DenseTab/test_A/img'

valid_num_workers = 0
valid_batch_size = 1

vocab = Vocab()

# model params
# backbone
arch = "res34"

pretrained_backbone = True
backbone_out_channels = (64, 128, 256, 512)

# fpn
fpn_out_channels = 256

# pan
pan_num_levels = 4
pan_in_dim = 256
pan_out_dim = 256

# row segment predictor
rs_scale = 1

# col segment predictor
cs_scale = 1

# divide predictor
dp_head_nums = 8
dp_scale = 1

# cells extractor params
ce_scale = 1 / 8
ce_pool_size = (3, 3)
ce_dim = 512
ce_head_nums = 8
ce_heads = 1

# decoder
embed_dim = 512
feat_dim = 512
lm_state_dim = 512
proj_dim = 512
cover_kernel = 7
att_threshold = 0.5
spatial_att_weight_loss_wight = 1.0

# merger
merger_dim = 512

# train params
base_lr = 0.0001
min_lr = 1e-6
weight_decay = 0

num_epochs = 200
sync_rate = 20

log_sep = 20

work_dir = "/home/cxh/cxh/Projects/DTSM/work_dir/temp_A"

train_checkpoint = None

eval_checkpoint = os.path.join(work_dir, 'best_f1_model.pth')

start_eval_epoch = 0