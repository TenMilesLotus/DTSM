import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
import tqdm
import json
import sys
sys.path.append('./')
sys.path.append('../')
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict
from libs.utils.cal_f1 import pred_result_to_table, table_to_relations, evaluate_f1
from libs.utils.comm import distributed, synchronize
from libs.utils.checkpoint import load_checkpoint, save_checkpoint
from libs.data import create_train_dataloader, create_valid_dataloader
from libs.utils.model_synchronizer import ModelSynchronizer
from libs.utils.time_counter import TimeCounter
from libs.utils.utils import is_simple_table
from libs.utils.utils import cal_mean_lr
from libs.utils.counter import Counter
from libs.utils import logger
from libs.model import build_model
from libs.configs import cfg, setup_config
import torch.distributed as dist

from libs.utils.teds_multiprocess import evaluate
from libs.utils.convert_form import save_per_html

import cv2
from PIL import Image
from torchvision.transforms import functional as F

metrics_name = ['f1']
best_metrics = [0.0]

def init():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='default')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    setup_config(args.cfg)
    os.environ['LOCAL_RANK'] = str(args.local_rank)
    num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    # num_gpus = 1
    distributed = num_gpus > 1
    if distributed:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()
    logger.setup_logger('Line Detect Model', cfg.work_dir, 'train.log')
    logger.info('Use config:%s' % args.cfg)

# 每个epoch
def train(cfg, epoch, dataloader, model, optimizer, scheduler, time_counter, synchronizer=None):
    model.train()
    counter = Counter(cache_nums=1000)
    for it, data_batch in enumerate(dataloader):
        ids = data_batch['ids']
        # text block
        images_size = data_batch['images_size']
        images = data_batch['images']
        # concat text block, image
        mask_paths = []
        for table in data_batch['tables']:
            mask_paths.append(table['image_path'].replace('img', 'text_mask'))
        text_masks = []
        for mask_path in mask_paths:
            text_mask = Image.open(mask_path).convert('RGB')
            old_w, old_h = text_mask.width, text_mask.height
            if max(old_w, old_h) > 500:
                if old_w > old_h:
                    new_w = 500
                    new_h = int(old_h / old_w * 500)
                    text_mask = text_mask.resize((new_w, new_h))
                else:
                    new_h = 500
                    new_w = int(old_w / old_h * 500)
                    text_mask = text_mask.resize((new_w, new_h))
            text_masks.append(text_mask)
        for i in range(len(text_masks)):
            text_masks[i] = F.to_tensor(text_masks[i])
            text_masks[i] = F.normalize(text_masks[i], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=False)
        text_masks = torch.stack(text_masks, 0)
        text_masks = text_masks.to(cfg.device)
        # images = torch.cat((images, text_masks), 1)
        images = images.to(cfg.device)
        cls_labels = data_batch['cls_labels'].to(cfg.device) 
        labels_mask = data_batch['labels_mask'].to(cfg.device)
        rows_fg_spans = data_batch['rows_fg_spans']
        rows_bg_spans = data_batch['rows_bg_spans']
        cols_fg_spans = data_batch['cols_fg_spans']
        cols_bg_spans = data_batch['cols_bg_spans'] 
        cells_spans = data_batch['cells_spans']
        divide_labels = data_batch['divide_labels'].to(cfg.device)
        layouts = data_batch['layouts'].to(cfg.device)
        merge_targets = data_batch['merge_targets'].to(cfg.device) 
        
        # text_info
        image_paths = []
        for table in data_batch['tables']:
            image_paths.append(table['image_path'])

        # try:
        optimizer.zero_grad()
        pred_result, result_info = model(
            images, images_size,
            cls_labels, labels_mask, layouts,
            rows_fg_spans, rows_bg_spans,
            cols_fg_spans, cols_bg_spans,
            cells_spans, divide_labels,
            merge_targets, text_masks
            , image_paths
        )
        loss = sum([val for key, val in result_info.items() if 'loss' in key])
        loss.backward()
        optimizer.step()
        scheduler.step()
        counter.update(result_info)
        # except:
        #     logger.info('Merge Error')
        #     continue

        if it % cfg.log_sep == 0:
            logger.info(
                '[Train][Epoch %03d Iter %04d][Memory: %.0f ][Mean LR: %f ][Left: %s] %s' %
                (
                    epoch,
                    it,
                    torch.cuda.max_memory_allocated()/1024/1024,
                    cal_mean_lr(optimizer),
                    time_counter.step(epoch, it + 1),
                    counter.format_mean(sync=False)
                )
            )

        if synchronizer is not None:
            synchronizer()
        if synchronizer is not None:
            synchronizer(final_align=True)


def valid(cfg, dataloader, model):
    model.eval()
    total_label_relations = list()
    total_pred_relations = list()
    total_relations_metric = list()

    pred_htmls = dict()
    label_htmls = dict()

    for it, data_batch in enumerate(tqdm.tqdm(dataloader)):
        ids = data_batch['ids']
        images_size = data_batch['images_size']
        images = data_batch['images']
        # concat text block, image
        mask_paths = []
        for table in data_batch['tables']:
            mask_paths.append(table['image_path'].replace('img', 'text_mask'))
        text_masks = []
        for mask_path in mask_paths:
            text_mask = Image.open(mask_path).convert('RGB')
            old_w, old_h = text_mask.width, text_mask.height
            if max(old_w, old_h) > 500:
                if old_w > old_h:
                    new_w = 500
                    new_h = int(old_h / old_w * 500)
                    text_mask = text_mask.resize((new_w, new_h))
                else:
                    new_h = 500
                    new_w = int(old_w / old_h * 500)
                    text_mask = text_mask.resize((new_w, new_h))
            text_masks.append(text_mask)
        for i in range(len(text_masks)):
            text_masks[i] = F.to_tensor(text_masks[i])
            text_masks[i] = F.normalize(text_masks[i], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=False)
        text_masks = torch.stack(text_masks, 0)
        text_masks = text_masks.to(cfg.device)
        # images = torch.cat((images, text_masks), 1)
        images = images.to(cfg.device)
        tables = data_batch['tables']

        pred_result, _ = model(images, images_size, text_masks=text_masks)
        pred_tables = [
            pred_result_to_table(tables[batch_idx],
                                 (pred_result[0][batch_idx], pred_result[1][batch_idx],
                                  pred_result[2][batch_idx], pred_result[3][batch_idx])
            )
            for batch_idx in range(len(ids))
        ]

        origin_img_path = tables[0]["image_path"]
        for table in pred_tables:
            table["img_name"] = os.path.basename(origin_img_path)

        pred_relations = [table_to_relations(table) for table in pred_tables]
        total_pred_relations.extend(pred_relations)

        pred_html = {str(ids[0]):{"html":save_per_html(pred_relations[0]["cells"])}}
        pred_htmls.update(pred_html)

        # label
        label_relations = []
        for table in tables:
            label_path = os.path.join(cfg.valid_data_dir, table['label_path'])
            with open(table['label_path'], 'r', encoding='utf-8') as f:
                label_relations.append(json.load(f))
        total_label_relations.extend(label_relations)

        label_html = {str(ids[0]):{"html":save_per_html(label_relations[0]["cells"])}}
        label_htmls.update(label_html)

    # cal P, R, F1
    total_relations_metric = evaluate_f1(total_label_relations, total_pred_relations, num_workers=40)
    P, R, F1 = np.array(total_relations_metric).mean(0).tolist()
    F1 = 2 * P * R / (P + R) if P != 0 and R != 0 else 0
    # print('[Valid] Total Type Mertric: Precision: %s, Recall: %s, F1-Score: %s' % (P, R, F1))
    logger.info('[Valid] Total Type Mertric: Precision: %s, Recall: %s, F1-Score: %s' % (P, R, F1))

    # teds_s = evaluate(pred_htmls, label_htmls, 40, True)
    # logger.info('[Valid] Total Type Mertric: TEDS-S %s' % (teds_s[0]))
    return (F1,)


def build_optimizer(cfg, model):
    params = list()
    for _, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.base_lr
        weight_decay = cfg.weight_decay
        params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
    optimizer = torch.optim.Adam(params, cfg.base_lr)
    return optimizer


def build_scheduler(cfg, optimizer, epoch_iters, start_epoch=0):
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cfg.num_epochs * epoch_iters,
        eta_min=cfg.min_lr,
        last_epoch=-1
        # last_epoch=-1 if start_epoch == 0 else start_epoch * epoch_iters
    )
    return scheduler


def main():
    init()

    train_dataloader = create_train_dataloader(
        cfg.vocab,
        cfg.train_lrcs_path,
        cfg.train_num_workers,
        cfg.train_max_batch_size,
        cfg.train_max_pixel_nums,
        cfg.train_bucket_seps,
        cfg.train_data_dir
    )

    logger.info(
        'Train dataset have %d samples, %d batchs' %
        (
            len(train_dataloader.dataset),
            len(train_dataloader.batch_sampler)
        )
    )

    valid_dataloader = create_valid_dataloader(
        cfg.vocab,
        cfg.valid_lrc_path,
        cfg.valid_num_workers,
        cfg.valid_batch_size,
        cfg.valid_data_dir
    )

    logger.info(
        'Valid dataset have %d samples, %d batchs with batch_size=%d' %
        (
            len(valid_dataloader.dataset),
            len(valid_dataloader.batch_sampler),
            valid_dataloader.batch_size
        )
    )

    model = build_model(cfg)
    model.cuda()

    if distributed():
        synchronizer = ModelSynchronizer(model, cfg.sync_rate)
    else:
        synchronizer = None

    epoch_iters = len(train_dataloader.batch_sampler)  # 为啥=0
    optimizer = build_optimizer(cfg, model)

    global metrics_name
    global best_metrics
    start_epoch = 0

    # finetune
    # resume_path = os.path.join(cfg.work_dir, 'latest_model.pth')
    resume_path = "/home/cxh/cxh/Projects/baseline-gai/work_dir/densetab/best_f1_model.pth"
    # if os.path.exists(resume_path):
    #     best_metrics, start_epoch = load_checkpoint(resume_path, model, optimizer)
    #     best_metrics = [0.0]
    #     start_epoch += 1
    #     logger.info('resume from: %s' % resume_path)
    # elif cfg.train_checkpoint is not None:
    #     load_checkpoint(cfg.train_checkpoint, model)
    #     logger.info('load checkpoint from: %s' % cfg.train_checkpoint)

    scheduler = build_scheduler(cfg, optimizer, epoch_iters, start_epoch)

    time_counter = TimeCounter(start_epoch, cfg.num_epochs, epoch_iters)
    time_counter.reset()

    for epoch in range(start_epoch, cfg.num_epochs):
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        train(cfg, epoch, train_dataloader, model, optimizer, scheduler, time_counter, synchronizer)

        if epoch >= cfg.start_eval_epoch:
            with torch.no_grad():
                metrics = valid(cfg, valid_dataloader, model)
            for metric_idx in range(len(metrics_name)):
                if metrics[metric_idx] > best_metrics[metric_idx]:
                    best_metrics[metric_idx] = metrics[metric_idx]
                    # choke
                    save_checkpoint(os.path.join(cfg.work_dir, 'best_%s_model.pth' % metrics_name[metric_idx]), model, optimizer, best_metrics, epoch)
                    logger.info('Save current model as best_%s_model' % metrics_name[metric_idx])

            save_checkpoint(os.path.join(cfg.work_dir, 'latest_model.pth'), model, optimizer, best_metrics, epoch)


if __name__ == '__main__':
    main()
