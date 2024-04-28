import sys
import json
sys.path.append('./')
sys.path.append('../')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import tqdm
import torch
import numpy as np
from libs.configs import cfg, setup_config
from libs.model import build_model
from libs.data import create_valid_dataloader
from libs.utils import logger
from libs.utils.cal_f1 import pred_result_to_table, table_to_relations, evaluate_f1
from libs.utils.checkpoint import load_checkpoint
from libs.utils.comm import synchronize, all_gather
from libs.utils.convert_form import save_per_html
from libs.utils.teds_multiprocess import evaluate
import cv2
import time
from PIL import Image
from torchvision.transforms import functional as F

def init():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrc", type=str, default="/home/cxh/cxh/Datasets/CombTab/annotations/dtsm_annotations/table.lrc")
    parser.add_argument("--cfg", type=str, default='default')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    setup_config(args.cfg)
    if args.lrc is not None:
        cfg.valid_lrc_path = args.lrc

    os.environ['LOCAL_RANK'] = str(args.local_rank)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    logger.setup_logger('Line Detect Model', cfg.work_dir, 'valid.log')
    logger.info('Use config: %s' % args.cfg)
    logger.info('Evaluate Dataset: %s' % cfg.valid_lrc_path)


def valid(cfg, dataloader, model):
    model.eval()
    total_label_relations = list()
    total_pred_relations = list()
    total_relations_metric = list()

    pred_htmls = dict()
    label_htmls = dict()

    vis_dir = "/home/cxh/cxh/Projects/baseline-gai/vis/testB"
    if not os.path.exists(vis_dir): os.makedirs(vis_dir)
    old_time = time.time()
    for it, data_batch in enumerate(tqdm.tqdm(dataloader)):
        ids = data_batch['ids']
        images_size = data_batch['images_size']
        images = data_batch['images'].to(cfg.device)
        tables = data_batch['tables']
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

        pred_result, _ = model(images, images_size, text_masks=text_masks)
        # for debug
        # if it == 2:
        #     break

        if it == 99:
            new_time = time.time()
            # print(new_time-old_time)
        # vis
        for i in range(len(ids)):
            # try:
                origin_img_path = tables[i]["image_path"]
                img_save_path = os.path.join(vis_dir, os.path.basename(origin_img_path))
                img_save = cv2.imread(origin_img_path)
                row_seg = pred_result[0][i]
                col_seg = pred_result[1][i]
                row_se = [[] for _ in range(len(row_seg))]   # merge
                col_se = [[] for _ in range(len(col_seg))]  # merge
                for sr, sc, er, ec in pred_result[3][i]:  # merge
                    if sr != er:  # merge
                        for j in range(sr, er):  # merge
                            row_se[j+1].append([col_seg[sc], col_seg[sc+1]])  # merge
                    elif sc != ec:  # merge
                        for j in range(sc, ec):  # merge
                            col_se[j+1].append([row_seg[sr], row_seg[sr+1]])  # merge
                for idx, c in enumerate(col_seg):
                    c *= img_save.shape[1]/images_size[i][0]
                    if col_se[idx]:
                        draw_list = [0, img_save.shape[0]]
                        for se in col_se[idx]:
                            draw_list.insert(-1, int(se[0]*img_save.shape[1]/images_size[i][0]))
                            draw_list.insert(-1, int(se[1]*img_save.shape[1]/images_size[i][0]))
                        for d in range(int(len(draw_list)/2)):
                            cv2.line(img_save, (int(c), int(draw_list[2*d])), (int(c), int(draw_list[2*d+1])), (128, 0, 0), 5)
                    else:
                        cv2.line(img_save, (int(c), 0), (int(c), img_save.shape[0]), (128, 0, 0), 5)
                
                for idx, r in enumerate(row_seg):
                    r *= img_save.shape[0]/images_size[i][1]
                    if row_se[idx]:
                        draw_list = [0, img_save.shape[1]]
                        for se in row_se[idx]:
                            draw_list.insert(-1, int(se[0]*img_save.shape[0]/images_size[i][1]))
                            draw_list.insert(-1, int(se[1]*img_save.shape[0]/images_size[i][1]))
                        for d in range(int(len(draw_list)/2)):
                            cv2.line(img_save, (int(draw_list[2*d]), int(r)), (int(draw_list[2*d+1]), int(r)), (0, 230, 230), 5)
                    else:
                        cv2.line(img_save, (0, int(r)), (img_save.shape[1], int(r)), (0, 230, 230), 5)
                cv2.imwrite(img_save_path, img_save)

                # print("img_save_path", img_save_path)
                # print("spans", pred_result[3][i])
            # except:
            #     pass

        # pred
        pred_tables = [
            pred_result_to_table(tables[batch_idx],
                (pred_result[0][batch_idx], pred_result[1][batch_idx], \
                    pred_result[2][batch_idx], pred_result[3][batch_idx])
            ) \
            for batch_idx in range(len(ids))
        ]
        # f1name
        for table in pred_tables:
            table["img_name"] = os.path.basename(origin_img_path)

        # dict(layout=layout(np), head_rows=[0,1], body_rows=body_rows, cells=[{'segmentation'=,'transcription'=}])
        pred_relations = [table_to_relations(table) for table in pred_tables]
        pred_html = {str(ids[0]):{"html":save_per_html(pred_relations[0]["cells"])}}
        pred_htmls.update(pred_html)

        total_pred_relations.extend(pred_relations)
        # label
        label_relations = []
        for table in tables:
            with open(table['label_path'], 'r', encoding='utf-8') as f:
                label_relations.append(json.load(f))

        label_html = {str(ids[0]):{"html":save_per_html(label_relations[0]["cells"])}}
        label_htmls.update(label_html)

        total_label_relations.extend(label_relations)


    # cal P, R, F1
    total_relations_metric = evaluate_f1(total_label_relations, total_pred_relations, num_workers=40)
    P, R, F1 = np.array(total_relations_metric).mean(0).tolist()
    F1 = 2 * P * R / (P + R)
    logger.info('[Valid] Total Type Mertric: Precision: %s, Recall: %s, F1-Score: %s' % (P, R, F1))

    teds_s = evaluate(pred_htmls, label_htmls, 40, True)
    logger.info('[Valid] Total Type Mertric: TEDS-S %s' % (teds_s[0]))

    # # return (F1, )
    # return (teds_s, )


def main():
    init()

    valid_dataloader = create_valid_dataloader(
        cfg.vocab,
        cfg.valid_lrc_path,
        cfg.valid_num_workers,
        cfg.valid_batch_size,
        cfg.valid_data_dir
    )
    logger.info(
        'Valid dataset have %d samples, %d batchs with batch_size=%d' % \
            (
                len(valid_dataloader.dataset),
                len(valid_dataloader.batch_sampler),
                valid_dataloader.batch_size
            )
    )

    model = build_model(cfg)
    model.cuda()
    
    load_checkpoint(cfg.eval_checkpoint, model)
    logger.info('Load checkpoint from: %s' % cfg.eval_checkpoint)

    with torch.no_grad():
        valid(cfg, valid_dataloader, model)


if __name__ == '__main__':
    main()
