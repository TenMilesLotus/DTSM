import os
import copy
import json
import pickle
import random
from torch._C import layout
import tqdm
import torch
import numpy as np
from PIL import Image
from .list_record_cache import ListRecordLoader
from libs.utils.format_translate import table_to_html


class LRCRecordLoader:
    def __init__(self, lrc_path, data_dir=''):
        self.loader = ListRecordLoader(lrc_path)
        self.data_root_dir = data_dir

    def __len__(self):
        return len(self.loader)

    def get_info(self, idx):
        table = self.loader.get_record(idx)
        image = Image.open(table['image_path']).convert('RGB')
        w = image.width
        h = image.height
        n_rows, n_cols = table['layout'].shape
        n_cells = n_rows * n_cols
        return w, h, n_cells

    def get_data(self, idx):
        table = self.loader.get_record(idx)
        img_path = os.path.join(self.data_root_dir, table['image_path'])
        image = Image.open(img_path).convert('RGB')
        return image, table


class Dataset:
    def __init__(self, loaders, transforms, max_edge=500):
        self.loaders = loaders
        self.transforms = transforms
        self.max_edge = max_edge

    def _match_loader(self, idx):
        offset = 0
        for loader in self.loaders:
            if len(loader) + offset > idx:
                return loader, idx - offset
            else:
                offset += len(loader)
        raise IndexError()
        
    def get_info(self, idx):
        loader, rela_idx = self._match_loader(idx)
        return loader.get_info(rela_idx)

    def __len__(self):
        return sum([len(loader) for loader in self.loaders])

    def __getitem__(self,idx):
        try:
            loader, rela_idx = self._match_loader(idx)
            image, table = loader.get_data(rela_idx)  # 这个table就是那个table
            # print('image_path', table['image_path'])
            old_w, old_h = image.width, image.height
            if max(old_w, old_h) > self.max_edge:
                if old_w > old_h:
                    new_w = self.max_edge
                    new_h = int(old_h / old_w * self.max_edge)
                    image = image.resize((new_w, new_h))
                    for cell in table["cells"]:
                        if cell["transcript"]:
                            cell["bbox"][0] = int(cell["bbox"][0] / old_w * new_w)
                            cell["bbox"][2] = int(cell["bbox"][2] / old_w * new_w)
                            cell["bbox"][1] = int(cell["bbox"][1] / old_h * new_h)
                            cell["bbox"][3] = int(cell["bbox"][3] / old_h * new_h)
                            cell["segmentation"][0][0][0] = int(cell["segmentation"][0][0][0] / old_w * new_w)
                            cell["segmentation"][0][1][0] = int(cell["segmentation"][0][1][0] / old_w * new_w)
                            cell["segmentation"][0][2][0] = int(cell["segmentation"][0][2][0] / old_w * new_w)
                            cell["segmentation"][0][3][0] = int(cell["segmentation"][0][3][0] / old_w * new_w)
                            cell["segmentation"][0][0][1] = int(cell["segmentation"][0][0][1] / old_h * new_h)
                            cell["segmentation"][0][1][1] = int(cell["segmentation"][0][1][1] / old_h * new_h)
                            cell["segmentation"][0][2][1] = int(cell["segmentation"][0][2][1] / old_h * new_h)
                            cell["segmentation"][0][3][1] = int(cell["segmentation"][0][3][1] / old_h * new_h)
                else:
                    new_h = self.max_edge
                    new_w = int(old_w / old_h * self.max_edge)
                    image = image.resize((new_w, new_h))
                    for cell in table["cells"]:
                        if cell["transcript"]:
                            cell["bbox"][0] = int(cell["bbox"][0] / old_w * new_w)
                            cell["bbox"][2] = int(cell["bbox"][2] / old_w * new_w)
                            cell["bbox"][1] = int(cell["bbox"][1] / old_h * new_h)
                            cell["bbox"][3] = int(cell["bbox"][3] / old_h * new_h)
                            cell["segmentation"][0][0][0] = int(cell["segmentation"][0][0][0] / old_w * new_w)
                            cell["segmentation"][0][1][0] = int(cell["segmentation"][0][1][0] / old_w * new_w)
                            cell["segmentation"][0][2][0] = int(cell["segmentation"][0][2][0] / old_w * new_w)
                            cell["segmentation"][0][3][0] = int(cell["segmentation"][0][3][0] / old_w * new_w)
                            cell["segmentation"][0][0][1] = int(cell["segmentation"][0][0][1] / old_h * new_h)
                            cell["segmentation"][0][1][1] = int(cell["segmentation"][0][1][1] / old_h * new_h)
                            cell["segmentation"][0][2][1] = int(cell["segmentation"][0][2][1] / old_h * new_h)
                            cell["segmentation"][0][3][1] = int(cell["segmentation"][0][3][1] / old_h * new_h)
            image, _, cls_label, \
                rows_fg_span, rows_bg_span, \
                    cols_fg_span, cols_bg_span, \
                        cells_span, divide = self.transforms(image, table) if 'layout' in table.keys() else self.transforms(image)
            return dict(
                id=idx,
                image_size=(image.shape[2], image.shape[1]),
                image=image,
                cls_label=cls_label,
                rows_fg_span=rows_fg_span,
                rows_bg_span=rows_bg_span,
                cols_fg_span=cols_fg_span,
                cols_bg_span=cols_bg_span,
                cells_span=cells_span,
                layout=table['layout'] if 'layout' in table.keys() else None,
                divide=divide,
                table=table,
                merge_target=table['merge_target'] if 'merge_target' in table.keys() else None
            )
        except Exception as e:
            print('Error occured while load data: %d' % idx)
            raise e


def collate_func(batch_data):
    batch_size = len(batch_data)

    image_dim = batch_data[0]['image'].shape[0]
    max_h = max([data['image'].shape[1] for data in batch_data])
    max_w = max([data['image'].shape[2] for data in batch_data])

    batch_id = list()
    batch_image_size = list()

    batch_image = torch.zeros([batch_size, image_dim, max_h, max_w], dtype=torch.float)
    batch_image_mask = torch.zeros([batch_size, 1, max_h, max_w], dtype=torch.float)
    batch_rows_fg_span = list()
    batch_rows_bg_span = list()
    batch_cols_fg_span = list()
    batch_cols_bg_span = list()
    batch_cells_span = list()
    batch_divide = list()
    tables = list()

    if all([(data['cls_label'] is None) and (data['layout'] is None) for data in batch_data]):
        batch_cls_label = list()
        batch_label_mask = list()
        batch_layout = list()
        batch_merge_target = list()
    else:
        assert not any([(data['cls_label'] is None) or (data['layout'] is None) for data in batch_data])
        max_label_length = max([data['cls_label'].shape[0] for data in batch_data])
        batch_cls_label = torch.zeros([batch_size, max_label_length], dtype=torch.long)
        batch_label_mask = torch.zeros([batch_size, max_label_length], dtype=torch.float)
        max_nr = max([data['layout'].shape[0] for data in batch_data])
        max_nc = max([data['layout'].shape[1] for data in batch_data])
        batch_layout = torch.full([batch_size, max_nr, max_nc], -1, dtype=torch.float)
        batch_merge_target = torch.full([batch_size, max_nr, max_nc], -1, dtype=torch.float)

    for batch_idx, data in enumerate(batch_data):
        batch_id.append(data['id'])
        batch_image_size.append(data['image_size'])

        _, cur_h, cur_w = data['image'].shape
        batch_image[batch_idx, :, :cur_h, :cur_w] = data["image"]
        batch_image_mask[batch_idx, :, :cur_h, :cur_w] = 1

        if (data['cls_label'] is None) and (data['layout'] is None) and (data['merge_target'] is None):
            batch_cls_label.append(data["cls_label"])
            batch_label_mask.append(None)
            batch_layout.append(data["layout"])
            batch_merge_target.append(data["merge_target"])
        else:
            label_length = data['cls_label'].shape[0]
            batch_cls_label[batch_idx, :label_length] = data['cls_label']
            batch_label_mask[batch_idx, :label_length] = 1.0
            layout_nr, layout_nc = data["layout" ].shape
            batch_layout[batch_idx, :layout_nr, :layout_nc] = torch.from_numpy(data['layout']).float()
            merge_target_nr, merge_target_nc = data["merge_target"].shape
            batch_merge_target[batch_idx, :merge_target_nr, :merge_target_nc] = torch.from_numpy(data['merge_target']).float()

        batch_rows_fg_span.append(data["rows_fg_span"])
        batch_rows_bg_span.append(data['rows_bg_span'])
        batch_cols_fg_span.append(data["cols_fg_span"])
        batch_cols_bg_span.append(data["cols_bg_span"])
        batch_cells_span.append(data["cells_span"])
        batch_divide.append(data["divide"])
        tables.append(data['table'])

    batch_divide = torch.tensor(batch_divide, dtype=torch.long) if batch_divide[0] is not None else batch_divide
    
    return dict(
        ids=batch_id,
        images_size=batch_image_size,
        images=batch_image,
        images_mask=batch_image_mask,
        cls_labels=batch_cls_label,
        labels_mask=batch_label_mask,
        rows_fg_spans=batch_rows_fg_span,
        rows_bg_spans=batch_rows_bg_span,
        cols_fg_spans=batch_cols_fg_span,
        cols_bg_spans=batch_cols_bg_span,
        cells_spans=batch_cells_span,
        divide_labels=batch_divide,
        layouts=batch_layout,
        tables=tables,
        merge_targets=batch_merge_target
    )