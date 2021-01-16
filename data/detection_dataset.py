from torch.utils.data import Dataset, DataLoader
from .augmentations import detection_crop
import numpy as np
import torch
import json
import cv2


class BBox:
    @staticmethod
    def fix_xyxy_bounds(bbox, bounds=(0,0,1,1)):
        # x(min), y(min)
        for idx in (0,1):
            if bbox[idx] < bounds[idx]:
                offset = bounds[idx] - bbox[idx]
                bbox[idx] = bounds[idx]
                bbox[idx+2] = bbox[idx+2] + offset
        # x(max), y(max)
        for idx in (2,3):
            if bbox[idx] > bounds[idx] - 0.01:
                offset = bbox[idx] - bounds[idx] + 0.01
                bbox[idx] = bounds[idx] - 0.01
                bbox[idx-2] = bbox[idx-2] - offset
        return bbox

    @staticmethod
    def coco_to_albu(bbox, image_shape):
        # xywh -> xy(min)xy(max)
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        # absolute -> relative
        bbox = [
            bbox[0]/image_shape[0], bbox[1]/image_shape[1],
            bbox[2]/image_shape[0], bbox[3]/image_shape[1]
        ]
        return np.array(bbox)

    @staticmethod
    def coco_to_pascal(bbox, image_shape):
        # xywh -> xy(min)xy(max)
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        return np.array(bbox)


class BrailleCocoDataset(Dataset):
    '''
    Takes COCO-formatted annotation file as input.
    Outputs boxes in specified format (pascal_voc by default)
    '''
    def __init__(
            self, ann_file,
            augment=None, read_from='ram', output_format='pascal_voc'
        ):
        with open(ann_file, 'r') as f:
            data = json.load(f)
        self.categories = {x['id']:x['name'] for x in data['categories']}
        self.bboxes, self.labels = self._read_annotations(data)
        self.read_from = read_from
        self.format = output_format
        self.augmentation = augment
        self.images = {
            x['id']: x['file_name'] if read_from == 'disk'
            else self._read_image(x['file_name'])
            for x in data['images']
        }
        self.id_mapping = dict(zip(self.images.keys(), range(len(self.images))))

    def _read_annotations(self, data):
        bboxes = {x['id']:[] for x in data['images']}
        labels = {x['id']:[] for x in data['images']}
        for entry in data['annotations']:
            bboxes[entry['image_id']].append(entry['bbox'])
            labels[entry['image_id']].append(entry['category_id'])
        return bboxes, labels

    def _read_image(self, path):
        # HWC (height x width x channels)
        image = cv2.imread(path)
        image = image/255
        return image

    def _transform(self, item):
        bounds = (0, 0, item['image'].shape[0], item['image'].shape[1])
        item['annotations'] = [
            BBox.fix_xyxy_bounds(box, bounds)
            for box in item['annotations']
        ]
        if self.augmentation is not None:
            transformed = self.augmentation(
                image=item['image'], bboxes=item['annotations'], category_ids=item['labels']
            )
            item = {
                'image': transformed['image'],
                'annotations': transformed['bboxes'],
                'labels': transformed['category_ids'],
            }
        return {key: np.array(item[key]) for key in item}

    def _format_output(self, item):
        if self.format == 'pascal_voc':
            format_fn = BBox.coco_to_pascal
        elif self.format == 'albumentations':
            format_fn = BBox.coco_to_albu
        elif self.format == 'coco':
            format_fn = lambda x,shape: x
        else:
            raise ValueError(f'unknown annotation format {self.format}')
        item['annotations'] = [
            format_fn(box, item['image'].shape)
            for box in item['annotations']
        ]
        return item

    def __getitem__(self, idx):
        idx = self.id_mapping[idx]
        image = self._read_image(self.images[idx]) if self.read_from == 'disk' else self.images[idx]
        item = {
            'image': image,
            'annotations': self.bboxes[idx],
            'labels': self.labels[idx],
        }
        try:
            item = self._transform(item)
        except ValueError:
            return {'image': None, 'annotations':None, 'labels': []}
        return self._format_output(item)

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    '''
    Discarding all items without annotations.
    Changing image format from BHWC to BCHW.
    Converting bbox axis order from xyxy to yxyx
    '''
    index = [idx for idx in range(len(batch)) if len(batch[idx]['labels'])>0]
    if len(index) == 0:
        return None
    image = np.stack([batch[idx]['image'] for idx in index])
    annos = [batch[idx]['annotations'] for idx in index]
    labels = [batch[idx]['labels'] for idx in index]
    return {
        'image': torch.tensor(image, dtype=torch.float32).permute(0,3,1,2),
        'annotations': [torch.tensor(x)[:,[1,0,3,2]].float() for x in annos],
        'labels': [torch.tensor(x).float() for x in labels],
    }


def get_coco_loaders(
        train_file, val_file,
        read_from='disk',
        batch_size=(12,4), crop_size=(384,384),
        num_workers=4
    ):
    '''
    Inputs:
        train_file[str]: path to COCO formatted annotation file with training examples
        val_file[str]: path to COCO formatted annotation file with validation examples
        read_from[str]: 'disk' or 'ram', whether to read every image from disk or ram
        batch_size[tuple]: tuple of trainining and evaluating batch sizes
        crop_size[tuple]: size of a random square image crop
    Outputs:
        train[DataLoader], val[DataLoader]
    '''
    aug = detection_crop(crop_size, format='coco')
    train_loader = DataLoader(
        BrailleCocoDataset(
            train_file, augment=aug,
            read_from=read_from, output_format='pascal_voc'),
        batch_size=batch_size[0], collate_fn=collate_fn, num_workers=num_workers
    )
    val_loader = DataLoader(
        BrailleCocoDataset(
            val_file, augment=aug,
            read_from=read_from, output_format='pascal_voc'
        ),
        batch_size=batch_size[1], collate_fn=collate_fn, num_workers=num_workers
    )
    return train_loader, val_loader