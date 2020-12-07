import numpy as np
import itertools
import os
import json


class BaseConverter:
    def __init__(self, labels, prefix='braille'):
        self.labels = labels
        self.prefix = prefix
        self.name_key = self.__class__.__name__.lower().replace('converter', '')

    def _flatten(self, l):
        return list(itertools.chain(*l))

    def _convert_entry(self, data, image_id):
        pass

    def _convert_records(self, records):
        return [self._convert_entry(entry, idx) for idx, entry in enumerate(records)]

    def save(self, data, path, name):
        with open(f'{path}/{self.prefix}_{self.name_key}_{name}_data.json', 'w') as fp:
            json.dump(data, fp)

    def convert(self, records, save=False):
        data = self._convert_records(records)
        if save:
            self._save(data)
        return data


class Detectron2Converter(BaseConverter):
    def _convert_entry(self, data, image_id):
        record = {
            'file_name': data["imagePath"],
            'height': data["imageHeight"],
            'width': data["imageWidth"],
            'image_id': image_id,
            'annotations': []
        }
        for anno in data['shapes']:
            bbox = {
                'bbox_mode': 0,
                'category_id': self.labels[anno['label']],
                'bbox': self._flatten(anno['points'])
            }
            record['annotations'].append(bbox)
        return record


class MMDetConverter(BaseConverter):
    def _convert_entry(self, data, image_id):
        record = {
            'filename': data["imagePath"],
            'height': data["imageHeight"],
            'width': data["imageWidth"],
            'text': data["transcription"]
        }
        boxes, labels = [], []
        for anno in data['shapes']:
            boxes.append(self._flatten(anno['points']))
            labels.append(self.labels[anno['label']],)

        record['ann'] = {
            'bboxes': boxes,
            'labels': labels
        }
        return record


class COCOConverter(BaseConverter):
    def _rect_area(self, points):
        points = self._flatten(points)
        return (points[2] - points[0])*(points[3] - points[1])

    def _coco_bbox(self, points):
        points = self._flatten(points)
        return points[0], points[1], points[2] - points[0], points[3] - points[1]

    def _convert_to_coco(self, records):
        data = {
            'info': {'year': 2020, 'description': 'dataset conversion'},
            'licenses': [{'id': 1, 'name': '-'}],
            'categories': [
                {'id':idx, 'name':label}
                for idx, label in enumerate(self.labels)
            ],
            'images': [],
            'annotations': []
        }

        annotation_count = 0
        for image_id, image_record in enumerate(records):
            image_dict = {
                "id": image_id, "license": 1,
                "height": image_record['imageHeight'], "width": image_record['imageWidth'],
                "file_name": image_record['imagePath']
            }
            data['images'].append(image_dict)
            for anno in image_record['shapes']:
                anno_dict = {
                    "id": annotation_count, "image_id": image_id,
                    "iscrowd": 0, "category_id": self.labels[anno['label']],
                    "bbox": self._coco_bbox(anno['points']),
                    "area": self._rect_area(anno['points'])
                }
                annotation_count += 1
                data['annotations'].append(anno_dict)
        return data

    def _convert_records(self, records):
        return self._convert_to_coco(records)


class PascalConverter(BaseConverter):
    def _to_xml_string(self, data_dict):
        load = ''
        for key in data_dict:
            load += f'<{key}>'
            if isinstance(data_dict[key], dict):
                load += f'{self._to_xml_string(data_dict[key])}</{key}>'
            else:
                load += f'{data_dict[key]}</{key}>'
        return(load)
    
    def _get_annotation_obj(self, shape):
        coords = self._flatten(shape['points'])
        obj_dict = {
            'name': shape['label'],
            'pose': 'Unspecified',
            'truncated': 0,
            'difficult': 0,
            'bndbox':{
                'xmin': coords[0],
                'ymin': coords[1],
                'xmax': coords[2],
                'ymax': coords[3]
            }
        }
        return self._to_xml_string({'object': obj_dict})

    def _convert_entry(self, data, image_id):
        record = {
            'folder': '/'.join(data['imagePath'].split('/')[:-1]),
            'filename': data['imagePath'].split('/')[-1],
            'path': data['imagePath'],
            'source': '-',
            'size': {
                'width': data['imageWidth'],
                'height': data['imageHeight'],
                'depth': 3
            },
            'segmented': 0,
        }
        record = self._to_xml_string(record)
        for shape in data['shapes']:
            record += self._get_annotation_obj(shape)
        return '<annotation>' + record + '</annotation>'

    def save(self, data, path, name):
        path = f'{path}/{self.prefix}_{self.name_key}_{name}_conversion'
        os.makedirs(path, exist_ok=True)
        for file_count in range(len(data)):
            with open(path + f'/{file_count}.xml', 'w') as f:
                f.write(data[file_count])
