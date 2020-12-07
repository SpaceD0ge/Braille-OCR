import itertools
from glob import glob
import json
import pickle
from sklearn.model_selection import train_test_split
import re
from os.path import abspath


class TextPreprocessor():
    '''Simple text preprocessing functionality.'''
    def __init__(self, translation_dict=None):
        self.repl = translation_dict
        if translation_dict is None:
            self.repl = {
                '::': ':',
                '..': '.',
                '/': '',
                '\n\n': '\n'
            }

    def process(self, label):
        label = label.lower()
        for key, value in self.repl.items():
            label = label.replace(key, value)
        return re.sub(' +', ' ', label)


class BrailleAnnotationsReader:
    '''
    Base Braille dataset. Derived from raw Labelme annotations.
    '''
    def __init__(self, prefix='braille', preproc=None):
        self.prefix = prefix
        self.preprocessor = preproc

    def _read_segmentation_records(self, file_names):
        '''Reads segmentation data entries into list of records and 
        dictionary of box labels.'''
        box_labels = []
        records = []
        for pic in file_names:
            with open(pic.replace('.jpg', '.json'), 'r', encoding='cp1251') as f:
                data = json.load(f)
                data['imagePath'] = abspath(pic)
                if self.preprocessor is not None:
                    for shape in data['shapes']:
                        shape['label'] = self.preprocessor.process(shape['label'])
                box_labels.extend([shape['label'] for shape in data['shapes']])
                records.append(data)
        box_labels = sorted(set(box_labels))
        box_labels = {box_labels[x]:x for x in range(len(box_labels))}
        return records, box_labels

    def _read_transcriptions(self, file_names):
        '''Reads full image transcriptions.'''
        labels = []
        for pic in file_names:
            with open(pic.replace('labeled.jpg', 'marked.txt'), 'r', encoding='cp1251') as f:
                label = f.read()
            if self.preprocessor is not None:
                label = self.preprocessor.process(label)
            labels.append(label)
        return labels

    def save(self, data, path, name):
        with open(path + f'/{name}.json', 'w') as fp:
            json.dump(data, fp)

    def get_records(self, image_list):
        transcriptions = self._read_transcriptions(image_list)
        records, box_labels = self._read_segmentation_records(image_list)
        for idx in range(len(records)):
            records[idx]['transcription'] = transcriptions[idx]
        return records, box_labels
