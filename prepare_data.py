from glob import glob
from sklearn.model_selection import train_test_split
from data.converters import MMDetConverter, COCOConverter, PascalConverter, Detectron2Converter
from data.base import BrailleAnnotationsReader, TextPreprocessor
import argparse


parser = argparse.ArgumentParser(description="Prepare data by converting and splitting Labelme annotations")
parser.add_argument("-root", type=str)
parser.add_argument("-image_list", type=str, default=None)
parser.add_argument("-save_to", type=str, default='./')
parser.add_argument("-train", type=float, default=0.9)
parser.add_argument("-val", type=float, default=0.1)
parser.add_argument("-test", type=float, default=0.0)
parser.add_argument(
    "-convert", type=str, default=None,
    choices=['mmdet', 'coco', 'pascal', 'detectron']
)


def split(records, train_part, val_part, test_part):
    train, test = train_test_split(
        records,
        train_size=train_part, test_size=val_part + test_part
    )
    if test_part > 0:
        coeff = 1/(val_part + test_part)
        test, val = train_test_split(
            test,
            train_size=coeff*val_part, test_size=coeff*test_part
        )
        return {'train': train, 'val': test, 'test': val}
    return {'train':train, 'val':test}


def get_converter(conv_type, labels):
    if conv_type is None:
        return None
    converters = {
        'mmdet': MMDetConverter,
        'coco': COCOConverter,
        'pascal': PascalConverter,
        'detectron': Detectron2Converter
    }
    return converters[conv_type](labels)


def read_image_list(root, image_list_file):
    if image_list_file is None:
        images = glob(f'{root}/**/*.labeled.jpg', recursive=True)
    else:
        with open(image_list_file) as fp:
            images = fp.read().split('\n')
    return images


if __name__ == "__main__":
    args = parser.parse_args()
    images = read_image_list(args.root, args.image_list)
    reader = BrailleAnnotationsReader(preproc=TextPreprocessor())
    records, labels = reader.get_records(images)
    print(f'{len(labels)} distinct labels found.')
    converter = get_converter(args.convert, labels)
    if args.train != 1.0:
        print(f'Splitting data by fractions of {args.train}-{args.val}-{args.test}')
        splits = split(records, args.train, args.val, args.test)
        for key in splits:
            if converter:
                splits[key] = converter.convert(splits[key])
                converter.save(splits[key], args.save_to, key)
            else:
                reader.save(splits[key], args.save_to, key)
    else:
        if converter:
            records = converter.convert(records)
            converter.save(records, args.save_to, 'full')
        else:    
            reader.save(records, args.save_to, 'full')
    print('Done!')
