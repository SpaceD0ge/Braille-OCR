_base_ = '../_base_/datasets/coco_detection.py'

dataset_type = 'CocoDataset'
batch_size = 4
data_root = 'data/'
classes = [
    '!', '##', "'", '(', '()', ')', '*', '+', ',', '-', '.',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
    '=', '>>', '?', 'a', 'cc', 'd', 'e', 'en', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'w', 'x', 'xx', 'y', 'z',
    '{', '|', '}', '~13456', '~3', '~46', '~5', '~56', '~6', '§',
    '«', '»', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й',
    'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'сс', 'т', 'у', 'ф',
    'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё'
]
data = dict(
    samples_per_gpu=batch_size,
    train=dict(
        type=dataset_type,
        ann_file='data/braille_coco_train_data.json',
        img_prefix='',
        classes=classes
    ),
    val=dict(
        type=dataset_type,
        ann_file='data/braille_coco_val_data.json',
        img_prefix='',
        classes=classes
    ),
    test=dict(
        type=dataset_type,
        ann_file='data/braille_coco_test_data.json',
        img_prefix='',
        classes=classes
    )
)
