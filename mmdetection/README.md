# Object detection for Braille annotations
Using the mmdetection framework.

## Install mmdetection:
```
pip install mmcv-full==latest+torch1.7.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -r requirements/build.txt; pip install -v -e .
```

Copy local 'configs' folder to mmdetection/configs

## Prepare data by converting it into the Coco format
Original files consist of a set of raw LabelMe annotations and corresponding images.
To convert them to a specified format and split into train-val-test parts, run this command:
```
python prepare_data.py -root [path to annotations] -save_to ./data \
-convert [coco|mmdet|detectron|pascal] -train 0.8 -val 0.1 -test 0.1
```
Coco files are saved to 'data/braille_coco_[train|test|val]_data.json' by default.

## Train a model
Pick a configuration file and run train.py script with it.

```
python mmdetection/tools/train.py mmdetection/configs/braille/braille_cascade_rcnn_coco.py
```

## Framework-specific actions

- show results
```
from mmdet.models import build_detector
from mmdet.apis import inference_detector, init_detector
from mmdet.apis import show_result_pyplot

model = init_detector(
    'path_to_config.py',
    'path_to_checkpoint.pth'
)
img = 'path_to_image.jpg'
result = inference_detector(model, img)
show_result_pyplot(model, img, result, fig_size=(64,48), score_thr=0.2)
```
- test metrics
```
cd ..; python mmdetection/tools/test.py 'path_to_config.py' 'path_to_checkpoint.pth' --eval bbox
```
