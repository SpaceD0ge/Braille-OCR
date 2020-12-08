# Object detection for Braille annotations
Simple scripts for training an object detection deep neural network.

## Getting started

#### Install preferred object detection framework
- detectron2:
```
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
```
- mmdetection:
```
pip install mmcv-full==latest+torch1.7.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; pip install -r requirements/build.txt; pip install -v -e .
```

#### Prepare data by converting it into the Coco format
Original files consist of a set of raw LabelMe annotations and corresponding images.
To convert them to a specified format and split into train-val-test parts, run this command:
```
python prepare_data.py -root [path to annotations] -save_to ./data \
-convert [coco|mmdet|detectron|pascal] -train 0.8 -val 0.1 -test 0.1
```
Coco files are saved to 'data/braille_coco_[train|test|val]_data.json' by default.

#### Train a model
Each method supports COCO format only.

- detectron2:

Change base path in configuration files first. 
It is set to "/usr/local/lib/python3.6/dist-packages/detectron2/model_zoo/configs" by default.
```
python detectron2/run.py --config-file /detectron2/braille_retinanet.yaml
```

- mmdetection:
```
python mmdetection/tools/train.py mmdetection/configs/braille/braille_cascade_rcnn_coco.py
```

## Framework-specific actions
#### mmdetection
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

#### detectron
- predict image
```
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2

img = cv2.imread('path_to_image.jpg')

cfg = get_cfg()
cfg.merge_from_file('path_to_config.yaml')
cfg.MODEL.WEIGHTS = 'path_to_checkpoint.pth'
predictor = DefaultPredictor(cfg)
outputs = predictor(img)["instances"]
```
- show results
```
from detectron2.utils.visualizer import Visualizer

v = Visualizer(img[:, :, ::-1], None, scale=1.2)
out = v.draw_instance_predictions(outputs.to("cpu"))
out.get_image()[:, :, ::-1]
```
- test metrics
```
python detectron2/run.py --config-file 'path_to_config.yaml' \
--eval-only MODEL.WEIGHTS 'path_to_checkpoint.pth'
```
