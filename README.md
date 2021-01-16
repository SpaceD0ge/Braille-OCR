# Object detection for Braille annotations
Simple scripts for training an object detection deep neural network.

## EfficientDet
A pytorch implementation based on Ross Wightman's work.

#### Prepare data by converting it into the Coco format
Original files consist of a set of raw LabelMe annotations and corresponding images.
To convert them to COCO format and split into train-val-test parts, run this command:
```
python prepare_data.py -root [path to annotations] -save_to ./ \
-convert coco -train 0.8 -val 0.1 -test 0.1
```
COCO files are saved to './braille_coco_[train|test|val]\_data.json' by default.

#### Edit a configuration file
An example configuration file 'effdet_infer.yaml' looks like this:
```yaml
base: effdet_train.yaml
process:
  checkpoint: 'checkpoints/effdet_d2/epoch=179.ckpt'
  gpus: 0
  precision: 32
model:
  image_size:
  - 1024
  - 768
data:
  test: 'braille_coco_test_data.json'
```
'base' keyword allows for editing new config files without unnecessary overlapping.

#### Train a model

```
python train.py -cfg confgis/effdet_train.yaml \
-model_type tf_efficientdet_d3 \
-train_file braille_coco_train_data.json \
-val_file braille_coco_val_data.json
```

#### Benchmark perfomance

```
python benchmark.py -cfg configs/effdet_infer.yaml \
-model_type tf_efficientdet_d3 \
-trials 10
```

#### Predict and visualize
(work in progress)

```
python predict.py -cfg configs/effdet_infer.yaml \
-model_type tf_efficientdet_d3 \
-checkpoint /checkpoints/effdet_d2/epoch=129.ckpt \
-test_file braille_coco_test_data.json \
-save_to ./images
```

#### Predicting images in Python
(work in progress)

```python
image_height, image_width = 1024, 768

config = ConfigReader('./').process('./configs/effdet_infer.yaml')
model_config = config['model']
model_config['image_size'] = [image_height, image_width]

model = DetectionModel.from_checkpoint(
	'[path_to_checkpoint.ckpt]',
	main_config=model_config
)

output = model({'image': image})
```


## MMDetection and Detectron2
Refer to 'mmdedection' and 'detectron2' subdirectories

## Coming soon
- Mean average precision metric for EfficientDet
- Flask server prototype
