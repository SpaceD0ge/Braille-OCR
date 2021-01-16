from configs.config import ConfigReader
from models.effdet import DetectionModel
from data.detection_dataset import BrailleCocoDataset
from data.augmentations import detection_resize
from tqdm.auto import tqdm
import torch
import argparse
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

parser = argparse.ArgumentParser(description="Train an object detection model")
parser.add_argument("-cfg", type=str, default="/configs/effdet_infer.yaml")
parser.add_argument("-root", type=str, default=".")
parser.add_argument("-save_to", type=str, default="./images")
parser.add_argument("-device", type=str, default=None)
args = parser.parse_args()


def visualize_bbox(img, bbox, color=255, thickness=1):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    return img


def visualize(image, bboxes, category_ids, categories):
    img = image.copy()
    for bbox in bboxes:
        img = visualize_bbox(img, bbox)
    font = ImageFont.truetype('configs/arial.ttf', 18)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = categories[category_id]
        draw.text(bbox[:2],  class_name, font = font, fill = (255, 255, 255))
    return np.array(img_pil)


def predict():
    configuration = ConfigReader(args.root).process(args.cfg)

    aug = detection_resize(configuration['model']['image_size'])
    dataset = BrailleCocoDataset(configuration['data']['val'], aug)
    
    if args.device is not None:
        device = args.device
    else:
        device = 'cpu' if configuration['process']['gpus'] == 0 else 'cuda'
    
    model = DetectionModel.load_from_checkpoint(
        configuration['process']['checkpoint'],
        strict=False,
        main_config = configuration['model']
    ).to(device)
    model.eval()
    
    for item_id in tqdm(range(0, len(dataset))):
        item = dataset[item_id]
        image = torch.tensor(item['image']).unsqueeze(0).to(device)
        image = image.permute(0,3,1,2).float()
        with torch.no_grad():
            outs = model({'image':image})['detections'].squeeze(0)
        outs = outs[torch.where(outs[:,4] > 0.1)[0]].cpu().numpy()
        result = visualize(
            (item['image']*255).astype(np.uint8),
            outs[:,:4],
            outs[:,-1],
            dataset.categories
        )
        cv2.imwrite(args.save_to + f'/{item_id}.jpg', result)

if __name__ == "__main__":
    predict()
