import albumentations as A


def detection_crop(crop_size, visibility=0.7, format='coco'):
    '''
    Takes a square crop ouf of an image
    formats:
        pascal_voc -> Xmin, Ymin, Xmax, Ymax
        coco -> Xmin, Ymin, Height, Width
        albumentations -> Xmin, Ymin, Xmax, Ymax (relative, between [0,1])
    '''
    aug = A.Compose(
        [A.RandomCrop(crop_size[0], crop_size[1]),],
        bbox_params = {
            'format': format, 'min_visibility':visibility, 'label_fields': ['category_ids']
        }
    )
    return aug

def detection_resize(image_size, format='coco'):
    '''
    Tries to resize an image without changing its aspect rate.
    '''
    aug = A.Compose(
        [
         A.LongestMaxSize(image_size[0]),
         A.PadIfNeeded(image_size[0], image_size[1]),
         A.Resize(image_size[0], image_size[1])
        ],
        bbox_params = {
            'format': format, 'label_fields': ['category_ids']
        }
    )
    return aug