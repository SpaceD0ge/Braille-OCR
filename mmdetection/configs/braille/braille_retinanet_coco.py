_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    'braille_data_coco.py'
]

total_epochs = 100
checkpoint_config = dict(interval=5)
optimizer = dict(type='SGD', lr=0.01)
lr_config = dict(step=[16, 22])

model = dict(
    bbox_head = dict(
        num_classes = 92
    )
)