_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/default_runtime.py',
    'braille_data_coco.py'
]

total_epochs = 100
checkpoint_config = dict(interval=5)
optimizer = dict(type='AdamW', lr=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11]
)

model = dict(
    roi_head = dict(
        bbox_head = [
            dict(type='Shared2FCBBoxHead', num_classes = 92),
            dict(type='Shared2FCBBoxHead', num_classes = 92),
            dict(type='Shared2FCBBoxHead', num_classes = 92),
        ]
    )
)