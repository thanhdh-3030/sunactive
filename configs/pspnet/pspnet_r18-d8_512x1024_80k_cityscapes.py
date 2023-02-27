_base_ = './pspnet_r50-d8_512x1024_80k_cityscapes.py'
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(
        depth=18,
        norm_cfg=norm_cfg),
    decode_head=dict(
        in_channels=512,
        channels=128,
        norm_cfg=norm_cfg
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
data =dict(
    # samples_per_gpu=8,
    # workers_per_gpu=8,
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        data_root='/home/dang.hong.thanh/datasets/Cityspaces',
        img_dir='images/train',

    ),
    val=dict(
        data_root='/home/dang.hong.thanh/datasets/Cityspaces',
        img_dir='images/val',

    ),
    test=dict(
        data_root='/home/dang.hong.thanh/datasets/Cityspaces',
        img_dir='images/val',

    ),
)