import sys,os
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)
import torch,monai
import torch.nn as nn
from lucidmodel.STUNet import STUNet

def load_model(modelname):
    if modelname =="unet_large":
        model = monai.networks.nets.UNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=192,
                    channels=(64,64,128,256,512),
                    strides=(2,2,2,2),
                    num_res_units=2,
                    norm=monai.networks.layers.Norm.BATCH,
                )
    elif modelname == "swinunetr":
        model = monai.networks.nets.SwinUNETR(
                img_size=(192,192,192),
                in_channels=1,
                out_channels=192,
                feature_size=48,
                drop_rate=0.1,
                attn_drop_rate=0.1,
                dropout_path_rate=0.1,
            )
    elif modelname=="STUNet_large":
        model = STUNet(1, 192, depth=[2,2,2,2,2,2], dims=[64, 128, 256, 512, 1024, 1024],
                    pool_op_kernel_sizes = ((2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2)),
               conv_kernel_sizes = ((3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)))
    return model
