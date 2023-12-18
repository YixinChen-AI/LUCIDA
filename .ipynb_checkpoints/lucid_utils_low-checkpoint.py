import sys,os
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)
import argparse,os
import SimpleITK as sitk
import numpy as np
import torch,monai
from datautils import resampleVolume,adjust_image_direction
from tqdm import tqdm
from lucidmodel.STUNet import STUNet
# import monai.transforms as transforms

def lucid(ct_path,outputdiranme = "lucid",check=True,modelname=None,modelweight=None,output=105):
    
    print(f"提供的NIfTI路径是:{ct_path}")
    file_path = os.path.dirname(os.path.abspath(__file__))
    
    ct_itk = sitk.ReadImage(os.path.join(ct_path))
    
    
    print("----------------direction check and spacing check------------------------")
    
    def create_direction_matrix(x_dir, y_dir, z_dir):
        # 创建一个3x3的方向矩阵
        direction_matrix = [
            x_dir, 0, 0,
            0, y_dir, 0,
            0, 0, z_dir
        ]
        return direction_matrix
    new_direction = (-1, 0, 0, 0, -1, 0, 0, 0, 1)
    direction_check = np.mean(np.abs(np.array(ct_itk.GetDirection()) - np.array(new_direction)))
    spacing_check = np.mean(np.abs(np.array(ct_itk.GetSpacing()) - np.array([1.5,1.5,1.5])))

    print("----------------pre-process <LUCID Standard Protocol>------------------------")
    
    if spacing_check < 0.05:
        print("spacing check: OK!!")
    else:
        print("spacing is",np.array(ct_itk.GetSpacing()))
        print("spacing need to be: [1.5,1.5,1.5]")
        ct_itk = resampleVolume([1.5,1.5,1.5],ct_itk,resamplemethod=sitk.sitkLinear)
        
    if direction_check < 0.05:
        print("direction check: OK!!")
    else:
        print("direction is",np.array(ct_itk.GetDirection()))
        print("direction need to be: ",new_direction)
            
        ct_itk = adjust_image_direction(ct_itk, new_direction)
        sitk.WriteImage(ct_itk, ct_path.replace(".nii.gz","_lucid.nii.gz"))
        print("standard protocol nii has been write in ",ct_path.replace(".nii.gz","lucid.nii.gz"))
    
    # print("CurvatureFlow!!")
    # ct_itk = sitk.CurvatureFlow(image1=ct_itk, timeStep=0.125, numberOfIterations=5)
    # print("LaplacianSharpening!!")
    # ct_itk = sitk.LaplacianSharpening(ct_itk)
    
    def scale_intensity_range(ct, a_min, a_max, b_min, b_max, clip):
        if clip:
            ct = torch.clamp(ct, min=a_min, max=a_max)
    
        # 线性缩放
        ct = (ct - a_min) / (a_max - a_min) * (b_max - b_min) + b_min
        return ct
        
    ct = sitk.GetArrayFromImage(ct_itk)
    ct = torch.tensor(ct).float().unsqueeze(0).unsqueeze(0)
    ct = scale_intensity_range(ct, a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True)
    
    print("----------------model loading------------------------")

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
    ckpt = torch.load(modelweight,map_location="cpu")
    model.load_state_dict(ckpt["model"])
        
    # model = Mega_model((192,192,192),in_channels=1,out_channels=text_features.shape[0],backbone = "STUNet_large",organ_embedding = text_features)
    # ckpt = torch.load(file_path+"/model_weight/lucid_unet_large_192/epoch_7.pth",map_location="cpu")
    # model.load_state_dict(ckpt["model"])
    model = model.to("cuda:0")
    model = model.eval()


    print("----------------Half-Precision inference------------------------")
    
    model = model.half()
    ct = ct.half()
    
    print("----------------sliding_window_inference------------------------")
    
    with torch.no_grad():
        wb_pred = monai.inferers.sliding_window_inference(
                    ct,(192,192,192),
                    sw_batch_size=1,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                    sw_device="cuda:0",
                    device="cpu",
                    progress=True)
        wb_pred = torch.sigmoid(wb_pred.float())
        wb_pred[wb_pred < 0.5] = 0
    
    print("----------------post-process------------------------")
    
    if not os.path.exists( os.path.join(os.path.dirname(ct_path),outputdiranme)):
        os.mkdir(os.path.join(os.path.dirname(ct_path),outputdiranme))
    
    combined = torch.argmax(wb_pred[0,:output],dim=0).detach().cpu().numpy()
    # 创建SimpleITK图像
    sitk_image = sitk.GetImageFromArray(combined)
    # 设置方向和像素间距
    sitk_image.SetDirection(ct_itk.GetDirection())
    sitk_image.SetSpacing(ct_itk.GetSpacing())
    sitk_image.SetOrigin(ct_itk.GetOrigin())
    print("----------------file saving------------------------")
    sitk.WriteImage(sitk_image, os.path.join(os.path.dirname(ct_path),outputdiranme,f"combined.nii.gz"))
    print("create combined nii.gz. ")
    