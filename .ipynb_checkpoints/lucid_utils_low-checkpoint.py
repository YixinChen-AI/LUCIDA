import sys,os
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path)
import argparse,os
import SimpleITK as sitk
import numpy as np
import torch,monai
import torch.nn as nn
from datautils import resampleVolume,adjust_image_direction
from tqdm import tqdm
from lucidmodel.STUNet import STUNet
from lucidutils import load_model

def lucid(ct_path,output_seg_path,output_stdct_path=None,check=True,modelname=None,modelweight=None,output=105,adaptor=None):
    
    
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

    if check:
        if direction_check < 0.05:
            print("direction check: OK!!")
        else:
            print("direction is",np.array(ct_itk.GetDirection()))
            print("direction need to be: ",new_direction)
                
            ct_itk = adjust_image_direction(ct_itk, new_direction)
            
    else:
        print("arg chech is set to False so no direction check!!")


    if output_stdct_path is not None:
        output_stdct_path_ = os.path.dirname(output_stdct_path)
        if not os.path.exists(output_stdct_path_):
            os.makedirs(output_stdct_path_)
            print(f"目录已创建：{output_stdct_path_}")
        sitk.WriteImage(ct_itk, output_stdct_path)
        print("standard protocol nii has been write in ",output_stdct_path)
    else:
        print("if need to save CT.nii.gz file in standard protocol (1.5mm), use arg <output_stdct_path>")
    
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
    if isinstance(modelname,list):
        print("emsemble mode!!")
        wb_preds = 0
        for mn,mn_ckpt in zip(modelname,modelweight):
            model = load_model(mn)
            ckpt = torch.load(mn_ckpt,map_location="cpu")
            model.load_state_dict(ckpt["model"])
    
            model = model.to("cuda:0")
            model = model.half()
            model = model.eval()
    
            ct = ct.half()
    
            with torch.no_grad():
                wb_pred = monai.inferers.sliding_window_inference(
                            ct,(192,192,192),
                            sw_batch_size=1,
                            predictor=model,
                            overlap=0.5,
                            mode="constant",
                            sw_device="cuda:0",
                            device="cpu",
                            progress=True)
                # wb_pred = torch.sigmoid(wb_pred.float())
                wb_preds += wb_pred
        wb_pred = wb_preds / len(modelname)
    else:    
        print("single model mode!!")
        model = load_model(modelname)
        ckpt = torch.load(modelweight,map_location="cpu")
        model.load_state_dict(ckpt["model"])
            
        model = model.to("cuda:0")
        model = model.half()
        model = model.eval()
    
        if adaptor is not None:
            print("-----------------Adaptor is used! use: {}------------------------------".format(adaptor["name"]))
            from adaptor import FourierTransform,Transform
            if adaptor["name"] == "FT":
                FT = FourierTransform(input_channel=2)
                FT.load_state_dict(torch.load(adaptor["ckpt"])["model"])
                FT = FT.to("cuda:0")
                FT = FT.eval()
                model = nn.Sequential(FT,model)
            if adaptor["name"] == "T":
                T = Transform()
                T.load_state_dict(torch.load(adaptor["ckpt"])["model"])
                T = T.half()
                T = T.to("cuda:0")
                T = T.eval()
                model = nn.Sequential(T,model)
        print("----------------Half-Precision inference------------------------")
        
        ct = ct.half()
    
        class SelectChannels(nn.Module):
            def __init__(self):
                super(SelectChannels, self).__init__()
            def forward(self, x):
                return x[:, :112]
        model = nn.Sequential(model,SelectChannels())
        
        print("----------------sliding_window_inference------------------------")
        
        with torch.no_grad():
            wb_pred = monai.inferers.sliding_window_inference(
                        ct,(192,192,192),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.5,
                        mode="constant",
                        sw_device="cuda:0",
                        device="cpu",
                        progress=True)
            # wb_pred = torch.sigmoid(wb_pred.float())
            # wb_pred[wb_pred < 0.5] = 0
    
    print("----------------post-process------------------------")
    
    combined = torch.argmax(wb_pred[0,:output],dim=0).detach().cpu().numpy()
    # 创建SimpleITK图像
    sitk_image = sitk.GetImageFromArray(combined)
    # 设置方向和像素间距
    sitk_image.SetDirection(ct_itk.GetDirection())
    sitk_image.SetSpacing(ct_itk.GetSpacing())
    sitk_image.SetOrigin(ct_itk.GetOrigin())
    print("----------------file saving------------------------")
    output_seg_path_ = os.path.dirname(output_seg_path)
    if not os.path.exists(output_seg_path_):
        os.makedirs(output_seg_path_)
        print(f"目录已创建：{output_seg_path_}")
    sitk.WriteImage(sitk_image, output_seg_path)
    print("create combined nii.gz. ")
    