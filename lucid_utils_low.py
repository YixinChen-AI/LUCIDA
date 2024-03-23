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

def lucid(ct_path,output_seg_path,output_stdct_path=None,check=True,modelname=None,modelweight=None,output=None,adaptor=None):
    
    
    print(f"提供的NIfTI路径是:{ct_path}")
    file_path = os.path.dirname(os.path.abspath(__file__))
    
    orict_itk = sitk.ReadImage(os.path.join(ct_path))
    ct_itk = sitk.ReadImage(os.path.join(ct_path))
    
    print("----------------direction check and spacing check------------------------")

    print("before processing, spacing:",ct_itk.GetSpacing())
    print("before processing, direction:",ct_itk.GetDirection())
    
    new_direction = (-1, 0, 0, 0, -1, 0, 0, 0, 1)
    # new_origin = ct_itk.GetOrigin()  # 保持原点不变
    new_spacing = (1.5,1.5,1.5)  # 保持间距不变
    # reference_image = sitk.Image(ct_itk.GetSize(), ct_itk.GetPixelID())
    # reference_image.SetOrigin(new_origin)
    # reference_image.SetSpacing(new_spacing)
    # reference_image.SetDirection(new_direction)
    # identity_transform = sitk.Transform(3, sitk.sitkIdentity)
    # ct_itk = sitk.Resample(ct_itk, reference_image, identity_transform, sitk.sitkLinear, -1000.0)


    
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


    print("after processing, spacing:",ct_itk.GetSpacing())
    print("after processing, direction:",ct_itk.GetDirection())


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

    # import matplotlib.pyplot as plt
    # plt.imshow(ct[20],cmap="gray")
    # plt.show()
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
                            overlap=0,
                            mode="constant",
                            sw_device="cuda:0",
                            device="cpu",
                            progress=True)
                # wb_pred = torch.sigmoid(wb_pred.float())
                # wb_pred[wb_pred < 0.5] = 0
                wb_preds += wb_pred
        wb_pred = wb_preds / len(modelname)
    else:    
        print("single model mode!!")
        model = load_model(modelname)
        ckpt = torch.load(modelweight,map_location="cpu")

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():  # 假设权重存储在'ckpt['model']'中
            name = k[7:] if k.startswith('module.') else k  # 移除 'module.' 前缀
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
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
            wb_pred = wb_pred[:,:145]
    print("----------------post-process------------------------")

    combined = torch.argmax(wb_pred[0],dim=0).detach().cpu()
    if output is not None:
        new_pred = torch.zeros_like(combined)
        for idx, labels in enumerate(output):
            if isinstance(labels, list):
                # 如果 labels 是一个列表，表示我们要取这些标签的并集
                mask = torch.zeros_like(combined, dtype=torch.bool)
                for label in labels:
                    mask |= (combined == label)
                new_pred[mask] = idx + 1  # 使用 idx + 1 作为新的标签索引
            else:
                # 否则，labels 就是一个单独的标签
                new_pred[combined == labels] = idx + 1
        combined = new_pred
    combined = combined.numpy()
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
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(orict_itk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    sitk_image = resampler.Execute(sitk_image)
    sitk_image = sitk.Cast(sitk_image, sitk.sitkUInt8)
    
    sitk.WriteImage(sitk_image, output_seg_path)
    print("create combined nii.gz. ",output_seg_path)
    