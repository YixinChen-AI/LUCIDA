import argparse,os
import SimpleITK as sitk
import numpy as np
import torch,monai
from models.Mega_model import Mega_model
from imICR import imICR_prompt
from datautils import resampleVolume,adjust_image_direction
from tqdm import tqdm

# import monai.transforms as transforms


parser = argparse.ArgumentParser(description='')
parser.add_argument("--ct", help='ct path, current version only support: nii.gz')
parser.add_argument("--gpu", default="0",help='used gpu')
parser.add_argument("--direction",default=False,help=" Bool. different machine CT have direction dilemma. For example, Directions (-1,-1,1) are same with (-1,1,1) from other machine.")
args = parser.parse_args()

ct_path = args.ct

print(f"提供的NIfTI路径是:{ct_path}")
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

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
if args.direction == "False" or not args.direction:
    new_direction = (-1, 0, 0, 0, -1, 0, 0, 0, 1)
else:
    x_dir, y_dir, z_dir = map(int, input("Enter the directions for x, y, z axes (e.g., 1, 1, -1): ").split(','))
    new_direction = create_direction_matrix(x_dir, y_dir, z_dir)

direction_check = np.mean(np.abs(np.array(ct_itk.GetDirection()) - np.array(new_direction)))
spacing_check = np.mean(np.abs(np.array(ct_itk.GetSpacing()) - np.array([1.5,1.5,1.5])))

if direction_check < 0.05 and spacing_check < 0.05:
    print("direction check: OK!!")
    print("spacing check: OK!!")

else:
    print("direction is",np.array(ct_itk.GetDirection()))
    print("spacing is",np.array(ct_itk.GetSpacing()))
    print("direction need to be: ",new_direction)
    print("spacing need to be: [1.5,1.5,1.5]")
    
    
    print("Doing MEGACT Standard Protocol...")
    ct_itk = resampleVolume([1.5,1.5,1.5],ct_itk,resamplemethod=sitk.sitkLinear)

    ct_transformed = adjust_image_direction(ct_itk, new_direction)
    sitk.WriteImage(ct_transformed, ct_path.replace(".nii.gz","_megact_tmp.nii.gz"))
    print("standard protocol nii has been write in ",ct_path.replace(".nii.gz","megact_tmp.nii.gz"))
    print("please restart Megact and use file with _megact_tmp.nii.gz in the save path")
    exit()
    

print("----------------pre-process------------------------")

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

text_features = torch.load("./prompt_weight/clip_RN101_norm_137x512_HCS_large.pth")
model = Mega_model((192,192,192),in_channels=1,out_channels=text_features.shape[0],backbone = "STUNet_large",organ_embedding = text_features)
ckpt = torch.load("./model_weight/pcnet_STUNet_large_clip_RN101_192/epoch_180.pth",map_location="cpu")
model.load_state_dict(ckpt["model"])
model = model.to("cuda:0")
model = model.eval()

print("----------------sliding_window_inference------------------------")

with torch.no_grad():
    wb_pred = monai.inferers.sliding_window_inference(
                ct,(192,192,192),
                sw_batch_size=1,
                predictor=model,
                overlap=0.75,
                mode="gaussian",
                sw_device="cuda:0",
                device="cpu",
                progress=True)
    wb_pred = torch.sigmoid(wb_pred)

print("----------------post-process------------------------")
print("create combined nii.gz. ")
imICR_prompt = list(imICR_prompt.keys())

if not os.path.exists( os.path.join(os.path.dirname(args.ct),"megact")):
    os.mkdir(os.path.join(os.path.dirname(args.ct),"megact"))

combined = torch.argmax(wb_pred[0,:105],dim=0).detach().cpu().numpy()
# 创建SimpleITK图像
sitk_image = sitk.GetImageFromArray(combined)
# 设置方向和像素间距
sitk_image.SetDirection(ct_itk.GetDirection())
sitk_image.SetSpacing(ct_itk.GetSpacing())
sitk_image.SetOrigin(ct_itk.GetOrigin())
sitk.WriteImage(sitk_image, os.path.join(os.path.dirname(args.ct),"megact",f"combined.nii.gz"))
exit()
print("start to create single ROI in mega file... ")

wb_pred_binary = wb_pred.detach().cpu().numpy() > 0.5


for i in tqdm(range(wb_pred_binary.shape[1])):  # 遍历105个通道
    channel_data = wb_pred_binary[0, i, :, :, :]  # 获取当前通道的数据
    channel_data = channel_data.astype(np.float32)  # 转换数据类型

    # 创建SimpleITK图像
    sitk_image = sitk.GetImageFromArray(channel_data)

    # 设置方向和像素间距
    sitk_image.SetDirection(ct_itk.GetDirection())
    sitk_image.SetSpacing(ct_itk.GetSpacing())
    sitk_image.SetOrigin(ct_itk.GetOrigin())

    # 保存为NIfTI文件
    sitk.WriteImage(sitk_image, os.path.join(os.path.dirname(args.ct),"megact",f"{i:03d}_{imICR_prompt[i]}.nii.gz"))

print(ct.shape)  

