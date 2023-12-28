import argparse,os
from lucid_utils_low import lucid

# import monai.transforms as transforms


parser = argparse.ArgumentParser(description='')
parser.add_argument("--ct", help='ct path, current version only support: nii.gz')
parser.add_argument("--gpu", default="0",help='used gpu. default:0')
args = parser.parse_args()

ct_path = args.ct

print(f"提供的NIfTI路径是:{ct_path}")
print(f"当前使用的GPU index:{args.gpu}")
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

lucid(ct_path,
      modelname="STUNet_large",
      modelweight="./model_weight/duct_STUNet_large_192e05.pth",
     output=112)
