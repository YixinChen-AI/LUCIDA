LUCID: 

## Usage
### step 1: installation
```
git clone git@github.com:YixinChen-AI/LUCID.git
cd LUCID
chmod 777 ./install.sh
./install.sh
```
还需要根据自己的CUDA版本安装torch，torch official website: https://pytorch.org/

### step 2: well-trained ckpt
将下载好的模型权重放到./model_weight/目录下.

## step 3: inference
1. 可以在python中调用
```
from lucid_utils_low import lucid
lucid("./testdata/autopet/CTres.nii.gz", # input ct nii.gz path
      outputpath="./testdata/totalseg/output123.nii.gz", # you can define the specific output segmentation nii.gz file to customized path. If you set None(default) to outputpath, the output file will be saved in "lucid_model_name/combiend.nii.gz".      modelname="STUNet_large",
      modelweight="./model_weight/lucid_STUNet_large_192e40.pth",
     output=112)
```

2. 可以在命令行中调用
```
python lucid.py --ct ct_nii_path --gpu 0
```

### step 4: output

输出结果会存储在ct_path同级目录下：

```
anypath/
│
├── ct1/
│ ├── ct.nii.gz
│ └── lucid
│  ├── combined.nii.gz

```

### update

1. v0.0. 2023/12/16.
    1. lucid_STUNet_large_192e65.pth, lucid_swinunetr_192e69, lucid_unet_large_192e65.pth.
    2. half inference. need 12GB GPU.
2. v0.1. 2023/12/28. duct_STUNet_large_192e05.pth.
    1. **Issue**: v0.0 cost too much CPU if input is too huge. Generaly for a whole-body CT image with 1.5mm spacing, the input could be larger than 700x300x300 and the inference proecss contain more than 100 192x192x192 patches. v0.0 need 2.2GB memory for each patches so for a large CT image, server need more than 128+ GB memory. **v0.1** remove sigmoid, which is useless in the argmax (1GB each patches); half inference.
