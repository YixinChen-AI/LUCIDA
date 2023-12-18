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
      modelname="STUNet_large",
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
