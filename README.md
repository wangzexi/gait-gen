# gait-gen

## 环境配置

```Powershell
# 创建python3.9虚拟环境
conda create --name py39 python=3.9
conda activate py39 

# Windows下安装pytorch以及cudatookit 11.3
# 需要支持nvidia显卡
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install pytorch-lightning
pip install einops
pip install matplotlib

# 训练模型
python train.py
```

## 细节

https://zexi.notion.site/818242089dd9404a8d3445309c5d90a1
