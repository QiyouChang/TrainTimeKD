This is the new repository for CMU 10623 Semester Project: Train-Time Knowledge distillation codespace. 

# Installation

```bash
git clone https://github.com/QiyouChang/TrainTimeKD.git
cd TrainTimeKD
conda env create -f environment.yml
conda activate vit
```

# Usage

## Method A: Vision Transformer (ViT)

To train ViT models solely, you need to specify the dataset config file in `configs/dataset` and the ViT config file in `config/vit`. For example, to train ViT-Base/16 on CIFAR-10 with mixed precision, you can run the following command:

```bash
python train_vit.py --model configs/vit/vit_base.yaml --data configs/dataset/cifar10.yaml --gpus 0 --set model.args.patch_size=16 --force --mixed
```

If you want to train on multiple gpus, then you can specify them in the `--gpus` argument. For example, to train ViT-Huge/14 on CIFAR-10 with mixed precision on gpus `0,1,2,3`, you can run the following command:

```bash
python train_vit.py --model configs/vit/vit_huge.yaml --data configs/dataset/cifar10.yaml --gpus 0,1,2,3 --set model.args.patch_size=14 --force --mixed
```

To resume training from a checkpoint (eg. epoch 100), you can run the following command:

```bash
python train_vit.py --model configs/vit/vit_base.yaml --data configs/dataset/cifar10.yaml --gpus 0 --set model.args.patch_size=16 resume=100 --force --mixed
```

## Method B: Valina Knowledge Distillation (vkd)

To train the valina knowledge distillation model, you need to specify the dataset config file in `configs/dataset` and the vkd config file `config/kd/vkd.yaml`. For example, to train vkd on CIFAR-10, you can run the following command:

```bash
python train_vkd.py --model configs/kd/vkd.yaml --data configs/dataset/cifar10.yaml --gpus 0 --force
```

## Method C: Co-learning Knowledge Distillation (ckd)

To train the valina knowledge distillation model, you need to specify the dataset config file in `configs/dataset` and the ckd config file `config/kd/ckd.yaml`. For example, to train ckd on CIFAR-10, you can run the following command:

```bash
python train_ckd.py --model configs/kd/ckd.yaml --data configs/dataset/cifar10.yaml --gpus 0 --force
```

# Results

Our training logs on 200 configurations trained over 2000 hours can be checked in wandb for [CIFAR-10](https://wandb.ai/baolef-cmu/CIFAR10) and [CIFAR-100](https://wandb.ai/baolef-cmu/CIFAR100).
