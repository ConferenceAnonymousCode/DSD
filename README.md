Deep-Supervision-Distillation
Source Code for anonymous conference submission, 'Overcoming Intermediate Layer Forgetting for Online Class-Incremental Continual Learning'.

### Requirements
pip install -r requirements.txt

### Data preparation
- CIFAR10 & CIFAR100 will be downloaded during the first run. (datasets/cifar10;/datasets/cifar100)
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download, and place it in datasets/mini_imagenet/

### Run on the datasets
sh run.sh
