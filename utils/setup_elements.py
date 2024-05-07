import torch
from models.resnet import Reduced_ResNet18, SupConResNet, ResNet34, ResNet50, ResNet101, ResNet152
from torchvision import transforms
import torch.nn as nn


default_trick = {'labels_trick': False, 'kd_trick': False, 'separated_softmax': False,
                 'review_trick': False, 'ncm_trick': False, 'kd_trick_star': False}


input_size_match = {
    'cifar100': [3, 32, 32],
    'cifar10': [3, 32, 32],
    'core50': [3, 128, 128],
    'mini_imagenet': [3, 84, 84],
    'openloris': [3, 50, 50]
}


n_classes = {
    'cifar100': 100,
    'cifar10': 10,
    'core50': 50,
    'mini_imagenet': 100,
    'openloris': 69
}


transforms_match = {
    'core50': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'mini_imagenet': transforms.Compose([
        transforms.ToTensor()]),
    'openloris': transforms.Compose([
            transforms.ToTensor()])
}


transforms_aug = {
    'cifar100': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
        ]),
    'cifar10': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
        ]),
    'mini_imagenet': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomResizedCrop(size=84, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821))
        ])
}

class cosLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(cosLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias = False)
        self.scale = 0.09

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.000001)

        L_norm = torch.norm(self.L.weight, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
        weight_normalized = self.L.weight.div(L_norm + 0.000001)
        cos_dist = torch.mm(x_normalized,weight_normalized.transpose(0,1))
        scores = cos_dist / self.scale
        return scores
    
def setup_architecture(params):
    nclass = n_classes[params.data]
    if params.agent in ['SCR', 'SCP']:
        if params.data == 'mini_imagenet':
            return SupConResNet(640, head=params.head)
        return SupConResNet(head=params.head)
    if params.agent == 'CNDPM':
        from models.ndpm.ndpm import Ndpm
        return Ndpm(params)
    if params.data == 'cifar100':
        model = Reduced_ResNet18(nclass)
        model.pcrLinear = cosLinear(160, nclass)
        model.linear = nn.Linear(160, nclass, bias=True)
        return model
    elif params.data == 'cifar10':
        model = Reduced_ResNet18(nclass)
        model.linear = nn.Linear(160, nclass, bias=True)
        return model
    elif params.data == 'mini_imagenet':
        model = Reduced_ResNet18(nclass)
        model.pcrLinear = cosLinear(640, nclass)
        model.linear = nn.Linear(640, nclass, bias=True)
        return model
     
def setup_opt(optimizer, model, lr, wd):
    if optimizer == 'SGD':
        optim = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                weight_decay=wd)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=wd)
    else:
        raise Exception('wrong optimizer name')
    return optim
