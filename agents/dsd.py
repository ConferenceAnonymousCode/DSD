import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, transforms_aug
from utils.utils import maybe_cuda
from utils.loss import SupConLoss, SupConLoss1
import copy
import torch.nn as nn
import torch.nn.functional as F
import json

class DeepSupervisionDistillation(ContinualLearner):
    def __init__(self, model, opt, params):
        super(DeepSupervisionDistillation, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.beta = params.beta
        self.temp = params.temp
        self.alpha = params.alpha
        self.theta = params.theta
        self.kld = nn.KLDivLoss(reduction="sum")
        self.contra_criterion = SupConLoss1()
    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        self.model = self.model.train()
        for ep in range(self.epoch):
            
        ############################ First Stage ############################################

            for i, batch_data in enumerate(train_loader):
                batch_x, batch_y = batch_data
                batch_x_aug = torch.stack([transforms_aug[self.data](batch_x[idx].cpu())
                                           for idx in range(batch_x.size(0))])
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_x_aug = maybe_cuda(batch_x_aug, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                batch_x_combine = torch.cat((batch_x, batch_x_aug))
                batch_y_combine = torch.cat((batch_y, batch_y))
                for j in range(self.mem_iters):
                    logits, feas, feat_list= self.model.pcrForward(batch_x_combine)
                    novel_loss = 0*self.criterion(logits, batch_y_combine)
                    self.opt.zero_grad()
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    if mem_x.size(0) > 0:
                        mem_x_aug = torch.stack([transforms_aug[self.data](mem_x[idx].cpu())
                                                 for idx in range(mem_x.size(0))])
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_x_aug = maybe_cuda(mem_x_aug, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_x_combine = torch.cat([mem_x, mem_x_aug])
                        mem_y_combine = torch.cat([mem_y, mem_y])
                        mem_logits, mem_fea, mem_fea_list= self.model.pcrForward(mem_x_combine)
                        combined_feas = torch.cat([mem_fea, feas])
                        combined_labels = torch.cat((mem_y_combine, batch_y_combine))
                        combined_feas_aug = self.model.pcrLinear.L.weight[combined_labels]

                        combined_feas_norm = torch.norm(combined_feas, p=2, dim=1).unsqueeze(1).expand_as(combined_feas)
                        combined_feas_normalized = combined_feas.div(combined_feas_norm + 0.000001)

                        combined_feas_aug_norm = torch.norm(combined_feas_aug, p=2, dim=1).unsqueeze(1).expand_as(
                            combined_feas_aug)
                        combined_feas_aug_normalized = combined_feas_aug.div(combined_feas_aug_norm + 0.000001)
                        cos_features = torch.cat([combined_feas_normalized.unsqueeze(1),
                                                  combined_feas_aug_normalized.unsqueeze(1)],
                                                 dim=1)
                        PSC = SupConLoss(temperature=self.temp, contrast_mode='proxy')
                        novel_loss += PSC(features=cos_features, labels=combined_labels)
                        
                        combined_x_batch = torch.cat([batch_x, mem_x])
                        combined_x_aug = torch.cat([batch_x_aug, mem_x_aug])
                        combined_x = torch.cat([combined_x_batch, combined_x_aug])
                        combined_y = torch.cat([batch_y, mem_y])
                        _, _, combined_feat_list= self.model.pcrForward(combined_x)                        
                        c_loss = 0
                        lbl_loss = 0
                        mtm_loss = 0
                        for index in range(len(combined_feat_list)):   
                            features = combined_feat_list[index]
                            if index > 0:
                                lbl_loss = distillation(features,feature_last_index)
                            feature_last_index = features.detach()   
                            f1, f2 = torch.split(features, [20, 20], dim=0)
                            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                            c_loss += self.contra_criterion(features=features, labels=combined_y) * self.alpha + lbl_loss * self.beta
                        novel_loss += c_loss
                        novel_loss.backward()
                        self.opt.step()
                        self.opt.zero_grad()
                        
    #######################################Second Stage######################################                        
                        _, _, combined_feat_list_new = self.model.pcrForward(combined_x)                
                        for index in range(len(combined_feat_list)):
                            features = combined_feat_list[index].detach()
                            features_new = combined_feat_list_new[index]                             
                            mtm_loss += torch.dist(features_new, features, p=2) * self.theta
                        mtm_loss.backward()
                        self.opt.step()
                        self.opt.zero_grad()
                self.buffer.update(batch_x, batch_y)
        self.after_train()
        
def distillation(student_scores, teacher_scores, T=2):
    p = F.log_softmax(student_scores / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / student_scores.shape[0]
    return l_kl

