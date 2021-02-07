import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch
import model.backbone as backbone
import torch.nn.functional as F
from torch.autograd import Function
import math
import numpy as np


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DPN(nn.Module):

    def __init__(self, num_classes, base_net='ResNet50', temperature=0.05):
        super(DPN, self).__init__()
        
        self.classes = num_classes
        self.tmp = temperature
        
        # Feature Extractor
        self.feature = backbone.network_dict[base_net]()
        
        # Bottleneck
        self.bottleneck = nn.Linear(2048, 256)
        
        # Class Classifier
        self.class_classifier = nn.Linear(256, self.classes, bias=False)
        
        # Domain Discriminator
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('fc1', nn.Linear(256, 1024))
        self.domain_classifier.add_module('relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dpt1', nn.Dropout(0.5))
        self.domain_classifier.add_module('fc2', nn.Linear(1024, 1024))
        self.domain_classifier.add_module('relu2', nn.ReLU(True))
        self.domain_classifier.add_module('dpt2', nn.Dropout(0.5))
        self.domain_classifier.add_module('fc3', nn.Linear(1024, 2))       

    def forward(self, x, alpha=0.0):
        
        x_feat = self.feature(x)
        x_feat = self.bottleneck(x_feat)
  
        # Normalize
        x_feat_norm = F.normalize(x_feat, dim=1)
        
        # Class Classifier Weight Normalize
        w = self.class_classifier.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.class_classifier.weight.data = w.div(norm.expand_as(w))
        
        # Class Classifier
        class_output = torch.div(self.class_classifier(x_feat_norm), self.tmp)
        
        if self.training == True:   
            # RevGrad
            reverse_feature = ReverseLayerF.apply(x_feat, alpha)   
            domain_output = self.domain_classifier(reverse_feature)
            
        else:
            domain_output = 0
            
        return x_feat, class_output, domain_output

