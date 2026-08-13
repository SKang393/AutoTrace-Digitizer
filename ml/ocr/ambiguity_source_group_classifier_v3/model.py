# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
from __future__ import annotations
import torch
from torch import nn
from .protocol import GLYPHS,SEED

class SourceGroupAmbiguityNet(nn.Module):
    def __init__(self,seed:int=SEED)->None:
        super().__init__(); generator=torch.Generator().manual_seed(seed)
        self.convolution=nn.Sequential(nn.Conv2d(1,12,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(12,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(24,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.classifier=nn.Sequential(nn.Linear(32*4*4+70,96),nn.ReLU(),nn.Linear(96,len(GLYPHS)))
        for module in self.modules():
            if isinstance(module,(nn.Conv2d,nn.Linear)):
                nn.init.xavier_uniform_(module.weight,generator=generator)
                if module.bias is not None: nn.init.zeros_(module.bias)
    def forward(self,value:torch.Tensor)->torch.Tensor:
        convolution=self.convolution(value).flatten(1); row=value.mean(dim=3).squeeze(1); column=value.mean(dim=2).squeeze(1)
        row_peak=value.amax(dim=3).squeeze(1); column_peak=value.amax(dim=2).squeeze(1)
        geometry=torch.cat(((row_peak>0.08).float().mean(dim=1,keepdim=True),(column_peak>0.08).float().mean(dim=1,keepdim=True),
                            row[:,:16].mean(dim=1,keepdim=True),row[:,16:].mean(dim=1,keepdim=True),column[:,:16].mean(dim=1,keepdim=True),column[:,16:].mean(dim=1,keepdim=True)),dim=1)
        return self.classifier(torch.cat((convolution,row,column,geometry),dim=1))*0.0625
__all__=["SourceGroupAmbiguityNet"]
