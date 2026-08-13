# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Exact full-line source-group extraction used by training and composition."""

from __future__ import annotations
import math
import numpy as np
from PIL import Image

IMAGE_SIZE = 32

def active_groups(crop: Image.Image) -> tuple[tuple[int,int,int,int], ...]:
    gray=np.asarray(crop.convert("L"),dtype=np.float32)
    if gray.ndim!=2 or min(gray.shape)==0: return ()
    edge=np.concatenate((gray[0],gray[-1],gray[:,0],gray[:,-1])); background=float(np.median(edge))
    contrast=max(0.0,background-float(np.percentile(gray,1))); foreground=gray<=background-max(10.0,contrast*0.30)
    coordinates=np.argwhere(foreground)
    if not len(coordinates): return ()
    top,left=coordinates.min(axis=0); bottom,right=coordinates.max(axis=0); active=np.flatnonzero(foreground.any(axis=0))
    gap=max(5,int(math.ceil(int(bottom-top+1)*0.40))); starts=[int(active[0])]; ends=[]
    for prior,current in zip(active[:-1],active[1:],strict=True):
        if int(current-prior-1)>=gap: ends.append(int(prior)); starts.append(int(current))
    ends.append(int(active[-1])); result=[]
    for group_left,group_right in zip(starts,ends,strict=True):
        rows=np.where(foreground[:,group_left:group_right+1].any(axis=1))[0]
        result.append((group_left,int(rows[0]),group_right+1,int(rows[-1])+1))
    return tuple(result)

def group_tensor(crop: Image.Image, groups: tuple[tuple[int,int,int,int],...], index: int) -> np.ndarray:
    if not 0<=index<len(groups): raise ValueError("Source-group index is out of range")
    left,top,right,bottom=groups[index]; gray=crop.convert("L").crop((left,top,right,bottom))
    line_height=max(group[3]-group[1] for group in groups); baseline=max(group[3] for group in groups)
    scale=11.5/max(1,line_height); glyph=gray.resize((max(1,round(gray.width*scale)),max(1,round(gray.height*scale))),Image.Resampling.BILINEAR)
    canvas=Image.new("L",(IMAGE_SIZE,IMAGE_SIZE),255); paste_x=(IMAGE_SIZE-glyph.width)//2; paste_y=int(round(21-(baseline-top)*scale)); canvas.paste(glyph,(paste_x,paste_y))
    return (1.0-np.asarray(canvas,dtype=np.float32)/255.0)[None,:,:].astype(np.float32)

__all__=["IMAGE_SIZE","active_groups","group_tensor"]
