# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fresh multi-group lines for the exact production source-group adapter."""
from __future__ import annotations
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
from typing import Any
import zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageEnhance,ImageFilter,ImageFont
from .crop import active_groups,group_tensor
from .protocol import COUNTS_PER_CLASS,GLYPHS,SEED

REPO_ROOT=Path(__file__).resolve().parents[3]
FONT_PATHS=(Path("src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"),Path("src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf"),Path("src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf"))
OFFSETS={"train":0,"validation":170_000,"sealed_public":520_000}
CONTEXT=("A","7","B","2","C","9")
def canonical_json_bytes(value:Any)->bytes:return (json.dumps(value,allow_nan=False,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n").encode()
def hash_bytes(value:bytes)->str:return sha256(value).hexdigest()

def _render(partition:str,label:int,index:int)->tuple[bytes,np.ndarray,dict[str,Any]]:
    seed=SEED+OFFSETS[partition]+label*10000+index; rng=random.Random(seed); np_rng=np.random.default_rng(seed); glyph=GLYPHS[label]
    font_relative=FONT_PATHS[(index*5+label*3+OFFSETS[partition])%3]; font_path=REPO_ROOT/font_relative; font_size=18+(index*11+label*7)%15; font=ImageFont.truetype(str(font_path),font_size)
    prefix=CONTEXT[(index+label)%len(CONTEXT)]; suffix=CONTEXT[(index*3+label+2)%len(CONTEXT)]; tokens=(prefix,glyph,suffix)
    ascent,descent=font.getmetrics(); line_height=ascent+descent; probe=ImageDraw.Draw(Image.new("L",(1,1)))
    advances=[max(1,int(round(probe.textlength(token,font=font)))) for token in tokens]; gap=max(6,int(math.ceil(font_size*0.42)))
    width=sum(advances)+gap*2+18; canvas=Image.new("L",(width,line_height+16),255); draw=ImageDraw.Draw(canvas); x=9; baseline=8+ascent; foreground=8+(index*13+label*17)%44
    for token,advance in zip(tokens,advances,strict=True): draw.text((x,baseline),token,font=font,anchor="ls",fill=foreground); x+=advance+gap
    canvas=canvas.transform(canvas.size,Image.Transform.AFFINE,(1.0,rng.uniform(-0.025,0.025),rng.uniform(-0.25,0.25),0.0,1.0,rng.uniform(-0.25,0.25)),resample=Image.Resampling.BICUBIC,fillcolor=255)
    if index%3==1: canvas=ImageEnhance.Contrast(canvas).enhance(0.92+0.08*rng.random())
    elif index%3==2: canvas=canvas.filter(ImageFilter.GaussianBlur(0.10+0.18*rng.random()))
    array=np.asarray(canvas,dtype=np.int16); array=np.clip(array+np_rng.integers(-2,3,size=array.shape,dtype=np.int16),0,255).astype(np.uint8); canvas=Image.fromarray(array,"L")
    groups=active_groups(canvas)
    if len(groups)!=3: raise RuntimeError(f"Expected three source groups, found {len(groups)}")
    tensor=group_tensor(canvas,groups,1); stream=BytesIO(); canvas.save(stream,format="PNG",compress_level=9); source=stream.getvalue(); sample_id=f"{partition}-source-group-{label}-{index:04d}"
    return source,tensor,{"sample_id":sample_id,"source_path":f"fixtures/{sample_id}.png","source_sha256":hash_bytes(source),"glyph":glyph,"label":label,"target_group_index":1,"group_count":3,"context_tokens":[prefix,suffix],"font_path":font_relative.as_posix(),"font_sha256":hash_bytes(font_path.read_bytes()),"font_size":font_size,"renderer_family":f"noto-full-line-source-group-v3-{partition}","degradation_family":f"source-group-affine-v3-{index%3}","private_or_article_image":False,"chandler_image":False}

def build_partition(partition:str)->tuple[bytes,bytes,np.ndarray,np.ndarray]:
    records=[];values=[];labels=[];stream=BytesIO()
    with zipfile.ZipFile(stream,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for label in range(len(GLYPHS)):
            for index in range(COUNTS_PER_CLASS[partition]):
                source,tensor,record=_render(partition,label,index); info=zipfile.ZipInfo(record["source_path"],date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;archive.writestr(info,source);records.append(record);values.append(tensor);labels.append(label)
    manifest={"schema":"graphreader.ocr-ambiguity-source-group-split.v1","partition":partition,"seed":SEED+OFFSETS[partition],"class_order":list(GLYPHS),"count_per_class":COUNTS_PER_CLASS[partition],"sample_count":len(records),"synthetic_only":True,"private_or_article_images":False,"chandler_included":False,"generalization_label_included":False,"samples":records}
    return canonical_json_bytes(manifest),stream.getvalue(),np.stack(values),np.asarray(labels,dtype=np.int64)
def split_fingerprint(partition:str)->str:
    manifest,archive,values,labels=build_partition(partition);digest=sha256();[digest.update(x) for x in (manifest,archive,np.ascontiguousarray(values).tobytes(),np.ascontiguousarray(labels).tobytes())];return digest.hexdigest()
def write_freeze(root:Path)->dict[str,Any]:
    if root.exists():raise RuntimeError(f"split root exists: {root}")
    summary={"schema":"graphreader.ocr-ambiguity-source-group-freeze.v1","partitions":{}}
    for partition in COUNTS_PER_CLASS:
        manifest,archive,values,labels=build_partition(partition);target=root/partition;target.mkdir(parents=True);(target/"private-manifest.json").write_bytes(manifest);(target/"fixtures.zip").write_bytes(archive)
        summary["partitions"][partition]={"sample_count":len(labels),"count_per_class":COUNTS_PER_CLASS[partition],"private_manifest_path":(target/"private-manifest.json").relative_to(REPO_ROOT).as_posix(),"private_manifest_sha256":hash_bytes(manifest),"fixture_archive_path":(target/"fixtures.zip").relative_to(REPO_ROOT).as_posix(),"fixture_archive_sha256":hash_bytes(archive),"fixture_archive_bytes":len(archive),"tensor_label_stream_sha256":hash_bytes(np.ascontiguousarray(values).tobytes()+np.ascontiguousarray(labels).tobytes()),"split_fingerprint":split_fingerprint(partition)}
    return summary
import math
__all__=["REPO_ROOT","build_partition","canonical_json_bytes","hash_bytes","split_fingerprint","write_freeze"]
