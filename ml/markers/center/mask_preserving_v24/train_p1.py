# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use V24 P1 fine-tune runner; execution is explicitly deferred."""
from __future__ import annotations
import argparse, hashlib, json, random, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import onnx, onnxruntime as ort, torch
from ml.markers.center.focal_confidence_v21.focal_loss import v21_loss
from ml.markers.center.scale_classifier_v16.model import ModelConfig, ScaleClassifierNet
from ml.markers.center.real_range_generator_v1.generator import audit as generator_audit, build_split
from ml.markers.center.metrics import center_metrics
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate, void_candidate
from .mask_preserving import extract_proposals, postprocess, prohibited_hits
from . import protocol

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = Path("ml/markers/center/mask_preserving_v24/training/p1.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/center/mask_preserving_v24/protocol.py"),
    Path("ml/markers/center/mask_preserving_v24/mask_preserving.py"),
    Path("ml/markers/center/mask_preserving_v24/FEASIBILITY.json"),
    Path("ml/markers/center/mask_preserving_v24/train_p1.py"),
    Path("ml/markers/center/focal_confidence_v21/focal_loss.py"),
    Path("ml/markers/center/scale_classifier_v16/model.py"),
    Path("ml/markers/center/real_range_generator_v1/generator.py"),
    Path("ml/markers/center/real_range_generator_v1/AUDIT.json"),
    Path("ml/markers/center/metrics.py"),
    Path("ml/policy/evidence-policy.json"),
    Path("ml/policy/acceptance-bars.json"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)

def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _configure(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

def _examples(scenes, maximum_negative_per_positive: int, generator: torch.Generator):
    values = [[] for _ in range(5)]
    for scene in scenes:
        proposals = extract_proposals(scene.tensor)
        coords = proposals.coordinates; centers = torch.tensor(scene.centers, dtype=torch.float32); radii = torch.tensor(scene.diameters, dtype=torch.float32) / 2.0
        distance = torch.cdist(coords, centers); nearest, nearest_index = distance.min(dim=1); labels = nearest.le(3.0).float()
        hard = torch.zeros(len(coords), dtype=torch.bool)
        for _, x, y in scene.hard_negatives:
            hard |= torch.cdist(coords, torch.tensor(((x, y),), dtype=torch.float32)).squeeze(1).le(8.0)
        positive = torch.nonzero(labels > .5).flatten(); negative = torch.nonzero(labels <= .5).flatten(); hard_indices = torch.nonzero(hard & (labels <= .5)).flatten()
        remaining = torch.tensor(sorted(set(negative.tolist()) - set(hard_indices.tolist())), dtype=torch.long)
        budget = max(len(positive) * maximum_negative_per_positive, len(hard_indices)); random_budget = max(0, budget-len(hard_indices))
        if len(remaining) > random_budget: remaining = remaining[torch.randperm(len(remaining), generator=generator)[:random_budget]]
        selected = torch.cat((positive, hard_indices, remaining)).unique(sorted=True)
        values[0].append(proposals.patches[selected]); values[1].append(labels[selected]); values[2].append((centers[nearest_index]-coords).index_select(0,selected)/4.0); values[3].append(radii[nearest_index].index_select(0,selected)); values[4].append(hard[selected])
    return tuple(torch.cat(part) for part in values)

def _evaluate(scenes, model, threshold):
    tp=fp=fn=dup=hits=truth=proposal_tp=0
    for scene in scenes:
        proposals=extract_proposals(scene.tensor); output=model(proposals.patches).detach().numpy(); predictions=postprocess(scene, proposals, output) if threshold == .25 else postprocess(scene, proposals, output)
        # Sensitivity thresholds are represented by filtering model probabilities here.
        predictions=tuple(p for p in predictions if p.confidence >= threshold)
        truth += len(scene.centers)
        edges=sorted((float(np.hypot(p.x-x,p.y-y)),i,j) for i,p in enumerate(predictions) for j,(x,y) in enumerate(scene.centers) if np.hypot(p.x-x,p.y-y)<=5)
        usedp=set(); usedt=set()
        for _,i,j in edges:
            if i not in usedp and j not in usedt: usedp.add(i); usedt.add(j)
        proposal_truths = {
            truth_index for x, y in proposals.coordinates.tolist()
            for truth_index, (truth_x, truth_y) in enumerate(scene.centers)
            if np.hypot(x-truth_x, y-truth_y) <= 5
        }
        metrics = center_metrics(predictions, scene.centers, 5.0)
        tp += len(usedp); fp += len(predictions)-len(usedp); fn += len(scene.centers)-len(usedt); proposal_tp += len(proposal_truths); dup += metrics.duplicate_count; hits += sum(prohibited_hits(predictions,scene).values())
    precision=tp/max(1,tp+fp); recall=tp/max(1,truth); f1=2*precision*recall/max(1e-12,precision+recall)
    return {"threshold":threshold,"scene_count":len(scenes),"proposal_true_positives":proposal_tp,"proposal_recall":proposal_tp/max(1,truth),"true_positives":tp,"false_positives":fp,"false_negatives":fn,"precision":precision,"recall":recall,"f1":f1,"duplicate_count":dup,"prohibited_structure_hits":hits}

def run(output_dir: Path, checkpoint: Path, v21_onnx: Path) -> dict:
    if output_dir.exists(): raise RuntimeError(f"candidate output already exists: {output_dir}")
    config=json.loads((REPO_ROOT/CONFIG_PATH).read_text(encoding="utf-8"))
    if _sha(checkpoint) != config["checkpoint_sha256"]: raise ValueError("V21 checkpoint hash changed")
    if _sha(v21_onnx) != config["v21_onnx_sha256"]: raise ValueError("V21 ONNX hash changed")
    for path_key, hash_key in (("feasibility_path","feasibility_sha256"),("generator_audit_path","generator_audit_sha256"),("evidence_policy_path","evidence_policy_sha256"),("acceptance_bars_path","acceptance_bars_sha256")):
        if _sha(REPO_ROOT/str(config[path_key])) != config[hash_key]: raise ValueError(f"bound input changed: {config[path_key]}")
    authorization=acquire_training_candidate(REPO_ROOT,task=protocol.TASK,revision=protocol.TRAINING_REVISION,candidate_id=protocol.TRAINING_CANDIDATE_ID,config_path=CONFIG_PATH,runner_source_paths=RUNNER_SOURCE_PATHS)
    output_dir.mkdir(parents=True); report_path=output_dir/"candidate-report.json"; started=time.perf_counter(); phase="initialization"
    try:
        _configure(config["seed"]); split_audit=generator_audit()
        if split_audit["splits"]["train"]["aggregate_sha256"] != config["train_split_sha256"] or split_audit["splits"]["dev"]["aggregate_sha256"] != config["dev_split_sha256"]: raise RuntimeError("corrected generator split identity changed")
        train=build_split("train"); dev=build_split("dev"); gen=torch.Generator().manual_seed(config["seed"]+1); patches,labels,offsets,radii,hard=_examples(train,config["maximum_negative_per_positive"],gen)
        if len(labels) != config["training_example_count_expected"] or int((labels>.5).sum()) != config["positive_example_count_expected"] or int(hard.sum()) != config["hard_negative_example_count_expected"]: raise RuntimeError("training example contract changed")
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False); model=ScaleClassifierNet(ModelConfig(seed=20260902)); model.load_state_dict(payload["state_dict"]); optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"]); steps=0; phase="training"; model.train()
        for _ in range(config["epochs"]):
            order=torch.randperm(len(labels),generator=gen)
            for start in range(0,len(labels),config["batch_size"]):
                index=order[start:start+config["batch_size"]]; loss=v21_loss(model.forward_raw(patches[index]),labels[index],offsets[index],radii[index],hard[index],positive_weight=config["positive_loss_weight"],hard_weight=config["hard_negative_loss_weight"],alpha=config["focal_alpha"],gamma=config["focal_gamma"]); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step(); steps+=1
        if steps != config["optimizer_steps_expected"] or steps > config["optimizer_steps_maximum"]: raise RuntimeError(f"optimizer step contract changed: {steps}")
        phase="dev"; model.eval(); comparisons=[_evaluate(dev,model,t) for t in [0.25,*config["selection_thresholds"]]]; selected=comparisons[0]; bar=config["acceptance_bar"]; dev_passed=selected["proposal_recall"]>=bar["proposal_recall_minimum"] and selected["precision"]>=bar["precision_minimum"] and selected["recall"]>=bar["recall_minimum"] and selected["duplicate_count"]<=bar["duplicates_maximum"] and selected["prohibited_structure_hits"]<=bar["prohibited_hits_maximum"]
        phase="export"; out_pt=output_dir/"marker-center-mask-preserving-v24-p1.pt"; torch.save({"state_dict":model.state_dict(),"config":model.export_contract()},out_pt); out_onnx=output_dir/"marker-center-mask-preserving-v24-p1.onnx"; torch.onnx.export(model,torch.zeros((1,3,33,33)),out_onnx,input_names=["candidate_patches"],output_names=["candidate_predictions"],dynamic_axes={"candidate_patches":{0:"candidate_count"},"candidate_predictions":{0:"candidate_count"}},opset_version=18,dynamo=False); onnx.checker.check_model(onnx.load(out_onnx)); session=ort.InferenceSession(str(out_onnx),providers=[protocol.PROVIDER]);
        if session.get_providers()[0] != protocol.PROVIDER: raise RuntimeError("CPUExecutionProvider was not selected")
        parity=[]; parity_source=extract_proposals(dev[0].tensor).patches
        for count in [1,8,37]:
            x=parity_source[:count].contiguous(); expected=model(x).detach().numpy(); actual=session.run(["candidate_predictions"],{"candidate_patches":x.numpy()})[0]; parity.append({"candidate_count":count,"maximum_absolute_error":float(np.max(np.abs(expected-actual)))})
        report={"schema":"graphreader.marker-center-mask-preserving-v24-candidate.v1","task":protocol.TASK,"revision":protocol.TRAINING_REVISION,"candidate_id":protocol.TRAINING_CANDIDATE_ID,"status":"dev_passed" if dev_passed and max(r["maximum_absolute_error"] for r in parity)<=config["onnx_parity_tolerance"] else "failed_dev","synthetic_only":True,"private_data":False,"real_dev_reads":0,"real_sealed_reads":0,"sealed_runs":0,"optimizer_steps":steps,"training_example_count":len(labels),"positive_example_count":int((labels>.5).sum()),"hard_negative_example_count":int(hard.sum()),"dev_comparisons":comparisons,"selected":selected,"dev_gate_passed":dev_passed,"checkpoint_sha256":_sha(out_pt),"onnx_sha256":_sha(out_onnx),"v21_checkpoint_sha256":config["checkpoint_sha256"],"v21_onnx_sha256":config["v21_onnx_sha256"],"onnx_provider":protocol.PROVIDER,"onnx_dynamic_candidate_counts":parity,"onnx_parity_maximum_absolute_error":max(r["maximum_absolute_error"] for r in parity),"elapsed_ms":round((time.perf_counter()-started)*1000,3),"production_approval":False,"release_eligible":False}
    except Exception as error:
        report={"schema":"graphreader.marker-center-mask-preserving-v24-failure.v1","task":protocol.TASK,"revision":protocol.TRAINING_REVISION,"candidate_id":protocol.TRAINING_CANDIDATE_ID,"status":"failed_runner","phase":phase,"exception_type":type(error).__name__,"exception_message":str(error),"synthetic_only":True,"private_data":False,"real_dev_reads":0,"real_sealed_reads":0,"sealed_runs":0}
        report_path.write_bytes(canonical_json_bytes(report)); void_candidate(authorization,error); raise
    report_path.write_bytes(canonical_json_bytes(report)); complete_training_candidate(authorization,status=report["status"],report_sha256=sha256_file(report_path)); return report

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--onnx",type=Path,required=True); a=p.parse_args(); print(json.dumps(run(a.output_dir.resolve(),a.checkpoint.resolve(),a.onnx.resolve()),indent=2,sort_keys=True))
if __name__ == "__main__": main()
