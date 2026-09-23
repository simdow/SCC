import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from .data import load_scale, preprocess
from .model import SCCNet
from .metrics import matrices, metrics, window_starts


def predict_scan(model, frames, scale):
    chunks, previous_end = [], -1
    for begin in window_starts(len(frames)):
        x = preprocess(frames[begin:begin+5]).unsqueeze(0).to(scale.device)
        with torch.no_grad():
            result = model(x)
            pose = ((result['heads'][0] + result['heads'][1]) * .5 * scale)[0]
        chunks.append(pose[max(0, previous_end-begin+1):].cpu().numpy())
        previous_end = begin+3
    prediction = np.concatenate(chunks).astype(np.float64)
    assert prediction.shape == (len(frames)-1, 6) and np.isfinite(prediction).all()
    return prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--scale-path', default='')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(40,50)))
    args = parser.parse_args()
    if not args.data_root:
        parser.error('--data-root is required')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error('Expose exactly one GPU with CUDA_VISIBLE_DEVICES')
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = SCCNet().cuda().eval()
    model.load_state_dict(saved['model'], strict=True)
    scale = load_scale(args.scale_path).cuda() if args.scale_path else torch.as_tensor(saved['scale'], dtype=torch.float32, device='cuda').reshape(1,1,6)
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Invalid checkpoint scale')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    scans = [p for s in args.subjects for p in sorted((Path(args.data_root)/f'{s:03d}').glob('*.h5'))]
    if not scans:
        raise ValueError('No scans found')
    for scan in scans:
        with h5py.File(scan) as handle:
            frames = handle['frames'][:]
            absolute = np.asarray(handle['tforms'], dtype=np.float64)
            calib = np.asarray(handle['T_calib'], dtype=np.float64)
        pred = predict_scan(model, frames, scale)
        truth = np.linalg.inv(absolute[:-1]) @ absolute[1:]
        cp = np.linalg.inv(calib) @ matrices(pred) @ calib
        cg = np.linalg.inv(calib) @ truth @ calib
        case = scan.parent.name+'_'+scan.stem
        rows.append(dict(case=case, pairs=len(pred), **metrics(cp,cg)))
        np.savez_compressed(out/(case+'.npz'), pose=pred, calibrated_prediction=cp, calibrated_truth=cg)
        (out/'per_scan.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps(dict(completed=len(rows), total=len(scans), case=case)), flush=True)
    keys = ('RTE','RRE','ATE','FD','FDR')
    summary = dict(n=len(rows), mean={k:float(np.mean([r[k] for r in rows])) for k in keys}, sd={k:float(np.std([r[k] for r in rows], ddof=0)) for k in keys}, sd_definition='population SD across scans, ddof=0', protocol='FP32, five frames, stride four, all adjacent predictions, overlap-trimmed tail, calibrated coordinates; geodesic RRE; no trajectory alignment')
    (out/'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
