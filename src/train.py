import argparse
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import SequenceDataset, load_scale, reverse_labels
from .losses import make_criteria, pose_loss
from .model import SCCNet, image_patch_mask


def objective(model, x, y, scale, criteria, epoch, mask_start, mask_weight, mask_rng):
    active = epoch > mask_start and mask_weight > 0
    mask = image_patch_mask(len(x), 4, side=4, ratio=.2, generator=mask_rng).to(x.device) if active else None
    output = model(x, epoch=101 if active else 0, patch_mask=mask)
    reverse = reverse_labels(y).to(x.device)
    forward_loss, _ = pose_loss(output['heads'], output['features'], y / scale, criteria)
    reverse_loss, _ = pose_loss(output['reverse_heads'], output['reverse_features'], reverse / scale, criteria)
    total = .5 * (forward_loss + reverse_loss)
    if active:
        state = random.getstate()
        try:
            auxiliary, _ = pose_loss(output['mask']['heads'], output['mask']['features'], y / scale, criteria)
        finally:
            random.setstate(state)
        total = total + mask_weight * auxiliary
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='')
    parser.add_argument('--scale-path', default='')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--mask-start', type=int, default=100)
    parser.add_argument('--mask-weight', type=float, default=.05)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', default='')
    args = parser.parse_args()
    if not args.data_root or not args.scale_path:
        parser.error('--data-root and --scale-path are required')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error('Expose exactly one GPU with CUDA_VISIBLE_DEVICES')
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists() and not args.resume:
        raise ValueError('Existing run requires an explicit --resume checkpoint')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    model = SCCNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=.8)
    criteria = make_criteria()
    scale = load_scale(args.scale_path).cuda()
    mask_rng = torch.Generator().manual_seed(4242)
    start, best = 0, float('inf')
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved.get('schema') != 'scc_net_baseline_v1':
            raise ValueError('Resume requires a checkpoint from this training entry point')
        for key in ('batch_size', 'mask_start', 'mask_weight', 'seed', 'data_root'):
            if saved['config'][key] != vars(args)[key]:
                raise ValueError(f'Resume configuration differs: {key}')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        if not torch.equal(saved['scale'].cpu(), scale.cpu()):
            raise ValueError('Resume scale differs')
        start, best = saved['epoch'], saved['best_validation_loss']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        mask_rng.set_state(saved['mask_rng'])
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))
    valid = DataLoader(SequenceDataset(args.data_root, range(30,40), fixed=True, windows=2), batch_size=args.batch_size, num_workers=args.num_workers)
    for epoch in range(start+1, args.epochs+1):
        loader = DataLoader(SequenceDataset(args.data_root, range(30), seed=args.seed, epoch=epoch), batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
        model.train()
        total = 0.
        for step, (x, y) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, x.cuda(), y.cuda(), scale, criteria, epoch, args.mask_start, args.mask_weight, mask_rng)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: epoch {epoch}, step {step}')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError(f'Nonfinite gradients: epoch {epoch}, step {step}')
            optimizer.step()
            total += float(loss.detach())
            if step % 20 == 0:
                print(json.dumps(dict(epoch=epoch, step=step, steps=len(loader), loss=float(loss.detach()))), flush=True)
        scheduler.step()
        rng_state = (torch.get_rng_state(), torch.cuda.get_rng_state(), random.getstate(), np.random.get_state())
        model.eval()
        errors = []
        with torch.no_grad():
            for x, y in valid:
                output = model(x.cuda())
                loss, _ = pose_loss(output['heads'], output['features'], y.cuda()/scale, criteria)
                errors.append((float(loss), len(x)))
        torch.set_rng_state(rng_state[0])
        torch.cuda.set_rng_state(rng_state[1])
        random.setstate(rng_state[2])
        np.random.set_state(rng_state[3])
        score = sum(v*n for v,n in errors)/sum(n for _,n in errors)
        improved = score < best
        best = min(best, score)
        record = dict(epoch=epoch, train_loss=total/len(loader), validation_loss=score, best_validation_loss=best)
        state = dict(schema='scc_net_baseline_v1', **record, config=vars(args), model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scale=scale.cpu(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), python_rng=random.getstate(), numpy_rng=np.random.get_state(), mask_rng=mask_rng.get_state())
        torch.save(state, out / 'last.partial')
        os.replace(out / 'last.partial', out / 'last.pt')
        if improved:
            torch.save(state, out / 'best.pt')
        if epoch % 20 == 0 or epoch in (args.mask_start, args.epochs):
            torch.save(state, out / f'epoch_{epoch:04d}.pt')
        (out / 'progress.json').write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
