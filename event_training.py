"""Small-memory training, train-only preprocessing, validation diagnostics."""
import json
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from event_data import Windows, TOKEN_NAMES
from event_models import EventModel, LegacyCNN
from event_evaluation import temporal_permutation


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(data, rows, args, state, shuffle=False, seed=0, drop=None):
    return DataLoader(Windows(data, rows, args.window, state, drop), batch_size=args.batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=args.device == 'cuda',
                      generator=torch.Generator().manual_seed(seed))


def objective(output, target, positive):
    return ((output[:, :3]-target[:, :3]).square().mean()
            + .25*(output[:, 3:6]-target[:, 3:6]).square().mean()
            + .25*nn.functional.binary_cross_entropy_with_logits(output[:, 6:], positive))


def task_loss(output, target, positive, training_objective='multitask'):
    if training_objective=='return_only':
        return (output[:,:3]-target[:,:3]).square().mean()
    if training_objective=='multitask':
        return objective(output,target,positive)
    raise ValueError(f'Unknown training objective: {training_objective}')


def selection_loss(output,target,positive,target_std,selection_metric='joint_loss'):
    if selection_metric=='return_mae':
        return ((output[:,:3]-target[:,:3])*target_std[:3]).abs().mean()
    if selection_metric=='joint_loss':
        return objective(output,target,positive)
    raise ValueError(f'Unknown checkpoint metric: {selection_metric}')


def weights_hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def infer(model, data, rows, args, state, drop=None):
    outputs, gate_rows = [], []
    model.eval()
    with torch.no_grad():
        for x, token, ids, mask, _, _ in loader(data, rows, args, state, drop=drop):
            output, gates = model(*[a.to(args.device, non_blocking=True) for a in [x, token, ids, mask]])
            outputs.append(output.cpu().float().numpy())
            gate_rows.append(gates.cpu().float().numpy())
    return np.concatenate(outputs), np.concatenate(gate_rows)


def fit(data, split, state, args, mode, seed, folder, drop=None):
    seed_all(seed)
    training_objective=getattr(args,'training_objective','multitask')
    selection_metric=getattr(args,'selection_metric','joint_loss')
    if training_objective not in ['multitask','return_only'] or selection_metric not in ['joint_loss','return_mae']:
        raise ValueError('Invalid training objective or selection metric')
    if training_objective=='return_only' and selection_metric!='return_mae':
        raise ValueError('Return-only requires return-based checkpoint selection; auxiliary heads are untrained')
    vocabulary = len(state['vocabulary'])
    model = (LegacyCNN(data['x'].shape[1], args.window, vocabulary, len(TOKEN_NAMES)) if mode == 'cnn'
             else EventModel(data['x'].shape[1], len(TOKEN_NAMES), vocabulary, data['groups'], mode, args.width))
    model.to(args.device)
    initial_hash=weights_hash(model.state_dict())
    target_std=torch.as_tensor(state['target_std'],device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3)
    amp = args.device == 'cuda' and not args.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best, best_epoch, best_state, stale = float('inf'), 0, None, 0
    history, gate_history = [], []
    train_loader = loader(data, split['train'], args, state, True, seed, drop)
    val_loader = loader(data, split['val'], args, state, drop=drop)
    started = time.perf_counter()
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    for epoch in range(args.epochs):
        model.train(); total = 0.; count = 0
        grad_sum=0.;grad_finite=0;grad_clipped=0;grad_nonfinite=0;scaler_skips=0
        for batch in train_loader:
            x, token, ids, mask, target, positive = [a.to(args.device, non_blocking=True) for a in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=amp):
                output, _ = model(x, token, ids, mask)
                loss = task_loss(output, target, positive, training_objective)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite train loss; reduce learning rate or disable AMP')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            gradient_norm=nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if getattr(args,'log_grad_norm',False):
                value=float(gradient_norm.detach())
                if np.isfinite(value):
                    grad_sum+=value;grad_finite+=1;grad_clipped+=int(value>1.)
                else:
                    grad_nonfinite+=1
            old_scale=scaler.get_scale() if getattr(args,'log_grad_norm',False) else None
            scaler.step(optimizer); scaler.update()
            if old_scale is not None:
                skipped=scaler.get_scale()<old_scale;scaler_skips+=int(skipped)
                if not np.isfinite(float(gradient_norm.detach())) and not skipped:
                    raise FloatingPointError('Nonfinite gradient norm without a confirmed GradScaler skip')
            total += float(loss.detach())*len(x); count += len(x)
        model.eval(); val_total = 0.; val_count = 0; gates_all = []; return_total=0.; task_total=0.
        with torch.no_grad():
            for batch in val_loader:
                x, token, ids, mask, target, positive = [a.to(args.device) for a in batch]
                output, gates = model(x, token, ids, mask)
                val_total += float(selection_loss(output,target,positive,target_std,selection_metric))*len(x); val_count += len(x)
                return_total += float(selection_loss(output,target,positive,target_std,'return_mae'))*len(x)
                task_total += float(task_loss(output,target,positive,training_objective))*len(x)
                gates_all.append(gates.cpu().numpy())
        val_loss = val_total/val_count
        if not np.isfinite(val_loss):
            raise FloatingPointError('Non-finite validation loss')
        scheduler.step(val_loss)
        record = dict(epoch=epoch+1, train_loss=total/count, val_loss=val_loss, lr=optimizer.param_groups[0]['lr'])
        record.update(val_return_mae=return_total/val_count,val_task_loss=task_total/val_count)
        if getattr(args,'log_grad_norm',False):
            record.update(gradient_norm_mean=grad_sum/grad_finite if grad_finite else None,
                          clipped_fraction=grad_clipped/grad_finite if grad_finite else None,
                          nonfinite_gradient_steps=grad_nonfinite,confirmed_scaler_skips=scaler_skips)
        history.append(record)
        if mode != 'cnn':
            values = np.concatenate(gates_all)
            for j, group in enumerate(data['groups']):
                gate_history.append(dict(epoch=epoch+1, group=group, mean=float(values[..., j].mean()),
                                         minimum=float(values[..., j].min()), maximum=float(values[..., j].max())))
        if val_loss < best-1e-6:
            best = val_loss; best_epoch = epoch+1; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        progress = dict(ticker=data['ticker'], mode=mode, seed=seed, ablation=drop,
                        training_objective=training_objective,selection_metric=selection_metric,
                        max_epochs=args.epochs, best_epoch=best_epoch, **record)
        # Dashboard tolerates a partial polling read and retries. Direct writes
        # also work in Windows app sandboxes that deny rename-over-existing.
        (Path(args.out)/'progress.json').write_text(json.dumps(progress), encoding='utf-8')
        (folder/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError('No finite checkpoint')
    model.load_state_dict(best_state); model.eval()
    record = dict(best_epoch=best_epoch, val_loss=best, epochs=len(history), seconds=time.perf_counter()-started,
                  parameters=sum(p.numel() for p in model.parameters()), amp=amp,
                  training_objective=training_objective,selection_metric=selection_metric,
                  initial_weights_sha256=initial_hash,best_weights_sha256=weights_hash(best_state),
                  peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20 if args.device == 'cuda' else 0.)
    torch.save(dict(state_dict=best_state, preprocess=state, mode=mode, groups=data['groups'],
                    features=data['names'], token_names=TOKEN_NAMES, window=args.window, width=args.width,
                    source_hash=data['source_hash'], seed=seed, drop=drop, record=record,
                    training_objective=training_objective,selection_metric=selection_metric,
                    calendar_summary=getattr(args, 'calendar_summary', None),
                    training_indices=split['train'], validation_indices=split['val']), folder/'best.pt')
    return model, record, history, gate_history


def decoded(output, state):
    regression = output[:, :6]*state['target_std']+state['target_mean']
    probability = 1/(1+np.exp(-np.clip(output[:, 6:], -40, 40)))
    return regression[:, :3], probability, np.exp(np.clip(regression[:, 3:], -15, 6))


def diagnostics(model, data, split, state, args, seed):
    """Validation only: continuous-input loss gradients and grouped block PI."""
    rows = split['val']
    base, _ = infer(model, data, rows, args, state)
    target = torch.from_numpy(((data['y'][rows]-state['target_mean'])/state['target_std']).astype('float32'))
    positive = torch.from_numpy((data['y'][rows, :3] > 0).astype('float32'))
    metric=getattr(args,'importance_metric','joint_loss')
    def diagnostic_loss(output,target,positive):
        return selection_loss(output,target,positive,torch.as_tensor(state['target_std'],device=output.device),metric)
    reference = float(diagnostic_loss(torch.from_numpy(base), target, positive))
    result = []
    for group in data['groups']:
        values = []
        for repeat in range(args.permutation_repeats):
            shuffled = temporal_permutation(data, rows, args.window, group, seed+repeat, block=5)
            pred, _ = infer(model, shuffled, rows, args, state)
            values.append(float(diagnostic_loss(torch.from_numpy(pred), target, positive))-reference)
        result.append(dict(group=group, method='temporal_permutation', value=float(np.mean(values)),
                           std=float(np.std(values)), repeats=len(values), baseline_loss=reference))
    gradients = {g: [] for g in data['groups']}
    for batch in loader(data, rows, args, state):
        x, token, ids, mask, target, positive = [a.to(args.device) for a in batch]
        x.requires_grad_(); token.requires_grad_()
        output, _ = model(x, token, ids, mask)
        gx, gt = torch.autograd.grad(diagnostic_loss(output, target, positive)*len(x), (x, token))
        for group, cols in data['groups'].items():
            sensitivity = gx[:, :, cols].abs().sum().item()/len(x)
            if group == 'broker':
                sensitivity += (gt.abs()*mask[..., None]).sum().item()/len(x)
            gradients[group].append((sensitivity, len(x)))
    for group, values in gradients.items():
        result.append(dict(group=group, method='loss_gradient', value=sum(v*n for v, n in values)/sum(n for _, n in values),
                           std=None, repeats=1, baseline_loss=reference))
    return result
