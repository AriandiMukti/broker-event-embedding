"""Small-memory next-session cross-entropy training and frozen selection."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from event_relation_model import RelationModel,RelationDataset
from event_training import seed_all,weights_hash


def configure_runtime(device):
    from event_objective_study import runtime
    device=runtime(device)
    # Evaluate in IEEE float32 across CPU/GPU; AMP training remains float16.
    # cuDNN TF32 convolution otherwise introduces ~1e-5 probability differences.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return device


def loader(data,rows,config,mode,seed,shuffle=False):
    return DataLoader(RelationDataset(data,rows,mode,seed),batch_size=config['batch_size'],shuffle=shuffle,num_workers=0,
        pin_memory=config['device']=='cuda',generator=torch.Generator().manual_seed(seed))


def infer(model,data,rows,config,mode,seed,remove_peer=None):
    ids=torch.as_tensor(data['ids'],device=config['device']);outputs=[]
    model.eval()
    with torch.no_grad():
        for own,ctx,tok,mask,_ in loader(data,rows,config,mode,seed):
            values=[v.to(config['device'],non_blocking=True) for v in [own,ctx,tok,mask]]
            logits=model(*values,ids,remove_peer=remove_peer)
            outputs.append(logits.float().softmax(-1).cpu().numpy())
    p=np.concatenate(outputs).astype('float64').clip(1e-8)
    return p/p.sum(-1,keepdims=True)


def fit(data,config,mode,seed,folder):
    seed_all(seed);model=RelationModel(int(data['vocabulary_size']),data['context'].shape[1],config['width'],mode).to(config['device'])
    before=weights_hash(model.state_dict());encoder_before=weights_hash(model.set_encoder.state_dict())
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=3)
    amp=config['device']=='cuda' and config['amp'];scaler=torch.amp.GradScaler('cuda',enabled=amp)
    train=np.flatnonzero(data['splits']=='train');val=np.flatnonzero(data['splits']=='val')
    trainloader=loader(data,train,config,mode,seed,True);valloader=loader(data,val,config,mode,seed)
    ids=torch.as_tensor(data['ids'],device=config['device'])
    best=float('inf');best_state=None;best_epoch=0;stale=0;history=[];started=time.perf_counter()
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    if config['device']=='cuda':torch.cuda.reset_peak_memory_stats()
    for epoch in range(config['epochs']):
        model.train();total=0.;n=0;norm_sum=0.;finite=0;nonfinite=0;skips=0;clipped=0
        for own,ctx,tok,mask,y in trainloader:
            own,ctx,tok,mask,y=[v.to(config['device'],non_blocking=True) for v in [own,ctx,tok,mask,y]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=config['device'],dtype=torch.float16,enabled=amp):
                logits=model(own,ctx,tok,mask,ids)
                loss=nn.functional.cross_entropy(logits.reshape(-1,5),y.reshape(-1))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite train loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=float(nn.utils.clip_grad_norm_(model.parameters(),1.).detach());scale=scaler.get_scale()
            scaler.step(optimizer);scaler.update();skipped=scaler.get_scale()<scale
            skips+=int(skipped)
            if np.isfinite(norm):norm_sum+=norm;finite+=1;clipped+=int(norm>1)
            else:
                nonfinite+=1
                if not skipped:raise FloatingPointError('Nonfinite norm without confirmed scaler skip')
            total+=float(loss.detach())*len(own);n+=len(own)
        model.eval();validation=0.;vn=0
        with torch.no_grad():
            for own,ctx,tok,mask,y in valloader:
                own,ctx,tok,mask,y=[v.to(config['device']) for v in [own,ctx,tok,mask,y]]
                logits=model(own,ctx,tok,mask,ids)
                value=nn.functional.cross_entropy(logits.reshape(-1,5),y.reshape(-1))
                validation+=float(value)*len(own);vn+=len(own)
        value=validation/vn
        if not np.isfinite(value):raise FloatingPointError('Nonfinite validation NLL')
        scheduler.step(value)
        row=dict(epoch=epoch+1,train_nll=total/n,validation_nll=value,lr=optimizer.param_groups[0]['lr'],
            gradient_norm_mean=norm_sum/finite if finite else None,clipped_fraction=clipped/finite if finite else None,
            nonfinite_gradient_norms=nonfinite,confirmed_scaler_skips=skips)
        history.append(row)
        if value<best-1e-6:
            best=value;best_epoch=epoch+1;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        (folder/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        (Path(config['out'])/'progress.json').write_text(json.dumps(dict(ticker=config.get('ticker'),fold=config.get('fold'),
            mode=mode,seed=seed,best_epoch=best_epoch,**row)),encoding='utf-8')
        if stale>=config['patience']:break
    if best_state is None:raise RuntimeError('No finite checkpoint')
    model.load_state_dict(best_state);model.eval();after=weights_hash(model.set_encoder.state_dict())
    if mode in ['own_only','frozen_peers']:assert encoder_before==after
    record=dict(mode=mode,seed=seed,best_epoch=best_epoch,epochs=len(history),validation_nll=best,
        seconds=time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        initial_sha256=before,initial_encoder_sha256=encoder_before,best_encoder_sha256=after,
        encoder_changed=encoder_before!=after,amp=amp,
        peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20 if config['device']=='cuda' else 0.)
    torch.save(dict(state_dict=best_state,record=record,mode=mode,seed=seed,width=config['width'],
        vocabulary=int(data['vocabulary_size']),context_features=data['context'].shape[1]),folder/'best.pt')
    return model,record,history


def restore(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=RelationModel(saved['vocabulary'],saved['context_features'],saved['width'],saved['mode'])
    model.load_state_dict(saved['state_dict']);model.to(device);model.eval()
    return model,saved
