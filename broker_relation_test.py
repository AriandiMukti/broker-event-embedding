"""ADRO: chronological masked-peer and next-session embedding experiments.

No target-broker input token; no broker-derived aggregate input features.
Same-day reconstruction measures association, NOT a forecast available pre-close.
"""
import argparse
import copy
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from torch import nn

from research import load_stock
from broker_tokens import fit_tokenizer, transform_tokens, STATES
from adro_token_embedding import EventEmbedding, scores

SEEDS = [11, 42, 123, 2026, 3050]
METHODS = ['full', 'brokers_only', 'ohlcv_only', 'shifted_peers', 'fixed_embedding']


def align_loaded_calendar(loaded, brokers_path):
    """Keep missing summary sessions as holes, never shorten trading lags."""
    df,good,cache,extras,snapshots,quality = loaded
    calendar = pd.DatetimeIndex(sorted(pd.Timestamp(p.name) for p in Path(brokers_path).iterdir()
        if p.is_dir() and len(p.name)==10 and p.name[4]=='-'))
    calendar = calendar[(calendar>=df.index.min())&(calendar<=df.index.max())]
    missing = calendar.difference(df.index)
    if not len(missing): return loaded
    positions = {d:i for i,d in enumerate(df.index)}
    new_cache=[];new_extras=[];new_snapshots=[]
    for day in calendar:
        i=positions.get(day)
        new_cache.append(cache[i] if i is not None else {})
        new_extras.append(extras[i] if i is not None else [1,0,0,0])
        new_snapshots.append(snapshots[i] if i is not None else None)
    quality=list(quality)+[dict(date=str(d.date()),issue='missing summary calendar session') for d in missing]
    return df.reindex(calendar),good.reindex(calendar,fill_value=False),new_cache,np.asarray(new_extras),new_snapshots,quality


def relation_tokens(loaded, train_stop, count=12, ticker='ADRO', clean_ohlcv=False):
    if clean_ohlcv:
        loaded = list(loaded)
        loaded[0] = loaded[0].copy()
        loaded[0].loc[~loaded[1], ['Open','High','Low','Close','Volume']] = np.nan
    tokenizer = fit_tokenizer(loaded, train_stop, count)
    tokenizer['ticker'] = ticker
    if clean_ohlcv: tokenizer['invalid_ohlcv_masked_before_context'] = True
    # Absolute per-broker thresholds prevent a masked target changing peer
    # size states via the old denominator (total absolute net market value).
    thresholds = {}
    for code in tokenizer['brokers']:
        values = [abs(s['values'].get(code, 0)) for s in loaded[4][:train_stop] if s is not None]
        thresholds[code] = float(np.quantile([v for v in values if v > 0], 2/3))
    tokenizer['large_absolute_net_threshold'] = thresholds
    tokenizer.pop('large_share_threshold')
    tokenizer['large_definition'] = 'absolute broker net value > active-day training 2/3 quantile; no market denominator'
    # Reuse context tokenization, then replace ALL broker labels and tokens.
    temporary = dict(tokenizer, large_share_threshold={c:1 for c in tokenizer['brokers']})
    daily, states, valid = transform_tokens(loaded, temporary)
    for i, snapshot in enumerate(loaded[4]):
        if snapshot is None: continue
        for j, code in enumerate(tokenizer['brokers']):
            value = snapshot['values'].get(code, 0)
            large = abs(value) > thresholds[code]
            state = 0 if value == 0 else ((2 if large else 1) if value > 0 else (4 if large else 3))
            states[i, j] = state; daily[i, j] = 1+5*j+state
    return tokenizer, daily, states, valid


def folds(n):
    # Disjoint test dates, expanding training, 21-session validation.
    edges = np.linspace(int(n * .6), n, 4, dtype=int)
    return [dict(fold=k+1, train_stop=int(a-21), val_stop=int(a), test_stop=int(b))
            for k, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]


def dataset(daily, states, valid, fold, task):
    brokers = states.shape[1]
    lag = 0 if task == 'same_day_masked' else 1
    target = np.arange(40+lag, fold['test_stop'])
    target = target[valid[target] & valid[target-lag] & (states[target] >= 0).all(axis=1)]
    source = target-lag
    # First six context tokens are OHLCV only. Retail_share and buyer_fraction
    # are deliberately absent: both incorporate broker-target activity.
    base = daily[source, :brokers+6].copy()
    splits = np.where(target < fold['train_stop'], 'train',
                      np.where(target < fold['val_stop'], 'val', 'test'))
    if any(not (splits == s).any() for s in ['train', 'val', 'test']):
        raise ValueError('Empty chronological split')
    return base, states[target], states[target-1], target, source, splits


def shift_peers(base, splits, brokers, seed):
    """Independent nonzero circular date shifts, within each temporal split.

    Retains each peer's marginal counts and most serial ordering; disrupts
    alignment to target and OHLCV. Not an exact independence/causality test.
    """
    rng = np.random.default_rng(seed + 90000)
    out = base.copy(); audit = {}
    for split in ['train', 'val', 'test']:
        ids = np.flatnonzero(splits == split)
        allowed = np.arange(5, len(ids)-4)
        if not len(allowed): raise ValueError('Need >=10 dates per split')
        offsets = rng.choice(allowed, brokers)
        for b, offset in enumerate(offsets):
            out[ids, b] = np.roll(base[ids, b], int(offset))
        audit[split] = offsets.tolist()
    return out, audit


def mask_inputs(base, brokers, method):
    # Each day is repeated once for every possible masked target broker.
    x = np.repeat(base[:, None, :], brokers, axis=1)
    for b in range(brokers): x[:, b, b] = 0
    if method == 'ohlcv_only': x[:, :, :brokers] = 0
    if method == 'brokers_only': x[:, :, brokers:] = 0
    return x


def tensor_inputs(x):
    days, brokers, width = x.shape
    return (torch.as_tensor(x.reshape(-1, 1, width), dtype=torch.long),
            torch.arange(brokers).repeat(days))


def logits_for(model, x):
    tx, broker_ids = tensor_inputs(x)
    logits = model(tx)
    return logits[torch.arange(len(tx)), broker_ids]


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return logits_for(model, x).softmax(-1).numpy().reshape(x.shape[0], x.shape[1], 5)


def fit(x, y, splits, vocabulary, dimension, seed, epochs, patience, freeze=False):
    torch.manual_seed(seed)
    model = EventEmbedding(vocabulary, y.shape[1], dimension, 1)
    if freeze: model.embedding.weight.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.03)
    train, val = splits == 'train', splits == 'val'
    yt = torch.as_tensor(y[train].reshape(-1), dtype=torch.long)
    yv = torch.as_tensor(y[val].reshape(-1), dtype=torch.long)
    best, stale, checkpoint, best_epoch = float('inf'), 0, None, None
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        loss = nn.functional.cross_entropy(logits_for(model, x[train]), yt)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
        model.eval()
        with torch.no_grad(): vl = float(nn.functional.cross_entropy(logits_for(model, x[val]), yv))
        if vl < best-1e-7:
            best, stale, best_epoch = vl, 0, epoch+1
            checkpoint = copy.deepcopy(model.state_dict())
        else: stale += 1
        if stale >= patience: break
    model.load_state_dict(checkpoint)
    return model, {'validation_nll': best, 'best_epoch': best_epoch, 'seed': seed}


def selected_logits(model, tx, broker_ids):
    """Same one-day forward, computing only the masked target's output head."""
    vectors = model.embedding(tx)
    nonpad = (tx != 0).unsqueeze(-1)
    daily = (vectors*nonpad).sum(2)/nonpad.sum(2).clamp(min=1)
    hidden = model.transform((daily*model.recency[None,:,None]).sum(1))
    candidates = model.embedding(model.output_ids[broker_ids])
    return (hidden[:,None,:]*candidates).sum(-1)+model.bias[broker_ids]


def fast_fit(x, y, splits, vocabulary, dimension, seed, epochs, patience, freeze=False):
    torch.manual_seed(seed)
    model = EventEmbedding(vocabulary, y.shape[1], dimension, 1)
    if freeze: model.embedding.weight.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.03)
    train, val = splits == 'train', splits == 'val'
    tx, ti = tensor_inputs(x[train]); vx, vi = tensor_inputs(x[val])
    yt = torch.as_tensor(y[train].reshape(-1), dtype=torch.long)
    yv = torch.as_tensor(y[val].reshape(-1), dtype=torch.long)
    best, stale, checkpoint, best_epoch = float('inf'), 0, None, None
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        loss = nn.functional.cross_entropy(selected_logits(model, tx, ti), yt)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
        model.eval()
        with torch.no_grad(): vl = float(nn.functional.cross_entropy(selected_logits(model, vx, vi), yv))
        if vl < best-1e-7:
            best, stale, best_epoch = vl, 0, epoch+1
            checkpoint = copy.deepcopy(model.state_dict())
        else: stale += 1
        if stale >= patience: break
    model.load_state_dict(checkpoint)
    return model, {'validation_nll': best, 'best_epoch': best_epoch, 'seed': seed}


def baseline(y, previous, splits):
    b = y.shape[1]; train = splits == 'train'
    freq = np.ones((b, 5)); trans = np.ones((b, 5, 5))
    for j in range(b):
        freq[j] += np.bincount(y[train, j], minlength=5)
        for before, after in zip(previous[train, j], y[train, j]):
            if before >= 0: trans[j, before, after] += 1
    freq /= freq.sum(-1, keepdims=True); trans /= trans.sum(-1, keepdims=True)
    constant = np.broadcast_to(freq, (len(y), b, 5))
    markov = trans[np.arange(b)[None, :], previous.clip(min=0)].copy()
    markov[previous < 0] = constant[previous < 0]
    return constant, markov


def losses(y, p):
    return -np.log(np.take_along_axis(p, y[:, :, None], axis=2)[:, :, 0].clip(1e-12))


def neighbor_sets(model, daily, train_stop, brokers):
    v = model.embedding.weight.detach().numpy()
    exposure = np.bincount(daily[40:train_stop, :len(brokers)].ravel(), minlength=len(v))
    # Exclude NET_ZERO and rare states; compare original space, not PCA.
    ids = [i for i in range(1, 1+5*len(brokers)) if (i-1) % 5 != 0 and exposure[i] >= 5]
    result = {}
    for q in ids:
        candidates = [i for i in ids if (i-1)//5 != (q-1)//5]
        if len(candidates) < 3: continue
        similarity = v[candidates] @ v[q] / np.maximum(np.linalg.norm(v[candidates], axis=1)*np.linalg.norm(v[q]), 1e-12)
        result[q] = [candidates[i] for i in np.argsort(-similarity)[:3]]
    return result


def stability(sets):
    rows = []
    for a, b in combinations(sets, 2):
        common = sorted(set(a) & set(b))
        for q in common:
            x, y = set(a[q]), set(b[q])
            rows.append(len(x & y)/len(x | y))
    return float(np.mean(rows)) if rows else None


def block_bootstrap(values, fold_ids, seed=2468, repeats=5000, block=5):
    """Resample dates in blocks within folds; brokers/seeds aren't independent N."""
    rng = np.random.default_rng(seed); unique = np.unique(fold_ids)
    groups = [values[fold_ids == f] for f in unique]
    draws = np.zeros(repeats)
    for r in range(repeats):
        sampled = []
        for g in groups:
            starts = rng.integers(0, len(g), int(np.ceil(len(g)/block)))
            ids = ((starts[:, None] + np.arange(block)) % len(g)).ravel()[:len(g)]
            sampled.extend(g[ids])
        draws[r] = np.mean(sampled)
    return {'gain_nll': float(np.mean(values)), 'ci_low': float(np.quantile(draws, .025)),
            'ci_high': float(np.quantile(draws, .975)), 'dates': len(values), 'block_sessions': block}


def run(args):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ticker = getattr(args, 'ticker', 'ADRO')
    loaded = load_stock(ticker, args.summary, args.brokers)
    loaded = align_loaded_calendar(loaded,args.brokers)
    dates = loaded[0].index
    metrics, daily_rows, audit_rows, fit_rows, stability_rows, neighbor_rows, pair_rows = [], [], [], [], [], [], []
    predictions = []
    for fold in folds(len(dates)):
        tokenizer, daily, states, valid = relation_tokens(loaded, fold['train_stop'], 12, ticker,
                                                       getattr(args,'clean_ohlcv',False))
        codes = tokenizer['brokers']; b = len(codes)
        folder = out / f"fold-{fold['fold']}"; folder.mkdir(exist_ok=True)
        (folder / 'tokenizer.json').write_text(json.dumps(tokenizer, indent=2), encoding='utf-8')
        for task in ['same_day_masked', 'next_day_peers']:
            base, y, prev, targets, source, splits = dataset(daily, states, valid, fold, task)
            test = splits == 'test'; yt = y[test]
            for i, t in enumerate(targets):
                audit_rows.append(dict(task=task, fold=fold['fold'], split=splits[i], target_date=str(dates[t].date()),
                                       source_date=str(dates[source[i]].date()), target_index=int(t)))
            constant, markov = baseline(y, prev, splits)
            methods_p = {}; models_by_method = {}
            for method in METHODS:
                ps, sets, fitted = [], [], []
                for seed in args.seeds:
                    context, shift_audit = shift_peers(base, splits, b, seed) if method == 'shifted_peers' else (base, None)
                    x = mask_inputs(context, b, method)
                    fit_function = fast_fit if getattr(args,'fast',False) else fit
                    model, record = fit_function(x, y, splits, len(tokenizer['tokens']), args.dimension, seed, args.epochs, args.patience,
                                        freeze=method == 'fixed_embedding')
                    p = predict(model, x[test]); ps.append(p); fitted.append((record['validation_nll'], seed, model))
                    record.update(task=task, fold=fold['fold'], method=method, offsets=shift_audit)
                    fit_rows.append(record)
                    metric = dict(task=task, fold=fold['fold'], method=method, seed=seed, **scores(yt, p))
                    metrics.append(metric)
                    neighbors = neighbor_sets(model, daily, fold['train_stop'], codes)
                    sets.append(neighbors)
                    if method in ['full', 'shifted_peers']:
                        for q, ns in neighbors.items():
                            for rank, neighbor in enumerate(ns, 1):
                                neighbor_rows.append(dict(task=task, fold=fold['fold'], method=method, seed=seed,
                                    query=tokenizer['tokens'][q], rank=rank, neighbor=tokenizer['tokens'][neighbor]))
                    if method == 'full':
                        loss_full = losses(yt, p)
                        # Inference ablation: remove one observed peer, leaving target masked.
                        # This is a model diagnostic, not causal identification.
                        for peer in range(b):
                            removed = x[test].copy(); removed[:, :, peer] = 0
                            delta = losses(yt, predict(model, removed))-loss_full
                            for target_b in range(b):
                                if target_b == peer: continue
                                pair_rows.append(dict(task=task, fold=fold['fold'], seed=seed, peer=codes[peer],
                                    target=codes[target_b], gain_nll=float(delta[:, target_b].mean())))
                    print(task, fold['fold'], method, seed, 'epoch', record['best_epoch'], 'nll', round(metric['nll'], 6), flush=True)
                methods_p[method] = ps; models_by_method[method] = fitted
                stability_rows.append(dict(task=task, fold=fold['fold'], method=method, top3_jaccard=stability(sets)))
            methods_p['frequency'] = [constant[test]]
            methods_p['own_previous_markov'] = [markov[test]]
            for method in ['frequency', 'own_previous_markov']:
                metrics.append(dict(task=task, fold=fold['fold'], method=method, seed=-1, **scores(yt, methods_p[method][0])))
            # Average LOSSES over seeds: does not silently switch to an ensemble model.
            for method, ps in methods_p.items():
                ls = np.mean([losses(yt, p) for p in ps], axis=0)
                ac = np.mean([(p.argmax(-1) == yt).astype(float) for p in ps], axis=0)
                for i, t in enumerate(targets[test]):
                    daily_rows.append(dict(task=task, fold=fold['fold'], date=str(dates[t].date()), method=method,
                                           nll=float(ls[i].mean()), state_accuracy=float(ac[i].mean())))
                for seed_i, p in enumerate(ps):
                    for i, t in enumerate(targets[test]):
                        for j, code in enumerate(codes):
                            predictions.append(dict(task=task, fold=fold['fold'], method=method,
                                seed=args.seeds[seed_i] if method in METHODS else -1, date=str(dates[t].date()),
                                broker=code, actual=int(yt[i,j]), **{STATES[k]:float(p[i,j,k]) for k in range(5)}))
            # Export one reproducible checkpoint, picked using validation alone.
            _, chosen_seed, chosen = min(models_by_method['full'], key=lambda v:v[0])
            torch.save(dict(state_dict=chosen.state_dict(), tokenizer=tokenizer, task=task, seed=chosen_seed,
                            dimension=args.dimension, context_days=1), folder/f'{task}.pt')
        (out/'training_runs.json').write_text(json.dumps(fit_rows, indent=2), encoding='utf-8')
    pd.DataFrame(metrics).to_csv(out/'metrics_by_seed.csv', index=False)
    d = pd.DataFrame(daily_rows); d.to_csv(out/'daily_test_scores.csv', index=False)
    pd.DataFrame(audit_rows).to_csv(out/'sample_audit.csv', index=False)
    pd.DataFrame(predictions).to_csv(out/'test_predictions.csv', index=False)
    pd.DataFrame(stability_rows).to_csv(out/'neighbor_stability.csv', index=False)
    pd.DataFrame(neighbor_rows).to_csv(out/'neighbors_by_seed.csv', index=False)
    pair = pd.DataFrame(pair_rows); pair.to_csv(out/'peer_ablation_by_seed.csv', index=False)
    pair_summary = pair.groupby(['task','peer','target'], as_index=False).gain_nll.mean()
    pair_summary.to_csv(out/'peer_ablation.csv', index=False)
    comparisons = []
    for task, group in d.groupby('task'):
        pivot = group.pivot(index=['fold','date'], columns='method', values='nll')
        for comparator in ['frequency','ohlcv_only','shifted_peers','fixed_embedding','own_previous_markov']:
            gain = (pivot[comparator]-pivot['full']).to_numpy()
            for block in [3,5,10]:
                comparisons.append(dict(task=task, comparator=comparator, **block_bootstrap(gain, pivot.index.get_level_values('fold').to_numpy(), block=block)))
    pd.DataFrame(comparisons).to_csv(out/'paired_block_bootstrap.csv', index=False)
    summary = d.groupby(['task','method'], as_index=False)[['nll','state_accuracy']].mean()
    summary.to_csv(out/'summary.csv', index=False)
    d.groupby(['task','fold','method'], as_index=False)[['nll','state_accuracy']].mean().to_csv(out/'summary_by_fold.csv', index=False)
    config = dict(vars(args), source_days=len(dates), source_start=str(dates[0].date()), source_end=str(dates[-1].date()),
                  folds=folds(len(dates)), source_quality=loaded[-1], context_features=['return_1','return_20','return_40','volume_relative20','range','body'],
                  excluded_context=['retail_share','buyer_fraction'], objective='5-state cross entropy; target broker token always absent',
                  test_use='historical exploratory reuse, not a new untouched confirmation set',
                  bootstrap='paired date-block resampling within folds; seed losses averaged; 5000 draws; approximate intervals, no causal claim')
    config['ticker'] = ticker
    config['calendar_aligned'] = True
    config['invalid_summary_dates'] = [str(d.date()) for d in loaded[0].index[~loaded[1]]]
    (out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    write_report(out, config, summary, pd.DataFrame(comparisons), pd.DataFrame(stability_rows), pair_summary)
    draw_chart(out, pd.DataFrame(comparisons), ticker, int(d[d.method=='full'].date.nunique()))
    print(summary.to_string(index=False), flush=True)


def write_report(out, config, summary, comparisons, stability_table, pairs):
    primary = comparisons[comparisons.block_sessions == 5]
    conclusions = []
    for task in ['same_day_masked','next_day_peers']:
        evidence = primary[(primary.task == task) & primary.comparator.isin(['ohlcv_only','shifted_peers','fixed_embedding'])]
        supported = bool((evidence.ci_low > 0).all())
        conclusions.append(f"- {task}: " + ('hasil mendukung pembelajaran dependensi statistik pada periode historis ini; full unggul terhadap OHLCV-only, konteks geser dan vektor acak tetap pada interval blok 5 sesi.' if supported else 'belum ada bukti konsisten bahwa embedding yang dilatih menambah informasi di atas semua kontrol; setidaknya satu interval perbandingan mencakup nol.'))
    lines = [f"# {config['ticker']}: apakah embedding belajar hubungan antarkejadian broker?", '',
        '## Kesimpulan eksperimen', '', *conclusions, '',
        f"Data: {config['source_days']} sesi ({config['source_start']} s.d. {config['source_end']}); tiga fold expanding, tanggal test tidak tumpang tindih; {len(config['seeds'])} seed.", '',
        '## Pertanyaan dan desain', '',
        '- same_day_masked: rekonstruksi state broker yang disembunyikan dari broker lain dan OHLCV hari yang sama. Ini asosiasi sesudah sesi selesai, bukan forecast sebelum penutupan.',
        '- next_day_peers: prediksi state sesi berikutnya dari broker lain dan OHLCV sesi sebelumnya. Token broker target sendiri tidak diberikan, termasuk state sebelumnya.',
        '- full = peer broker + enam konteks OHLCV; brokers_only dan ohlcv_only memisahkan sumber informasi.',
        '- fixed_embedding: konteks sama seperti full dan inisialisasi seed sama, tetapi vektor input/output acak dibekukan; hanya transform linear dan bias yang dilatih, dengan budget epoch yang sama. Ini kontrol representasi, bukan penyamaan jumlah parameter trainable; perbedaan optimisasi/underfitting masih dapat memengaruhi hasil.',
        '- shifted_peers: setiap broker digeser tanggal secara independen dalam train/validasi/test. Marginal state dipertahankan, keterkaitan tanggal dirusak. Kontrol juga merusak keselarasan peer dengan OHLCV dan regime; ini bukan uji permutasi exact. Pergeseran circular adalah kontrol offline yang dapat memakai observasi bertanggal lebih akhir dalam split, bukan prosedur prediksi live.',
        '- Target broker selalu PAD pada input, termasuk semua varian acak. Retail share dan buyer fraction tidak digunakan karena mengandung kontribusi target. State normal/besar dihitung dari absolute net value broker dibanding quantile train broker itu sendiri, TANPA denominator pasar. Ini mengubah definisi besar dari eksperimen lama agar target tidak mengubah state peer melalui denominator.',
        '- Broker, batas besar/kecil dan quantile OHLCV hanya di-fit pada train masing-masing fold. Checkpoint hanya dipilih dari validasi. Tidak ada pemilihan seed berdasarkan test.',
        '- Frekuensi dan Markov memakai label train saja. own_previous_markov mendapat state target sesi sebelumnya; ia benchmark yang lebih kuat, bukan input setara.',
        '', '## Hasil test', '', '| Tugas | Metode | NLL (turun lebih baik) | Akurasi state |', '|---|---|---:|---:|']
    for r in summary.to_dict('records'):
        lines.append(f"| {r['task']} | {r['method']} | {r['nll']:.6f} | {r['state_accuracy']:.2%} |")
    lines += ['', 'Angka adalah rata-rata loss/akurasi seed per tanggal lalu rata-rata tanggal; bukan memilih test terbaik atau probabilitas ensemble.', '',
              '## Selisih berpasangan dan ketidakpastian', '',
              'Gain = NLL pembanding minus NLL full; positif berarti full lebih baik. Bootstrap blok tanggal dalam tiap fold, 5.000 pengambilan ulang. Broker dan seed tidak dihitung sebagai sampel independen. Interval ini bersyarat pada model yang sudah di-fit dan bukan mencakup semua ketidakpastian training.', '',
              '| Tugas | Pembanding | Gain NLL | Interval 95% blok 5 sesi | Tanggal |', '|---|---|---:|---|---:|']
    for r in comparisons[comparisons.block_sessions == 5].to_dict('records'):
        lines.append(f"| {r['task']} | {r['comparator']} | {r['gain_nll']:.6f} | [{r['ci_low']:.6f}, {r['ci_high']:.6f}] | {r['dates']} |")
    lines += ['', 'Sensitivitas panjang blok 3 dan 10 sesi tersedia di paired_block_bootstrap.csv. Banyak perbandingan bersifat eksploratif; interval bukan p-value terkontrol untuk semua hipotesis.', '',
              '## Stabilitas tetangga embedding', '',
              'Rata-rata Jaccard top-3 lintas seed di ruang embedding asli, token lintas broker yang muncul >=5 kali train dan bukan NET_ZERO. Tidak membandingkan koordinat PCA. Jaccard tinggi saja tidak membuktikan relasi ekonomi; frekuensi dan objective juga memengaruhi geometri.', '',
              '| Tugas | Metode | Mean Jaccard top-3 |', '|---|---|---:|']
    for r in stability_table[stability_table.method.isin(['full','shifted_peers'])].groupby(['task','method'],as_index=False).top3_jaccard.mean().to_dict('records'):
        lines.append(f"| {r['task']} | {r['method']} | {r['top3_jaccard']:.4f} |")
    lines += ['', '## Diagnostik peer', '',
              'peer_ablation.csv mengukur kenaikan test NLL ketika satu peer dihapus dari input. Positif berarti model bergantung pada peer itu secara membantu. Ini penghapusan input di luar distribusi training, eksploratif dan bukan bukti sebab-akibat/koordinasi. Peringkat pasangan tidak digunakan memilih model.', '',
              '## Batas kesimpulan', '',
              'Hubungan yang dipelajari berarti dependensi statistik atas state net broker, bukan identitas investor, niat, atau koordinasi. Keseimbangan net buy/sell pasar tetap dapat menimbulkan hubungan mekanis. State besar memakai nilai net absolut dengan threshold per broker, tanpa normalisasi terhadap pasar. Angka tidak dapat dibandingkan langsung dengan eksperimen lama yang memakai threshold porsi pasar.',
              'Jika full tidak mengalahkan OHLCV-only dan kontrol shifted secara konsisten dengan interval gain di atas nol, bukti relasi yang berguna belum kuat. fixed_embedding menguji nilai tambah vektor yang dilatih dibanding proyeksi acak tetap. Jika hanya same-day membaik, jangan mengubahnya menjadi klaim prediksi sesi berikutnya.',
              'Periode historis ini telah digunakan pada eksperimen sebelumnya. Ini walk-forward eksploratif yang diperluas, bukan test baru yang benar-benar belum pernah dilihat dalam pengembangan. Fold berikutnya memakai masa test fold sebelumnya sebagai training, sesuai expanding window.',
              'Data relatif pendek, threshold state dan pilihan broker berubah antar-fold sesuai train. NET_ZERO berarti net nol/tidak tercatat, bukan gross tidak aktif. Tidak ada target harga, rekomendasi transaksi, atau backtest profit di eksperimen ini.', '',
              '## Menjalankan ulang', '', '```powershell', f"python multi_broker_relations.py --tickers {config['ticker']}", 'python -m unittest test_broker_relations.py test_broker_tokens.py -v', '```', '']
    (out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')


def draw_chart(out, comparisons, ticker='ADRO', dates=84):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11, 'font.family':'DejaVu Sans'})
    names = {'frequency':'Frekuensi train', 'ohlcv_only':'OHLCV saja',
             'shifted_peers':'Konteks broker digeser', 'fixed_embedding':'Embedding acak dibekukan'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True)
    for ax, task, title in zip(axes, ['same_day_masked','next_day_peers'],
                              ['Rekonstruksi hari yang sama', 'Prediksi sesi berikutnya']):
        group = comparisons[(comparisons.task == task) & (comparisons.block_sessions == 5)].set_index('comparator')
        for i, (method, label) in enumerate(names.items()):
            r = group.loc[method]; color = '#087f8c' if r.ci_low > 0 else '#7c8594'
            ax.errorbar(r.gain_nll, i, xerr=[[r.gain_nll-r.ci_low],[r.ci_high-r.gain_nll]],
                        fmt='o', color=color, capsize=5, linewidth=2, markersize=7)
        ax.set_yticks(range(4), names.values()); ax.invert_yaxis()
        ax.axvline(0, color='#b03a48', linestyle='--', linewidth=1)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=18)
        ax.set_xlabel('Gain NLL full terhadap pembanding\nPositif = full lebih baik', labelpad=12)
        ax.grid(axis='x', alpha=.18); ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    fig.suptitle(f'{ticker} | Apakah konteks broker membawa informasi?', fontsize=17, fontweight='bold', y=.97)
    fig.text(.5,.025, f'Maks. {dates} tanggal test per tugas | 3 fold expanding | 5 seed | Interval bootstrap blok 5 sesi, 95%\nUji historis eksploratif; hubungan statistik bukan bukti koordinasi atau kemampuan memprediksi harga.', ha='center', fontsize=10, color='#536170')
    fig.tight_layout(rect=[.015,.12,.99,.89], w_pad=3)
    fig.savefig(out/'relation_evidence.png', dpi=160, facecolor='white'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', default=r'data/Daily/Summary')
    parser.add_argument('--brokers', default=r'data/Daily/All')
    parser.add_argument('--ticker',default='ADRO')
    parser.add_argument('--out', default=None)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--dimension', type=int, default=8)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = parser.parse_args()
    if len(args.ticker)!=4 or not args.ticker.isalpha() or args.ticker!=args.ticker.upper():parser.error('Use a four-letter uppercase ticker')
    if args.out is None:args.out=f'runs/{args.ticker.lower()}-relations-v1'
    if min(args.epochs, args.patience, args.dimension) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive sizes and unique seeds required')
    run(args)
