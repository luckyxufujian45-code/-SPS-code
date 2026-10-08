"""Particle-event detection. Supervised labels are synthetic unless supplied."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from sklearn.ensemble import IsolationForest, RandomForestClassifier


def read_trace(path):
    text = None
    for encoding in ('utf-8-sig', 'gb18030', 'utf-16'):
        try:
            text = Path(path).read_text(encoding=encoding)
            break
        except UnicodeError:
            continue
    if text is None:
        raise ValueError('Cannot decode CSV')
    lines = text.splitlines()
    delimiter = ';' if lines[0].lower().startswith('sep=;') else ','
    if lines[0].lower().startswith('sep='):
        delimiter = lines.pop(0)[4:]
    rows = list(csv.reader(lines, delimiter=delimiter))
    header = rows.pop(0)
    if len(header) == 3 and header[1].startswith('Time'):
        ti, yi = 1, 2
    elif len(header) == 2:
        ti, yi = 0, 1
    else:
        raise ValueError('Expected time,signal or Number;Time isotope;isotope columns')
    data = np.array([[float(r[ti]), float(r[yi])] for r in rows if r])
    if len(data) < 100 or not np.isfinite(data).all():
        raise ValueError('At least 100 finite samples are required')
    if np.any(np.diff(data[:, 0]) <= 0) or np.any(data[:, 1] < 0):
        raise ValueError('Times must increase and signal must be nonnegative')
    return data[:, 0], data[:, 1]


def features(y, window=101):
    # Robust local normalization accommodates slow baseline drift; no unit conversion.
    baseline = median_filter(y, size=window, mode='nearest')
    scale = median_filter(np.abs(y - baseline), size=window, mode='nearest') * 1.4826
    floor = max(float(np.median(scale)), 1e-9)
    scale = np.maximum(scale, floor)
    z = (y - baseline) / scale
    p = np.pad(z, 2, mode='edge')
    neighbors = np.column_stack([p[i:i + len(y)] for i in range(5)])
    x = np.column_stack((neighbors, neighbors.max(axis=1),
                         np.maximum(neighbors, 0).sum(axis=1), z - neighbors[:, 1]))
    return x, baseline, z


def inject(background, rng):
    y = background.copy()
    labels = np.zeros(len(y), dtype=int)
    scale = max(float(np.median(np.abs(y - np.median(y))) * 1.4826), 1.0)
    # One to three sampled points: exploratory priors, not measured particle shapes.
    for center in range(60, len(y) - 60, 43):
        width = int(rng.integers(1, 4))
        amplitude = np.exp(rng.uniform(np.log(2), np.log(300))) * scale
        shape = np.array([1.0]) if width == 1 else (
            np.array([1.0, .45]) if width == 2 else np.array([.4, 1.0, .35]))
        y[center:center + width] += amplitude * shape
        labels[center:center + width] = 1
    return y, labels


def fit_models(background, labels=None, seed=42):
    rng = np.random.default_rng(seed)
    # Hold out the final third of the background for threshold calibration.
    cut = len(background) * 2 // 3
    if cut < 300 or len(background) - cut < 200:
        raise ValueError('Background needs at least 600 points')
    train, calibration = background[:cut], background[cut:]
    if labels is None:
        mixed, target = inject(train, rng)
        xf, _, _ = features(mixed)
        clean, _, _ = features(train)
        xf = np.concatenate([xf, clean])
        target = np.concatenate([target, np.zeros(len(clean), dtype=int)])
        source = 'synthetic particle injections into measured background'
    else:
        signal, target = labels
        xf, _, _ = features(signal)
        if set(np.unique(target)) != {0, 1}:
            raise ValueError('Labels must contain both background=0 and particle=1')
        source = 'user supplied point labels'
    rf = RandomForestClassifier(n_estimators=150, max_depth=12,
                                class_weight='balanced', n_jobs=-1, random_state=seed)
    rf.fit(xf, target)
    clean, _, _ = features(train)
    iso = IsolationForest(n_estimators=150, contamination='auto',
                          n_jobs=-1, random_state=seed).fit(clean)
    return rf, iso, calibration, source


def score_models(models, signal):
    rf, iso = models
    x, baseline, z = features(signal)
    return {'supervised': rf.predict_proba(x)[:, 1],
            'unsupervised': -iso.score_samples(x)}, baseline, z


def events(mask, times, signal, baseline, scores, unit='exported_signal'):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    dt = float(np.median(np.diff(times)))
    return [dict(start_index=int(a), end_index=int(b - 1), start_time=float(times[a]),
                 end_time=float(times[b - 1]), duration_s=float(times[b - 1] - times[a] + dt),
                 peak=float(signal[a:b].max()),
                 net_signal_sum=float(np.maximum(signal[a:b] - baseline[a:b], 0).sum()),
                 net_counts_estimate=(float(np.maximum(signal[a:b] - baseline[a:b], 0).sum()) * dt
                                      if unit == 'cps' else
                                      float(np.maximum(signal[a:b] - baseline[a:b], 0).sum()) if unit == 'counts' else ''),
                 score=float(scores[a:b].max())) for a, b in zip(starts, stops)]


def write_results(out, times, signal, models, calibration, source, alpha, unit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scores, baseline, z = score_models(models, signal)
    cal, _, cal_z = score_models(models, calibration)
    report = {'training_label_source': source, 'signal_unit': unit,
              'sampling_interval_s': float(np.median(np.diff(times))),
              'calibration_target_point_false_positive_rate': alpha,
              'limitations': 'Event identity and real precision/recall require independent labels. Adjacent positive points are merged; coincident particles cannot be resolved reliably.',
              'models': {}}
    masks = {}
    for name, values in scores.items():
        # Include positive-excursion gating in calibration, as in deployment.
        eligible = np.where(cal_z > 0, cal[name], -np.inf)
        threshold = float(np.quantile(eligible, 1 - alpha, method='higher'))
        mask = (values > threshold) & (z > 0)
        masks[name] = mask
        found = events(mask, times, signal, baseline, values, unit)
        fields = ['start_index', 'end_index', 'start_time', 'end_time', 'duration_s',
                  'peak', 'net_signal_sum', 'net_counts_estimate', 'score']
        with (out / (name + '_events.csv')).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(found)
        report['models'][name] = {'event_count': len(found), 'threshold': threshold,
                                  'calibration_point_fpr': float(((cal[name] > threshold) & (cal_z > 0)).mean())}
    with (out / 'point_predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'signal', 'baseline', 'supervised_score', 'supervised_flag',
                         'unsupervised_anomaly_score', 'unsupervised_flag'])
        writer.writerows(zip(times, signal, baseline, scores['supervised'], masks['supervised'].astype(int),
                             scores['unsupervised'], masks['unsupervised'].astype(int)))
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    # Show a bounded segment with the strongest signal, preserving original samples.
    peak = int(np.argmax(signal))
    a, b = max(0, peak - 250), min(len(signal), peak + 251)
    for ax, name in zip(axes, scores):
        ax.plot(times[a:b], signal[a:b], linewidth=.8, label='signal')
        ax.plot(times[a:b], baseline[a:b], label='local baseline')
        selected = np.flatnonzero(masks[name][a:b]) + a
        ax.scatter(times[selected], signal[selected], c='red', s=18, label='detected points')
        ax.set_title(name)
        ax.set_ylabel(unit)
        ax.legend()
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(out / 'detections.png', dpi=160)
    plt.close(fig)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return masks, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--background', help='Confirmed particle-free ionic solution CSV')
    parser.add_argument('--sample', help='Sample CSV')
    parser.add_argument('--labels', help='Optional CSV: time,signal,label (0 or 1)')
    parser.add_argument('--unit', default='exported_signal', help='Confirmed input unit; no conversion')
    parser.add_argument('--out', default='results')
    parser.add_argument('--alpha', type=float, default=.001)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    if not 0 < args.alpha < .1:
        parser.error('--alpha must be between 0 and 0.1')
    truth = None
    labeled = None
    if args.demo:
        rng = np.random.default_rng(2026)
        background = rng.poisson(20, 24000).astype(float)
        sample, truth = inject(rng.poisson(20, 12000).astype(float), rng)
        times = np.arange(len(sample)) * .003
        args.unit = 'counts (synthetic)'
    else:
        if not args.background or not args.sample:
            parser.error('Provide --background and --sample, or --demo')
        bg_times, background = read_trace(args.background)
        times, sample = read_trace(args.sample)
        if not np.isclose(np.median(np.diff(bg_times)), np.median(np.diff(times)), rtol=.01):
            parser.error('Background and sample sampling intervals must match')
        if args.labels:
            d = np.loadtxt(args.labels, delimiter=',', skiprows=1, ndmin=2)
            if d.shape[1] != 3 or not np.isfinite(d).all() or np.any(np.diff(d[:, 0]) <= 0):
                parser.error('Labels require finite, ordered time,signal,label columns')
            labeled = d[:, 1], d[:, 2]
    rf, iso, calibration, source = fit_models(background, labeled)
    masks, report = write_results(args.out, times, sample, (rf, iso), calibration,
                                  source, args.alpha, args.unit)
    if truth is not None:
        report['synthetic_point_metrics'] = {}
        for name, mask in masks.items():
            tp = int((mask & (truth == 1)).sum())
            fp = int((mask & (truth == 0)).sum())
            fn = int((~mask & (truth == 1)).sum())
            report['synthetic_point_metrics'][name] = dict(precision=tp / max(tp + fp, 1),
                                                          recall=tp / max(tp + fn, 1), tp=tp, fp=fp, fn=fn)
        Path(args.out, 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
