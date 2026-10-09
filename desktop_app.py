"""Portable desktop front end for a single particle detector."""
import csv
import json
import os
import queue
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from spicpms import read_trace, features, inject, events


def train(background, mode, label_path=None):
    cut = len(background) * 2 // 3
    if cut < 300 or len(background) - cut < 200:
        raise ValueError('纯离子背景至少需要 600 个采样点。')
    bg, calibration = background[:cut], background[cut:]
    clean, _, _ = features(bg)
    source = 'pure ionic background (no particle labels)'
    if mode == 'supervised':
        if label_path:
            d = np.loadtxt(label_path, delimiter=',', skiprows=1, ndmin=2, encoding='utf-8-sig')
            if d.shape[1] != 3 or len(d) < 100 or not np.isfinite(d).all() or np.any(np.diff(d[:, 0]) <= 0) or np.any(d[:, 1] < 0):
                raise ValueError('标注 CSV 必须包含 time,signal,label 三列及至少 100 个有效、按时间排序的点。')
            x, _, _ = features(d[:, 1])
            target = d[:, 2]
            if set(np.unique(target)) != {0, 1}:
                raise ValueError('标注必须同时包含背景 0 和颗粒 1。')
            source = 'user supplied point labels'
        else:
            signal, target = inject(bg, np.random.default_rng(42))
            x, _, _ = features(signal)
            x = np.concatenate([x, clean])
            target = np.r_[target, np.zeros(len(clean), dtype=int)]
            source = 'synthetic particle injections into measured background'
        model = RandomForestClassifier(n_estimators=150, max_depth=12,
                                       class_weight='balanced', n_jobs=-1, random_state=42).fit(x, target)
    else:
        model = IsolationForest(n_estimators=150, contamination='auto', n_jobs=-1,
                                random_state=42).fit(clean)
    return model, calibration, source


def score(model, mode, y):
    x, baseline, z = features(y)
    scores = model.predict_proba(x)[:, 1] if mode == 'supervised' else -model.score_samples(x)
    return scores, baseline, z


def load_trace(path, interval_ms=None):
    times, signal = read_trace(path)
    file_interval = float(np.median(np.diff(times)))
    if interval_ms is not None:
        if not np.isfinite(interval_ms) or interval_ms <= 0:
            raise ValueError('采样间隔必须是有效正数（毫秒）。')
        times = times[0] + np.arange(len(signal)) * interval_ms / 1000
    return times, signal, file_interval


def calibrate_standard(path, diameter_nm, unit='cps', interval_ms=None):
    if not np.isfinite(diameter_nm) or diameter_nm <= 0:
        raise ValueError('标准颗粒粒径必须是正数。')
    times, signal, _ = load_trace(path, interval_ms)
    _, baseline, z = features(signal)
    # A common strict reference selection keeps both model calibrations comparable.
    reference_events = events(z > 8, times, signal, baseline, z, unit)
    counts = np.array([e['net_counts_estimate'] for e in reference_events], dtype=float)
    if len(counts) < 5 or not np.isfinite(counts).all() or np.median(counts) <= 0:
        raise ValueError('标准样品中不足 5 个可用强事件（背景归一化信号 > 8），无法校准。')
    return dict(reference_file=str(Path(path).resolve()), reference_diameter_nm=diameter_nm,
                reference_median_net_counts=float(np.median(counts)), reference_event_count=len(counts),
                reference_sampling_interval_s=float(np.median(np.diff(times))),
                reference_selection='local robust z > 8; adjacent points merged; median net integrated counts',
                assumption='Same composition, density and instrument sensitivity; diameter proportional to cube root of net counts')


def analyze(model, mode, calibration, source, sample, out, alpha=.001, unit='cps', background_path=None, mass_factor=None, density=None, size_calibration=None, interval_ms=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times, signal, file_interval = load_trace(sample, interval_ms)
    scores, baseline, z = score(model, mode, signal)
    cal_scores, _, cal_z = score(model, mode, calibration)
    threshold = float(np.quantile(np.where(cal_z > 0, cal_scores, -np.inf), 1 - alpha, method='higher'))
    if not np.isfinite(threshold):
        raise ValueError('背景波动不足，无法校准检测阈值，请提供更长且有代表性的离子背景。')
    mask = (scores > threshold) & (z > 0)
    found = events(mask, times, signal, baseline, scores, unit)
    sizes = None
    if size_calibration is not None:
        sizes = []
        for event in found:
            diameter = size_calibration['reference_diameter_nm'] * (event['net_counts_estimate'] / size_calibration['reference_median_net_counts']) ** (1 / 3)
            event['equivalent_sphere_diameter_nm'] = diameter
            sizes.append(diameter)
    elif mass_factor is not None:
        if density is None or not np.isfinite([mass_factor, density]).all() or min(mass_factor, density) <= 0:
            raise ValueError('质量系数和密度必须是有效正数。')
        sizes = []
        for event in found:
            mass = event['net_counts_estimate'] * mass_factor
            diameter = (6 * mass * 1e6 / (np.pi * density)) ** (1 / 3)
            event.update(particle_mass_fg=mass, equivalent_sphere_diameter_nm=diameter)
            sizes.append(diameter)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'events.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['start_index', 'end_index', 'start_time', 'end_time',
            'duration_s', 'peak', 'net_signal_sum', 'net_counts_estimate', 'score']
        if mass_factor is not None and size_calibration is None:
            fields += ['particle_mass_fg']
        if sizes is not None:
            fields += ['equivalent_sphere_diameter_nm']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(found)
    with (out / 'point_predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'signal', 'baseline', 'score', 'detected'])
        writer.writerows(zip(times, signal, baseline, scores, mask.astype(int)))
    report = dict(model=mode, sample=str(Path(sample).resolve()), background=background_path,
                  training_source=source, unit=unit, sampling_interval_s=float(np.median(np.diff(times))),
                  file_sampling_interval_s=file_interval, timing_source='manual interval' if interval_ms is not None else 'CSV time column',
                  alpha=alpha, threshold=threshold, candidate_event_count=len(found),
                  calibration_point_fpr=float(((cal_scores > threshold) & (cal_z > 0)).mean()),
                  mass_factor_fg_per_count=mass_factor, particle_density_g_cm3=density,
                  size_calibration=size_calibration,
                  size_status='calibrated equivalent sphere diameter' if sizes is not None else 'not calibrated; signal distribution only',
                  limitation='Candidate events require independent validation; no measured real-data accuracy. Counts integration assumes continuous equal-duration sampling.')
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    peak = int(np.argmax(signal))
    a, b = max(0, peak - 250), min(len(signal), peak + 251)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    overview, ax = axes
    overview.plot(times, signal, lw=.5, label='Full signal')
    overview.plot(times, baseline, lw=.5, label='Local background')
    event_peaks = [e['start_index'] + int(np.argmax(signal[e['start_index']:e['end_index'] + 1])) for e in found]
    overview.scatter(times[event_peaks], signal[event_peaks], color='red', s=18, label='Candidate particle events')
    overview.set(xlabel='Time (s)', ylabel=unit, title=f'{mode}: {len(found)} candidate events (full trace)')
    overview.legend()
    ax.plot(times[a:b], signal[a:b], lw=.8, label='Signal')
    ax.plot(times[a:b], baseline[a:b], label='Local background')
    selected = np.flatnonzero(mask[a:b]) + a
    ax.scatter(times[selected], signal[selected], color='red', s=20, label='Candidate points')
    ax.set(xlabel='Time (s)', ylabel=unit, title='Detail near strongest signal')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'detections.png', dpi=150)
    plt.close(fig)
    values = np.array(sizes if sizes is not None else [e['net_counts_estimate'] for e in found], dtype=float)
    label = 'Equivalent sphere diameter (nm)' if sizes is not None else 'Net integrated signal (estimated counts; NOT particle diameter)'
    hist, edges = np.histogram(values, bins=20)
    with (out / 'distribution.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['bin_left_nm' if sizes is not None else 'bin_left_counts', 'bin_right_nm' if sizes is not None else 'bin_right_counts', 'event_count'])
        w.writerows(zip(edges[:-1], edges[1:], hist))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=edges, color='#377eb8', edgecolor='white')
    ax.set(xlabel=label, ylabel='Candidate event count', title=f'N = {len(found)}')
    fig.tight_layout()
    fig.savefig(out / ('size_distribution.png' if sizes is not None else 'signal_distribution.png'), dpi=150)
    plt.close(fig)
    stats = {'candidate_particle_count': len(found), 'diameter_available': sizes is not None,
             'mean_diameter_nm': float(values.mean()) if sizes is not None and len(values) else None,
             'median_diameter_nm': float(np.median(values)) if sizes is not None and len(values) else None,
             'std_diameter_nm': float(values.std()) if sizes is not None and len(values) else None}
    (out / 'particle_statistics.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out / 'particle_statistics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['statistic', 'value'])
        w.writerows(stats.items())
    return report


def launch(mode):
    from signal_gui import launch as open_window
    open_window(mode)


def entry(mode):
    import multiprocessing
    multiprocessing.freeze_support()
    if '--self-test' in sys.argv:
        import tkinter as tk
        output = Path(sys.argv[sys.argv.index('--self-test') + 1])
        output.mkdir(parents=True, exist_ok=True)
        try:
            root = tk.Tk()
            root.withdraw()
            root.update()
            root.destroy()
            rng = np.random.default_rng(2026)
            background = rng.poisson(20, 3000).astype(float)
            sample, _ = inject(rng.poisson(20, 1500).astype(float), rng)
            p = output / 'synthetic_sample.csv'
            with p.open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['time', 'signal'])
                w.writerows(zip(np.arange(len(sample)) * .003, sample / .003))
            model, calibration, source = train(background / .003, mode)
            size_info = calibrate_standard(p, 100, interval_ms=1.5)
            report = analyze(model, mode, calibration, source, p, output, size_calibration=size_info, interval_ms=1.5)
            from signal_gui import SignalApp
            root = tk.Tk()
            root.withdraw()
            app = SignalApp(root, mode)
            app.results.append(app.read_result(output, report))
            app.select_result(0)
            app.xmin.set('0.1')
            app.xmax.set('0.5')
            app.bins.set('12')
            app.log_y.set(True)
            app.redraw()
            root.update_idletasks()
            assert len(app.table.get_children()) == report['candidate_event_count']
            assert np.allclose(app.figures['signal'].axes[0].get_xlim(), [.1, .5])
            assert abs(report['sampling_interval_s'] - .0015) < 1e-9
            app.figures['signal'].savefig(output / 'interactive_signal_test.png')
            app.figures['distribution'].savefig(output / 'interactive_distribution_test.png')
            app.close()
            (output / 'self_test.json').write_text(json.dumps({'ok': True, 'interactive_ui': True, 'report': report}, indent=2), encoding='utf-8')
        except Exception:
            (output / 'self_test.json').write_text(json.dumps({'ok': False, 'error': traceback.format_exc()}), encoding='utf-8')
            sys.exit(1)
    else:
        launch(mode)
