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


def calibrate_standard(path, diameter_nm, unit='cps'):
    if not np.isfinite(diameter_nm) or diameter_nm <= 0:
        raise ValueError('标准颗粒粒径必须是正数。')
    times, signal = read_trace(path)
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


def analyze(model, mode, calibration, source, sample, out, alpha=.001, unit='cps', background_path=None, mass_factor=None, density=None, size_calibration=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times, signal = read_trace(sample)
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
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    root = tk.Tk()
    title = '监督学习 · 随机森林' if mode == 'supervised' else '无监督学习 · Isolation Forest'
    root.title('单颗粒 ICP-MS — ' + title)
    root.geometry('860x750')
    root.minsize(800, 700)
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill='both', expand=True)
    frame.columnconfigure(1, weight=1)
    ttk.Label(frame, text='单颗粒 ICP-MS 信号识别', font=('Microsoft YaHei UI', 17, 'bold')).grid(row=0, column=0, columnspan=3, sticky='w')
    ttk.Label(frame, text=title + '  |  默认 3 ms、cps；时间间隔从 CSV 读取').grid(row=1, column=0, columnspan=3, sticky='w', pady=(5, 15))
    bg, output, labels = tk.StringVar(), tk.StringVar(value=str(Path.home() / 'Documents' / 'ICPMS_results')), tk.StringVar()
    standard = tk.StringVar()
    samples = []
    sample_text = tk.StringVar(value='尚未选择样品')

    def choose_bg():
        p = filedialog.askopenfilename(title='选择匹配的纯离子溶液 CSV', filetypes=[('CSV', '*.csv')])
        if p:
            bg.set(p)

    def choose_samples():
        p = filedialog.askopenfilenames(title='选择一个或多个样品 CSV', filetypes=[('CSV', '*.csv')])
        if p:
            samples[:] = p
            sample_text.set(f'已选择 {len(p)} 个文件：' + Path(p[0]).name)

    def choose_output():
        p = filedialog.askdirectory(title='选择结果保存文件夹')
        if p:
            output.set(p)

    for row, text, var, callback in [(2, '纯离子背景', bg, choose_bg), (3, '样品文件', sample_text, choose_samples), (4, '保存文件夹', output, choose_output)]:
        ttk.Label(frame, text=text).grid(row=row, column=0, sticky='w', padx=(0, 10), pady=8)
        ttk.Entry(frame, textvariable=var, state='readonly' if row == 3 else 'normal').grid(row=row, column=1, sticky='ew')
        ttk.Button(frame, text='选择', command=callback).grid(row=row, column=2, padx=(10, 0))
    def choose_standard():
        p = filedialog.askopenfilename(title='选择已知粒径标准：本项目 20ppt-* 为 100 nm 银颗粒', filetypes=[('CSV', '*.csv')])
        if p:
            standard.set(p)
    ttk.Label(frame, text='颗粒标准（可选）').grid(row=5, column=0, sticky='w', pady=8)
    ttk.Entry(frame, textvariable=standard).grid(row=5, column=1, sticky='ew')
    ttk.Button(frame, text='选择', command=choose_standard).grid(row=5, column=2, padx=(10, 0))
    if mode == 'supervised':
        def choose_labels():
            p = filedialog.askopenfilename(title='独立训练标注 CSV：time,signal,label', filetypes=[('CSV', '*.csv')])
            if p:
                labels.set(p)
        ttk.Label(frame, text='训练标注（可选）').grid(row=6, column=0, sticky='w', pady=8)
        ttk.Entry(frame, textvariable=labels).grid(row=6, column=1, sticky='ew')
        ttk.Button(frame, text='选择', command=choose_labels).grid(row=6, column=2, padx=(10, 0))
    settings = ttk.Frame(frame)
    settings.grid(row=7, column=0, columnspan=3, sticky='w', pady=12)
    ttk.Label(settings, text='背景逐点误检目标').pack(side='left')
    alpha = tk.StringVar(value='0.001')
    ttk.Entry(settings, textvariable=alpha, width=10).pack(side='left', padx=10)
    ttk.Label(settings, text='强度单位').pack(side='left')
    unit = tk.StringVar(value='cps')
    ttk.Combobox(settings, textvariable=unit, values=['cps', 'counts'], state='readonly', width=8).pack(side='left', padx=10)
    ttk.Label(settings, text='标准粒径 nm').pack(side='left')
    known_size = tk.StringVar(value='100')
    ttk.Entry(settings, textvariable=known_size, width=10).pack(side='left', padx=10)
    note = ('未选训练标注时使用模拟脉冲标签训练；真实准确率尚未验证。' if mode == 'supervised' else '模型检测背景异常；异常事件需进一步确认是否为颗粒。')
    ttk.Label(frame, text=note + '\n输出：全程信号图（红点标记事件）、颗粒候选数量、粒径分布图及统计表。\n本项目可选择 20ppt-* 文件，标准粒径填 100 nm；留空时仅输出积分信号分布。\n粒径校准要求标准与样品颗粒组成、密度和仪器灵敏度一致，输出等效球径。', wraplength=790).grid(row=8, column=0, columnspan=3, sticky='w', pady=8)
    bar = ttk.Progressbar(frame, mode='indeterminate')
    bar.grid(row=10, column=0, columnspan=3, sticky='ew', pady=10)
    log = tk.Text(frame, height=9, wrap='word', state='disabled')
    log.grid(row=11, column=0, columnspan=3, sticky='nsew')
    frame.rowconfigure(11, weight=1)
    messages = queue.Queue()
    state = {'running': False, 'last': None}

    def append(text):
        log.configure(state='normal')
        log.insert('end', text + '\n')
        log.see('end')
        log.configure(state='disabled')

    def run():
        if state['running']:
            return
        try:
            a = float(alpha.get())
            if not 0 < a < .1:
                raise ValueError('误检目标必须大于 0 且小于 0.1。')
            if not Path(bg.get()).is_file() or not samples or not output.get().strip():
                raise ValueError('请选择纯离子背景、样品和保存文件夹。')
            background_file, sample_files, label_file, u = bg.get(), list(samples), labels.get().strip(), unit.get()
            standard_file = standard.get().strip()
            diameter_nm = float(known_size.get()) if standard_file else None
            if standard_file and (not Path(standard_file).is_file() or not np.isfinite(diameter_nm) or diameter_nm <= 0):
                raise ValueError('请选择有效标准文件并填写正的标准粒径。')
            if any(Path(p).resolve() == Path(background_file).resolve() for p in sample_files):
                raise ValueError('样品与训练背景必须是不同文件。')
            if label_file and any(Path(p).resolve() == Path(label_file).resolve() for p in sample_files):
                raise ValueError('训练标注与待识别样品必须来自独立文件。')
            destination = Path(output.get()) / (mode + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        except Exception as e:
            messagebox.showerror('输入需要检查', str(e))
            return
        state['running'] = True
        start.configure(state='disabled')
        bar.start()
        append('正在训练模型并校准背景阈值……')

        def work():
            try:
                bt, background = read_trace(background_file)
                dt = float(np.median(np.diff(bt)))
                model, calibration, source = train(background, mode, label_file or None)
                size_info = calibrate_standard(standard_file, diameter_nm, u) if standard_file else None
                if size_info:
                    if not np.isclose(size_info['reference_sampling_interval_s'], dt, rtol=.01):
                        raise ValueError('标准颗粒与背景的采样间隔不一致。')
                    messages.put(('log', f'粒径校准：{size_info["reference_event_count"]} 个标准事件，净计数中位数 {size_info["reference_median_net_counts"]:.3f}，已知粒径 {diameter_nm:g} nm'))
                summary = []
                for i, p in enumerate(sample_files):
                    t, _ = read_trace(p)
                    if not np.isclose(np.median(np.diff(t)), dt, rtol=.01):
                        raise ValueError(f'{Path(p).name} 与背景采样间隔不一致。')
                    result = analyze(model, mode, calibration, source, p,
                                     destination / f'{i+1:03d}_{Path(p).stem}', a, u, background_file, size_calibration=size_info)
                    summary.append({'sample': p, 'candidate_events': result['candidate_event_count']})
                    messages.put(('log', f'{Path(p).name}：{result["candidate_event_count"]} 个候选事件；间隔 {result["sampling_interval_s"]*1000:.3f} ms'))
                with (destination / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=['sample', 'candidate_events'])
                    writer.writeheader()
                    writer.writerows(summary)
                messages.put(('done', str(destination)))
            except Exception:
                destination.mkdir(parents=True, exist_ok=True)
                (destination / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
                messages.put(('error', traceback.format_exc().splitlines()[-1] + f'\n详情：{destination / "error.log"}'))
        threading.Thread(target=work, daemon=True).start()

    def open_results():
        if state['last']:
            os.startfile(state['last'])

    actions = ttk.Frame(frame)
    actions.grid(row=9, column=0, columnspan=3, sticky='w')
    start = ttk.Button(actions, text='开始识别', command=run)
    start.pack(side='left', padx=(0, 10))
    ttk.Button(actions, text='打开结果文件夹', command=open_results).pack(side='left')

    def poll():
        try:
            while True:
                kind, text = messages.get_nowait()
                append(text if kind == 'log' else ('完成：' if kind == 'done' else '失败：') + text)
                if kind in ('done', 'error'):
                    state['running'] = False
                    start.configure(state='normal')
                    bar.stop()
                    if kind == 'done':
                        state['last'] = text
                    else:
                        messagebox.showerror('识别失败', text)
        except queue.Empty:
            pass
        root.after(150, poll)

    def close():
        if state['running'] and not messagebox.askyesno('正在运行', '关闭软件将中断当前识别，确定关闭？'):
            return
        root.destroy()
    root.protocol('WM_DELETE_WINDOW', close)
    poll()
    root.mainloop()


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
            size_info = calibrate_standard(p, 100)
            report = analyze(model, mode, calibration, source, p, output, size_calibration=size_info)
            (output / 'self_test.json').write_text(json.dumps({'ok': True, 'report': report}, indent=2), encoding='utf-8')
        except Exception:
            (output / 'self_test.json').write_text(json.dumps({'ok': False, 'error': traceback.format_exc()}), encoding='utf-8')
            sys.exit(1)
    else:
        launch(mode)
