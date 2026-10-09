"""Interactive ICP-MS processing and result inspection window."""
import csv
import json
import os
import queue
import threading
import traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from desktop_app import train, analyze, calibrate_standard, load_trace


class SignalApp:
    def __init__(self, root, mode):
        self.root, self.mode = root, mode
        self.samples, self.results = [], []
        self.current = None
        self.running = False
        self.messages = queue.Queue()
        self.last_folder = None
        name = '监督学习 / 随机森林' if mode == 'supervised' else '无监督学习 / Isolation Forest'
        root.title('单颗粒 ICP-MS 信号处理 — ' + name)
        root.geometry('1380x900')
        root.minsize(1120, 780)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        left = ttk.Frame(root, padding=12, width=325)
        left.grid(row=0, column=0, sticky='ns')
        left.columnconfigure(0, weight=1)
        ttk.Label(left, text='信号处理参数', font=('Microsoft YaHei UI', 15, 'bold')).grid(row=0, sticky='w', pady=(0, 8))
        ttk.Label(left, text=name).grid(row=1, sticky='w', pady=(0, 8))
        self.background, self.standard, self.labels = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.sample_text = tk.StringVar(value='尚未选择样品')
        self.output = tk.StringVar(value=str(Path.home() / 'Documents' / 'ICPMS_results'))
        row = 2
        for label, variable, callback in [('纯离子背景 CSV', self.background, lambda: self.pick(self.background)),
                                         ('样品 CSV（可多选）', self.sample_text, self.pick_samples),
                                         ('已知粒径标准 CSV（可选）', self.standard, lambda: self.pick(self.standard)),
                                         ('导出目录', self.output, self.pick_output)]:
            ttk.Label(left, text=label).grid(row=row, sticky='w', pady=(7, 2))
            line = ttk.Frame(left)
            line.grid(row=row+1, sticky='ew')
            ttk.Entry(line, textvariable=variable, width=28, state='readonly' if variable == self.sample_text else 'normal').pack(side='left', fill='x', expand=True)
            ttk.Button(line, text='选择', command=callback, width=5).pack(side='left', padx=(4, 0))
            row += 2
        if mode == 'supervised':
            ttk.Label(left, text='训练标注 CSV（可选）').grid(row=row, sticky='w', pady=(7, 2))
            line = ttk.Frame(left)
            line.grid(row=row+1, sticky='ew')
            ttk.Entry(line, textvariable=self.labels, width=28).pack(side='left', fill='x', expand=True)
            ttk.Button(line, text='选择', command=lambda: self.pick(self.labels), width=5).pack(side='left', padx=(4, 0))
            row += 2
        self.timing = tk.StringVar(value='读取 CSV 时间列')
        self.interval = tk.StringVar(value='3')
        self.unit = tk.StringVar(value='cps')
        self.alpha = tk.StringVar(value='0.001')
        self.diameter = tk.StringVar(value='100')
        parameters = ttk.LabelFrame(left, text='采集与识别', padding=8)
        parameters.grid(row=row, sticky='ew', pady=12)
        ttk.Label(parameters, text='时间设置').grid(row=0, column=0, sticky='w')
        timing = ttk.Combobox(parameters, textvariable=self.timing, values=['读取 CSV 时间列', '手动采样间隔'], state='readonly', width=19)
        timing.grid(row=0, column=1, pady=4)
        ttk.Label(parameters, text='采样间隔 ms').grid(row=1, column=0, sticky='w')
        self.interval_entry = ttk.Entry(parameters, textvariable=self.interval, width=21, state='disabled')
        self.interval_entry.grid(row=1, column=1, pady=4)
        timing.bind('<<ComboboxSelected>>', lambda _: self.interval_entry.configure(state='normal' if self.timing.get() == '手动采样间隔' else 'disabled'))
        ttk.Label(parameters, text='强度单位').grid(row=2, column=0, sticky='w')
        ttk.Combobox(parameters, textvariable=self.unit, values=['cps', 'counts'], state='readonly', width=19).grid(row=2, column=1, pady=4)
        for r, label, var in [(3, '逐点误检目标', self.alpha), (4, '标准粒径 nm', self.diameter)]:
            ttk.Label(parameters, text=label).grid(row=r, column=0, sticky='w')
            ttk.Entry(parameters, textvariable=var, width=21).grid(row=r, column=1, pady=4)
        row += 1
        self.start = ttk.Button(left, text='开始识别 / 重新计算', command=self.run)
        self.start.grid(row=row, sticky='ew', pady=4)
        ttk.Button(left, text='打开导出目录', command=self.open_folder).grid(row=row+1, sticky='ew', pady=4)
        self.progress = ttk.Progressbar(left, mode='indeterminate')
        self.progress.grid(row=row+2, sticky='ew', pady=5)
        self.status = tk.StringVar(value='选择数据并设置参数后开始识别。')
        ttk.Label(left, textvariable=self.status, wraplength=290).grid(row=row+3, sticky='w', pady=8)
        ttk.Label(left, text='识别数是候选事件数。\n无标注的监督模型使用模拟标签。\n粒径校准要求标准与样品灵敏度一致。\n手动时间会重建时间轴并用于 cps 积分。', wraplength=290, foreground='#666666').grid(row=row+4, sticky='w', pady=6)

        right = ttk.Frame(root, padding=(0, 12, 12, 12))
        right.grid(row=0, column=1, sticky='nsew')
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)
        sample_bar = ttk.Frame(right)
        sample_bar.grid(row=0, sticky='ew')
        ttk.Label(sample_bar, text='当前结果').pack(side='left')
        self.result_name = tk.StringVar()
        self.result_picker = ttk.Combobox(sample_bar, textvariable=self.result_name, state='readonly', width=45)
        self.result_picker.pack(side='left', padx=8)
        self.result_picker.bind('<<ComboboxSelected>>', lambda _: self.select_result(self.result_picker.current()))
        self.info = tk.StringVar(value='暂无结果：识别完成后，信号图和统计会直接显示在这里。')
        ttk.Label(right, textvariable=self.info, wraplength=1000, font=('Microsoft YaHei UI', 10)).grid(row=1, sticky='w', pady=10)
        controls = ttk.LabelFrame(right, text='可视化设置（仅改变显示；识别参数修改后请重新计算）', padding=6)
        controls.grid(row=2, sticky='ew', pady=(0, 8))
        fields_line, toggle_line = ttk.Frame(controls), ttk.Frame(controls)
        fields_line.pack(fill='x')
        toggle_line.pack(fill='x', pady=(5, 0))
        self.xmin, self.xmax, self.ymax = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.bins = tk.StringVar(value='20')
        self.log_y, self.markers, self.show_bg = tk.BooleanVar(), tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        for label, variable, width in [('起始 s', self.xmin, 8), ('结束 s', self.xmax, 8), ('强度上限', self.ymax, 9), ('分布箱数', self.bins, 5)]:
            ttk.Label(fields_line, text=label).pack(side='left', padx=(0, 3))
            ttk.Entry(fields_line, textvariable=variable, width=width).pack(side='left', padx=(0, 8))
        for text, var in [('对数强度', self.log_y), ('颗粒标记', self.markers), ('背景线', self.show_bg)]:
            ttk.Checkbutton(toggle_line, text=text, variable=var, command=self.redraw).pack(side='left')
        ttk.Button(toggle_line, text='应用', command=self.redraw).pack(side='left', padx=5)
        ttk.Button(toggle_line, text='全程', command=self.reset_view).pack(side='left')
        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=3, sticky='nsew')
        self.figures, self.canvases = {}, {}
        for key, label in [('signal', '信号图'), ('distribution', '粒径 / 信号分布')]:
            panel = ttk.Frame(self.notebook)
            self.notebook.add(panel, text=label)
            fig = Figure(figsize=(10, 6), dpi=100)
            canvas = FigureCanvasTkAgg(fig, master=panel)
            toolbar = NavigationToolbar2Tk(canvas, panel, pack_toolbar=False)
            toolbar.pack(side='bottom', fill='x')
            canvas.get_tk_widget().pack(fill='both', expand=True)
            self.figures[key], self.canvases[key] = fig, canvas
        table_panel = ttk.Frame(self.notebook)
        self.notebook.add(table_panel, text='颗粒事件统计表')
        cols = ('index', 'time', 'duration', 'peak', 'counts', 'diameter')
        self.table = ttk.Treeview(table_panel, columns=cols, show='headings')
        for key, label in zip(cols, ['序号', '时间 s', '持续 ms', '峰值', '净积分计数', '等效粒径 nm']):
            self.table.heading(key, text=label)
            self.table.column(key, width=115, stretch=True)
        scroll = ttk.Scrollbar(table_panel, orient='vertical', command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.table.pack(fill='both', expand=True)
        self.table.bind('<<TreeviewSelect>>', self.focus_event)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.poll_id = root.after(100, self.poll)

    def pick(self, var):
        p = filedialog.askopenfilename(filetypes=[('CSV', '*.csv')])
        if p:
            var.set(p)

    def pick_samples(self):
        paths = filedialog.askopenfilenames(filetypes=[('CSV', '*.csv')])
        if paths:
            self.samples = list(paths)
            self.sample_text.set(f'{len(paths)} 个文件：{Path(paths[0]).name}')

    def pick_output(self):
        p = filedialog.askdirectory()
        if p:
            self.output.set(p)

    def run(self):
        if self.running:
            return
        try:
            bg, standard, labels = self.background.get().strip(), self.standard.get().strip(), self.labels.get().strip()
            samples = list(self.samples)
            alpha, diameter, unit = float(self.alpha.get()), float(self.diameter.get()), self.unit.get()
            interval = float(self.interval.get()) if self.timing.get() == '手动采样间隔' else None
            if not 0 < alpha < .1 or not np.isfinite(diameter) or diameter <= 0:
                raise ValueError('误检目标需在 0～0.1 之间，标准粒径必须是正数。')
            if interval is not None and (not np.isfinite(interval) or interval <= 0):
                raise ValueError('采样间隔必须是正数（ms）。')
            if not samples or not Path(bg).is_file() or not self.output.get().strip():
                raise ValueError('请选择背景、样品及导出目录。')
            if any(Path(p).resolve() == Path(bg).resolve() for p in samples):
                raise ValueError('背景与待识别样品不能是同一个文件。')
            if labels and any(Path(p).resolve() == Path(labels).resolve() for p in samples):
                raise ValueError('训练标注与样品必须独立。')
            destination = Path(self.output.get()) / (self.mode + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        except Exception as e:
            messagebox.showerror('参数需要检查', str(e))
            return
        self.running = True
        self.results = []
        self.result_picker.configure(values=[])
        self.result_name.set('')
        self.current = None
        self.info.set('正在处理新数据，结果将在完成后显示。')
        self.table.delete(*self.table.get_children())
        for key in self.figures:
            self.figures[key].clear()
            self.canvases[key].draw_idle()
        self.start.configure(state='disabled')
        self.progress.start()
        self.status.set('正在训练和校准……')

        def worker():
            try:
                bt, background, _ = load_trace(bg, interval)
                dt = float(np.median(np.diff(bt)))
                model, calibration, source = train(background, self.mode, labels or None)
                size = calibrate_standard(standard, diameter, unit, interval) if standard else None
                if size and not np.isclose(size['reference_sampling_interval_s'], dt, rtol=.01):
                    raise ValueError('标准与背景时间间隔不同，请检查文件或使用明确的手动间隔。')
                summaries = []
                for i, p in enumerate(samples):
                    t, _, _ = load_trace(p, interval)
                    if not np.isclose(np.median(np.diff(t)), dt, rtol=.01):
                        raise ValueError(f'{Path(p).name} 的采样间隔与背景不同。')
                    folder = destination / f'{i+1:03d}_{Path(p).stem}'
                    report = analyze(model, self.mode, calibration, source, p, folder, alpha, unit, bg, size_calibration=size, interval_ms=interval)
                    self.messages.put(('result', self.read_result(folder, report)))
                    summaries.append({'sample': p, 'candidate_events': report['candidate_event_count']})
                with (destination / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
                    w = csv.DictWriter(f, fieldnames=['sample', 'candidate_events'])
                    w.writeheader()
                    w.writerows(summaries)
                self.messages.put(('done', str(destination)))
            except Exception:
                destination.mkdir(parents=True, exist_ok=True)
                (destination / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
                self.messages.put(('error', traceback.format_exc().splitlines()[-1]))
        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def read_result(folder, report):
        points = np.loadtxt(folder / 'point_predictions.csv', delimiter=',', skiprows=1, encoding='utf-8-sig', ndmin=2)
        with (folder / 'events.csv').open(encoding='utf-8-sig') as f:
            found = list(csv.DictReader(f))
        stats = json.loads((folder / 'particle_statistics.json').read_text(encoding='utf-8'))
        return dict(folder=folder, report=report, points=points, events=found, stats=stats)

    def select_result(self, index):
        if not 0 <= index < len(self.results):
            return
        self.current = self.results[index]
        r, p, s = self.current['report'], self.current['points'], self.current['stats']
        size_text = f'平均粒径 {s["mean_diameter_nm"]:.2f} nm；中位数 {s["median_diameter_nm"]:.2f} nm' if s['mean_diameter_nm'] is not None else '未校准粒径：显示净积分计数分布'
        self.info.set(f'{Path(r["sample"]).name}  |  {len(p):,} 点  |  时长 {p[-1,0]-p[0,0]:.3f} s  |  间隔 {r["sampling_interval_s"]*1000:g} ms  |  单位 {r["unit"]}\n候选颗粒 {s["candidate_particle_count"]} 个  |  {size_text}\n时间来源：{r["timing_source"]}；模型阈值 {r["threshold"]:.5f}；背景校准逐点误检率 {r["calibration_point_fpr"]:.4%}')
        self.table.delete(*self.table.get_children())
        for i, e in enumerate(self.current['events']):
            self.table.insert('', 'end', iid=str(i), values=(i+1, f'{float(e["start_time"]):.6f}', f'{float(e["duration_s"])*1000:.3f}', f'{float(e["peak"]):.3f}', f'{float(e["net_counts_estimate"]):.3f}', f'{float(e["equivalent_sphere_diameter_nm"]):.2f}' if 'equivalent_sphere_diameter_nm' in e else '未校准'))
        self.reset_view()

    def reset_view(self):
        self.xmin.set('')
        self.xmax.set('')
        self.ymax.set('')
        self.redraw()

    def redraw(self):
        if self.current is None:
            return
        try:
            points, found = self.current['points'], self.current['events']
            lo = float(self.xmin.get()) if self.xmin.get().strip() else points[0, 0]
            hi = float(self.xmax.get()) if self.xmax.get().strip() else points[-1, 0]
            upper = float(self.ymax.get()) if self.ymax.get().strip() else None
            bins = int(self.bins.get())
            if not np.isfinite([lo, hi]).all() or hi <= lo or not 1 <= bins <= 500 or (upper is not None and (not np.isfinite(upper) or upper <= 0)):
                raise ValueError('起始时间需小于结束时间；强度上限需为正数；分布箱数需为 1～500。')
            fig = self.figures['signal']
            fig.clear()
            ax = fig.add_subplot(111)
            ax.plot(points[:,0], points[:,1], lw=.65, label='Signal')
            if self.show_bg.get():
                ax.plot(points[:,0], points[:,2], lw=.65, label='Background')
            if self.markers.get() and found:
                indices = [int(e['start_index']) + int(np.argmax(points[int(e['start_index']):int(e['end_index'])+1,1])) for e in found]
                ax.scatter(points[indices,0], points[indices,1], s=23, color='#e34a33', label='Candidate particles')
            ax.set(xlabel='Time (s)', ylabel=self.current['report']['unit'], xlim=(lo,hi))
            visible = points[(points[:,0] >= lo) & (points[:,0] <= hi),1]
            top = upper if upper is not None else max(float(visible.max()) * 1.1, 1.) if len(visible) else max(float(points[:,1].max()) * 1.1, 1.)
            if self.log_y.get():
                positive = visible[visible > 0]
                bottom = max(float(positive.min()) * .8, 1e-9) if len(positive) else 1e-3
                ax.set_yscale('log', nonpositive='clip')
                ax.set_ylim(min(bottom, top / 10), top)
            else:
                ax.set_ylim(0, top)
            ax.legend(loc='upper right')
            fig.tight_layout()
            self.canvases['signal'].draw_idle()
            fig = self.figures['distribution']
            fig.clear()
            ax = fig.add_subplot(111)
            calibrated = self.current['stats']['diameter_available']
            key = 'equivalent_sphere_diameter_nm' if calibrated else 'net_counts_estimate'
            values = [float(e[key]) for e in found]
            ax.hist(values, bins=bins, color='#377eb8', edgecolor='white')
            ax.set(xlabel='Equivalent sphere diameter (nm)' if calibrated else 'Net integrated signal (counts; not particle diameter)', ylabel='Candidate event count', title=f'N = {len(values)}; {bins} bins')
            fig.tight_layout()
            self.canvases['distribution'].draw_idle()
        except Exception as e:
            messagebox.showerror('可视化参数需要检查', str(e))

    def focus_event(self, _):
        selection = self.table.selection()
        if not selection or self.current is None:
            return
        e = self.current['events'][int(selection[0])]
        t = self.current['points'][:,0]
        pad = max(.02, self.current['report']['sampling_interval_s'] * 10)
        self.xmin.set(f'{max(t[0],float(e["start_time"])-pad):.6f}')
        self.xmax.set(f'{min(t[-1],float(e["end_time"])+pad):.6f}')
        self.redraw()
        self.notebook.select(0)

    def poll(self):
        try:
            while True:
                kind, data = self.messages.get_nowait()
                if kind == 'result':
                    self.results.append(data)
                    self.result_picker.configure(values=[f'{i+1}. {Path(r["report"]["sample"]).name}' for i,r in enumerate(self.results)])
                    self.result_picker.current(len(self.results)-1)
                    self.select_result(len(self.results)-1)
                    self.status.set(f'已完成 {len(self.results)} / {len(self.samples)} 个样品')
                else:
                    self.running = False
                    self.start.configure(state='normal')
                    self.progress.stop()
                    if kind == 'done':
                        self.last_folder = data
                        self.status.set('识别完成。右侧可缩放、查看统计并切换样品；结果已导出。')
                    else:
                        self.status.set('失败：' + data)
                        messagebox.showerror('识别失败', data)
        except queue.Empty:
            pass
        self.poll_id = self.root.after(100, self.poll)

    def open_folder(self):
        if self.last_folder:
            os.startfile(self.last_folder)

    def close(self):
        if self.running and not messagebox.askyesno('正在识别', '关闭将中断当前处理，确定关闭？'):
            return
        self.root.after_cancel(self.poll_id)
        self.root.destroy()


def launch(mode):
    root = tk.Tk()
    SignalApp(root, mode)
    root.mainloop()
