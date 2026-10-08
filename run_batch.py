"""Run both detectors on all sample CSVs; select nearest log-median ionic reference."""
import csv
import json
from pathlib import Path
import numpy as np
from spicpms import read_trace, fit_models, write_results


def main():
    root = Path('260915icpms源数据')
    references = {}
    for path in root.rglob('*.csv'):
        if path.parent.name == '标准曲线':
            t, y = read_trace(path)
            references[path] = (t, y)
    if not references:
        raise ValueError('No confirmed ionic references found')
    out = Path('results_real')
    out.mkdir(exist_ok=True)
    cached, summary = {}, []
    for path in sorted(root.rglob('*.csv')):
        if path in references:
            continue
        t, y = read_trace(path)
        sample_median = max(float(np.median(y)), 1e-9)
        reference = min(references, key=lambda p: abs(np.log(max(float(np.median(references[p][1])), 1e-9) / sample_median)))
        bt, bg = references[reference]
        if not np.isclose(np.median(np.diff(bt)), np.median(np.diff(t)), rtol=.01):
            raise ValueError('Sampling intervals differ')
        if reference not in cached:
            cached[reference] = fit_models(bg)
        rf, iso, calibration, source = cached[reference]
        _, report = write_results(out / path.stem, t, y, (rf, iso), calibration,
                                  source, .001, 'cps')
        report['sample_file'] = str(path.resolve())
        report['ionic_reference_file'] = str(reference.resolve())
        report['reference_selection'] = 'Nearest log median signal; does not guarantee matrix/noise matching'
        report['reference_median_cps'] = float(np.median(bg))
        report['sample_median_cps'] = sample_median
        (out / path.stem / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        summary.append(dict(sample=path.stem, reference=reference.stem,
                            supervised_events=report['models']['supervised']['event_count'],
                            unsupervised_events=report['models']['unsupervised']['event_count'],
                            sample_median_cps=sample_median, reference_median_cps=float(np.median(bg))))
        print(path.stem, summary[-1]['supervised_events'], summary[-1]['unsupervised_events'], flush=True)
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)


if __name__ == '__main__':
    main()
