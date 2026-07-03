#!/usr/bin/env python3
import csv, sys, statistics as st

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ('', None) else 0.0) for k, v in r.items()})
    return rows

def pct(xs, p):
    xs = sorted(xs)
    if not xs: return float('nan')
    i = min(len(xs)-1, int(round(p*(len(xs)-1))))
    return xs[i]

def med(xs): return st.median(xs) if xs else float('nan')

def summarize(path, warmup=5):
    rows = load(path)
    n = len(rows)
    tot = [r['total_ms'] for r in rows]
    wall_s = sum(tot)/1000.0
    body = rows[warmup:]
    steady_total = [r['total_ms'] for r in body]
    print(f"=== {path} ===")
    print(f"frames={n}  wall(sum total_ms)={wall_s:.1f}s  fps(n/wall)={n/wall_s:.2f}")
    print(f"frame total_ms: median={med(steady_total):.1f} p90={pct(steady_total,0.9):.1f}")
    nerf_wait = sum(r['nerf_wait_ms'] for r in rows)/1000.0
    print(f"nerf_wait sum={nerf_wait:.1f}s ({100*nerf_wait/wall_s:.1f}% of wall)")
    for stage in ['save_result_ms','save_result_wait_ms','loftr_predict_ms','ransac_ms','select_kf_ms',
                  'optimize_gpu_ms','make_frame_ms','find_corres_ref_ms','find_corres_local_ms']:
        xs = [r.get(stage, 0.0) for r in body]
        # split first-half vs second-half to see drift
        h = len(xs)//2
        print(f"  {stage:22s} median={med(xs):6.2f}  1stHalf={med(xs[:h]):6.2f} 2ndHalf={med(xs[h:]):6.2f}")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        summarize(p)
        print()
