import csv, math, sys

def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'),)*3
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (p, c-h, c+h)

def fmt(k, n):
    if not n: return "n=0"
    p, lo, hi = wilson(k, n)
    return f"{p:.3f} [{lo:.3f},{hi:.3f}] (n={n})"

CONF = ['Europe', 'Northern America']
ORDER = CONF + ['South Asia','East Asia','Southeast Asia','Latin America & Caribbean',
                'MENA','Sub-Saharan Africa','Oceania','Central Asia','NO COUNTRY']

for path in sys.argv[1:]:
    rows = {r['region']: {k: int(v) if k != 'region' else v for k, v in r.items()}
            for r in csv.DictReader(open(path))}
    print(f"\n=== {path} ===")
    print(f"{'region':<28} {'precision(post-botfilter)':<28} {'precision(raw)':<28} {'recall':<28} {'bot-pred share of fires':<24} tier")
    for reg in ORDER:
        r = rows.get(reg)
        if not r: continue
        tier = ('CONFIRMATORY' + (' (PASS n>=300)' if r['truth_n'] >= 300 else ' (UNDERPOWERED)')) if reg in CONF \
               else 'exploratory' + (' [n>=300]' if r['truth_n'] >= 300 else ' [n<300]')
        botshare = f"{r['fires_botpred']/r['fires']:.3f}" if r['fires'] else "n/a"
        print(f"{reg:<28} {fmt(r['correct_nonbot'], r['fires_nonbot']):<28} {fmt(r['correct'], r['fires']):<28} "
              f"{fmt(r['correct'], r['truth_n']):<28} {botshare:<24} {tier}")
