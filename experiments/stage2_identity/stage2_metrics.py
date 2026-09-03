import csv
import math
import sys
from collections import defaultdict

EA = {'CN','JP','KR','KP','TW','HK','MO','MN'}
SEA = {'ID','MY','PH','SG','TH','VN','MM','KH','LA','BN','TL'}
SA = {'IN','PK','BD','LK','NP','BT','MV','AF'}
LATAM = {'MX','GT','HN','SV','NI','CR','PA','CU','DO','HT','JM','TT','BB','BS','BZ','GY','SR',
         'CO','VE','EC','PE','BO','BR','PY','UY','AR','CL','PR','MQ','GP','GF','AW','CW'}
MENA = {'DZ','BH','EG','IR','IQ','IL','JO','KW','LB','LY','MA','OM','PS','QA','SA','SY','TN','TR','AE','YE'}
SSA = {'AO','BJ','BW','BF','BI','CV','CM','CF','TD','KM','CG','CD','CI','DJ','GQ','ER','SZ','ET',
       'GA','GM','GH','GN','GW','KE','LS','LR','MG','MW','ML','MR','MU','MZ','NA','NE','NG','RW',
       'ST','SN','SC','SL','SO','ZA','SS','SD','TZ','TG','UG','ZM','ZW'}
NAM = {'US','CA','BM'}
OC = {'AU','NZ','FJ','PG','SB','VU','WS','TO','NC','PF','GU','AS','KI','FM','MH','NR','PW','TV'}
CAS = {'KZ','KG','TJ','TM','UZ'}

def region(c):
    for name, s in [('East Asia',EA),('Southeast Asia',SEA),('South Asia',SA),
                    ('Latin America & Caribbean',LATAM),('MENA',MENA),
                    ('Sub-Saharan Africa',SSA),('Northern America',NAM),
                    ('Oceania',OC),('Central Asia',CAS)]:
        if c in s:
            return name
    return 'Europe' if c else 'NO COUNTRY'

def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'), float('nan'))
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (p, c - h, c + h)

def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{p:.3f} [{lo:.3f},{hi:.3f}] (n={n})" if n else "n=0"

def main(val_csv, cov_csv):

    fires = defaultdict(int); correct = defaultdict(int); total = defaultdict(int)
    for row in csv.DictReader(open(val_csv)):
        r = region(row['country'])
        total[r] += 1
        if row['rule_fires'] == 'true':
            fires[r] += 1
            if row['correct'] == 'true':
                correct[r] += 1

    cov_n = defaultdict(int); cov_active = defaultdict(int); cov_reach = defaultdict(int)
    fp_pat = defaultdict(int); fp_vol = defaultdict(int); fp_den = defaultdict(int)
    for row in csv.DictReader(open(cov_csv)):
        r = region(row['country'])
        cov_n[r] += 1
        active = row['active_as_pusher'] == 'true' or row['reachable_via_resolver'] == 'true'
        if active:
            cov_active[r] += 1
            fp_den[r] += 1
            if row['bot_pattern_fp'] == 'true': fp_pat[r] += 1
            if row['bot_volume_fp'] == 'true': fp_vol[r] += 1
        if row['reachable_via_resolver'] == 'true':
            cov_reach[r] += 1

    confirmatory = ['Europe', 'Northern America']
    gs = ['South Asia','East Asia','Southeast Asia','Latin America & Caribbean','MENA','Sub-Saharan Africa']
    order = confirmatory + gs + ['Oceania','Central Asia']
    print(f"{'region':<28} {'precision':<28} {'recall':<28} {'truth_n':>7} {'tier'}")
    for r in order:
        if r in confirmatory:
            tier = 'CONFIRMATORY (PASS)' if total[r] >= 300 else 'CONFIRMATORY (UNDERPOWERED)'
        else:
            tier = 'exploratory' + (' [n<300]' if total[r] < 300 else ' [n>=300]')
        print(f"{r:<28} {fmt(correct[r], fires[r]):<28} {fmt(correct[r], total[r]):<28} {total[r]:>7} {tier}")
    print(f"\n{'region':<28} {'coverage(any activity)':<28} {'bot FP (pattern|volume)'}")
    for r in order:
        print(f"{r:<28} {fmt(cov_active[r], cov_n[r]):<28} {fmt(fp_pat[r], fp_den[r])} | {fmt(fp_vol[r], fp_den[r])}")
    print("\nG3 judgment inputs: compare Global-South rows against Europe/N-America rows —")
    print("the STOP branch triggers on inadequate OR geographically skewed accuracy (prereg §5.1).")

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'g3_validation_rows.csv',
         sys.argv[2] if len(sys.argv) > 2 else 'g3_coverage_rows.csv')
