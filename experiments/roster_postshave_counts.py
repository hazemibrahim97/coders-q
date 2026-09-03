import csv
import json
import os
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
ROSTER = os.path.join(ROOT, "data", "orcid_github_roster_2025.json")
VAL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "roster_github_validation.csv")
SSA = os.path.join(ROOT, "data", "ssa_github_validation.csv")

EA = {'CN','JP','KR','KP','TW','HK','MO','MN'}
SEA = {'ID','MY','PH','SG','TH','VN','MM','KH','LA','BN','TL'}
SA = {'IN','PK','BD','LK','NP','BT','MV','AF'}
LATAM = {'MX','GT','HN','SV','NI','CR','PA','CU','DO','HT','JM','TT','BB','BS','BZ','GY','SR',
         'CO','VE','EC','PE','BO','BR','PY','UY','AR','CL','PR','MQ','GP','GF','AW','CW'}
MENA = {'DZ','BH','EG','IR','IQ','IL','JO','KW','LB','LY','MA','OM','PS','QA','SA','SY','TN','TR','AE','YE'}
SSA_SET = {'AO','BJ','BW','BF','BI','CV','CM','CF','TD','KM','CG','CD','CI','DJ','GQ','ER','SZ','ET',
       'GA','GM','GH','GN','GW','KE','LS','LR','MG','MW','ML','MR','MU','MZ','NA','NE','NG','RW',
       'ST','SN','SC','SL','SO','ZA','SS','SD','TZ','TG','UG','ZM','ZW'}
NAM = {'US','CA','BM'}
OC = {'AU','NZ','FJ','PG','SB','VU','WS','TO','NC','PF','GU','AS','KI','FM','MH','NR','PW','TV'}
CAS = {'KZ','KG','TJ','TM','UZ'}

def region(c):
    for name, s in [('East Asia',EA),('Southeast Asia',SEA),('South Asia',SA),
                    ('Latin America & Caribbean',LATAM),('MENA',MENA),
                    ('Sub-Saharan Africa',SSA_SET),('Northern America',NAM),
                    ('Oceania',OC),('Central Asia',CAS)]:
        if c in s:
            return name
    return 'Europe' if c else 'NO COUNTRY'

BASE09 = {'South Asia':3456,'East Asia':None,'Southeast Asia':None,'Latin America & Caribbean':2676,
          'MENA':1245,'Sub-Saharan Africa':481,'Europe':8880,'Northern America':6229,
          'Oceania':None,'Central Asia':None,'NO COUNTRY':2112}

def main():
    with open(ROSTER) as f:
        roster = json.load(f)
    login_country = {}
    for r in roster:
        lg = (r.get("login") or "").strip().strip("/").lower()
        if lg and lg not in login_country:
            login_country[lg] = (r.get("country") or "").strip()

    val = {}
    for path in (VAL, SSA):
        if os.path.exists(path):
            with open(path) as f:
                for row in csv.reader(f):
                    if row and row[0]:
                        val[row[0].strip().lower()] = (row[1] if len(row) > 1 else "",
                                                       row[2] if len(row) > 2 else "")

    reg_total = defaultdict(int); live = defaultdict(int); missing = defaultdict(int)
    org = defaultdict(int); other = defaultdict(int); unval = defaultdict(int)
    unmapped = defaultdict(int)
    for lg, c in login_country.items():
        r = region(c)
        if r == 'Europe' and c and c not in NAM:
            if not c.isalpha() or len(c) != 2:
                unmapped[c] += 1
        reg_total[r] += 1
        v = val.get(lg)
        if v is None:
            unval[r] += 1
        elif v[0] == 'OK' and v[1] == 'User':
            live[r] += 1
        elif v[0] == 'OK' and v[1] == 'Organization':
            org[r] += 1
        elif v[0] == 'missing':
            missing[r] += 1
        else:
            other[r] += 1

    order = ['Europe','Northern America','South Asia','East Asia','Southeast Asia',
             'Latin America & Caribbean','MENA','Sub-Saharan Africa','Oceania','Central Asia','NO COUNTRY']
    print(f"validated logins loaded: {len(val)} | roster distinct logins: {len(login_country)}\n")
    hdr = f"{'region':<28}{'roster':>8}{'liveUser':>9}{'missing':>8}{'org':>5}{'other':>7}{'unval':>7}{'09~tot':>8}{'floor':>7}"
    print(hdr); print('-'*len(hdr))
    for r in order:
        b = BASE09.get(r)
        floor = 'PASS' if live[r] >= 300 else ('--' if r == 'NO COUNTRY' else 'BELOW')
        print(f"{r:<28}{reg_total[r]:>8}{live[r]:>9}{missing[r]:>8}{org[r]:>5}{other[r]:>7}{unval[r]:>7}"
              f"{(str(b) if b else '—'):>8}{floor:>7}")
    tot_live = sum(live.values()); tot = sum(reg_total.values())
    print('-'*len(hdr))
    print(f"{'TOTAL':<28}{tot:>8}{tot_live:>9}")
    if unmapped:
        print(f"\n[warn] odd country codes bucketed to Europe catch-all: {dict(unmapped)}")
    print("\nShaving = liveUser/roster. GN (Europe, N.America) is the confirmatory tier; GS regions"
          "\nare exploratory (G3 reframe). This roster-backbone count is separate from the G3 >=300"
          "\ntruth-row floor (in-window noreply-validatable), which only GN clears.")

if __name__ == '__main__':
    main()
