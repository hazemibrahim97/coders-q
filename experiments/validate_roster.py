import csv
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROSTER = os.path.join(BASE, "data", "orcid_github_roster_2025.json")
SSA_CSV = os.path.join(BASE, "data", "ssa_github_validation.csv")
OUT_CSV = os.path.join(BASE, "data", "roster_github_validation.csv")

TOKEN = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
if not TOKEN:
    sys.exit("no gh auth token")

with open(ROSTER) as f:
    roster = json.load(f)
logins = sorted({r["login"].strip().strip("/").lower() for r in roster if r.get("login")})

done = set()
for path in (SSA_CSV, OUT_CSV):
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.reader(f):
                if row:
                    done.add(row[0].lower())

todo = [l for l in logins if l not in done]
print(f"{len(logins)} distinct logins, {len(done)} already validated, {len(todo)} to do", flush=True)

out = open(OUT_CSV, "a", newline="")
w = csv.writer(out)

PACE = 0.75
for i, login in enumerate(todo):
    req = urllib.request.Request(
        f"https://api.github.com/users/{urllib.parse.quote(login)}",
        headers={"Authorization": f"Bearer {TOKEN}",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "codersq-roster-validation"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            d = json.load(resp)
            remaining = int(resp.headers.get("X-RateLimit-Remaining", "1000"))
            w.writerow([login, "OK", d.get("type", ""), d.get("created_at", ""),
                        d.get("public_repos", "")])
    except urllib.error.HTTPError as e:
        remaining = int(e.headers.get("X-RateLimit-Remaining", "1000") or "1000")
        if e.code == 404:
            w.writerow([login, "missing", "", "", ""])
        elif e.code in (403, 429) and remaining == 0:
            reset = int(e.headers.get("X-RateLimit-Reset", time.time() + 300))
            wait = max(30, reset - time.time() + 5)
            print(f"rate-limited at {i}/{len(todo)}, sleeping {wait:.0f}s", flush=True)
            out.flush()
            time.sleep(wait)
            todo.append(login)
        else:
            w.writerow([login, f"error_{e.code}", "", "", ""])
    except Exception as ex:
        w.writerow([login, "error_net", "", "", ""])
    if i % 500 == 0:
        out.flush()
        print(f"{i}/{len(todo)} done", flush=True)
    time.sleep(PACE)

out.close()
print("COMPLETE", flush=True)
