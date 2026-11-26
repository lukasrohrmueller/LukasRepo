#!/usr/bin/env python3
"""Standortsuche um Faulbach: DEM-Rasterpunkte, die Faulbach UND bestehende Repeater erreichen.
Gleiche Annahmen/Kriterien wie step2_los.py (Fresnel >= 60 %, Reserve >= Mindestreserve).
Nutzung: python3 step3_sites.py --radius-km 12 --site-h 10 --zh 10"""
import argparse, json, math
import numpy as np
import step2_los as s

ap = argparse.ArgumentParser()
ap.add_argument("--radius-km", type=float, default=12)
ap.add_argument("--grid-m", type=float, default=300)
ap.add_argument("--site-h", type=float, default=10, help="ANNAHME Masthoehe neuer Standort")
ap.add_argument("--zh", type=float, default=10, help="ANNAHME Antennenhoehe Faulbach")
ap.add_argument("--max-hop-km", type=float, default=30)
ap.add_argument("--min-margin-db", type=float, default=10)
ap.add_argument("--top", type=int, default=12)
a = ap.parse_args()

dem, m = s.load_dem()
reps = json.load(open(s.HERE / "data/repeaters_filtered.json"))
reserve = lambda r: 22 + 4 - r["fspl"] - r["diff"] + 130   # gleiche Annahmen wie step2 (Defaults)
ok = lambda r: reserve(r) >= a.min_margin_db
F = s.FAULBACH
dlat = a.grid_m / 111320
dlon = a.grid_m / (111320 * math.cos(math.radians(F[0])))
N = int(a.radius_km * 1000 / a.grid_m)
rows = []
for i in range(-N, N + 1):
    for j in range(-N, N + 1):
        p = (F[0] + i * dlat, F[1] + j * dlon)
        if s.hav_km(F, p) > a.radius_km or s.hav_km(F, p) < 0.3:
            continue
        r0 = s.analyze(dem, m, F, p, a.zh, a.site_h)
        if not ok(r0):
            continue                                   # erreicht Faulbach nicht
        hits = []
        for n in reps:
            q = (n["lat"], n["lon"])
            if s.hav_km(p, q) <= a.max_hop_km:
                r = s.analyze(dem, m, p, q, a.site_h, 10)
                if ok(r):
                    hits.append((n["adv_name"], r["km"], reserve(r)))
        if hits:
            rows.append((p, float(s.sample(dem, m, *p)), r0["km"], hits))
print(f"Rasterpunkte mit Link nach Faulbach UND >=1 Repeater: {len(rows)}")
# Ranking: Zahl erreichbarer Repeater, danach kuerzester Abstand zu Faulbach
rows.sort(key=lambda x: (-len(x[3]), x[2]))
for p, z, d, hits in rows[: a.top]:
    west = [h for h in hits if True]
    print(f"\n({p[0]:.4f}, {p[1]:.4f}) Hoehe {z:.0f} m, {d:.1f} km von Faulbach, erreicht {len(hits)} Repeater, z.B.:")
    for nm, km, rs in sorted(hits, key=lambda h: h[1])[:4]:
        print(f"    {nm[:30]:30} {km:5.1f} km  Reserve {rs:.0f} dB")
