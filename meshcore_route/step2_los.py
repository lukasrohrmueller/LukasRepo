#!/usr/bin/env python3
"""Schritt 2/3 (Kern): Sichtlinie, Fresnel, Link-Reserve, Routing ab Faulbach.

Daten:  Copernicus GLO-30 *DSM* (Oberflaechenmodell: enthaelt Wald/Bebauung),
        lokal in data/dem/*.tif, wird einmal zu data/dem_crop.npy zusammengesetzt.
Modell: Erdkruemmung k=4/3, Freiraumdaempfung, 1. Fresnel-Zone (Kriterium: >= 60 % frei).
ALLE Funk-/Antennen-Parameter sind ANNAHMEN (siehe Argumente) - keine Messwerte!

Beispiel:
  python3 step2_los.py --max-hop-km 30 --rep-h 10
"""
import argparse, heapq, json, math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R_EARTH = 6371000.0
FREQ_MHZ = 869.618                     # aus den Karten-Daten (alle Repeater)
LAMBDA = 299792458.0 / (FREQ_MHZ * 1e6)

# --- Koordinaten (Ortsmitten per OSM/Nominatim; NICHT die Antennenstandorte) ---
FAULBACH = (49.7848, 9.4411)
MAINZ = (49.9995, 8.2736)


# ---------------------------------------------------------------- DEM
def load_dem():
    """DEM-Kacheln zu einem Ausschnitt zusammensetzen und als .npy cachen."""
    npy, meta = HERE / "data/dem_crop.npy", HERE / "data/dem_crop.json"
    if not npy.exists():
        import rasterio
        from rasterio.merge import merge
        srcs = [rasterio.open(p) for p in sorted((HERE / "data/dem").glob("dem_*.tif"))]
        arr, tr = merge(srcs, bounds=(8.1, 49.5, 9.95, 50.2))
        np.save(npy, arr[0].astype("float32"))
        meta.write_text(json.dumps({"a": tr.a, "e": tr.e, "c": tr.c, "f": tr.f}))
    return np.load(npy, mmap_mode="r"), json.loads(meta.read_text())


def sample(dem, m, lat, lon):
    """Bilineare Hoehe (m ue. NN) an Koordinaten (Arrays erlaubt)."""
    col = (np.asarray(lon) - m["c"]) / m["a"] - 0.5
    row = (np.asarray(lat) - m["f"]) / m["e"] - 0.5
    r0, c0 = np.floor(row).astype(int), np.floor(col).astype(int)
    fr, fc = row - r0, col - c0
    z = lambda r, c: np.asarray(dem[r, c], dtype="float64")
    return (z(r0, c0) * (1 - fr) * (1 - fc) + z(r0, c0 + 1) * (1 - fr) * fc
            + z(r0 + 1, c0) * fr * (1 - fc) + z(r0 + 1, c0 + 1) * fr * fc)


# ---------------------------------------------------------------- Geometrie
def hav_km(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH / 1000 * math.asin(math.sqrt(h))


def analyze(dem, m, a, b, ha, hb, k=4 / 3, step=30.0):
    """Link a->b. a,b=(lat,lon). ha,hb = Antennenhoehe ueber DSM-Oberflaeche (m)."""
    D = hav_km(a, b) * 1000
    n = max(int(D / step), 4)
    t = np.linspace(0, 1, n + 1)
    lat = a[0] + (b[0] - a[0]) * t          # gerade Linie in Lat/Lon (ok < ~60 km)
    lon = a[1] + (b[1] - a[1]) * t
    z = sample(dem, m, lat, lon)
    d = t * D
    z1, z2 = z[0] + ha, z[-1] + hb
    los = z1 + (z2 - z1) * t
    bulge = d * (D - d) / (2 * k * R_EARTH)             # Erdkruemmung (k=4/3)
    f1 = np.sqrt(LAMBDA * d * (D - d) / D)              # Radius 1. Fresnel-Zone
    clr = los - (z + bulge)                             # Freiraum ueber Hindernis
    ratio = clr[1:-1] / f1[1:-1]                        # Anteil der Fresnel-Zone, der frei ist
    i = int(np.argmin(ratio))
    # Beugungsdaempfung: dominante Kante (Knife-Edge, ITU-R P.526). Naeherung: nur EIN Hindernis,
    # mehrere Kanten wuerden mehr Daempfung ergeben (also eher zu optimistisch bei Huegelketten).
    v = (-clr[1:-1]) * np.sqrt(2 * D / (LAMBDA * d[1:-1] * (D - d[1:-1])))
    vm = float(v.max())
    diff = 0.0 if vm <= -0.78 else 6.9 + 20 * math.log10(math.sqrt((vm - 0.1) ** 2 + 1) + vm - 0.1)
    fspl = 32.45 + 20 * math.log10(FREQ_MHZ) + 20 * math.log10(D / 1000)
    return {"km": D / 1000, "min_ratio": float(ratio[i]), "worst_km": float(d[i + 1] / 1000),
            "clear_m": float(clr[i + 1]), "fspl": fspl, "diff": diff}


# ---------------------------------------------------------------- Routing
def bottleneck(adj, src, key):
    """Pfad mit kleinstem 'schlimmsten' Kantenwert (Minimax). key(edge)-> Kosten, kleiner=besser.
    Tie-Break: Hopzahl. Gibt (best, prev) fuer alle Knoten zurueck."""
    best = {src: (0.0, 0)}
    prev = {}
    pq = [(0.0, 0, src)]
    while pq:
        c, h, u = heapq.heappop(pq)
        if best[u] != (c, h):
            continue
        for v, e in adj[u].items():
            cand = (max(c, key(e)), h + 1)
            if v not in best or cand < best[v]:
                best[v] = cand
                prev[v] = u
                heapq.heappush(pq, (cand[0], cand[1], v))
    return best, prev


def path_to(prev, src, dst):
    p = [dst]
    while p[-1] != src:
        p.append(prev[p[-1]])
    return p[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", nargs=3, type=float, default=[*MAINZ, 10.0], metavar=("LAT", "LON", "H"))
    ap.add_argument("--end", nargs=3, type=float, default=[*FAULBACH, 10.0], metavar=("LAT", "LON", "H"))
    ap.add_argument("--rep-h", type=float, default=10.0, help="ANNAHME Repeater-Antennenhoehe ueber DSM (m)")
    ap.add_argument("--max-hop-km", type=float, default=30.0)
    ap.add_argument("--fresnel", type=float, default=None,
                    help="optional: harte Bedingung, freier Anteil der 1. Fresnel-Zone (z.B. 0.6). Standard: aus, Beugung zaehlt")
    # ANNAHMEN Link-Budget (nicht gemessen):
    ap.add_argument("--tx-dbm", type=float, default=22.0)
    ap.add_argument("--gain-dbi", type=float, default=2.0, help="je Antenne")
    ap.add_argument("--sens-dbm", type=float, default=-130.0,
                    help="Empfindlichkeit; Schaetzung fuer SF8/62,5 kHz (-174+10log10(BW)+NF6-10)")
    ap.add_argument("--min-margin-db", type=float, default=10.0, help="Mindestreserve (Fading) zur Akzeptanz")
    a = ap.parse_args()

    dem, m = load_dem()
    reps = json.loads((HERE / "data/repeaters_filtered.json").read_text())
    # Duplikate (gleiche Position auf ~10 m) zusammenfassen, juengsten Eintrag behalten
    uniq = {}
    for n in sorted(reps, key=lambda x: x["_age_days"], reverse=True):
        uniq[(round(n["lat"], 4), round(n["lon"], 4))] = n
    nodes = [{"id": "START", "name": "START (Mainz)", "lat": a.start[0], "lon": a.start[1], "h": a.start[2]},
             {"id": "ZIEL", "name": "ZIEL (Faulbach)", "lat": a.end[0], "lon": a.end[1], "h": a.end[2]}]
    for i, n in enumerate(uniq.values()):
        nodes.append({"id": f"R{i}", "name": n["adv_name"], "lat": n["lat"], "lon": n["lon"],
                      "h": a.rep_h, "age": n["_age_days"]})
    print(f"{len(reps)} Repeater -> {len(nodes)-2} nach Duplikat-Zusammenfassung")
    print(f"ANNAHMEN: Repeater-Antenne {a.rep_h} m, START {a.start[2]} m, ZIEL {a.end[2]} m ueber DSM; "
          f"TX {a.tx_dbm} dBm, {a.gain_dbi} dBi je Antenne, Empfindlichkeit {a.sens_dbm} dBm, "
          f"Mindestreserve {a.min_margin_db} dB, Fresnel-Pflicht: {a.fresnel}, Beugung Knife-Edge, max {a.max_hop_km} km/Hop\n")

    # Kanten: nur Paare unter max-hop-km; Link gilt bei freier Fresnel-Zone UND genug Reserve
    adj = {n["id"]: {} for n in nodes}
    tried = ok = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            A, B = nodes[i], nodes[j]
            if hav_km((A["lat"], A["lon"]), (B["lat"], B["lon"])) > a.max_hop_km:
                continue
            r = analyze(dem, m, (A["lat"], A["lon"]), (B["lat"], B["lon"]), A["h"], B["h"])
            tried += 1
            r["reserve"] = a.tx_dbm + 2 * a.gain_dbi - r["fspl"] - r["diff"] - a.sens_dbm
            if (a.fresnel is None or r["min_ratio"] >= a.fresnel) and r["reserve"] >= a.min_margin_db:
                adj[A["id"]][B["id"]] = adj[B["id"]][A["id"]] = r
                ok += 1
    print(f"Paare geprueft: {tried}, davon nutzbar: {ok}")
    (HERE / "data/edges.json").write_text(json.dumps(
        {"nodes": nodes, "edges": [[u, v, e] for u in adj for v, e in adj[u].items() if u < v]}))

    byid = {n["id"]: n for n in nodes}
    # --- Kritische Stelle: Nachbarn von Faulbach
    print("\n== Faulbach (ZIEL): erreichbare Repeater (Fresnel frei, Reserve ok) ==")
    for v, e in sorted(adj["ZIEL"].items(), key=lambda kv: kv[1]["km"]):
        print(f"  {byid[v]['name'][:30]:30} {e['km']:5.1f} km  Fresnel-frei {e['min_ratio']:.2f}  Beugung {e['diff']:.0f} dB  Reserve {e['reserve']:.0f} dB")
    if not adj["ZIEL"]:
        # Naechster Kandidat, der knapp scheitert: zeigt, woran es liegt
        print("  KEINE. Naechste Repeater und woran die Strecke scheitert:")
        near = sorted(nodes[2:], key=lambda n: hav_km(FAULBACH, (n["lat"], n["lon"])))[:6]
        for n in near:
            r = analyze(dem, m, (a.end[0], a.end[1]), (n["lat"], n["lon"]), a.end[2], n["h"])
            print(f"  {n['name'][:30]:30} {r['km']:5.1f} km  Fresnel-frei {r['min_ratio']:.2f} "
                  f"(Engstelle bei {r['worst_km']:.1f} km, Freiraum {r['clear_m']:.0f} m) Beugung {r['diff']:.0f} dB")

    # --- Pfade ab Faulbach
    for title, key in (("kuerzester laengster Hop", lambda e: e["km"]),
                       ("beste schwaechste Reserve", lambda e: -e["reserve"])):
        best, prev = bottleneck(adj, "ZIEL", key)
        print(f"\n== Pfad Faulbach -> Mainz, Kriterium: {title} ==")
        if "START" not in best:
            comp = set(best)
            c2, _ = bottleneck(adj, "START", key)
            gap = min(((hav_km((byid[x]["lat"], byid[x]["lon"]), (byid[y]["lat"], byid[y]["lon"])), x, y)
                       for x in comp for y in c2), default=None)
            print(f"  KEIN Pfad. Von Faulbach erreichbar: {len(comp)-1} Repeater; von Mainz: {len(c2)-1}.")
            if gap:
                print(f"  Groesste Luecke (engste Stelle zwischen beiden Teilnetzen): {gap[0]:.1f} km "
                      f"zwischen '{byid[gap[1]]['name']}' und '{byid[gap[2]]['name']}'")
            continue
        p = path_to(prev, "ZIEL", "START")
        print(f"  Hops: {len(p)-1}")
        for u, v in zip(p, p[1:]):
            e = adj[u][v]
            print(f"  {byid[u]['name'][:26]:26} -> {byid[v]['name'][:26]:26} {e['km']:5.1f} km  "
                  f"Fresnel {e['min_ratio']:.2f}  Beugung {e['diff']:.0f} dB  Reserve {e['reserve']:.0f} dB")
        worst = min((adj[u][v] for u, v in zip(p, p[1:])), key=lambda e: e["reserve"])
        print(f"  Engpass: laengste Strecke {max(adj[u][v]['km'] for u,v in zip(p,p[1:])):.1f} km, "
              f"schwaechste Reserve {worst['reserve']:.0f} dB")


if __name__ == "__main__":
    main()
