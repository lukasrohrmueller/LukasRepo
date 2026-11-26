#!/usr/bin/env python3
"""Schritt 1: MeshCore-Knotenliste laden, cachen, filtern, Datenlage zeigen.

Quelle (aus dem Quellcode von map.meshcore.io ermittelt, src/config.js):
    https://map.meshcore.io/api/v1/nodes?binary=1&short=1   (msgpack, Kurzschluessel)
Der im alten README genannte Endpoint meshcore.dev/api/v1/nodes lieferte HTTP 500.

Typen laut Karten-Config: 1=Client, 2=Repeater, 3=Room Server, 4=Sensor.
Kurzschluessel: pk,t,n,la,id,ud,p,l,s (siehe KEYMAP); lat/lon behalten ihre Namen.

Nutzung:
    python3 step1_fetch.py                 # nutzt Cache, falls vorhanden
    python3 step1_fetch.py --refresh       # erzwingt neuen Abruf
    python3 step1_fetch.py --max-age-days 30 --bbox 49.6 50.1 8.2 9.6
"""
import argparse, json, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import msgpack
import requests

URL = "https://map.meshcore.io/api/v1/nodes?binary=1&short=1"
KEYMAP = {"pk": "public_key", "t": "type", "n": "adv_name", "la": "last_advert",
          "id": "inserted_date", "ud": "updated_date", "p": "params",
          "l": "link", "s": "source"}
TYPE_NAMES = {1: "Client", 2: "Repeater", 3: "Room Server", 4: "Sensor"}
HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "nodes_raw.json"            # Rohdaten-Cache
FILTERED = HERE / "data" / "repeaters_filtered.json"


def _jsonable(o):
    """msgpack liefert bytes/datetime -> JSON-tauglich machen."""
    if isinstance(o, bytes):
        return o.hex()
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, dict):
        return {(KEYMAP.get(k, k)): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    return o


def fetch(refresh):
    """Laedt Rohdaten (oder Cache). Cache vermeidet erneuten Abruf."""
    if RAW.exists() and not refresh:
        d = json.loads(RAW.read_text())
        print(f"[cache] {RAW} (abgerufen {d['fetched_at']})")
        return d
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    nodes = msgpack.unpackb(r.content, raw=False, timestamp=3)
    d = {"fetched_at": datetime.now(timezone.utc).isoformat(), "source": URL,
         "nodes": _jsonable(nodes)}
    RAW.parent.mkdir(exist_ok=True)
    RAW.write_text(json.dumps(d))
    print(f"[abruf] {len(d['nodes'])} Knoten gespeichert in {RAW}")
    return d


def parse_ts(v):
    """Zeitstempel robust lesen (ISO-String, ggf. Unix-Sekunden/ms)."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v / 1000 if v > 1e11 else v, timezone.utc)
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--max-age-days", type=float, default=30)
    ap.add_argument("--bbox", nargs=4, type=float, default=[49.6, 50.1, 8.2, 9.6],
                    metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    a = ap.parse_args()

    d = fetch(a.refresh)
    nodes = d["nodes"]
    print("Felder im 1. Eintrag:", sorted(nodes[0].keys()))
    print("Beispiel:", json.dumps(nodes[0])[:400])
    print("Typen gesamt:", {TYPE_NAMES.get(k, k): v
                            for k, v in Counter(n.get("type") for n in nodes).items()})

    la0, la1, lo0, lo1 = a.bbox
    now = datetime.now(timezone.utc)
    inbox = [n for n in nodes if n.get("lat") is not None and n.get("lon") is not None
             and la0 <= n["lat"] <= la1 and lo0 <= n["lon"] <= lo1]
    rep = [n for n in inbox if n.get("type") in (2, 3)]   # Repeater + Room Server

    # Alter: updated_date (nur bei Uploader-Knoten 'u...' aussagekraeftig), sonst last_advert
    for n in rep:
        ts = parse_ts(n.get("updated_date")) or parse_ts(n.get("last_advert"))
        n["_age_days"] = round((now - ts).total_seconds() / 86400, 1) if ts else None
        n["_manual"] = not str(n.get("source", "")).startswith("u")
    fresh = [n for n in rep if n["_age_days"] is not None and n["_age_days"] <= a.max_age_days]

    print(f"\nBBox {a.bbox}: {len(inbox)} Knoten, davon Repeater/Room Server: {len(rep)}")
    print(f"Typen in BBox: { {TYPE_NAMES.get(k, k): v for k, v in Counter(n.get('type') for n in inbox).items()} }")
    ages = sorted(n["_age_days"] for n in rep if n["_age_days"] is not None)
    print(f"Ohne Zeitstempel: {sum(n['_age_days'] is None for n in rep)}; "
          f"manuell eingetragen (Datum nicht aussagekraeftig): {sum(n['_manual'] for n in rep)}")
    for lim in (5, 10, 20, 30, 90, 365):
        print(f"  Alter <= {lim:>3} d: {sum(x <= lim for x in ages)}")
    print(f"=> nach Filter (<= {a.max_age_days} d): {len(fresh)} Repeater/Room Server")
    FILTERED.write_text(json.dumps(fresh, indent=1))
    print(f"Gespeichert: {FILTERED}")
    print("\nname | typ | lat | lon | alter_d | quelle | freq")
    for n in sorted(fresh, key=lambda x: x["lat"]):
        print(f"{n.get('adv_name','?')[:28]:28} | {TYPE_NAMES.get(n['type'])} | {n['lat']:.4f} | "
              f"{n['lon']:.4f} | {n['_age_days']} | {n.get('source')} | "
              f"{(n.get('params') or {}).get('freq')}")


if __name__ == "__main__":
    sys.exit(main())
