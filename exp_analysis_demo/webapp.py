#!/usr/bin/env python3
"""Local web demo for the beach bird project (no extra dependencies).

    python exp_analysis_demo/webapp.py            # then open http://127.0.0.1:8000
    python exp_analysis_demo/webapp.py --open     # opens the browser for you

Serves one page that shows, for any test image, the same four panels the video
demo uses (ground truth / method 1 / method 2 / method 3), with a confidence
slider, a search box over the 240 test images, arrow-key navigation, and a
drag-and-drop box for your own picture.

Everything is stdlib http.server + Pillow; the models are the ones in
local checkpoints (see demo.py).  First request after start loads and warms up the three
detectors, which takes ~15 s.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "exp_analysis_demo"))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import demo  # noqa: E402  (reuses load_models / draw_panel / frame_for)

IMG_DIR = os.path.join("common", "dataset", "images", "test")
LBL_DIR = os.path.join("common", "dataset", "labels", "test")
MANIFEST = os.path.join("common", "dataset", "dataset_manifest.csv")

STATE: dict = {}
LOCK = threading.Lock()
CACHE: dict = {}
CACHE_MAX = 48


class Args:
    yolo_weights = "method2/results/train_runs/single_stage_baseline/weights/best.pt"
    m3_weights = "method3/output/default_params.pth"
    m3_config = "method3/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml"
    conf = 0.25


def _check_data() -> None:
    """Explain a missing dataset instead of a FileNotFoundError traceback."""
    missing = [d for d in (IMG_DIR, LBL_DIR) if not os.path.isdir(d)]
    if not missing:
        return
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("")
    print("[x] no test images found - this folder has no dataset/")
    for d in missing:
        print("    missing: %s" % os.path.abspath(d))
    print("")
    print("    The repository does not ship images (dataset/ is git-ignored).")
    print("    Two ways to run the demo:")
    print("      1) start it from the demo bundle, which contains six test frames")
    print("      2) point it at an existing dataset:")
    print("         python exp_analysis_demo/webapp.py --data C:/path/to/dataset_root")
    print("")
    print("      dataset_root must contain images/test/*.jpg and labels/test/*.txt")
    print("")
    raise SystemExit(2)


def boot() -> None:
    _check_data()
    from exp_analysis_demo.eval_all import load_gt_yolo, load_manifest

    manifest = load_manifest(MANIFEST) if os.path.exists(MANIFEST) else {}
    gt_all = load_gt_yolo(LBL_DIR, MANIFEST if os.path.exists(MANIFEST) else None, IMG_DIR)
    names = sorted(os.listdir(IMG_DIR))
    names = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))]

    runners = demo.load_models(Args)
    warm = Image.open(os.path.join(IMG_DIR, names[0])).convert("RGB")
    warm_bgr = np.array(warm)[:, :, ::-1].copy()
    for key, fn in runners.items():
        fn(warm_bgr if key == "method1_ml" else warm)
    print("[web] ready: %d images, 3 detectors warm" % len(names), flush=True)

    # load_manifest keeps w/h/count/group/split only; the density label the UI
    # shows comes straight from the CSV
    bins = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                bins[os.path.basename(row["image_path"])] = {
                    "bin": row.get("density_bin", ""),
                    "count": int(row.get("object_count") or 0),
                }
    STATE.update({
        "names": names,
        "gt": gt_all,
        "manifest": manifest,
        "bins": bins,
        "runners": runners,
        "args": Args,
    })


def panel_jpeg(name: str, conf: float) -> bytes:
    key = (name, round(conf, 3))
    if key in CACHE:
        return CACHE[key]
    with LOCK:
        args = STATE["args"]
        args.conf = conf
        frame = demo.frame_for(name, IMG_DIR, STATE["gt"], STATE["runners"], conf, hint=False)
        buf = io.BytesIO()
        frame.convert("RGB").save(buf, "JPEG", quality=88)
        data = buf.getvalue()
    if len(CACHE) > CACHE_MAX:
        CACHE.pop(next(iter(CACHE)))
    CACHE[key] = data
    return data


def upload_jpeg(raw: bytes, conf: float) -> bytes:
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    if max(pil.size) > 1280:
        pil.thumbnail((1280, 1280), Image.LANCZOS)
    bgr = np.array(pil)[:, :, ::-1].copy()
    with LOCK:
        runs = [(k, STATE["runners"][k]) for k in ("method1_ml", "method2_yolo", "method3_rtdetr")]
        args = STATE["args"]
        args.conf = conf
        results = []
        for key, fn in runs:
            pred, scores, ms = fn(bgr if key == "method1_ml" else pil)
            results.append((key, pred, scores, ms))

    w = pil.size[0]
    canvas = Image.new("RGB", (3 * demo.PANEL + 4 * demo.MARGIN, demo.PANEL + demo.CAPTION + demo.HEADER),
                       (18, 18, 20))
    d = ImageDraw.Draw(canvas)
    d.text((demo.MARGIN, 18), "Your image - %d x %d px" % pil.size, font=demo.F_BIG, fill=(255, 255, 255))
    d.text((demo.MARGIN, 52), "no ground truth in an uploaded picture, so only predictions are drawn",
           font=demo.F_SMALL, fill=(170, 170, 175))
    sx = demo.PANEL / float(w) if w else 1.0
    pretty = {"method1_ml": "Method 1 - classic CV", "method2_yolo": "Method 2 - YOLOv8s",
              "method3_rtdetr": "Method 3 - RT-DETRv2"}
    thumbs = {"method1_ml": "creates boxes from contrast", "method2_yolo": "one box per confident bird",
              "method3_rdetr": "one query per bird, no NMS"}
    for i, (key, pred, scores, ms) in enumerate(results):
        base = Image.new("RGB", (demo.PANEL, demo.PANEL))
        base.paste(pil.resize((demo.PANEL, demo.PANEL), Image.BICUBIC), (0, 0))
        n = len(pred)
        cap = "%d boxes" % n  # no per-call timing: only the table is authoritative
        canvas.paste(demo.draw_panel(base, np.zeros((0, 4)), pred * sx, np.zeros((n,), bool), 0,
                                     pretty[key], cap, thumbs.get(key, "")),
                     (demo.MARGIN + i * (demo.PANEL + demo.MARGIN), demo.HEADER))
    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, "JPEG", quality=88)
    return buf.getvalue()


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>Beach Bird Detection - demo</title>
<meta name="viewport" content="width=1500">
<style>
:root{--bg:#12131a;--panel:#1b1d26;--line:#2b2e3b;--fg:#e8e9ee;--dim:#9aa0ad;--accent:#4ea1ff;--good:#2fd07a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 "Segoe UI",system-ui,sans-serif}
header{display:flex;align-items:center;gap:18px;padding:14px 20px;border-bottom:1px solid var(--line);position:sticky;top:0;background:#14161e;z-index:5}
h1{font-size:17px;margin:0;font-weight:600;letter-spacing:.2px}
h1 small{display:block;font-size:12px;color:var(--dim);font-weight:400;margin-top:2px}
.ctrl{display:flex;align-items:center;gap:10px;margin-left:auto;color:var(--dim);font-size:13px}
input[type=range]{width:160px;accent-color:var(--accent)}
main{display:grid;grid-template-columns:268px 1fr;gap:0;height:calc(100vh - 64px)}
body.wide main{grid-template-columns:0 1fr}
body.wide aside{display:none}
button{background:#1e2130;color:#dfe2ea;border:1px solid var(--line);border-radius:7px;padding:5px 10px;font-size:13px;cursor:pointer}
button:hover{border-color:var(--accent);color:#fff}
aside{border-right:1px solid var(--line);overflow:auto;padding:12px}
#search{width:100%;padding:8px 10px;border-radius:8px;border:1px solid var(--line);background:#0f1016;color:var(--fg);margin-bottom:10px}
.thumb{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:7px 9px;border-radius:8px;cursor:pointer;color:#c9ccd6;font-size:13px}
.thumb:hover{background:#1e2130}
.thumb.sel{background:#243250;color:#fff;box-shadow:inset 2px 0 0 var(--accent)}
.thumb .n{color:var(--dim);font-variant-numeric:tabular-nums}
section{padding:16px 20px;overflow:auto}
#stage{width:100%;border-radius:10px;border:1px solid var(--line);display:block;background:#000;min-height:200px}
#meta{color:var(--dim);font-size:13px;margin:8px 0 14px}
table{border-collapse:collapse;font-size:13px;width:100%;max-width:820px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:500}
td.ap{color:var(--good);font-variant-numeric:tabular-nums}
#drop{margin-top:18px;border:1.5px dashed #39405a;border-radius:10px;padding:18px;text-align:center;color:var(--dim);font-size:13px}
#drop.hot{border-color:var(--accent);color:var(--accent)}
.kbd{display:inline-block;border:1px solid var(--line);border-radius:5px;padding:0 5px;font-size:12px;color:#c9ccd6}
</style></head>
<body>
<header>
  <h1>Beach Bird Detection - one test image, four views
    <small>COMP9517 26T3 &middot; M1 classic CV &middot; M2 YOLOv8s &middot; M3 RT-DETRv2</small></h1>
  <div class="ctrl">
    <button id="wide" title="hide the image list">Wide view</button>
    <span>confidence</span><input id="conf" type="range" min="5" max="60" step="5" value="25">
    <span id="confv">0.25</span>
    <span class="kbd">&larr;</span><span class="kbd">&rarr;</span>
  </div>
</header>
<main>
  <aside>
    <input id="search" placeholder="search by file name or bird count, e.g. 1790 or 66">
    <div id="list"></div>
  </aside>
  <section>
    <a id="zoom" target="_blank"><img id="stage" alt="comparison"></a>
    <div id="meta">loading…</div>
    <table>
      <thead><tr><th>Method</th><th>mAP@0.5</th><th>Training</th><th>Inference</th></tr></thead>
      <tbody id="stats"></tbody>
    </table>
    <div id="drop">Drop any image here to run all three models on it (local only, nothing is uploaded)</div>
  </section>
</main>
<script>
let items = [], cur = -1, conf = 0.25;
const listEl = document.getElementById('list'), stage = document.getElementById('stage');
const meta = document.getElementById('meta'), drop = document.getElementById('drop');

function render() {
  const q = document.getElementById('search').value.trim().toLowerCase();
  listEl.innerHTML = '';
  items.forEach((it, i) => {
    if (q && !(it.name.toLowerCase().includes(q) || String(it.count) === q)) return;
    const d = document.createElement('div');
    d.className = 'thumb' + (i === cur ? ' sel' : '');
    d.innerHTML = '<span>' + it.name.replace('.jpg','') + '</span><span class="n">' + it.count + ' birds</span>';
    d.onclick = () => show(i);
    listEl.appendChild(d);
  });
}
function show(i) {
  cur = i; const it = items[i];
  const u = '/api/panel?name=' + encodeURIComponent(it.name) + '&conf=' + conf;
  stage.src = u; document.getElementById('zoom').href = u;
  meta.textContent = it.name + ' &middot; ' + it.count + ' birds &middot; ' + (it.bin ? 'density ' + it.bin : '');
  render();
  history.replaceState(null, '', '#' + it.name);
}
document.getElementById('conf').oninput = e => {
  conf = e.target.value / 100;
  document.getElementById('confv').textContent = conf.toFixed(2);
  if (cur >= 0) stage.src = '/api/panel?name=' + encodeURIComponent(items[cur].name) + '&conf=' + conf;
};
document.getElementById('search').oninput = render;
document.getElementById('wide').onclick = () => document.body.classList.toggle('wide');
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' && cur < items.length - 1) { show(cur + 1); e.preventDefault(); }
  if (e.key === 'ArrowLeft' && cur > 0) { show(cur - 1); e.preventDefault(); }
});
['dragover','dragenter'].forEach(ev => drop.addEventListener(ev, e => {e.preventDefault(); drop.classList.add('hot');}));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => {e.preventDefault(); drop.classList.remove('hot');}));
drop.addEventListener('drop', async e => {
  const f = e.dataTransfer.files[0]; if (!f) return;
  meta.textContent = 'running ' + f.name + ' …';
  const r = await fetch('/api/upload?conf=' + conf, {method:'POST', body: f});
  stage.src = URL.createObjectURL(await r.blob());
  meta.textContent = f.name + ' &middot; uploaded image (no ground truth, predictions only)';
  cur = -1; render();
});
fetch('/api/images').then(r => r.json()).then(j => {
  items = j.images; render();
  const h = decodeURIComponent(location.hash.slice(1));
  const i = items.findIndex(x => x.name === h);
  show(i >= 0 ? i : 0);
});
fetch('/api/stats').then(r => r.json()).then(j => {
  const pretty = {method1_ml:'Method 1 - classic CV', method2_yolo:'Method 2 - YOLOv8s', method3_rtdetr:'Method 3 - RT-DETRv2'};
  document.getElementById('stats').innerHTML = j.rows.map(r =>
    '<tr><td>' + (pretty[r.method] || r.method) + '</td><td class="ap">' + r.ap50 +
    '</td><td>' + r.train + '</td><td>' + r.ms + '</td></tr>').join('');
});
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "BeachBirdDemo/1.0"

    def log_message(self, fmt, *args):  # quieter console
        if "/api/panel" not in (args[0] if args else ""):
            sys.stderr.write("[web] %s\n" % (fmt % args))

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path == "/":
            return self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))

        if u.path == "/api/images":
            bins = STATE["bins"]
            out = []
            for n in STATE["names"]:
                info = bins.get(n, {})
                out.append({"name": n, "count": int(info.get("count") or 0),
                            "bin": info.get("bin", "")})
            return self._send(200, "application/json",
                              json.dumps({"images": out}).encode("utf-8"))

        if u.path == "/api/panel":
            name = (q.get("name") or [""])[0]
            conf = float((q.get("conf") or ["0.25"])[0])
            if name not in STATE["names"]:
                return self._send(404, "text/plain", b"unknown image")
            try:
                data = panel_jpeg(name, conf)
            except Exception as exc:  # noqa: BLE001
                return self._send(500, "text/plain", ("render failed: %s" % exc).encode())
            return self._send(200, "image/jpeg", data)

        if u.path == "/api/stats":
            rows = []
            path = os.path.join(REPO, "results", "eval", "summary.csv")
            if os.path.exists(path):
                with open(path, newline="", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        if r["method"] not in ("method1_ml", "method2_yolo", "method3_rtdetr"):
                            continue
                        ap = float(r["ap50"]) if r.get("ap50") else 0.0
                        tt = r.get("train_time_s") or ""
                        if tt not in ("", "0", "0.0"):
                            tt = "%.1f min" % (float(tt) / 60)
                        elif r["method"] == "method1_ml":
                            tt = "none (no learning)"
                        else:
                            tt = "not measured"
                        ms = r.get("inference_ms_per_image") or ""
                        rows.append({"method": r["method"], "ap50": "%.4f" % ap,
                                     "train": tt, "ms": ("%.1f ms" % float(ms)) if ms else "n/a"})
            return self._send(200, "application/json", json.dumps({"rows": rows}).encode("utf-8"))

        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path != "/api/upload":
            return self._send(404, "text/plain", b"not found")
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return self._send(400, "text/plain", b"empty body")
        raw = self.rfile.read(n)
        conf = float((q.get("conf") or ["0.25"])[0])
        try:
            data = upload_jpeg(raw, conf)
        except Exception as exc:  # noqa: BLE001
            return self._send(500, "text/plain", ("render failed: %s" % exc).encode())
        return self._send(200, "image/jpeg", data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="open the browser once ready")
    ap.add_argument("--data", help="dataset root holding images/test and labels/test")
    args = ap.parse_args(argv)

    if args.data:
        global IMG_DIR, LBL_DIR, MANIFEST
        IMG_DIR = os.path.join(args.data, "images", "test")
        LBL_DIR = os.path.join(args.data, "labels", "test")
        MANIFEST = os.path.join(args.data, "dataset_manifest.csv")

    t0 = time.time()
    print("[web] booting detectors ...", flush=True)
    boot()
    print("[web] boot took %.1f s" % (time.time() - t0), flush=True)

    # Windows lets a second process bind the same port unless this is off,
    # which makes a stale server answer while you think you restarted it
    ThreadingHTTPServer.allow_reuse_address = False
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = "http://%s:%d/" % (args.host, args.port)
    print("[web] serving %s  (Ctrl+C to stop)" % url, flush=True)
    if args.open:
        import threading as th
        import webbrowser
        th.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
