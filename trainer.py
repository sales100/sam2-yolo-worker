"""Optomech OMS YOLO trainer, shared by the office PC agent and the Runpod Pod fallback.

The trainer never holds storage credentials. The OMS hands it short-lived presigned
GET URLs (manifest, images, parent .pt) and, at the end, presigned PUT URLs for the
outputs. Every call to the OMS is signed: X-OMS-Signature = hex HMAC-SHA256(key, raw body).
  * office PC: key = the PC's server.key, node = PC1 / PC2
  * Runpod Pod: key = a one-time per-run token (the OMS stores only its SHA-256), node = run:<id>

Modes
  python trainer.py --pod            Runpod Pod (env OMS_URL, OMS_RUN_ID, OMS_RUN_TOKEN)
  python trainer.py --spec job.json  standalone smoke test (URLs inside the spec, no OMS)
  import trainer; trainer.Trainer(...) office agent (agent.py)
"""
import csv
import glob
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

YOLOV5_DIR = os.environ.get("YOLOV5_DIR", "/opt/yolov5")
WORK_ROOT = os.environ.get("TRAIN_WORK", "/workspace/train" if os.path.isdir("/workspace") else os.path.join(os.getcwd(), "train"))
PROGRESS_EVERY = 20  # seconds
VERSION = "train-1"


class Cancelled(Exception):
    pass


class Oms:
    """Signed client for /api/mv/gpu-agent/<op>."""

    def __init__(self, base: str, node: str, key: str):
        self.base, self.node, self.key = base.rstrip("/"), node, key.encode()

    def post(self, op: str, body: dict, timeout=60) -> dict:
        body = dict(body, node=self.node, ts=int(time.time() * 1000))
        raw = json.dumps(body, separators=(",", ":")).encode()
        sig = hmac.new(self.key, raw, hashlib.sha256).hexdigest()
        req = urllib.request.Request(f"{self.base}/api/mv/gpu-agent/{op}", data=raw, method="POST",
                                     headers={"Content-Type": "application/json", "X-OMS-Signature": sig})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")


class Offline:
    """Standalone stand-in for the OMS (smoke tests): prints progress, uses pre-issued PUT URLs."""

    def __init__(self, spec):
        self.spec = spec

    def post(self, op, body, timeout=60):
        if op == "progress":
            print("PROGRESS", json.dumps({k: body.get(k) for k in ("epoch", "epochs", "metrics", "phase")}), flush=True)
            return {"cancel": False}
        if op == "uploads":
            return {"urls": {n: self.spec["uploads"][n] for n in body["names"] if n in self.spec.get("uploads", {})}}
        if op == "complete":
            print("COMPLETE", json.dumps({k: v for k, v in body.items() if k != "log_tail"})[:6000], flush=True)
            return {"ok": True}
        return {}


def get(url, dest, limit=2 * 1024 ** 3):
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        n = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            n += len(b)
            if n > limit:
                raise RuntimeError("download too large")
            f.write(b)
    os.replace(tmp, dest)


def put(url, path):
    with open(path, "rb") as f:
        data = f.read()
    req = urllib.request.Request(url, data=data, method="PUT", headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=900) as r:
        if r.status not in (200, 201, 204):
            raise RuntimeError(f"upload failed {r.status}")
    return len(data)


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:80] or "x"


class Trainer:
    def __init__(self, oms, spec: dict, log=print):
        self.oms, self.spec, self.log = oms, spec, log
        self.run_id = spec["run_id"]
        self.dir = os.path.join(WORK_ROOT, safe(self.run_id))
        self.proc = None
        self.cancel = False
        self.tail = []
        os.makedirs(self.dir, exist_ok=True)

    # ---------- helpers ----------
    def say(self, line):
        line = str(line).rstrip()
        self.tail.append(line)
        self.tail = self.tail[-200:]
        self.log(line)

    def progress(self, phase, epoch=None, epochs=None, metrics=None):
        try:
            r = self.oms.post("progress", {"run_id": self.run_id, "phase": phase, "epoch": epoch, "epochs": epochs,
                                           "metrics": metrics or {}, "log_tail": "\n".join(self.tail[-60:])[-8000:]})
            if r.get("cancel"):
                self.cancel = True
        except Exception as e:  # network blips must not kill a multi-hour run
            self.log(f"progress post failed: {e}")
        if self.cancel:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
            raise Cancelled()

    @property
    def family(self):
        return self.spec["family"]

    @property
    def task(self):
        return self.spec["task"]

    # ---------- dataset ----------
    def build_dataset(self):
        s = self.spec
        man_path = os.path.join(self.dir, "manifest.json")
        if not os.path.exists(man_path):
            get(s["manifest_url"], man_path, 512 * 1024 ** 2)
        man = json.load(open(man_path, encoding="utf-8"))
        if s.get("manifest_sha256"):
            h = hashlib.sha256(open(man_path, "rb").read()).hexdigest()
            if h != s["manifest_sha256"]:
                raise RuntimeError("manifest checksum mismatch")
        classes = sorted(man["classes"], key=lambda c: c["class_index"])
        names = [c["name"] for c in classes]
        cache = os.path.join(self.dir, "img")
        os.makedirs(cache, exist_ok=True)
        urls = s["images"]
        items = [i for i in man["items"] if i.get("file_id") in urls]
        self.say(f"dataset: {len(items)} images, {len(names)} classes")

        def fetch(i):
            ext = os.path.splitext(i.get("file_name") or "")[1].lower() or ".jpg"
            p = os.path.join(cache, i["file_id"] + ext)
            if not os.path.exists(p):
                get(urls[i["file_id"]], p, 200 * 1024 ** 2)
            return p

        with ThreadPoolExecutor(8) as ex:
            paths = list(ex.map(fetch, items))
        root = os.path.join(self.dir, "ds")
        shutil.rmtree(root, ignore_errors=True)
        splits = {"train": "train", "valid": "val", "test": "test"}
        for i, p in zip(items, paths):
            sp = splits.get(i.get("split"), "train")
            stem = i["file_id"]
            ext = os.path.splitext(p)[1]
            if self.task == "classify":
                ci = i.get("image_tag_class_index")
                if ci is None or not (0 <= ci < len(names)):
                    continue
                d = os.path.join(root, sp, safe(names[ci]))
                os.makedirs(d, exist_ok=True)
                shutil.copyfile(p, os.path.join(d, stem + ext))
            else:
                os.makedirs(os.path.join(root, "images", sp), exist_ok=True)
                os.makedirs(os.path.join(root, "labels", sp), exist_ok=True)
                shutil.copyfile(p, os.path.join(root, "images", sp, stem + ext))
                lines = []
                for sh in i.get("shapes") or []:
                    g = sh.get("geometry") or {}
                    if sh.get("shape_type", "box") != "box" or not all(k in g for k in ("x", "y", "w", "h")):
                        continue
                    x, y, w, h = (max(0.0, min(1.0, float(g[k]))) for k in ("x", "y", "w", "h"))
                    if w <= 0 or h <= 0:
                        continue
                    lines.append(f"{sh['class_index']} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
                with open(os.path.join(root, "labels", sp, stem + ".txt"), "w") as f:
                    f.write("\n".join(lines))
        if self.task == "classify":
            for sp in ("train", "val", "test"):
                for n in names:  # every split needs every class folder
                    os.makedirs(os.path.join(root, sp, safe(n)), exist_ok=True)
            self.data = root
        else:
            for sp in ("train", "val", "test"):
                os.makedirs(os.path.join(root, "images", sp), exist_ok=True)
            if not os.listdir(os.path.join(root, "images", "val")):  # tiny sets: validate on train
                shutil.rmtree(os.path.join(root, "images", "val"))
                shutil.copytree(os.path.join(root, "images", "train"), os.path.join(root, "images", "val"))
                shutil.copytree(os.path.join(root, "labels", "train"), os.path.join(root, "labels", "val"), dirs_exist_ok=True)
            self.data = os.path.join(root, "data.yaml")
            with open(self.data, "w", encoding="utf-8") as f:
                f.write(f"path: {json.dumps(root)}\ntrain: images/train\nval: images/val\ntest: images/test\n")
                f.write(f"nc: {len(names)}\nnames: {json.dumps(names)}\n")
        self.names = names
        pat = os.path.join(root, "test", "*", "*") if self.task == "classify" else os.path.join(root, "images", "test", "*")
        self.has_test = bool(glob.glob(pat))

    # ---------- training ----------
    def out_dir(self):
        return os.path.join(self.dir, "out", "run")

    def command(self, resume: bool):
        s, hyp = self.spec, self.spec.get("hyperparameters") or {}
        epochs, imgsz, batch = int(s["epochs"]), int(s["imgsz"]), int(hyp.get("batch", -1))
        weights = s.get("weights_local") or s["base_model"]  # parent .pt (fine-tune) or e.g. yolov5s.pt
        last = os.path.join(self.out_dir(), "weights", "last.pt")
        proj = os.path.join(self.dir, "out")
        if self.family == "yolov5":
            if self.task == "classify":
                return [sys.executable, "classify/train.py", "--model", weights, "--data", self.data, "--epochs", str(epochs),
                        "--img", str(imgsz), "--batch-size", str(batch if batch > 0 else 32), "--project", proj, "--name", "run",
                        "--exist-ok", "--workers", "2"], YOLOV5_DIR
            if resume and os.path.exists(last):
                return [sys.executable, "train.py", "--resume", last], YOLOV5_DIR
            cmd = [sys.executable, "train.py", "--data", self.data, "--weights", weights, "--epochs", str(epochs), "--imgsz", str(imgsz),
                   "--batch-size", str(batch), "--project", proj, "--name", "run", "--exist-ok", "--workers", "2",
                   "--patience", str(int(hyp.get("patience", 100)))]
            if hyp.get("freeze"):
                cmd += ["--freeze", str(int(hyp["freeze"]))]
            if hyp.get("optimizer"):
                cmd += ["--optimizer", str(hyp["optimizer"])]
            if hyp.get("cache"):
                cmd += ["--cache", "disk"]
            return cmd, YOLOV5_DIR
        args = dict(data=self.data, epochs=epochs, imgsz=imgsz, batch=batch, project=proj, name="run", exist_ok=True, workers=2,
                    patience=int(hyp.get("patience", 100)), plots=True)
        for k in ("lr0", "lrf", "momentum", "weight_decay", "optimizer", "freeze", "mosaic", "fliplr", "hsv_h", "hsv_s", "hsv_v", "degrees"):
            if k in hyp:
                args[k] = hyp[k]
        code = ("from ultralytics import YOLO; import json,sys; a=json.loads(sys.argv[2]); "
                "YOLO(sys.argv[1]).train(**a) if sys.argv[3]=='0' else YOLO(sys.argv[1]).train(resume=True)")
        if resume and os.path.exists(last):
            return [sys.executable, "-c", code, last, "{}", "1"], None
        return [sys.executable, "-c", code, weights, json.dumps(args), "0"], None

    def read_results(self):
        p = os.path.join(self.out_dir(), "results.csv")
        if not os.path.exists(p):
            return None, {}
        rows = list(csv.DictReader(open(p, encoding="utf-8", errors="replace")))
        if not rows:
            return None, {}
        r = {k.strip(): v.strip() for k, v in rows[-1].items() if k}
        out = {}
        for k, v in r.items():
            try:
                out[k] = round(float(v), 5)
            except ValueError:
                pass
        ep = int(out.get("epoch", len(rows) - 1)) + (0 if self.family == "yolov5" else 0)
        return ep, out

    def train(self, resume=False):
        cmd, cwd = self.command(resume)
        self.say("$ " + " ".join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd))[:400])
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", env=env)

        def pump():
            for line in self.proc.stdout:
                if line.strip() and "it/s]" not in line and "s/it]" not in line:
                    self.say(line)

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        epochs = int(self.spec["epochs"])
        while self.proc.poll() is None:
            time.sleep(PROGRESS_EVERY)
            ep, m = self.read_results()
            self.progress("training", None if ep is None else ep + 1, epochs, m)
        t.join(5)
        if self.proc.returncode != 0:
            raise RuntimeError(f"training exited with code {self.proc.returncode}")
        ep, m = self.read_results()
        self.final = m
        self.progress("training", None if ep is None else ep + 1, epochs, m)

    # ---------- evaluation ----------
    V5_ROW = re.compile(r"^\s*(\S.*?)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")

    def evaluate(self, weights, split):
        """Per-class metrics. Returns {"all": {...}, "classes": [{class_index, name, precision, recall, map50, map50_95}]}"""
        imgsz = str(self.spec["imgsz"])
        if self.task == "classify":
            if self.family == "yolov5":
                r = subprocess.run([sys.executable, "classify/val.py", "--weights", weights, "--data", self.data, "--img", imgsz],
                                   cwd=YOLOV5_DIR, capture_output=True, text=True, errors="replace")
                m = re.findall(r"\s+all\s+\d+\s+([\d.]+)\s+([\d.]+)", r.stdout + r.stderr)
                return {"all": {"top1": float(m[-1][0]), "top5": float(m[-1][1])} if m else {}, "classes": []}
            out = subprocess.run([sys.executable, "-c", "from ultralytics import YOLO;import sys,json;m=YOLO(sys.argv[1]).val(data=sys.argv[2],split=sys.argv[3],imgsz=int(sys.argv[4]),plots=False);"
                                  "print('EVALJSON'+json.dumps({'top1':float(m.top1),'top5':float(m.top5)}))", weights, self.data, split, imgsz],
                                 capture_output=True, text=True, errors="replace")
            j = re.findall(r"EVALJSON(.*)", out.stdout)
            return {"all": json.loads(j[-1]) if j else {}, "classes": []}
        if self.family == "yolov5":
            r = subprocess.run([sys.executable, "val.py", "--data", self.data, "--weights", weights, "--task", split, "--imgsz", imgsz,
                                "--verbose", "--project", os.path.join(self.dir, "val"), "--name", split, "--exist-ok"],
                               cwd=YOLOV5_DIR, capture_output=True, text=True, errors="replace")
            res = {"all": {}, "classes": []}
            by_name = {n: i for i, n in enumerate(self.names)}
            for line in (r.stdout + "\n" + r.stderr).splitlines():
                mm = self.V5_ROW.match(line)
                if not mm:
                    continue
                name = mm.group(1).strip()
                row = {"images": int(mm.group(2)), "instances": int(mm.group(3)), "precision": float(mm.group(4)),
                       "recall": float(mm.group(5)), "map50": float(mm.group(6)), "map50_95": float(mm.group(7))}
                if name == "all":
                    res["all"] = row
                elif name in by_name:
                    res["classes"].append(dict(row, class_index=by_name[name], name=name))
            return res
        code = ("from ultralytics import YOLO;import sys,json;m=YOLO(sys.argv[1]).val(data=sys.argv[2],split=sys.argv[3],imgsz=int(sys.argv[4]),plots=False);b=m.box;"
                "print('EVALJSON'+json.dumps({'all':{'precision':float(b.mp),'recall':float(b.mr),'map50':float(b.map50),'map50_95':float(b.map)},"
                "'classes':[{'class_index':int(c),'name':m.names[int(c)] if hasattr(m,'names') else str(c),'precision':float(b.p[i]),'recall':float(b.r[i]),'map50':float(b.ap50[i]),'map50_95':float(b.ap[i])} for i,c in enumerate(b.ap_class_index)]}))")
        out = subprocess.run([sys.executable, "-c", code, weights, self.data, split, imgsz], capture_output=True, text=True, errors="replace")
        j = re.findall(r"EVALJSON(.*)", out.stdout)
        if not j:
            self.say((out.stdout + out.stderr)[-1500:])
        return json.loads(j[-1]) if j else {"all": {}, "classes": []}

    def regression(self, new_eval, parent_pt, split):
        """Compare the new model with its parent on the parent's (old) classes."""
        old = self.evaluate(parent_pt, split)
        n_old = int(self.spec.get("parent_class_count") or len(self.names))
        tol = float((self.spec.get("hyperparameters") or {}).get("regression_tolerance", 0.03))
        newc = {c["class_index"]: c for c in new_eval.get("classes", [])}
        rows, worst = [], 0.0
        for c in old.get("classes", []):
            if c["class_index"] >= n_old:
                continue
            nc = newc.get(c["class_index"], {})
            d = round(nc.get("map50", 0.0) - c["map50"], 4)
            worst = min(worst, d)
            rows.append({"class_index": c["class_index"], "name": c["name"], "parent_map50": c["map50"], "new_map50": nc.get("map50"), "delta": d})
        return {"split": split, "tolerance": tol, "worst_delta": worst, "passed": worst >= -tol, "classes": rows, "parent_all": old.get("all")}

    # ---------- main ----------
    def run(self, resume=False):
        s = self.spec
        try:
            self.progress("preparing")
            self.build_dataset()
            if s.get("parent_pt_url"):
                pp = os.path.join(self.dir, "parent.pt")
                if not os.path.exists(pp):
                    get(s["parent_pt_url"], pp)
                s["weights_local"] = pp
            self.progress("training", 0, int(s["epochs"]))
            self.train(resume=resume)
            wdir = os.path.join(self.out_dir(), "weights")
            best = os.path.join(wdir, "best.pt")
            if not os.path.exists(best):
                best = os.path.join(wdir, "last.pt")
            self.progress("evaluating")
            split = "test" if self.has_test else "val"
            ev = self.evaluate(best, split)
            ev["split"] = split
            reg = None
            if s.get("weights_local") and self.task == "detect":
                reg = self.regression(ev, s["weights_local"], split)
            self.progress("exporting")
            onnx = None
            try:
                from handler import export_file
                onnx, _, _, shape = export_file(best, self.family, int(s["imgsz"]))
            except Exception as e:
                self.say(f"ONNX export failed (the .pt is still saved): {e}")
            files = {"best.pt": best, "last.pt": os.path.join(wdir, "last.pt"), "results.csv": os.path.join(self.out_dir(), "results.csv")}
            if onnx:
                files["best.onnx"] = onnx
            plots = os.path.join(self.dir, "plots")
            shutil.rmtree(plots, ignore_errors=True)
            os.makedirs(plots)
            for p in glob.glob(os.path.join(self.out_dir(), "*.png")) + glob.glob(os.path.join(self.out_dir(), "*.jpg")):
                shutil.copy(p, plots)
            if os.listdir(plots):
                files["plots.zip"] = shutil.make_archive(os.path.join(self.dir, "plots"), "zip", plots)
            files = {k: v for k, v in files.items() if os.path.exists(v)}
            self.progress("uploading")
            urls = self.oms.post("uploads", {"run_id": self.run_id, "names": list(files),
                                             "sizes": {k: os.path.getsize(v) for k, v in files.items()}}).get("urls", {})
            uploaded = {}
            for n, p in files.items():
                if n in urls:
                    uploaded[n] = {"bytes": put(urls[n], p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
            self.oms.post("complete", {"run_id": self.run_id, "status": "succeeded", "metrics": self.final, "eval": ev,
                                       "regression": reg, "files": uploaded, "names": self.names,
                                       "log_tail": "\n".join(self.tail)[-18000:]}, timeout=180)
            shutil.rmtree(self.dir, ignore_errors=True)
            return "succeeded"
        except Cancelled:
            self.oms.post("complete", {"run_id": self.run_id, "status": "cancelled", "log_tail": "\n".join(self.tail)[-18000:]})
            shutil.rmtree(self.dir, ignore_errors=True)
            return "cancelled"
        except Exception as e:
            self.say(f"ERROR: {e}")
            try:
                self.oms.post("complete", {"run_id": self.run_id, "status": "failed", "error": str(e)[:1000],
                                           "log_tail": "\n".join(self.tail)[-18000:]})
            except Exception:
                pass
            return "failed"


def run_standalone(spec: dict):
    t = Trainer(Offline(spec), spec)
    return {"status": t.run()}


def main():
    if "--spec" in sys.argv:
        spec = json.load(open(sys.argv[sys.argv.index("--spec") + 1]))
        print(run_standalone(spec))
        return
    if "--pod" in sys.argv:
        oms = Oms(os.environ.get("OMS_URL", ""), "run:" + os.environ.get("OMS_RUN_ID", ""), os.environ.get("OMS_RUN_TOKEN", ""))
        if os.environ.get("OMS_SPEC_URL"):  # smoke test: spec fetched from a presigned URL, nothing reported to the OMS
            with urllib.request.urlopen(os.environ["OMS_SPEC_URL"], timeout=60) as r:
                spec = json.loads(r.read())
            print(run_standalone(spec), flush=True)
        else:
            spec = oms.post("job", {"run_id": os.environ["OMS_RUN_ID"]}).get("job")
            if spec:
                Trainer(oms, spec).run(resume=True)
        while True:  # the OMS terminates the Pod after "complete"; never let Runpod restart the job
            time.sleep(3600)


if __name__ == "__main__":
    main()
