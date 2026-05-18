"""app.py — Web UI server for Lab 5 bilateral denoising filter.

Usage:
    python 05/app.py
Then open http://localhost:5173
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread

LAB5 = Path(__file__).resolve().parent
WEB = LAB5 / "web"
OUTPUTS = LAB5 / "outputs"
PORT = 5173


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # suppress request logs

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", (WEB / "index.html").read_bytes())
            return

        if path.startswith("/api/images/"):
            # Отдаём только PNG из каталога outputs, без доступа к произвольным путям.
            name = path[len("/api/images/"):]
            if "/" in name or "\\" in name or not name.endswith(".png"):
                self._send(403, "text/plain", b"Forbidden")
                return
            img_path = OUTPUTS / name
            if not img_path.exists():
                self._send(404, "text/plain", b"Not found")
                return
            self._send(200, "image/png", img_path.read_bytes())
            return

        if path == "/api/status":
            data = {
                "has_aov": (OUTPUTS / "aov.npz").exists(),
                "has_reference": (OUTPUTS / "reference.npz").exists(),
                "has_filtered": (OUTPUTS / "filtered_web.png").exists(),
                "has_debug": (OUTPUTS / "debug_objid.png").exists(),
            }
            self._send(200, "application/json", json.dumps(data).encode())
            return

        self._send(404, "text/plain", b"Not found")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/api/run-filter":
            self._send(200, "application/json", json.dumps(run_filter(body)).encode())
        elif self.path == "/api/render-noisy":
            self._send(200, "application/json", json.dumps(render_noisy()).encode())
        else:
            self._send(404, "text/plain", b"Not found")


def run_filter(params: dict) -> dict:
    out_path = OUTPUTS / "filtered_web.png"
    cmd = [
        sys.executable, str(LAB5 / "main.py"),
        "--aov", str(OUTPUTS / "aov.npz"),
        "--output", str(out_path),
        "--mode", params.get("mode", "mean"),
        "--sigma-s", str(params.get("sigma_s", 3.0)),
        "--sigma-n", str(params.get("sigma_n", 0.3)),
        "--sigma-z", str(params.get("sigma_z", 0.1)),
        "--sigma-c", str(params.get("sigma_c", 0.0)),
        "--radius", str(params.get("radius", 3)),
        "--dump-debug",
    ]
    if params.get("split_direct_indirect"):
        cmd.append("--split-direct-indirect")
    if params.get("energy_normalize") == "object":
        cmd.extend(["--energy-normalize", "object"])
    ref = OUTPUTS / "reference.npz"
    if ref.exists():
        cmd.extend(["--reference", str(ref)])

    result = subprocess.run(cmd, capture_output=True, text=True)

    # main.py сохраняет рядом текстовый отчёт; отсюда UI читает метрики.
    txt_path = OUTPUTS / "filtered_web.txt"
    metrics: dict = {}
    energy: list = []

    if txt_path.exists():
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if m := re.match(r"psnr_noisy_db:\s*([\d.]+)", line):
                metrics["psnr_noisy"] = float(m.group(1))
            elif m := re.match(r"psnr_filtered_db:\s*([\d.]+)", line):
                metrics["psnr_filtered"] = float(m.group(1))
            elif m := re.match(r"l1_noisy:\s*([\d.]+)", line):
                metrics["l1_noisy"] = float(m.group(1))
            elif m := re.match(r"l1_filtered:\s*([\d.]+)", line):
                metrics["l1_filtered"] = float(m.group(1))
            elif m := re.match(
                r"obj_id=(\d+)\s+energy_in=([\d.]+)\s+energy_out=([\d.]+)\s+ratio=([\d.]+)", line
            ):
                energy.append({
                    "obj_id": int(m.group(1)),
                    "energy_in": float(m.group(2)),
                    "energy_out": float(m.group(3)),
                    "ratio": float(m.group(4)),
                })

    if energy:
        metrics["energy"] = energy

    return {
        "success": result.returncode == 0,
        "metrics": metrics,
        "log": result.stdout,
        "error": result.stderr if result.returncode != 0 else "",
    }


def render_noisy() -> dict:
    aov_path = OUTPUTS / "aov.npz"
    aov_preview = OUTPUTS / "aov.png"
    noisy_preview = OUTPUTS / "noisy.png"
    cmd = [
        sys.executable, str(LAB5 / "render_aov.py"),
        "--width", "500", "--height", "500",
        "--samples", "4", "--max-depth", "5",
        "--output", str(aov_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    success = result.returncode == 0
    error = result.stderr if not success else ""

    if success:
        # UI использует noisy.png, поэтому после рендера синхронизируем
        # его со свежим preview, который создал render_aov.py.
        if aov_preview.exists():
            shutil.copyfile(aov_preview, noisy_preview)
        else:
            success = False
            error = f"missing preview after render: {aov_preview}"

    return {"success": success, "log": result.stdout, "error": error}


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    WEB.mkdir(exist_ok=True)

    # При первом запуске UI подготавливаем стартовый filtered_web.png,
    # если AOV уже существует на диске.
    if not (OUTPUTS / "filtered_web.png").exists() and (OUTPUTS / "aov.npz").exists():
        print("Generating initial filtered image...", flush=True)
        run_filter({})

    server = HTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"\n  Bilateral Denoiser UI -> {url}\n")
    Thread(target=lambda: (__import__("time").sleep(0.6), webbrowser.open(url)),
           daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
