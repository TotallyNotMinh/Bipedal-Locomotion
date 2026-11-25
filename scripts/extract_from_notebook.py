"""Lấy file zip (checkpoint) được nhúng trong output của notebook ra máy cá nhân.

Dùng khi chạy notebook trên VS Code nối với kernel Kaggle: cell "Tải về máy" nhúng
file zip (base64) vào output, VS Code lưu output đó vào file .ipynb trên máy bạn.
Script này đọc .ipynb và giải mã ra file zip.

Chạy TRÊN MÁY CÁ NHÂN (sau khi đã Ctrl+S notebook):
    python scripts/extract_from_notebook.py kaggle_remote.ipynb --out downloads/
"""

import argparse
import base64
import json
import re
from pathlib import Path

PATTERN = re.compile(r'download="([^"]+)"\s+href="data:application/zip;base64,([A-Za-z0-9+/=]+)"')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("notebook", nargs="?", default="kaggle_remote.ipynb")
    p.add_argument("--out", default="downloads")
    args = p.parse_args()

    nb = json.loads(Path(args.notebook).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    found = 0
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            html = output.get("data", {}).get("text/html")
            if not html:
                continue
            html = "".join(html) if isinstance(html, list) else html
            for name, b64 in PATTERN.findall(html):
                target = out_dir / Path(name).name
                target.write_bytes(base64.b64decode(b64))
                print(f"[✓] {target} ({target.stat().st_size / 1e6:.1f} MB)")
                found += 1

    if not found:
        print("Không tìm thấy file nhúng. Đã chạy cell 'Tải về máy' và Ctrl+S notebook chưa?")


if __name__ == "__main__":
    main()
