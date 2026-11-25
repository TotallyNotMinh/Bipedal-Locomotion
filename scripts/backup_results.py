"""Sao lưu / khôi phục kết quả train lên một Kaggle Dataset riêng tư.

Mục đích: khi phiên Kaggle bị ngắt thì /kaggle/working bị xóa. Script này nén
thư mục kết quả (metrics, config, checkpoints) rồi đẩy lên một Kaggle Dataset
private, để phiên sau tải về và train tiếp (restore_checkpoint_path).

Cần KAGGLE_USERNAME, KAGGLE_KEY (kaggle.com → Settings → API → Create New Token).
Lấy từ Kaggle Secrets nếu có; nếu không (vd. chạy qua VS Code) thì hỏi nhập tay 1 lần.

Dùng trong notebook:
    from scripts.backup_results import start_auto_backup, backup_now
    start_auto_backup(interval_min=15)   # chạy nền, tự backup mỗi 15 phút
    backup_now()                         # backup ngay lập tức

Dùng từ dòng lệnh:
    python scripts/backup_results.py backup
    python scripts/backup_results.py restore --dest results/flat
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_RESULTS = "results/flat"
DEFAULT_SLUG = "g1-m2-results"
ARCHIVE_NAME = "results_flat.zip"
STAGE_DIR = Path("/kaggle/tmp/backup_stage") if Path("/kaggle").exists() else Path("/tmp/backup_stage")

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def _load_kaggle_creds() -> str:
    """Đưa KAGGLE_USERNAME/KAGGLE_KEY vào env (từ Kaggle Secrets nếu cần). Trả về username."""
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        try:
            from kaggle_secrets import UserSecretsClient  # Kaggle Secrets (nếu đã attach)
            client = UserSecretsClient()
            os.environ["KAGGLE_USERNAME"] = client.get_secret("KAGGLE_USERNAME")
            os.environ["KAGGLE_KEY"] = client.get_secret("KAGGLE_KEY")
        except Exception:
            # Chạy qua VS Code: nhập tay 1 lần, chỉ giữ trong RAM của kernel,
            # KHÔNG ghi vào notebook / git.
            import getpass
            print("[backup] Không đọc được Kaggle Secrets → nhập tay (lấy trong kaggle.json).")
            os.environ["KAGGLE_USERNAME"] = input("KAGGLE_USERNAME: ").strip()
            os.environ["KAGGLE_KEY"] = getpass.getpass("KAGGLE_KEY: ").strip()
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        raise RuntimeError("Thiếu KAGGLE_USERNAME / KAGGLE_KEY.")
    return os.environ["KAGGLE_USERNAME"]


def _kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    if shutil.which("kaggle") is None:
        subprocess.check_call(["pip", "install", "-q", "kaggle"])
    res = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{res.stdout}\n{res.stderr}")
    return res


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

def _should_skip(path: Path) -> bool:
    # Bỏ checkpoint đang ghi dở (orbax ghi vào thư mục tmp rồi mới rename).
    return any("tmp" in part.lower() for part in path.parts)


def make_archive(results_dir: str | Path, out_zip: str | Path) -> int:
    """Nén results_dir vào out_zip (đường dẫn tương đối giữ nguyên). Trả về số file."""
    results_dir = Path(results_dir)
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = out_zip.with_suffix(".zip.part")
    n = 0
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(results_dir.rglob("*")):
            if f.is_file() and not _should_skip(f.relative_to(results_dir)):
                zf.write(f, arcname=str(f.relative_to(results_dir.parent)))
                n += 1
    tmp_zip.replace(out_zip)
    return n


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

def backup_now(results_dir: str = DEFAULT_RESULTS, slug: str = DEFAULT_SLUG,
               message: str | None = None) -> bool:
    """Nén kết quả và đẩy lên Kaggle Dataset private (tạo mới nếu chưa có)."""
    results_dir_p = Path(results_dir)
    if not results_dir_p.exists() or not any(results_dir_p.rglob("*")):
        print(f"[backup] Chưa có gì trong {results_dir}, bỏ qua.")
        return False

    with _lock:
        user = _load_kaggle_creds()
        dataset_id = f"{user}/{slug}"
        if STAGE_DIR.exists():
            shutil.rmtree(STAGE_DIR)
        STAGE_DIR.mkdir(parents=True)

        n = make_archive(results_dir_p, STAGE_DIR / ARCHIVE_NAME)
        size_mb = (STAGE_DIR / ARCHIVE_NAME).stat().st_size / 1e6
        with open(STAGE_DIR / "dataset-metadata.json", "w") as f:
            json.dump({"title": slug, "id": dataset_id,
                       "licenses": [{"name": "CC0-1.0"}]}, f)

        msg = message or f"auto backup {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
        # Thử tạo version mới; nếu dataset chưa tồn tại thì tạo mới (mặc định private).
        res = _kaggle("datasets", "version", "-p", str(STAGE_DIR), "-m", msg, check=False)
        if res.returncode != 0:
            res2 = _kaggle("datasets", "create", "-p", str(STAGE_DIR), check=False)
            if res2.returncode != 0:
                raise RuntimeError(
                    f"Upload thất bại.\n[version] {res.stdout}{res.stderr}\n"
                    f"[create] {res2.stdout}{res2.stderr}"
                )
        print(f"[backup] ✓ {n} file ({size_mb:.1f} MB) → kaggle.com/datasets/{dataset_id}")
        return True


def start_auto_backup(interval_min: float = 15, results_dir: str = DEFAULT_RESULTS,
                      slug: str = DEFAULT_SLUG) -> threading.Thread:
    """Chạy backup_now() định kỳ trong một thread nền (daemon)."""
    _load_kaggle_creds()  # báo lỗi ngay nếu thiếu secret, thay vì lỗi im lặng trong thread

    def _loop():
        while True:
            time.sleep(interval_min * 60)
            try:
                backup_now(results_dir, slug)
            except Exception as e:  # không để lỗi mạng làm chết thread
                print(f"[backup] ✗ lỗi: {e}")

    t = threading.Thread(target=_loop, name="auto-backup", daemon=True)
    t.start()
    print(f"[backup] Auto backup mỗi {interval_min} phút → dataset '{slug}' (private)")
    return t


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------

def restore(dest_parent: str = "results", slug: str = DEFAULT_SLUG) -> Path:
    """Lấy bản backup mới nhất và giải nén vào dest_parent (vd. results/ → results/flat/...).

    Ưu tiên dataset đã gắn vào notebook (/kaggle/input/<slug>), nếu không thì tải bằng API.
    """
    dest_parent_p = Path(dest_parent)
    dest_parent_p.mkdir(parents=True, exist_ok=True)

    candidates = [Path("/kaggle/input") / slug]
    if Path("/kaggle/input").exists():
        candidates += [d for d in Path("/kaggle/input").glob(f"**/{slug}") if d.is_dir()]
    input_dir = next((d for d in candidates if d.exists()), None)
    if input_dir is not None:
        src = input_dir
        print(f"[restore] Dùng dataset đã gắn: {src}")
    else:
        user = _load_kaggle_creds()
        src = STAGE_DIR.parent / "backup_download"
        if src.exists():
            shutil.rmtree(src)
        _kaggle("datasets", "download", f"{user}/{slug}", "-p", str(src), "--unzip")
        print(f"[restore] Đã tải {user}/{slug} → {src}")

    # Kaggle có thể tự giải nén file zip; xử lý cả hai trường hợp.
    zips = list(src.rglob(ARCHIVE_NAME))
    if zips:
        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(dest_parent_p)
    else:
        for d in src.rglob("flat"):
            if d.is_dir():
                shutil.copytree(d, dest_parent_p / "flat", dirs_exist_ok=True)
                break
        else:
            raise FileNotFoundError(f"Không tìm thấy {ARCHIVE_NAME} hoặc thư mục flat/ trong {src}")

    runs = sorted((dest_parent_p / "flat").glob("g1_flat_*"))
    print(f"[restore] ✓ Khôi phục {len(runs)} run vào {dest_parent_p / 'flat'}")
    for r in runs:
        ckpts = sorted((d for d in (r / "checkpoints").glob("*") if d.name.isdigit()),
                       key=lambda d: int(d.name)) if (r / "checkpoints").exists() else []
        last = ckpts[-1].name if ckpts else "—"
        print(f"    {r.name}: {len(ckpts)} checkpoint, mới nhất {last}")
    return dest_parent_p / "flat"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backup")
    b.add_argument("--results", default=DEFAULT_RESULTS)
    b.add_argument("--slug", default=DEFAULT_SLUG)
    b.add_argument("-m", "--message", default=None)
    r = sub.add_parser("restore")
    r.add_argument("--dest", default="results")
    r.add_argument("--slug", default=DEFAULT_SLUG)
    a = p.parse_args()
    if a.cmd == "backup":
        backup_now(a.results, a.slug, a.message)
    else:
        restore(a.dest, a.slug)


if __name__ == "__main__":
    main()
