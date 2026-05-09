from pathlib import Path
import subprocess
import csv
import time


SLIDES_DIR = Path("./slides")
ZARR_DIR = Path("./zarr")
LOG_FILE = Path("./zarr_conversion_log.csv")

TILE_WIDTH = 518
TILE_HEIGHT = 518
MAX_WORKERS = 1
FILL_COLOR = 255


def convert_one_slide(slide_path: Path):
    slide_id = slide_path.stem
    out_path = ZARR_DIR / f"{slide_id}.zarr"

    if out_path.exists():
        return slide_id, "skipped_exists", str(out_path), ""

    cmd = [
        "/home/jovyan/.local/bin/isyntax2raw",
        "write_tiles",
        str(slide_path),
        str(out_path),
        "--tile_width",
        str(TILE_WIDTH),
        "--tile_height",
        str(TILE_HEIGHT),
        "--max_workers",
        str(MAX_WORKERS),
        "--fill_color",
        str(FILL_COLOR),
    ]

    print(f"\nConverting: {slide_path.name}")
    print(" ".join(cmd), flush=True)

    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        elapsed = round(time.time() - start, 2)

        if result.returncode == 0:
            return slide_id, "success", str(out_path), f"{elapsed}s"
        else:
            return slide_id, "failed", str(out_path), result.stderr[-1000:]

    except Exception as e:
        return slide_id, "error", str(out_path), str(e)


def main():
    ZARR_DIR.mkdir(parents=True, exist_ok=True)

    slides = sorted(SLIDES_DIR.glob("*.isyntax"))

    if not slides:
        raise SystemExit(f"No .isyntax files found in {SLIDES_DIR}")

    print(f"Found {len(slides)} iSyntax files.")

    write_header = not LOG_FILE.exists()

    with LOG_FILE.open("a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["slide_id", "status", "zarr_path", "message"])

        for i, slide_path in enumerate(slides, start=1):
            print(f"\n[{i}/{len(slides)}] Processing {slide_path.name}")

            slide_id, status, zarr_path, message = convert_one_slide(slide_path)

            writer.writerow([slide_id, status, zarr_path, message])
            f.flush()

            print(f"[{slide_id}] {status}")


if __name__ == "__main__":
    main()
