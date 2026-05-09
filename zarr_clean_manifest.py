from pathlib import Path
import csv
import numpy as np
import cv2
import zarr
from tqdm import tqdm
from PIL import Image

ZARR_DIR = Path("./zarr")
MANIFEST = Path("./clean_manifest.csv")
QC_DIR = Path("./qc")

TILE_SIZE = 518
STRIDE = 518
THUMBNAIL_DOWNSAMPLE = 64

WHITE_THRESHOLD = 245
MAX_WHITE_FRACTION = 0.90

NOISE_DILATION_PX = 100
MIN_COMPONENT_AREA = 5
MIN_MAIN_FRACTION = 0.001


def white_fraction(rgb, threshold=245):
    return float(np.all(rgb >= threshold, axis=2).mean())


def boxes_intersect(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or ax0 >= bx1 or ay1 <= by0 or ay0 >= by1)


def find_level0_array(root):
    candidates = []

    def walk(group, prefix=""):
        for key in group.keys():
            item = group[key]
            path = f"{prefix}/{key}" if prefix else key

            if isinstance(item, zarr.Array):
                shape = item.shape
                if len(shape) == 5:
                    h, w = shape[-2], shape[-1]
                    candidates.append((path, h, w, shape))
                elif len(shape) == 3 and shape[-1] in [3, 4]:
                    h, w = shape[0], shape[1]
                    candidates.append((path, h, w, shape))
                elif len(shape) == 3 and shape[0] in [3, 4]:
                    h, w = shape[1], shape[2]
                    candidates.append((path, h, w, shape))

            elif isinstance(item, zarr.Group):
                walk(item, path)

    walk(root)

    if not candidates:
        raise RuntimeError("No valid image array found in this Zarr.")

    candidates.sort(key=lambda x: x[1] * x[2], reverse=True)

    path, h, w, shape = candidates[0]
    print(f"Selected array: {path}, shape={shape}, width={w}, height={h}")

    return root[path], path, h, w


def read_rgb_region(arr, x, y, width, height):
    shape = arr.shape

    if len(shape) == 5:
        patch = arr[0, :3, 0, y:y + height, x:x + width]
        patch = np.asarray(patch)
        patch = np.transpose(patch, (1, 2, 0))

    elif len(shape) == 3 and shape[-1] in [3, 4]:
        patch = arr[y:y + height, x:x + width, :3]
        patch = np.asarray(patch)

    elif len(shape) == 3 and shape[0] in [3, 4]:
        patch = arr[:3, y:y + height, x:x + width]
        patch = np.asarray(patch)
        patch = np.transpose(patch, (1, 2, 0))

    else:
        raise ValueError(f"Unsupported Zarr shape: {shape}")

    return patch.astype(np.uint8)


def make_thumbnail_strided(arr, downsample):
    shape = arr.shape

    if len(shape) == 5:
        thumb = arr[0, :3, 0, ::downsample, ::downsample]
        thumb = np.asarray(thumb)
        thumb = np.transpose(thumb, (1, 2, 0))

    elif len(shape) == 3 and shape[-1] in [3, 4]:
        thumb = arr[::downsample, ::downsample, :3]
        thumb = np.asarray(thumb)

    elif len(shape) == 3 and shape[0] in [3, 4]:
        thumb = arr[:3, ::downsample, ::downsample]
        thumb = np.asarray(thumb)
        thumb = np.transpose(thumb, (1, 2, 0))

    else:
        raise ValueError(f"Unsupported Zarr shape: {shape}")

    return thumb.astype(np.uint8)


def tissue_mask(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask_gray = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    _, mask_sat = cv2.threshold(
        sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = cv2.bitwise_or(mask_gray, mask_sat)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def save_qc(slide_id, thumb, mask, labels, stats, main_label, valid_labels):
    QC_DIR.mkdir(exist_ok=True)

    overlay = thumb.copy()

    for label in valid_labels:
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        color = (0, 255, 0) if label == main_label else (255, 0, 0)

        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            overlay,
            f"{label}:{area}",
            (x, max(10, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
        )

    Image.fromarray(thumb).save(QC_DIR / f"{slide_id}_thumbnail.png")
    Image.fromarray(mask).save(QC_DIR / f"{slide_id}_mask.png")
    Image.fromarray(overlay).save(QC_DIR / f"{slide_id}_components.png")


def process_zarr(zarr_path, writer):
    slide_id = zarr_path.stem.replace(".zarr", "")

    print(f"\nProcessing slide: {slide_id}")
    print(f"Zarr path: {zarr_path}")

    root = zarr.open(str(zarr_path), mode="r")
    arr, arr_path, height0, width0 = find_level0_array(root)

    print(f"[{slide_id}] Level 0 size: width={width0}, height={height0}")

    thumb = make_thumbnail_strided(arr, THUMBNAIL_DOWNSAMPLE)

    print(f"[{slide_id}] Thumbnail shape: {thumb.shape}")

    if thumb.ndim != 3 or thumb.shape[2] < 3:
        raise ValueError(f"Invalid thumbnail shape: {thumb.shape}")

    mask = tissue_mask(thumb)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8, ltype=cv2.CV_32S
    )

    valid_labels = [
        label for label in range(1, n_labels)
        if stats[label, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA
    ]

    if not valid_labels:
        print(f"[{slide_id}] No valid tissue components found.")
        return 0

    main_label = max(
        valid_labels,
        key=lambda label: stats[label, cv2.CC_STAT_AREA],
    )

    main_area = stats[main_label, cv2.CC_STAT_AREA]

    print(
        f"[{slide_id}] Components={len(valid_labels)}, "
        f"main_label={main_label}, main_area={main_area}"
    )

    save_qc(slide_id, thumb, mask, labels, stats, main_label, valid_labels)

    main_mask = labels == main_label

    noise_boxes = []

    for label in valid_labels:
        if label == main_label:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        x0 = max(0, x * THUMBNAIL_DOWNSAMPLE - NOISE_DILATION_PX)
        y0 = max(0, y * THUMBNAIL_DOWNSAMPLE - NOISE_DILATION_PX)
        x1 = min(width0, (x + w) * THUMBNAIL_DOWNSAMPLE + NOISE_DILATION_PX)
        y1 = min(height0, (y + h) * THUMBNAIL_DOWNSAMPLE + NOISE_DILATION_PX)

        noise_boxes.append((x0, y0, x1, y1))

    mx = stats[main_label, cv2.CC_STAT_LEFT]
    my = stats[main_label, cv2.CC_STAT_TOP]
    mw = stats[main_label, cv2.CC_STAT_WIDTH]
    mh = stats[main_label, cv2.CC_STAT_HEIGHT]

    main_x0 = max(0, mx * THUMBNAIL_DOWNSAMPLE)
    main_y0 = max(0, my * THUMBNAIL_DOWNSAMPLE)
    main_x1 = min(width0, (mx + mw) * THUMBNAIL_DOWNSAMPLE)
    main_y1 = min(height0, (my + mh) * THUMBNAIL_DOWNSAMPLE)

    print(
        f"[{slide_id}] Main box Level 0: "
        f"({main_x0}, {main_y0}) - ({main_x1}, {main_y1})"
    )
    print(f"[{slide_id}] Noise boxes: {len(noise_boxes)}")

    kept = 0
    skipped_not_main = 0
    skipped_noise = 0
    skipped_white = 0

    y_stop = main_y1 - TILE_SIZE + 1
    x_stop = main_x1 - TILE_SIZE + 1

    if y_stop <= main_y0 or x_stop <= main_x0:
        print(f"[{slide_id}] Main region is smaller than tile size.")
        return 0

    y_positions = range(main_y0, y_stop, STRIDE)
    x_positions = range(main_x0, x_stop, STRIDE)

    for y in tqdm(y_positions, desc=f"Manifest {slide_id}"):
        for x in x_positions:
            tile_box = (x, y, x + TILE_SIZE, y + TILE_SIZE)

            tx0 = max(0, x // THUMBNAIL_DOWNSAMPLE)
            ty0 = max(0, y // THUMBNAIL_DOWNSAMPLE)
            tx1 = min(main_mask.shape[1], (x + TILE_SIZE) // THUMBNAIL_DOWNSAMPLE + 1)
            ty1 = min(main_mask.shape[0], (y + TILE_SIZE) // THUMBNAIL_DOWNSAMPLE + 1)

            if tx1 <= tx0 or ty1 <= ty0:
                skipped_not_main += 1
                continue

            main_fraction = float(main_mask[ty0:ty1, tx0:tx1].mean())

            if main_fraction < MIN_MAIN_FRACTION:
                skipped_not_main += 1
                continue

            if any(boxes_intersect(tile_box, nb) for nb in noise_boxes):
                skipped_noise += 1
                continue

            patch = read_rgb_region(
                arr,
                x=x,
                y=y,
                width=TILE_SIZE,
                height=TILE_SIZE,
            )

            if patch.shape[0] != TILE_SIZE or patch.shape[1] != TILE_SIZE:
                continue

            wf = white_fraction(patch, threshold=WHITE_THRESHOLD)

            if wf > MAX_WHITE_FRACTION:
                skipped_white += 1
                continue

            writer.writerow([
                slide_id,
                str(zarr_path),
                arr_path,
                x,
                y,
                x + TILE_SIZE,
                y + TILE_SIZE,
                TILE_SIZE,
                round(wf, 6),
                round(main_fraction, 6),
            ])

            kept += 1

    print(
        f"[{slide_id}] DONE | kept={kept}, "
        f"skipped_not_main={skipped_not_main}, "
        f"skipped_noise={skipped_noise}, "
        f"skipped_white={skipped_white}"
    )

    return kept


def main():
    zarr_files = sorted(ZARR_DIR.glob("*.zarr"))

    if not zarr_files:
        raise SystemExit(f"No .zarr folders found in {ZARR_DIR}")

    print(f"Found {len(zarr_files)} Zarr slides.")

    with MANIFEST.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "slide_id",
            "zarr_path",
            "array_path",
            "x0",
            "y0",
            "x1",
            "y1",
            "tile_size",
            "white_fraction",
            "main_fraction",
        ])

        total_kept = 0

        for zarr_path in zarr_files:
            try:
                kept = process_zarr(zarr_path, writer)
                total_kept += kept
                f.flush()

            except Exception as e:
                print(f"[FAILED] {zarr_path}: {e}")

    print("\nClean manifest created:")
    print(MANIFEST)
    print(f"Total clean patches: {total_kept}")
    print("QC images saved in ./qc/")


if __name__ == "__main__":
    main()
