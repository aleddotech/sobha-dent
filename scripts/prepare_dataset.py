#!/usr/bin/env python3
"""Merge Roboflow YOLO exports into a 2-class dent/scratch dataset."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "yolo"
NAMES = ["dent", "scratch"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def class_id(name: str) -> int | None:
    n = name.lower().replace("_", " ").replace("-", " ")
    if "scratch" in n or "scratche" in n:
        return 1
    if "dent" in n:
        return 0
    return None


def load_names(data_yaml: Path) -> list[str]:
    names: list[str] = []
    text = data_yaml.read_text(encoding="utf-8", errors="ignore")
    in_names = False
    for line in text.splitlines():
        if line.strip().startswith("names:"):
            rest = line.split(":", 1)[1].strip()
            if rest.startswith("["):
                inner = rest.strip("[]")
                return [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
            in_names = True
            continue
        if in_names:
            if not line.startswith(" ") and not line.startswith("\t") and line.strip() and not line.strip().startswith("-"):
                break
            if ":" in line and not line.strip().startswith("-"):
                names.append(line.split(":", 1)[1].strip().strip("'\""))
            elif line.strip().startswith("-"):
                names.append(line.split("-", 1)[1].strip().strip("'\""))
    return names


def iter_split_dirs(root: Path) -> list[tuple[str, Path, Path]]:
    found = []
    for split in ("train", "valid", "val", "test"):
        img_dir = None
        for cand in (root / split / "images", root / split):
            if cand.is_dir() and any(p.suffix.lower() in IMG_EXT for p in cand.iterdir() if p.is_file()):
                img_dir = cand
                break
        if img_dir is None:
            continue
        lbl_dir = root / split / "labels"
        if not lbl_dir.is_dir():
            sibling = img_dir.parent / "labels"
            lbl_dir = sibling if sibling.is_dir() else img_dir
        dest_split = "val" if split in {"valid", "val", "test"} else "train"
        found.append((dest_split, img_dir, lbl_dir))
    return found


def convert_label(src: Path, dst: Path, mapping: dict[int, int]) -> int:
    kept = 0
    lines_out = []
    if src.exists():
        for line in src.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            old = int(float(parts[0]))
            if old not in mapping:
                continue
            lines_out.append(" ".join([str(mapping[old]), *parts[1:]]))
            kept += 1
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return kept


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)

    n_img = {"train": 0, "val": 0}
    n_box = {"train": 0, "val": 0}

    for proj in sorted(p for p in RAW.iterdir() if p.is_dir()):
        yamls = list(proj.rglob("data.yaml"))
        if not yamls:
            print(f"no data.yaml in {proj.name}")
            continue
        names = load_names(yamls[0])
        mapping = {}
        for i, name in enumerate(names):
            cid = class_id(name)
            if cid is not None:
                mapping[i] = cid
        print(f"{proj.name}: {names} -> {mapping}")
        if not mapping:
            continue
        ds_root = yamls[0].parent
        for split, img_dir, lbl_dir in iter_split_dirs(ds_root):
            for img in img_dir.iterdir():
                if not img.is_file() or img.suffix.lower() not in IMG_EXT:
                    continue
                stem = f"{proj.name}_{img.stem}"
                lbl = lbl_dir / f"{img.stem}.txt"
                out_lbl = OUT / "labels" / split / f"{stem}.txt"
                kept = convert_label(lbl, out_lbl, mapping)
                if kept == 0 and split == "train":
                    out_lbl.unlink(missing_ok=True)
                    continue
                shutil.copy2(img, OUT / "images" / split / f"{stem}{img.suffix.lower()}")
                n_img[split] += 1
                n_box[split] += kept

    yaml = OUT / "dataset.yaml"
    yaml.write_text(
        f"path: {OUT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: dent\n"
        "  1: scratch\n"
    )
    print(f"wrote {yaml}")
    print(f"train images={n_img['train']} boxes={n_box['train']}")
    print(f"val images={n_img['val']} boxes={n_box['val']}")
    print("real-world unlabeled eval stays in data/val_realworld (no boxes)")


if __name__ == "__main__":
    main()
