"""YOLOboxer - local web dashboard for labelling images in YOLO format.

Run:  python yoloboxer.py [IMAGE_DIR] [--host 127.0.0.1] [--port 8000]
Then open http://127.0.0.1:8000

Pick an image directory in the UI (or pass it on the command line).
Labels are written next to the images:

    <IMAGE_DIR>/labels/<image stem>.txt   one line per box: class cx cy w h (normalised 0-1)
    <IMAGE_DIR>/labels/classes.txt        one class name per line, line number = class id

The backend is a plain FastAPI app; the UI lives in ./static (index.html, app.js, style.css).
"""
import argparse
import os
from collections import Counter
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
STATIC_DIR = Path(__file__).parent / "static"

# these are the user's own files, and large images are common; lift Pillow's bomb guard
Image.MAX_IMAGE_PIXELS = None

app = FastAPI(title="YOLOboxer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def no_cache(request, call_next):
    # make the browser revalidate every file, so UI changes show up on a normal reload
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


# --- workspace: the currently opened image directory ------------------------

class Workspace:
    def __init__(self):
        self.image_dir: Path | None = None

    @property
    def labels_dir(self) -> Path:
        return self.image_dir / "labels"

    @property
    def classes_file(self) -> Path:
        return self.labels_dir / "classes.txt"

    def require(self) -> Path:
        if self.image_dir is None:
            raise HTTPException(400, "no image directory opened")
        return self.image_dir

    def images(self) -> list[str]:
        d = self.require()
        return sorted(p.name for p in d.iterdir()
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    def check_image(self, name: str) -> Path:
        if name not in set(self.images()):
            raise HTTPException(404, "unknown image")
        return self.image_dir / name

    def label_path(self, name: str) -> Path:
        return self.labels_dir / (Path(name).stem + ".txt")

    def read_boxes(self, name: str) -> list[dict] | None:
        """Boxes from the image's label file, or None if it has no label file yet."""
        lp = self.label_path(name)
        if not lp.exists():
            return None
        boxes = []
        for line in lp.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            c, cx, cy, w, h = parts
            boxes.append({"cls": int(c), "cx": float(cx), "cy": float(cy),
                          "w": float(w), "h": float(h)})
        return boxes

    def read_classes(self) -> list[str]:
        if not self.classes_file.exists():
            return []
        return [line.strip() for line in self.classes_file.read_text().splitlines() if line.strip()]

    def write_classes(self, classes: list[str]):
        self.labels_dir.mkdir(exist_ok=True)
        self.classes_file.write_text("\n".join(classes) + ("\n" if classes else ""))

    def remap_labels(self, mapping: dict[int, int]) -> int:
        """Rewrite the class id of every box in every label file through `mapping`
        (old id -> new id; ids not in it are kept). Returns the number of files changed."""
        if not self.labels_dir.is_dir():
            return 0
        changed = 0
        for lp in sorted(self.labels_dir.glob("*.txt")):
            if lp == self.classes_file:
                continue
            text = lp.read_text()
            lines = []
            for line in text.splitlines():
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in mapping:
                    line = f"{mapping[int(parts[0])]} {parts[1]}"
                lines.append(line)
            new = "\n".join(lines) + ("\n" if lines else "")
            if new != text:
                tmp = lp.with_name(lp.name + ".tmp")
                tmp.write_text(new)
                os.replace(tmp, lp)   # never leave a half-written label file behind
                changed += 1
        return changed


ws = Workspace()


def count_images(d: Path) -> int:
    try:
        return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    except OSError:
        return 0


def image_size(path: Path) -> tuple[int, int] | None:
    """Width and height as the browser shows the image, i.e. after EXIF rotation."""
    try:
        with Image.open(path) as im:
            w, h = im.size
            if im.getexif().get(0x0112) in (5, 6, 7, 8):   # orientation: rotated 90 or 270
                w, h = h, w
            return w, h
    except OSError:
        return None


# --- models -----------------------------------------------------------------

class Box(BaseModel):
    cls: int      # class id (line index in classes.txt)
    cx: float     # centre x, normalised 0-1
    cy: float     # centre y, normalised 0-1
    w: float      # width, normalised 0-1
    h: float      # height, normalised 0-1


class SavePayload(BaseModel):
    boxes: list[Box]


class OpenPayload(BaseModel):
    path: str


class ClassesPayload(BaseModel):
    classes: list[str]


class ReorderPayload(BaseModel):
    order: list[int]   # order[new id] = old id; a permutation of the current class ids


# --- routes -----------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
def api_state():
    """Current directory, its images (with done-flag) and classes."""
    if ws.image_dir is None:
        return {"dir": None, "images": [], "classes": []}
    images = []
    for n in ws.images():
        lp = ws.label_path(n)
        done = lp.exists()
        # empty = saved with no boxes (a background image)
        images.append({"name": n, "done": done, "empty": done and lp.stat().st_size == 0})
    return {"dir": str(ws.image_dir), "images": images, "classes": ws.read_classes()}


@app.post("/api/open")
def api_open(payload: OpenPayload):
    path = Path(payload.path).expanduser()
    if not path.is_dir():
        raise HTTPException(400, f"not a directory: {path}")
    ws.image_dir = path.resolve()
    return api_state()


@app.get("/api/browse")
def api_browse(path: str = "~"):
    """List sub-directories of `path` so the UI can navigate the filesystem."""
    d = Path(path).expanduser().resolve()
    if not d.is_dir():
        raise HTTPException(400, f"not a directory: {d}")
    try:
        subdirs = sorted((p for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")),
                         key=lambda p: p.name.lower())
    except PermissionError:
        raise HTTPException(403, f"permission denied: {d}")
    return {
        "path": str(d),
        "parent": str(d.parent) if d.parent != d else None,
        "images": count_images(d),
        "dirs": [{"name": p.name, "images": count_images(p)} for p in subdirs],
    }


@app.get("/image/{name}")
def image(name: str):
    return FileResponse(ws.check_image(name))


@app.get("/api/label/{name}")
def api_label(name: str):
    ws.check_image(name)
    return {"boxes": ws.read_boxes(name) or []}


@app.post("/api/label/{name}")
def api_save_label(name: str, payload: SavePayload):
    ws.check_image(name)
    ws.labels_dir.mkdir(exist_ok=True)
    lines = [f"{b.cls} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}" for b in payload.boxes]
    ws.label_path(name).write_text("\n".join(lines) + ("\n" if lines else ""))
    return {"ok": True, "count": len(lines)}


@app.get("/api/stats")
def api_stats():
    """Dataset statistics from the saved label files."""
    classes = ws.read_classes()
    instances = Counter()     # class id -> number of boxes
    images_with = Counter()   # class id -> number of images containing it
    measured = Counter()      # class id -> boxes in images whose size could be read
    area = Counter()          # class id -> summed box area, px
    with_boxes = empty = unlabelled = 0
    for name in ws.images():
        boxes = ws.read_boxes(name)
        if boxes is None:
            unlabelled += 1
            continue
        if boxes:
            with_boxes += 1
            size = image_size(ws.image_dir / name)
            if size:
                for b in boxes:
                    measured[b["cls"]] += 1
                    area[b["cls"]] += b["w"] * size[0] * b["h"] * size[1]
        else:
            empty += 1
        ids = [b["cls"] for b in boxes]
        instances.update(ids)
        images_with.update(set(ids))

    def avg_px(c=None):
        n = sum(measured.values()) if c is None else measured[c]
        total = sum(area.values()) if c is None else area[c]
        return round(total / n) if n else None

    # every class in classes.txt, plus any id used in a label file but missing from it
    ids = list(range(len(classes))) + sorted(c for c in instances if c >= len(classes))
    return {
        "images": {"total": with_boxes + empty + unlabelled, "with_boxes": with_boxes,
                   "empty": empty, "unlabelled": unlabelled},
        "boxes": sum(instances.values()),
        "avg_px": avg_px(),
        "classes": [{"id": c, "name": classes[c] if c < len(classes) else None,
                     "instances": instances[c], "images": images_with[c], "avg_px": avg_px(c)}
                    for c in ids],
    }


@app.post("/api/classes")
def api_save_classes(payload: ClassesPayload):
    ws.require()
    classes = [c.strip() for c in payload.classes if c.strip()]
    ws.write_classes(classes)
    return {"classes": classes}


@app.post("/api/classes/reorder")
def api_reorder_classes(payload: ReorderPayload):
    """Reorder the classes and renumber the boxes in all label files to match."""
    ws.require()
    classes = ws.read_classes()
    if sorted(payload.order) != list(range(len(classes))):
        raise HTTPException(400, "order must be a permutation of the class ids")
    mapping = {old: new for new, old in enumerate(payload.order) if old != new}
    files = ws.remap_labels(mapping)
    ws.write_classes([classes[old] for old in payload.order])
    return {"classes": ws.read_classes(), "files": files}


# --- entry point ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YOLOboxer image labeller")
    parser.add_argument("image_dir", nargs="?", help="directory with images to open on start")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.image_dir:
        api_open(OpenPayload(path=args.image_dir))
    print(f"YOLOboxer on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
