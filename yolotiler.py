"""YOLOtiler - local web dashboard for cutting labelled images into overlapping tiles.

Run:  python yolotiler.py [IMAGE_DIR] [--host 127.0.0.1] [--port 8001]
Then open http://127.0.0.1:8001

Pick an image directory in the UI (or pass it on the command line), set the tile size and
overlap, and check how the YOLOboxer labels end up in each tile. Export writes the tiles in
the same layout YOLOboxer uses, so the output directory can be opened in YOLOboxer again:

    <OUT>/<image stem>_x<left>_y<top><ext>    the tile, cut from the image at pixel (left, top)
    <OUT>/labels/<tile stem>.txt              boxes clipped to the tile, normalised to the tile
    <OUT>/labels/classes.txt                  copied from the image directory
    <OUT>/tiling.json                         the settings used

The tile grid is computed twice: in static/tiler.js for the preview and here for the export.
tile_starts and clip_box mirror tileStarts and clipBox there; keep them in step.
"""
import argparse
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from yoloboxer import STATIC_DIR, OpenPayload, Workspace, api_browse, count_images, image_size

app = FastAPI(title="YOLOtiler")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.get("/api/browse")(api_browse)


@app.middleware("http")
async def no_cache(request, call_next):
    # make the browser revalidate every file, so UI changes show up on a normal reload
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


ws = Workspace()


# --- tiling -----------------------------------------------------------------

class TileSettings(BaseModel):
    w: int = Field(ge=1)               # tile width, px
    h: int = Field(ge=1)               # tile height, px
    ox: int = Field(ge=0)              # horizontal overlap between neighbouring tiles, px
    oy: int = Field(ge=0)              # vertical overlap, px
    min_vis: float = Field(ge=0, le=100)   # keep a cut box if at least this % of it is in the tile
    # the overlap as the user typed it, in % of the tile size (x, y), if they did; only
    # recorded in tiling.json, the tiles are cut by ox and oy
    overlap_pct: tuple[float, float] | None = None


def tile_starts(length: int, size: int, overlap: int) -> list[int]:
    """Tile offsets along one axis. The last tile is moved back to end at the image edge,
    so it overlaps its neighbour by more than `overlap` instead of running off the image."""
    if size >= length:
        return [0]
    stride = max(1, size - overlap)
    return list(range(0, length - size, stride)) + [length - size]


def tiles_for(width: int, height: int, s: TileSettings) -> list[tuple[int, int, int, int]]:
    """(left, top, width, height) of every tile, row by row. Tiles never exceed the image."""
    tw, th = min(s.w, width), min(s.h, height)
    return [(x, y, tw, th)
            for y in tile_starts(height, th, s.oy)
            for x in tile_starts(width, tw, s.ox)]


def clip_box(b: dict, tile: tuple[int, int, int, int], width: int, height: int,
             min_vis: float) -> str | None:
    """A YOLO box of the image cut to `tile`, as a label line normalised to the tile,
    or None if it misses the tile or less than `min_vis` % of it is left."""
    tx, ty, tw, th = tile
    x0, x1 = (b["cx"] - b["w"] / 2) * width, (b["cx"] + b["w"] / 2) * width
    y0, y1 = (b["cy"] - b["h"] / 2) * height, (b["cy"] + b["h"] / 2) * height
    ix0, ix1 = max(x0, tx), min(x1, tx + tw)
    iy0, iy1 = max(y0, ty), min(y1, ty + th)
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    area = (x1 - x0) * (y1 - y0)
    vis = (ix1 - ix0) * (iy1 - iy0) / area if area > 0 else 0
    if vis * 100 < min_vis - 1e-9:
        return None
    cx, cy = ((ix0 + ix1) / 2 - tx) / tw, ((iy0 + iy1) / 2 - ty) / th
    return f"{b['cls']} {cx:.6f} {cy:.6f} {(ix1 - ix0) / tw:.6f} {(iy1 - iy0) / th:.6f}"


def export_dir(out: str) -> Path:
    """The output directory for `out`; a relative path is taken relative to the image dir."""
    src = ws.require()
    d = Path(out).expanduser()
    if not d.is_absolute():
        d = src / d
    d = d.resolve()
    if d in (src, ws.labels_dir.resolve()):
        raise HTTPException(400, "the output directory must differ from the image directory")
    if d.exists() and not d.is_dir():
        raise HTTPException(400, f"not a directory: {d}")
    return d


# --- models -----------------------------------------------------------------

class ExportStartPayload(BaseModel):
    out: str
    settings: TileSettings


class ExportImagePayload(BaseModel):
    out: str
    name: str
    settings: TileSettings
    skip_empty: bool = False   # leave out tiles without any box


# --- routes -----------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "tiler.html")


@app.get("/api/state")
def api_state():
    """Current directory, its images (with done-flag and size) and classes."""
    if ws.image_dir is None:
        return {"dir": None, "images": [], "classes": []}
    images = []
    for n in ws.images():
        lp = ws.label_path(n)
        done = lp.exists()
        size = image_size(ws.image_dir / n)
        images.append({"name": n, "done": done, "empty": done and lp.stat().st_size == 0,
                       "w": size[0] if size else None, "h": size[1] if size else None})
    return {"dir": str(ws.image_dir), "images": images, "classes": ws.read_classes()}


@app.post("/api/open")
def api_open(payload: OpenPayload):
    path = Path(payload.path).expanduser()
    if not path.is_dir():
        raise HTTPException(400, f"not a directory: {path}")
    ws.image_dir = path.resolve()
    return api_state()


@app.get("/image/{name}")
def image(name: str):
    return FileResponse(ws.check_image(name))


@app.get("/api/label/{name}")
def api_label(name: str):
    ws.check_image(name)
    return {"boxes": ws.read_boxes(name) or []}


@app.get("/api/export/check")
def api_export_check(out: str):
    """Where `out` resolves to and how many images are already there."""
    d = export_dir(out)
    return {"out": str(d), "images": count_images(d) if d.is_dir() else 0}


@app.post("/api/export/start")
def api_export_start(payload: ExportStartPayload):
    """Create the output directory, copy classes.txt and record the settings."""
    d = export_dir(payload.out)
    out = Workspace()
    out.image_dir = d
    d.mkdir(parents=True, exist_ok=True)
    out.write_classes(ws.read_classes())
    record = {"source": str(ws.image_dir), **payload.settings.model_dump()}
    (d / "tiling.json").write_text(json.dumps(record, indent=2) + "\n")
    return {"out": str(d)}


@app.post("/api/export/image")
def api_export_image(payload: ExportImagePayload):
    """Cut one image into tiles and write them with their clipped labels.
    An image without a label file gives tiles without label files (still unlabelled)."""
    src = ws.check_image(payload.name)
    s = payload.settings
    if s.ox >= s.w or s.oy >= s.h:
        raise HTTPException(400, "overlap must be smaller than the tile size")
    out = Workspace()
    out.image_dir = export_dir(payload.out)
    out.labels_dir.mkdir(parents=True, exist_ok=True)
    boxes = ws.read_boxes(payload.name)
    stem, ext = Path(payload.name).stem, Path(payload.name).suffix
    save_opts = {"quality": 95} if ext.lower() in {".jpg", ".jpeg", ".webp"} else {}

    written = with_boxes = 0
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)   # labels were drawn on the rotated image
        width, height = im.size
        for tile in tiles_for(width, height, s):
            lines = [line for b in boxes or []
                     if (line := clip_box(b, tile, width, height, s.min_vis))]
            if payload.skip_empty and not lines:
                continue
            x, y, tw, th = tile
            name = f"{stem}_x{x}_y{y}{ext}"
            im.crop((x, y, x + tw, y + th)).save(out.image_dir / name, **save_opts)
            if boxes is not None:
                out.label_path(name).write_text("\n".join(lines) + ("\n" if lines else ""))
            written += 1
            with_boxes += bool(lines)
    return {"tiles": written, "with_boxes": with_boxes}


# --- entry point ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YOLOtiler image tiler")
    parser.add_argument("image_dir", nargs="?", help="directory with images to open on start")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.image_dir:
        api_open(OpenPayload(path=args.image_dir))
    print(f"YOLOtiler on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
