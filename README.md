# YOLOboxer
A lightweight, local web app for drawing bounding boxes on images and saving them in
YOLO format. Point it at a folder of images, define your classes, and label with mouse
and keyboard. Labels and `classes.txt` are written to a `labels/` folder next to the
images, ready for YOLO training. No accounts, no cloud, no build step.

## Run

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python yoloboxer.py [IMAGE_DIR] [--port 8000]
```

Open http://127.0.0.1:8000, pick an image directory (type the path or use *browse…*),
add your classes in the right panel and start drawing boxes.

## Output

Labels are written next to the images:

```
IMAGE_DIR/
  img001.jpg
  labels/
    classes.txt     one class name per line; line number = class id
    img001.txt      one box per line: class cx cy w h (normalised 0-1)
```

An image saved with no boxes gets an empty label file and counts as labelled (a background
image); it is marked blue instead of green in the image list.

## Keys

| key | action |
| --- | --- |
| drag | draw a box with the current class |
| click | select a box |
| `1`…`0`, `a`…`z` | pick class / re-class the selected box |
| `Del` / `Backspace` | delete selected box |
| `Esc` | deselect |
| pinch / `Ctrl`+scroll, `+` `-` | zoom in / out |
| two-finger scroll / middle-drag | pan the zoomed image |
| double-click / *fit* button | reset zoom |
| `←` `→` | previous / next image |
| `Ctrl+S` | save |
| `Enter` | save + next |

## Code layout

- `yoloboxer.py` – FastAPI backend (directory browsing, images, labels, classes)
- `static/index.html`, `static/style.css`, `static/app.js` – the UI, plain HTML/CSS/JS, no build step
