"""Dataset loading utilities"""
from pathlib import Path
import cv2
import numpy as np

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEFAULT_DATASET = (
    Path(__file__).resolve().parent
    / "dataset"
)

def read_boxes(path: Path, width: int, height: int) -> np.ndarray:
    
    txt = path.with_suffix('.txt')
    xml = path.with_suffix('.xml')
    if txt.exists():
        boxes = []
        for line in txt.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f'Expected YOLO class cx cy w h in {txt}: {line}')
            _, cx, cy, w, h = map(float, parts)
            if not all(0 <= v <= 1 for v in (cx, cy, w, h)):
                raise ValueError(f'Not normalized YOLO: {txt}')
            boxes.append([max(0, (cx-w/2)*width), max(0, (cy-h/2)*height),
                          min(width, (cx+w/2)*width), min(height, (cy+h/2)*height)])
        return np.asarray(boxes, dtype=float).reshape(-1, 4)
    if xml.exists():
        import xml.etree.ElementTree as ET
        root = ET.parse(xml).getroot()
        boxes = []
        for obj in root.findall('.//object'):
            box = obj.find('bndbox')
            if box is not None:
                boxes.append([float(box.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')])
        return np.asarray(boxes, dtype=float).reshape(-1, 4)
    raise FileNotFoundError(f'Missing label {txt} or {xml}')

def load_dataset(dataset: Path, split: str):
    
    images = dataset / 'images' / split
    labels = dataset / 'labels' / split
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f'Missing {images} or {labels}')
    paths = sorted(p for p in images.rglob('*') if p.suffix.lower() in EXTS)
    if not paths:
        raise FileNotFoundError(f'No images in {images}')
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f'Cannot read image: {path}')
        h, w = image.shape[:2]
        gt = read_boxes(labels / path.relative_to(images), w, h)
        yield path, image, gt
