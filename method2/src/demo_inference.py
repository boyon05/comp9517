"""Visual inference demonstration using a trained YOLOv8 model

Default: yolov8s.pt, confidence=0.25, image size=640, output=results"""

import argparse
import sys
import os
import time
from pathlib import Path
try:
    from ultralytics import YOLO
except ImportError:
    print('[Error] ultralytics is required. Install via: pip install ultralytics')
    sys.exit(1)

def run_demo(source, model_path='yolov8s.pt', conf=0.25, imgsz=640, output_dir='results/demo_output', show=False):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not source or not os.path.exists(source):
        fallback_sample = Path(__file__).resolve().parent.parent / 'results' / 'augmentation_demo.png'
        if fallback_sample.exists():
            print(f"[Notice] Source '{source}' not specified or not found. Using default sample: {fallback_sample.name}")
            source = str(fallback_sample)
        else:
            print(f"[Error] Source path '{source}' does not exist.")
            return
    print('=' * 65)
    print('YOLOv8s Detector Inference Demo')
    print(f'Model:      {model_path}')
    print(f'Source:     {source}')
    print(f'Confidence: {conf}')
    print('=' * 65)
    model = YOLO(model_path)
    start_time = time.time()
    results = model.predict(source=source, conf=conf, imgsz=imgsz, save=True, project=str(out_dir), name='predictions', exist_ok=True, verbose=False)
    total_time = time.time() - start_time
    total_imgs = len(results) if isinstance(results, list) else 1
    fps = total_imgs / total_time if total_time > 0 else 0
    print(f'\n[Demo Finished Successfully]')
    print(f'Processed {total_imgs} image(s) in {total_time:.3f}s (~{fps:.1f} FPS)')
    for r in results[:5]:
        num_birds = len(r.boxes)
        print(f'  -> File: {Path(r.path).name} | Detected Targets: {num_birds}')
    print(f"Annotated outputs saved to: {out_dir / 'predictions'}")

def main():
    project_root = Path(__file__).resolve().parent.parent
    default_src = str(project_root / 'results' / 'augmentation_demo.png')
    default_out = str(project_root / 'results' / 'demo_output')
    best_weights = project_root / 'results' / 'ablation_study' / 'runs' / 'Strategy_2_Advanced' / 'weights' / 'best.pt'
    default_model = str(best_weights) if best_weights.exists() else 'yolov8s.pt'
    parser = argparse.ArgumentParser(description='Live Inference Demo for Video Presentation')
    parser.add_argument('--source', type=str, default=default_src, help='Path to test image, folder, or video')
    parser.add_argument('--model', type=str, default=default_model, help='Trained weights or base model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--output_dir', type=str, default=default_out, help='Output dir')
    parser.add_argument('--show', action='store_true', help='Show output in a window')
    args = parser.parse_args()
    run_demo(args.source, args.model, args.conf, args.imgsz, args.output_dir, args.show)

if __name__ == '__main__':
    main()
