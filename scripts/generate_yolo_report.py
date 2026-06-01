import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kagglehub
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO

# Allow script execution from the scripts/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_utils import parse_xmls, LABEL_MAP
from src.models.modules.potholes_yolo import PotholeYoloModule

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

DEFAULT_CKPT = Path('logs/pothole_yolo_detection_v1_20260601_190124/checkpoints/epoch=4-step=225.ckpt')
DEFAULT_OUTPUT_DIR = Path('outputs/yolo_report')


def resolve_dataset_root():
    cached = Path.home() / '.cache' / 'kagglehub' / 'datasets' / 'idanbaru' / 'annotated-potholes-with-severity-levels' / 'versions' / '1'
    if cached.exists():
        return cached
    print('Dataset not found in KaggleHub cache. Downloading dataset...')
    dataset_path = kagglehub.dataset_download('idanbaru/annotated-potholes-with-severity-levels')
    return Path(dataset_path)


def get_predictions(model: DetectionModel, image_path: Path, conf: float = 0.1, max_det: int = 100):
    # Support both YOLO wrapper (predict(source=...,...)) and DetectionModel.predict(x)
    try:
        results = model.predict(source=str(image_path), conf=conf, max_det=max_det, verbose=False)
    except TypeError:
        # DetectionModel.predict signature expects positional 'x' as image array/tensor
        import cv2 as _cv2
        img = _cv2.imread(str(image_path))
        if img is None:
            return []
        results = model.predict(img)

    if len(results) == 0:
        return []

    boxes = results[0].boxes
    if boxes is None or len(boxes.data) == 0:
        return []

    data = boxes.data.cpu().numpy()
    # data layout: x1, y1, x2, y2, score, cls
    # apply conf and max_det filtering in case DetectionModel.predict ignored them
    if data.shape[1] >= 5:
        scores = data[:, 4]
        keep_mask = scores >= conf
        data = data[keep_mask]

    if data.shape[0] > 0:
        # sort by score desc and limit to max_det
        order = (-data[:, 4]).argsort()
        order = order[:max_det]
        data = data[order]

    predictions = []
    for row in data:
        x1, y1, x2, y2, score, cls = row[:6]
        predictions.append({
            'x1': float(x1),
            'y1': float(y1),
            'x2': float(x2),
            'y2': float(y2),
            'score': float(score),
            'class': int(cls),
            'label': LABEL_NAMES.get(int(cls), str(int(cls)))
        })
    return predictions


def draw_boxes(image: np.ndarray, boxes, color, label_prefix=None):
    for box in boxes:
        x1, y1, x2, y2 = map(int, [box['x1'], box['y1'], box['x2'], box['y2']])
        text = box.get('label', '')
        if label_prefix is not None:
            score = box.get('score')
            suffix = f' {score:.2f}' if score is not None else ''
            text = f'{label_prefix}: {text}{suffix}'
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = text[:32]
            txt_size = cv2.getTextSize(text, font, 0.5, 1)[0]
            cv2.rectangle(image, (x1, y1 - txt_size[1] - 6), (x1 + txt_size[0] + 4, y1), color, -1)
            cv2.putText(image, text, (x1 + 2, y1 - 4), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def build_comparison_image(image_path: Path, gt_boxes, pred_boxes):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    gt_image = image.copy()
    pred_image = image.copy()

    draw_boxes(gt_image, gt_boxes, color=(0, 255, 0), label_prefix='GT')
    draw_boxes(pred_image, pred_boxes, color=(0, 0, 255), label_prefix='PRED')

    if gt_image.shape != pred_image.shape:
        pred_image = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]))

    combined = np.concatenate([gt_image, pred_image], axis=1)
    return combined


def make_gt_boxes(df: pd.DataFrame, image_path: Path):
    rows = df[df['file'] == str(image_path)]
    boxes = []
    for _, row in rows.iterrows():
        boxes.append({
            'x1': float(row['xmin']),
            'y1': float(row['ymin']),
            'x2': float(row['xmax']),
            'y2': float(row['ymax']),
            'class': int(row['label']),
            'label': LABEL_NAMES.get(int(row['label']), str(int(row['label'])))
        })
    return boxes


def save_dataset_summary(df: pd.DataFrame, output_dir: Path):
    labels = df['label'].map(lambda v: LABEL_NAMES.get(int(v), str(int(v))))
    counts = labels.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind='bar', color=['#4caf50', '#ff9800', '#f44336'], ax=ax)
    ax.set_title('Dataset label distribution')
    ax.set_xlabel('Pothole severity')
    ax.set_ylabel('Image annotation count')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    path = output_dir / 'dataset_label_distribution.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def iou(box_a, box_b):
    xA = max(box_a['x1'], box_b['x1'])
    yA = max(box_a['y1'], box_b['y1'])
    xB = min(box_a['x2'], box_b['x2'])
    yB = min(box_a['y2'], box_b['y2'])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    area_a = max(0.0, box_a['x2'] - box_a['x1']) * max(0.0, box_a['y2'] - box_a['y1'])
    area_b = max(0.0, box_b['x2'] - box_b['x1']) * max(0.0, box_b['y2'] - box_b['y1'])
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_image_detection_metrics(gt_boxes, pred_boxes, iou_threshold=0.5):
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0

    matched = [False] * len(gt_boxes)
    tp = 0
    fp = 0

    for pred in sorted(pred_boxes, key=lambda x: x['score'], reverse=True):
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if matched[idx]:
                continue
            score = iou(pred, gt)
            if score > best_iou:
                best_iou = score
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold and pred['class'] == gt_boxes[best_idx]['class']:
            tp += 1
            matched[best_idx] = True
        else:
            fp += 1

    fn = int(sum(1 for matched_flag in matched if not matched_flag))
    return tp, fp, fn


def compute_dataset_metrics(all_gt, all_pred, iou_threshold=0.5):
    tp = fp = fn = 0
    for gt_boxes, pred_boxes in zip(all_gt, all_pred):
        image_tp, image_fp, image_fn = compute_image_detection_metrics(gt_boxes, pred_boxes, iou_threshold)
        tp += image_tp
        fp += image_fp
        fn += image_fn

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = tp / max(tp + fp + fn, 1)
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'images_evaluated': len(all_gt),
    }


def save_metrics_report(metrics, output_dir: Path):
    lines = [
        f"Images evaluated: {metrics['images_evaluated']}",
        f"True positives: {metrics['tp']}",
        f"False positives: {metrics['fp']}",
        f"False negatives: {metrics['fn']}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1 score: {metrics['f1']:.4f}",
        f"Detection accuracy: {metrics['accuracy']:.4f}",
    ]
    path = output_dir / 'detection_metrics.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return path


def save_heatmap(centers, output_dir: Path, img_size=(640, 640), bins=64):
    if len(centers) == 0:
        return None

    xs = np.array([c[0] for c in centers])
    ys = np.array([c[1] for c in centers])
    heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=bins, range=[[0, img_size[0]], [0, img_size[1]]])
    heatmap = np.flipud(heatmap.T)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(heatmap, cmap='hot', extent=[0, img_size[0], 0, img_size[1]], origin='lower')
    ax.set_title('Prediction center heatmap')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax)
    path = output_dir / 'prediction_heatmap.png'
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def find_metrics_csv(run_dir: Path):
    for candidate in run_dir.rglob('metrics.csv'):
        return candidate
    return None


def save_loss_curve(metrics_csv: Path, output_dir: Path):
    df = pd.read_csv(metrics_csv)
    loss_columns = [c for c in df.columns if 'loss' in c.lower()]
    if not loss_columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    for col in loss_columns:
        ax.plot(df[col], label=col)
    ax.set_title('Loss curve')
    ax.set_xlabel('Step or epoch index')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    path = output_dir / 'loss_curve.png'
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description='Generate YOLO inference report for pothole detection.')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CKPT, help='YOLO Lightning checkpoint path')
    parser.add_argument('--samples', type=int, default=5, help='Number of sample images to visualize')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_DIR, help='Directory to save generated outputs')
    parser.add_argument('--conf', type=float, default=0.1, help='Confidence threshold for predictions')
    parser.add_argument('--max_det', type=int, default=100, help='Maximum number of detections per image')
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    dataset_root = resolve_dataset_root()
    print('Using dataset root:', dataset_root)
    df = parse_xmls(str(dataset_root))
    save_dataset_summary(df, output_dir)

    model = PotholeYoloModule.load_from_checkpoint(str(args.checkpoint))
    model.eval()

    # For inference prefer the YOLO wrapper when the original config was a .pt file
    inference_model = None
    cfg = getattr(model.hparams, 'model_cfg', None)
    if isinstance(cfg, str) and cfg.endswith('.pt'):
        inference_model = YOLO(cfg)
    else:
        inference_model = model.model

    if not hasattr(inference_model, 'predict'):
        raise RuntimeError('Loaded model has no predict() method. Provide a model with a predict API.')

    unique_images = [Path(img) for img in sorted(df['file'].unique())]
    existing_images = [img for img in unique_images if img.exists()]
    if len(existing_images) == 0:
        raise FileNotFoundError('No valid image files found for the dataset.')

    sample_images = existing_images[: min(args.samples, len(existing_images))]
    all_gt = []
    all_pred = []
    heatmap_centers = []
    prediction_counts = []

    for image_path in existing_images:
        gt_boxes = make_gt_boxes(df, image_path)
        pred_boxes = get_predictions(inference_model, image_path, conf=args.conf, max_det=args.max_det)
        all_gt.append(gt_boxes)
        all_pred.append(pred_boxes)
        prediction_counts.append(len(pred_boxes))
        heatmap_centers.extend([((b['x1'] + b['x2']) / 2.0, (b['y1'] + b['y2']) / 2.0) for b in pred_boxes])

    metrics = compute_dataset_metrics(all_gt, all_pred)
    report_path = save_metrics_report(metrics, output_dir)
    print('Saved detection metrics:', report_path)

    heatmap_path = save_heatmap(heatmap_centers, output_dir)
    if heatmap_path is not None:
        print('Saved prediction heatmap:', heatmap_path)

    run_dir = args.checkpoint.parent.parent
    metrics_csv = find_metrics_csv(run_dir)
    if metrics_csv is not None:
        loss_path = save_loss_curve(metrics_csv, output_dir)
        if loss_path is not None:
            print('Saved loss curve:', loss_path)
    else:
        print('No metrics.csv found under', run_dir)

    for idx, image_path in enumerate(sample_images, start=1):
        gt_boxes = make_gt_boxes(df, image_path)
        pred_boxes = get_predictions(inference_model, image_path, conf=args.conf, max_det=args.max_det)
        comparison = build_comparison_image(image_path, gt_boxes, pred_boxes)
        out_path = output_dir / f'sample_{idx:02d}_{image_path.stem}.png'
        cv2.imwrite(str(out_path), comparison)
        print(f'Saved sample comparison: {out_path} (predictions={len(pred_boxes)})')

    counts_path = output_dir / 'prediction_counts.txt'
    with open(counts_path, 'w', encoding='utf-8') as f:
        for image_path, count in zip(existing_images, prediction_counts):
            f.write(f'{Path(image_path).name}: {count}\n')
    print('Saved prediction counts:', counts_path)
    print('Generated report in:', output_dir)

if __name__ == '__main__':
    main()
