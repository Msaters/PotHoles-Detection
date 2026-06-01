import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.modules.potholes_yolo import PotholeYoloModule
from src.data_utils import parse_xmls, PotholeDatasetYolo
import torch

ckpt = Path('logs/pothole_yolo_detection_v1_20260601_190124/checkpoints/epoch=4-step=225.ckpt')
print('checkpoint exists', ckpt.exists())
model = PotholeYoloModule.load_from_checkpoint(str(ckpt))
model.eval()

root = Path.home() / '.cache' / 'kagglehub' / 'datasets' / 'idanbaru' / 'annotated-potholes-with-severity-levels' / 'versions' / '1'
df = parse_xmls(str(root))
print('parsed rows', len(df), 'images', df['file'].nunique())

ds = PotholeDatasetYolo(df, img_size=640)
img, target = ds[0]
print('img', img.shape, 'target', target.shape)

with torch.no_grad():
    out = model(img.unsqueeze(0))

print('out type', type(out))
try:
    print('len', len(out))
except Exception as exc:
    print('len error', exc)
print('out repr', repr(out)[:1000])
if hasattr(out, '__len__') and len(out) > 0:
    print('out[0] type', type(out[0]))
    try:
        print('out[0] shape', out[0].shape)
    except Exception:
        print('out[0] repr', repr(out[0])[:500])
