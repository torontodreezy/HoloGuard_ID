"""MIDV-Holo annotations and a lazy PyTorch dataset for prepared images."""
from dataclasses import dataclass
from pathlib import Path
import json
import math
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent
MARKUP_DIR = ROOT.parent / 'dataset/map/markup/markup'
IMAGES_DIR = ROOT.parent / 'dataset/images'

@dataclass(frozen=True)
class Annotation:
    json_path: Path
    image_path: Path
    template_name: str
    points: tuple
    relative_image: Path


def get_image_path(json_path, markup_dir=MARKUP_DIR, images_dir=IMAGES_DIR):
    return Path(images_dir) / Path(json_path).relative_to(markup_dir).with_suffix('')


def load_annotations(markup_dir=MARKUP_DIR, images_dir=IMAGES_DIR, on_error=None):
    """Yield one record per template; report invalid JSON/templates via callback(path, message)."""
    markup_dir, images_dir = Path(markup_dir), Path(images_dir)
    if not markup_dir.is_dir() or not images_dir.is_dir():
        raise FileNotFoundError('Markup or images directory does not exist')
    def error(path, message):
        if on_error:
            on_error(path, str(message))
    for path in sorted(markup_dir.rglob('*.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            templates = data['document']['templates']
            if not isinstance(templates, dict) or not templates:
                raise ValueError('templates must be a non-empty object')
            image = get_image_path(path, markup_dir, images_dir)
            if not image.is_file():
                raise ValueError(f'image not found: {image}')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            error(path, exc)
            continue
        for name, template in templates.items():
            try:
                quad = template['template_quad']
                if not isinstance(quad, list) or len(quad) != 4 or any(not isinstance(p, list) or len(p) != 2 for p in quad):
                    raise ValueError('template_quad must contain four coordinate pairs')
                points = tuple(tuple(float(v) for v in p) for p in quad)
                if not all(math.isfinite(v) for p in points for v in p):
                    raise ValueError('coordinates must be finite')
                yield Annotation(path, image, name, points, image.relative_to(images_dir))
            except (ValueError, TypeError, KeyError) as exc:
                error(path, f'template {name}: {exc}')


class ResizePad:
    """Resize while retaining proportions and pad to a square."""
    def __init__(self, size):
        if not isinstance(size, int) or size < 1:
            raise ValueError('size must be a positive integer')
        self.size = size

    def __call__(self, image):
        from torchvision.transforms import functional as F
        w, h = image.size
        scale = self.size / max(w, h)
        image = F.resize(image, [max(1, round(h * scale)), max(1, round(w * scale))], antialias=True)
        w, h = image.size
        return F.pad(image, [(self.size-w)//2, (self.size-h)//2,
                             self.size-w-(self.size-w)//2, self.size-h-(self.size-h)//2])


def resnet18_transform(size=224):
    """Aspect-preserving resize and padding, then ImageNet normalization."""
    from torchvision import transforms
    return transforms.Compose([ResizePad(size), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class HologramDataset(Dataset):
    """Read prepared images lazily from a JSON split manifest."""
    def __init__(self, manifest, split, transform=None):
        self.manifest = Path(manifest).resolve()
        data = json.loads(self.manifest.read_text(encoding='utf-8'))
        if split not in ('train', 'validation', 'test'):
            raise ValueError('split must be train, validation or test')
        self.samples = [x for x in data['samples'] if x['split'] == split]
        self.transform = transform if transform is not None else resnet18_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image
        sample = self.samples[index]
        with Image.open(self.manifest.parent / sample['path']) as image:
            image = image.convert('RGB')
            image = self.transform(image)
        return image, sample['label']
