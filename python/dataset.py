import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


DATASET_ROOT = Path("/home/bladerunner/CLionProjects/engines/dataset")
IMAGES_ROOT = DATASET_ROOT / "images"

CLASS_NAMES = [
    "origins",
    "copy_without_holo",
    "pseudo_holo_copy",
    "photo_holo_copy",
    "photo_replacement",
]

CLASS_DIRS = [
    IMAGES_ROOT / "origins",
    IMAGES_ROOT / "fraud" / "copy_without_holo",
    IMAGES_ROOT / "fraud" / "pseudo_holo_copy",
    IMAGES_ROOT / "fraud" / "photo_holo_copy",
    IMAGES_ROOT / "fraud" / "photo_replacement",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class HologramDataset(Dataset):
    def __init__(self, transform=None, samples=None):
        self.transform = transform

        if samples is not None:
            self.samples = list(samples)
            return

        self.samples = []

        for label, directory in enumerate(CLASS_DIRS):
            if not directory.is_dir():
                raise FileNotFoundError(f"Directory not found: {directory}")

            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError("Dataset contains no images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    groups = {}

    for image_path, label in dataset.samples:
        document_type = image_path.parent.parent.name
        document_id = image_path.parent.name

        group_key = (document_type, document_id)

        if group_key not in groups:
            groups[group_key] = []

        groups[group_key].append((image_path, label))

    group_keys = sorted(groups.keys())

    rng = random.Random(seed)
    rng.shuffle(group_keys)

    train_end = int(len(group_keys) * train_ratio)
    val_end = train_end + int(len(group_keys) * val_ratio)

    train_keys = group_keys[:train_end]
    val_keys = group_keys[train_end:val_end]
    test_keys = group_keys[val_end:]

    def collect_samples(keys):
        samples = []

        for key in keys:
            samples.extend(groups[key])

        return samples

    train_dataset = HologramDataset(
        transform=dataset.transform,
        samples=collect_samples(train_keys),
    )

    val_dataset = HologramDataset(
        transform=dataset.transform,
        samples=collect_samples(val_keys),
    )

    test_dataset = HologramDataset(
        transform=dataset.transform,
        samples=collect_samples(test_keys),
    )

    return train_dataset, val_dataset, test_dataset



if __name__ == "__main__":
    dataset = HologramDataset()

    train, val, test = split_dataset(dataset)

    print(f"Total: {len(dataset)}")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test: {len(test)}")

    for name, subset in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
    ]:
        print(f"\n{name}:")
        for label, class_name in enumerate(CLASS_NAMES):
            count = sum(y == label for _, y in subset.samples)
            print(f"  {class_name}: {count}")
