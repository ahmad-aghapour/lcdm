from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CenterCropLongEdge:
    """Crop the image to a square using the shorter side."""

    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        input_dir: str,
        image_size: int = 256,
        file_ext: str = "png",
        max_images: int = 1000,
        center_crop: bool = False,
    ):
        self.input_dir = Path(input_dir)
        self.image_size = int(image_size)
        self.file_ext = file_ext
        self.max_images = int(max_images)
        self.center_crop = bool(center_crop)

        self.files = sorted(self.input_dir.glob(f"*.{self.file_ext}"))[: self.max_images]

        tfms = []
        if self.center_crop:
            tfms.append(CenterCropLongEdge())
        tfms.extend(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )
        self.transform = transforms.Compose(tfms)

    def __len__(self):
        return len(self.files)

    def load_tensor(self, path: str | Path):
        path = Path(path)
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        path = self.files[idx]
        return {
            "image": self.load_tensor(path),
            "name": path.name,
            "index": idx,
            "path": str(path),
            "id": path.name,
        }
