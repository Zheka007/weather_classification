from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

CLASS_TO_TARGET = {"rain": 0, "fog": 1, "snow": 2}

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted(
            os.listdir(image_dir),
            key=lambda x: int(os.path.splitext(x)[0])
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name

def get_transforms(mean, std):
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAutocontrast(p=0.3),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        "val": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }

def build_idx_to_target(class_to_idx):
    return {
        class_to_idx[class_name]: target_label
        for class_name, target_label in CLASS_TO_TARGET.items()
    }