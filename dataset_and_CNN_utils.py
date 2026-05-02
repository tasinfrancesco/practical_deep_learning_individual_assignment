#Create the dataset class functions.
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class MultiLabelImageDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        classes,
        image_col="image",
        labels_col="labels",
        transform=None,
    ):
        self.csv_file = Path(csv_file)
        self.image_dir = Path(image_dir)
        self.classes = list(classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        self.image_col = image_col
        self.labels_col = labels_col
        self.transform = transform

        self.df = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.df)

    def _encode_labels(self, labels_value):
        labels = str(labels_value).split()

        target = torch.zeros(self.num_classes, dtype=torch.float32)

        for label in labels:
            if label in self.class_to_idx:
                target[self.class_to_idx[label]] = 1.0

        return target

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.image_dir / Path(str(row[self.image_col])).name
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        labels = self._encode_labels(row[self.labels_col])

        return image, labels
    

def make_data_loader(
    csv_file,
    image_dir,
    classes,
    batch_size=32,
    image_size=224,
    num_workers=8,
    shuffle=True,
    prefetch_factor=2,
):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = MultiLabelImageDataset(
        csv_file=csv_file,
        image_dir=image_dir,
        classes=classes,
        image_col="image",
        labels_col="labels",
        transform=transform,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=torch.cuda.is_available(),
    )

    return data_loader

def make_augmented_train_loader(
    csv_file,
    image_dir,
    classes,
    batch_size=32,
    image_size=224,
    num_workers=4,
    prefetch_factor=2,
):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.75, 1.0),
            ratio=(0.85, 1.15),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
    ])

    dataset = MultiLabelImageDataset(
        csv_file=csv_file,
        image_dir=image_dir,
        classes=classes,
        image_col="image",
        labels_col="labels",
        transform=train_transform,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        dataset,
        **loader_kwargs,
    )

    return train_loader

class FlexibleCNN(nn.Module):
        def __init__(self, num_layers,  num_filters, kernel_size, num_classes, batch_norm_included=False, image_size=224, dropout_rate=0.5):
            super().__init__()

            layers = []

            for _ in range(num_layers):
                    layers.append(
                        nn.Conv2d(
                            in_channels=3 if not layers else num_filters,
                            out_channels=num_filters,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                        )
                    )
                    if batch_norm_included:
                        layers.append(nn.BatchNorm2d(num_filters))
                    layers.append(nn.ReLU())
                    layers.append(nn.MaxPool2d(kernel_size=2))
                    
            self.features = nn.Sequential(*layers)

            final_size = image_size // (2**num_layers)
            flattened_dim = num_filters * final_size * final_size

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(flattened_dim, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x