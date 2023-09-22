from torchvision import datasets
from torch.utils.data import DataLoader

def get_data_loaders(train_dir, test_dir, val_dir, transform, batch_size=32):
    # Create datasets for training, testing, and validation
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    # Create data loaders to efficiently load and batch the data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Return the data loaders for training, testing, and validation
    return train_loader, test_loader, val_loader
