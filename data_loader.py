from torchvision import datasets, transforms
import torch.utils.data as data
import torch
import os

    
def load_training(root_path, data_dir, batch_size, kwargs):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    normalize = transforms.Normalize(mean=mean, std=std)
    
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
         ], p=0.8),
         transforms.RandomGrayscale(p=0.2),
         transforms.ToTensor(),
         normalize
        ])
    
    data = ImageInstance(root = root_path + data_dir, transform = transform)
    
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, 
                                               shuffle=True, drop_last=True, **kwargs)
    return train_loader


def load_testing(root_path, data_dir, batch_size, kwargs):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         normalize
        ])
    
    data = datasets.ImageFolder(root=root_path + data_dir, 
                                transform=transform)
    
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    
    return test_loader


class ImageInstance(data.Dataset):
    def __init__(self, root, transform=None):
        super(ImageInstance, self).__init__()
        self.imagenet_dir = os.path.join(root)
        self.dataset = datasets.ImageFolder(self.imagenet_dir, transform)

    def __getitem__(self, index):
        image_data = list(self.dataset.__getitem__(index))
        # important to return the index!
        data = [index] + image_data
        return tuple(data)

    def __len__(self):
        return len(self.dataset)
