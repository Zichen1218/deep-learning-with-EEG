from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from src.config import CONFIG

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root="./data",download=True,train=True,transform=transform)
val_dataset = datasets.MNIST(root="./data",train=False,download=True,transform = transform)

train_dataloader = DataLoader(train_dataset,batch_size=CONFIG['batch_size'],shuffle=True,num_workers=0)
val_dataloader = DataLoader(val_dataset,batch_size=CONFIG['batch_size'],shuffle=False,num_workers=0)