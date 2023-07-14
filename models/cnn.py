import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(7*7*32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
    

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        flatten = out.reshape(out.size(0), -1)
        out = self.fc1(flatten)
        out = self.softmax(self.fc2(out))
        
        return out
    
if __name__ == "__main__":
    x = torch.randn([1, 1, 28, 28])
    model = CNN(10)
    print(model(x).size())

        