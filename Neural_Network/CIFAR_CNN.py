import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available():
    device = torch.device(device='cuda')
else:
    device = torch.device(device='cpu')

learning_rate = 0.001
batch_size = 4
total_epoch = 2


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = CIFAR10(root='./data', transform=transform, download=True, train=True)
test_dataset = CIFAR10(root='./data', transform=transform, train=False)

train_loader = DataLoader(dataset= train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset = test_dataset, shuffle=False, batch_size=batch_size)

classes = ('plance', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

examples = iter(train_loader)
samples, labels = next(examples)


def imshow(img):
    img = img / 1.0  # unnormalize if you normalized (not needed here)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # CxHxW → HxWxC
    plt.show()

imshow(torchvision.utils.make_grid(samples))

class ConvoNetwork(nn.Module):
    def __init__(self):
        super(ConvoNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 100)
        self.fc2 = nn.Linear(100, 84)
        self.fc3 = nn.Linear(84,10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x1 = self.pool(x1)
        x2 = self.relu(self.conv2(x1))
        x2 = self.pool(x2)

#       Flatten the layer before passing into linear layer
        x2 = x2.view(-1, 16*5*5)
        out1 = self.relu(self.fc1(x2))
        out1 = self.relu(self.fc2(out1))
        out1 = self.fc3(out1)
        return out1
    
model = ConvoNetwork().to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

total_steps = len(train_loader)
for epoch in range(total_epoch):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        criterion = loss(output, labels)

        optimizer.zero_grad()
        criterion.backward()

        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'epoch : {epoch + 1}, step= {i + 1} / {total_steps}, loss : {criterion.item():.5f}')

with torch.no_grad():
    n_correct = 0     
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)  
        _, predictions = torch.max(output, dim = 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            prediction = predictions[i]
            if(label == prediction):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    accuracy = 100.0 * n_correct / n_samples
    for i in range(10):
        class_accuracy = 100.0 * n_class_correct[i] / n_class_samples[i]

from PIL import Image
import torch.nn.functional as F
img = Image.open("bird.jpg")  

transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),
])

model = ConvoNetwork()  
model.load_state_dict(torch.load("cifar_model.pth"))
model.eval()

with torch.no_grad():
    output = model(img_tensor)
    probs = F.softmax(output, dim=1)
    predicted = output.argmax(1).item()

print(f"Predicted: {classes[predicted]}")