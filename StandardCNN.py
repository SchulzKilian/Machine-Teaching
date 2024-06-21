import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchviz import make_dot

class ResNet50(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(ResNet50, self).__init__()

        self.resnet = models.resnet50(pretrained=True)

        # Modify the first convolutional layer to accept input_channels
        if input_channels != 3:
            self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the fully connected layer with a flexible one
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

    def train_model(model, train_loader, criterion, optimizer, num_epochs):
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                try:
                    assert not torch.any(torch.isnan(outputs)), "Model output contains NaN values."
                except:
                    print("The new loop fucked it up")
                    exit()
                loss = criterion(outputs, labels,epoch)
                loss.backward()
                try:
                    assert not torch.any(torch.isnan(model(inputs))), "Model output contains only NaN values."
                except:
                    print("The backward fucked it up")
                    exit()
                optimizer.step()
                try:
                    assert not torch.any(torch.isnan(model(inputs))), "Model output contains only NaN values."
                except:
                    print("The optimizer fucked it up")
                    exit()
                running_loss += loss.item()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')




class SimpleCNN(nn.Module):
    def __init__(self, classes, num_input_channels=3):
        super(SimpleCNN, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_classes = classes
        self.conv1 = nn.Conv2d(num_input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # Adjusted input size
        self.fc2 = nn.Linear(128, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)  # Properly flatten the output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    def train_model(model, train_loader, criterion, optimizer, num_epochs):
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels,epoch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
        make_dot(loss, params=dict(model.named_parameters())).render("computation_graph", format="png")




class SimplestCNN(nn.Module):
    def __init__(self, classes, num_input_channels=3):
        super(SimplestCNN, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_classes = classes
        self.conv1 = nn.Conv2d(num_input_channels, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 8 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(model, train_loader, criterion, optimizer, num_epochs):
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                print(outputs)
                loss = criterion(outputs, labels,epoch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')