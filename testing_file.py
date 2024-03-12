import torch
import torch.nn as nn
import torch.optim as optim
#from tfds.datasets import stanford_dogs
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
#from GuidedCNN import GuidedCNN
from StandardCNN import ResNet50
from MachinePunishment import PunisherLoss





# Define training parameters
batch_size = 16
learning_rate = 0.001
num_epochs = 3
data_size = 5000



class SubsetDataset(Dataset):
    def __init__(self, dataset, num_samples):
        self.dataset = dataset
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.dataset[idx]

# Load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
subset_train_dataset = SubsetDataset(train_dataset, data_size)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Train each model
models = [ResNet50(10)]
for model in models:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = PunisherLoss(2,train_dataset,model)
    model.train_model(train_loader, criterion, optimizer, num_epochs)

# Define a function to test a model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Test each model
for i, model in enumerate(models):
    accuracy = test_model(model, test_loader)
    print(f'Accuracy of model {i+1}: {accuracy}%')
