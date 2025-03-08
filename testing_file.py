import torch
import torch.nn as nn
import torch.optim as optim
#from tfds.datasets import stanford_dogs
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
#from GuidedCNN import GuidedCNN
from StandardCNN import ResNet50, SimpleCNN, SimplestCNN
from MachinePunishment import PunisherLoss
import os
from PIL import Image

doglabels = [34, 21, 6, 33, 8, 1, 24, 27, 12, 10, 28, 7]


# Define training parameters
batch_size = 4
learning_rate = 0.1
num_epochs = 10
data_size = 1600
arg = "catsvdogs"

class CustomImageDataset(Dataset):
    def __init__(self, folder_path, num_samples, transform):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images from the specified folder
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):  # Check for image file extensions
                file_path = os.path.join(folder_path, filename)
                if filename.islower() and len(self.images) < num_samples // 2:
                    self.images.append(file_path)
                    self.labels.append(0)  # Label 0 for lowercase filenames
                elif filename.isupper() and len(self.images) < num_samples:
                    self.images.append(file_path)
                    self.labels.append(1)  # Label 1 for uppercase filenames
            
            # Stop if we have enough samples
            if len(self.images) >= num_samples:
                break

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_raw = self.images[idx]
        image = Image.open(image_raw)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


class SubsetDataset(Dataset):
    def __init__(self, dataset, num_samples):
        self.transform = transform
        self.dataset = dataset
        self.num_samples = num_samples


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):


        if arg !=  "catsvdogs":
            return self.dataset[idx]
        image, label = self.dataset[idx]
        if label in doglabels:
            return image, 0
        
        else:
            return image, 1
            


# Load data

if arg == "numbers":
    channels = 1
    classes = 10
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

elif arg == "cars":
    channels = 3
    classes = 196
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to match the input size of ResNet50
    transforms.ToTensor(),          # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])
    test_dataset = datasets.StanfordCars(root='./cars', split= "test", transform=transform, download = True)
    train_dataset = datasets.StanfordCars(root='./cars', split="train",transform=transform, download=True)

elif arg == "pets":
    channels = 3
    classes = 37
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to match the input size of ResNet50
    transforms.ToTensor(),          # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])
    test_dataset = datasets.OxfordIIITPet(root='./pets', split= "test", transform=transform, download = True)
    train_dataset = datasets.OxfordIIITPet(root='./pets',transform=transform, download=True)

elif arg == "catsvdogs":
    folder_path = './pets/oxford-iiit-pet/images'
    channels = 3
    classes = 2
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to match the input size of ResNet50
    transforms.ToTensor(),          # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
    
])
    full_dataset = CustomImageDataset(folder_path, num_samples=data_size, transform=transform)
    
    # Split into train/test (80-20 split)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])


if arg != "catsvdogs":
    train_dataset = SubsetDataset(train_dataset, data_size)
    test_dataset = SubsetDataset(test_dataset,data_size)






train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def decide_callback(epoch, number):
    return False
    if number in [1,2] and epoch % 3 == 0:
        return True
    else:
        return False

# Train each model
models = [SimplestCNN(classes,channels), ]
for model in models:
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = PunisherLoss(train_dataset,model, decide_callback, optimizer=optimizer)
    # criterion = nn.CrossEntropyLoss()
    model.train_model(train_loader, criterion, optimizer, num_epochs)
    # model.train_model(criterion, optimizer)

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

    torch.save(model.state_dict(), 'trained_model.pth')


    