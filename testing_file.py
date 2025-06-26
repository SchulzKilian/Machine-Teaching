import torch
import torch.nn as nn
import torch.optim as optim
#from tfds.datasets import stanford_dogs
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
#from GuidedCNN import GuidedCNN
from torchvision.models import resnet34, resnet18
from collections import Counter
# from StandardCNN import ResNet50, SimpleCNN, SimplestCNN
from MachinePunishment import PunisherLoss
import os
from TestInterface import classify_image 
from PIL import Image
import matplotlib.pyplot as plt




# Define training parameters
batch_size = 20
learning_rate = 0.01
num_epochs = 25
data_size = float('inf')  # Use float('inf') to load all available data
arg = "pets"

class CustomImageDataset(Dataset):
    def __init__(self, folder_path, num_samples, transform):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images from the specified folder
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):  # Check for image file extensions
                file_path = os.path.join(folder_path, filename)
                # Your logic for splitting based on case and num_samples is preserved
                if filename[0].islower() and len(self.images) < num_samples // 2:
                    self.images.append(file_path)
                    self.labels.append(0)  # Label 0 for lowercase filenames
                elif filename[0].isupper() and len(self.images) < num_samples:
                    self.images.append(file_path)
                    self.labels.append(1)  # Label 1 for uppercase filenames
            
            # Stop if we have enough samples
            if len(self.images) >= num_samples:
                break

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_raw = self.images[idx]
        # MINIMAL CHANGE 1: Added .convert('RGB') for robustness.
        # This prevents crashes from non-RGB images without changing the logic.
        image = Image.open(image_raw).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

class SubsetDataset(Dataset):
    # YOUR SUBSETDATASET CLASS IS UNCHANGED
    def __init__(self, dataset, num_samples=None, num_labels=None, selected_labels=None, transform=None):
        self.dataset = dataset
        self.transform = transform
        
        label_to_indices = {}
        for i in range(len(dataset)):
            try: # Added try-except for safety
                _, label = dataset[i]
                if label not in label_to_indices:
                    label_to_indices[label] = []
                label_to_indices[label].append(i)
            except:
                continue

        all_labels = sorted(list(label_to_indices.keys()))
        
        if selected_labels is not None:
            self.included_labels = set(selected_labels)
        elif num_labels is not None:
            self.included_labels = set(all_labels[:num_labels])
        else:
            self.included_labels = set(all_labels)
        
        self.indices = []
        for label in self.included_labels:
            if label in label_to_indices:
                self.indices.extend(label_to_indices[label])
        
        if num_samples is not None and num_samples < len(self.indices):
            self.indices = self.indices[:num_samples]
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.indices)} samples")
        original_idx = self.indices[idx]
        item = self.dataset[original_idx]
        if self.transform is not None:
            image, label = item
            image = self.transform(image)
            return image, label
        return item

# Load data - YOUR LOGIC IS UNCHANGED
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.StanfordCars(root='./cars', split= "test", transform=transform, download = True)
    train_dataset = datasets.StanfordCars(root='./cars', split="train",transform=transform, download=True)

elif arg == "pets":
    channels = 3
    classes = 37
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.OxfordIIITPet(root='./pets', split= "test", transform=transform, download = True)
    train_dataset = datasets.OxfordIIITPet(root='./pets',transform=transform, download=True)

elif arg == "catsvdogs":
    folder_path = './pets/oxford-iiit-pet/images'
    channels = 3
    classes = 2
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = CustomImageDataset(folder_path, num_samples=data_size, transform=transform)
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# YOUR SUBSETTING LOGIC IS UNCHANGED
if arg != "catsvdogs":
    train_dataset = SubsetDataset(train_dataset)
    test_dataset = SubsetDataset(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def decide_callback(epoch, number):
    return False
    if number == 0:
        return True
    return False
    # Return true 
    


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Your model initialization list
models = [resnet18(weights=None)] # weights=None means training from scratch   

# This loop now contains the full training logic
for model in models:
    train_losses = []
    val_losses = []
    # Adapt final layer for your number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes)
    model.to(device) # Move model to the correct device

    # Setup optimizer and criterion as you had them
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = PunisherLoss(train_dataset,model, decide_callback, optimizer=optimizer)

    print("\n--- Starting Training ---")
    model.train() # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            # Your criterion requires the epoch, as in your original PunisherLoss
            loss = criterion(outputs, labels, epoch) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # --- Validation Pass ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels, epoch) # Using your criterion for validation loss
                running_val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(test_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Print both losses
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        model.train() # Switch back to training mode
    print("--- Training Finished ---")
    # --- Plotting the Losses ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('loss_plot.png')  




# YOUR TESTING LOGIC IS UNCHANGED
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def print_top_labels(dataset, name):
    try:
        if isinstance(dataset, SubsetDataset):
            labels = [label for _, label in dataset.dataset][:len(dataset)]
        else:
            labels = dataset.labels[:len(dataset)]
    except AttributeError:
        if isinstance(dataset, torch.utils.data.Subset):
            # Correctly handle PyTorch's Subset class
            labels = [dataset.dataset.labels[i] for i in dataset.indices]
        else:
            labels = [label for _, label in dataset] # Fallback for other iterable datasets
    
    label_counts = Counter(labels)
    print(f"\nLabel distribution in {name}:")
    for label, count in label_counts.most_common(37):
        print(f"Label {label}: {count} occurrences")

print_top_labels(train_dataset, "training set")
print_top_labels(test_dataset, "testing set")

for i, model in enumerate(models):
    accuracy = test_model(model, test_loader)
    print(f'Accuracy of model {i+1}: {accuracy:.2f}%')
 
    torch.save(model.state_dict(), 'trained_model.pth')

    try:
        if input("\nDo you want to open the interface? (y/n): ").lower() == 'y':
            model.load_state_dict(torch.load('trained_model.pth'))
            model.eval()
            classify_image(model, transform)
    except EOFError:
        print("\nSkipping interface in non-interactive mode.")
