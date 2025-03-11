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
from TestInterface import classify_image
from PIL import Image

doglabels = [34, 21, 6, 33, 8, 1, 24, 27, 12, 10, 28, 7]


# Define training parameters
batch_size = 4
learning_rate = 0.1
num_epochs = 10
data_size = 1600
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
    def __init__(self, dataset, num_samples=None, num_labels=None, selected_labels=None, transform=None):
        """
        Create a dataset subset based on labels and/or number of samples.
        
        Args:
            dataset: Original dataset
            num_samples: Maximum number of samples to include
            num_labels: How many different labels to include
            selected_labels: Which specific labels to include (overrides num_labels)
            transform: Optional transform to apply to the data
        """
        self.dataset = dataset
        self.transform = transform
        
        # Identify all unique labels and their indices
        label_to_indices = {}
        for i in range(len(dataset)):
            _, label = dataset[i]
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(i)
        
        all_labels = sorted(list(label_to_indices.keys()))
        print(f"DEBUG: All available labels: {all_labels}")
        
        # Determine which labels to include
        if selected_labels is not None:
            self.included_labels = set(selected_labels)
            print(f"DEBUG: Using selected labels: {self.included_labels}")
        elif num_labels is not None:
            self.included_labels = set(all_labels[:num_labels])
            print(f"DEBUG: Using first {num_labels} labels: {self.included_labels}")
        else:
            self.included_labels = set(all_labels)
            print(f"DEBUG: Using all labels: {self.included_labels}")
        
        # Create list of indices ONLY for the included labels
        self.indices = []
        for label in self.included_labels:
            if label in label_to_indices:
                self.indices.extend(label_to_indices[label])
        
        # Limit to num_samples if specified
        if num_samples is not None and num_samples < len(self.indices):
            self.indices = self.indices[:num_samples]
            
        # Verify our filtering worked correctly
        self._verify_label_counts()
    
    def _verify_label_counts(self):
        """Debug method to check label distribution"""
        counts = {}
        for idx in self.indices:
            _, label = self.dataset[idx]
            counts[label] = counts.get(label, 0) + 1
        
        print("DEBUG: Final label distribution in subset:")
        for label in sorted(counts.keys()):
            print(f"  Label {label}: {counts[label]} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.indices)} samples")
            
        original_idx = self.indices[idx]
        item = self.dataset[original_idx]
        
        # Apply transform if provided
        if self.transform is not None:
            image, label = item
            image = self.transform(image)
            return image, label
        
        return item

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
    train_dataset = SubsetDataset(train_dataset, selected_labels=[1,2,3,4,5,6])
    test_dataset = SubsetDataset(test_dataset,selected_labels = [1,2,3,4,5,6])






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


from collections import Counter

# Get label frequencies function
def print_top_labels(dataset, name):
    try:
        if isinstance(dataset, SubsetDataset):
            labels = [label for _, label in dataset.dataset][:len(dataset)]
        else:
            labels = dataset.labels[:len(dataset)]
    except AttributeError:
        labels = [label for _, label in dataset]
    
    label_counts = Counter(labels)
    print(f"\nTop 5 labels in {name}:")
    for label, count in label_counts.most_common(37):
        print(f"Label {label}: {count} occurrences")

print_top_labels(train_dataset, "training set")
print_top_labels(test_dataset, "testing set")
# Test each model
for i, model in enumerate(models):
    accuracy = test_model(model, test_loader)
    print(f'Accuracy of model {i+1}: {accuracy}%')
 
    torch.save(model.state_dict(), 'trained_model.pth')


    if input("\nDo you want to open the interface? (y/n): ").lower() == 'y':
        model.load_state_dict(torch.load('trained_model.pth'))
        model.eval()

        classify_image(model, transform)