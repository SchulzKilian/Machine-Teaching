import argparse
from PIL import Image
import torch

def classify_image(model, transform):
    image_path = input("Please input the path to the image")
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tensor)
            _, pred = torch.max(outputs, 1)
            
        print(f"Predicted Breed: {pred.item()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Example usage (replace with your model/transform)
    from torchvision import transforms
    
    # Model/transform setup
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to match the input size of ResNet50
    transforms.ToTensor(),          # Convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])
    model = torch.nn.Sequential()  # Replace with your trained model
    
    # Command-line interface
    parser = argparse.ArgumentParser(description='Classify pet breeds')
    parser.add_argument('image_path', type=str, help='Path to input image')
    args = parser.parse_args()
    
    # Run classification
    classify_image(model, transform, args.image_path)