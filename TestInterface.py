import argparse
from PIL import Image
import torch



breeds = {2:"Yorkshire Terrier", 3:"Siamese", 4:"Russian Blue", 5:"Ragdoll", 6:"Persian", 7:"Maine",8:"British Shorthair", 9:"Bengal", 10:"Beagle", 11:"Basset Hound", 12:"Boxer", 13:"Chihuahua", 14:"Dachshund", 15:"Dalmatian", 16:"German Shepherd", 17:"Golden Retriever", 18:"Great Dane", 19:"Husky", 20:"Labrador Retriever", 21:"Poodle", 22:"Pug", 23:"Rottweiler", 24:"Schnauzer", 25:"Shih Tzu", 26:"Siberian Husky", 27:"Staffordshire Bull Terrier", 28:"Welsh Corgi", 29:"Yorkshire Terrier", 30:"Siamese", 31:"Russian Blue", 32:"Ragdoll", 33:"Persian"}
def classify_image(model, transform, image_path=None):
    if not image_path:
        image_path = input("Please input the path to the image")
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tensor)
            # print(outputs)
            _, pred = torch.max(outputs, 1)
            if pred.item()  in breeds:
                pred = breeds[pred.item()]
            
        print(f"Predicted Breed: {pred}")
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
    from torchvision.models import resnet18
    import torch.nn as nn
    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 37)
    model.load_state_dict(torch.load("trained_model.pth"))
# Replace with your trained model
    image_path= """/home/kilianschulz/Programming/Machine-Teaching/pets/oxford-iiit-pet/images/Maine_Coon_143.jpg"""


    
    # Run classification
    classify_image(model, transform, image_path)




