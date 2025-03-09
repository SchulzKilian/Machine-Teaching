import tkinter as tk
from tkinterdnd2 import TkinterDnD
from PIL import Image
import torch

class DragDropInterface(TkinterDnD.Tk):
    def __init__(self, model, transform):
        super().__init__()
        self.model = model
        self.transform = transform
        
        self.title("Pet Breed Classifier")
        self.geometry("400x200")
        
        self.label = tk.Label(self, text="Drag & Drop Image Here", bg='lightgrey', width=50, height=10)
        self.label.pack(pady=20)
        
        self.result_label = tk.Label(self, text="Prediction: ")
        self.result_label.pack()
        
        self.label.drop_target_register('DND_Files')
        self.label.dnd_bind('<<Drop>>', self.process_drop)

    def process_drop(self, event):
        file_path = event.data.strip('{}')
        try:
            image = Image.open(file_path)
            tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(tensor)
                _, pred = torch.max(output, 1)
            
            self.result_label.config(text=f"Predicted Breed: {pred.item()}")
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}")