import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PunisherLoss(nn.Module):
    def __init__(self, threshold: int, training_dataset, default_loss = None):
        super(PunisherLoss, self).__init__()
        self.threshold = threshold
        self.marked_pixels = [] 
        self.training_dataset = training_dataset
        if not default_loss:
            self.default_loss = nn.CrossEntropyLoss()
        else:
            self.default_loss = default_loss



    def  forward(self, inputs, targets, epoch):
        if self.threshold == 0:
            return self.custom_loss_function(inputs, targets, self.training_dataset)
        elif epoch%self.threshold==0:
            return self.custom_loss_function(inputs, targets, self.training_dataset)
        else:
            return self.default_loss(inputs, targets)


    def backward(self):
        # Compute gradients of the loss
        self.zero_grad()  # Clear accumulated gradients
        self.backward()  # Backpropagation to compute gradients

        for param in self.parameters():

            param.data -= 0.01 * param.grad  


    def custom_loss_function(self, inputs, targets, training_dataset):
        root = tk.Tk()
        root.title("Mark Pixels")

        # Convert the input tensor to a NumPy array
        image_np = inputs.squeeze().numpy() * 255  # Assuming grayscale image, and un-normalizing

        # Convert the NumPy array to a PIL Image
        image_pil = Image.fromarray(image_np.astype(np.uint8))

        image_tk = ImageTk.PhotoImage(image_pil)

        def mark_pixels(event):
            x, y = event.x, event.y
            self.marked_pixels.append((x,y))
            image_draw.point((x, y), fill='red')

        # Create a new window to display the image for marking
        window = tk.Toplevel(root)
        window.title("Mark Pixels")
        canvas = tk.Canvas(window, width=image_np.shape[1], height=image_np.shape[0])
        canvas.pack()
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

        image_draw = ImageDraw.Draw(image_pil)
        canvas.bind("<Button-1>", mark_pixels)


        def close_window():
            root.destroy()


        close_button = tk.Button(root, text="Continue", command=close_window)
        close_button.pack()

        root.mainloop()
        return self

