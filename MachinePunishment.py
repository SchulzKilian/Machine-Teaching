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
        return None


    def custom_loss_function(self, inputs, targets, training_dataset, amount=1):
        root = tk.Tk()
        root.title("Mark Pixels")

        # Create a new window to display the images for marking
        window = tk.Toplevel(root)
        window.title("Mark Pixels")
        window.geometry("800x600")  # Set the window size

        # Create a canvas to display the image
        canvas = tk.Canvas(window, bg="white")
        canvas.place(relwidth=1, relheight=1)

        # Load and display each image on the canvas
        for idx in np.random.choice(len(training_dataset), size=amount, replace=False):
            # Get the image and convert it to a NumPy array
            image, _ = training_dataset[idx]
            image_np = image.squeeze().detach().numpy() * 255  # Assuming grayscale image, and un-normalizing

            # Convert the NumPy array to a PIL Image
            image_pil = Image.fromarray(image_np.astype(np.uint8))

            # Convert the PIL Image to a Tkinter-compatible format
            image_tk = ImageTk.PhotoImage(image_pil)

            # Display the image on the canvas
            canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

            # Prevent the image_tk from being garbage-collected prematurely
            canvas.image = image_tk

            def mark_pixels(event, canvas=canvas):
                x, y = event.x, event.y
                self.marked_pixels.append((x, y))

            canvas.bind("<Button-1>", mark_pixels)

        def close_window():
            root.destroy()

        close_button = tk.Button(window, text="Continue", command=close_window)
        close_button.place(relx=0.5, rely=0.95, anchor=tk.CENTER)  # Place the button at the bottom center of the window

        root.mainloop()
        return self