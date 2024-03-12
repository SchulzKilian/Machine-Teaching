import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PunisherLoss(nn.Module):
    red_color = "#FF0001"
    def __init__(self, threshold: int, training_dataset, model, default_loss = None):
        super(PunisherLoss, self).__init__()
        self.threshold = threshold
        self.epochs = []
        self.model = model
        self.marked_pixels = set()
        self.radius = 15
        self.training_dataset = training_dataset
        if not default_loss:
            self.default_loss = nn.CrossEntropyLoss()
        else:
            self.default_loss = default_loss

    def setradius(self, radius):
        self.radius = radius

    def forward(self, inputs, targets, epoch):
        print(epoch)
        if epoch%self.threshold==0 and epoch not in self.epochs and epoch is not 0:
            print("custom")
            self.epochs.append(epoch)
            return self.custom_loss_function(inputs, targets, self.training_dataset)
        
        else:
            print("default")
            return self.default_loss(inputs, targets)


    def backward(self):
        return None


    def custom_loss_function(self, inputs, targets, training_dataset, amount=1):
        root = tk.Tk()
        root.title("Mark Pixels")

        

        # Create a new window to display the images for marking
        window = tk.Toplevel(root)
        window.title("Mark Pixels")
        window.geometry("800x800")  # Set the window size

        # Create a canvas to display the image
        canvas = tk.Canvas(window, bg="white")
        canvas.place(relwidth=10, relheight=10)

        # Load and display each image on the canvas
        for idx in np.random.choice(len(training_dataset), size=amount, replace=False):
            image, label = training_dataset[idx]
            image_np = image.squeeze().detach().numpy() * 255  # Assuming grayscale image, and un-normalizing

            # Convert the NumPy array to a PIL Image
            image_pil = Image.fromarray(image_np.astype(np.uint8))

            # Calculate the saliency map for the current image
            saliency_map = self.compute_saliency_map(image.unsqueeze(0),label)  # Assuming 'model' is your trained model
            
            # Convert the saliency map to a PIL Image
            saliency_map_pil = Image.fromarray((saliency_map * 255).astype(np.uint8))

            # Convert the PIL Image to a Tkinter-compatible format
            width, height = saliency_map_pil.size

            scaling_factor = 800//max(width,height)
            new_width = width * scaling_factor
            new_height = height * scaling_factor
            print("height from "+ str(height)+ " to " +str(new_height))
            print("width from "+ str(width)+ " to " +str(new_width))
            # Resize the image
            saliency_map_tk = ImageTk.PhotoImage(saliency_map_pil.resize((new_width, new_height)))

            # Display the saliency map on the canvas
            canvas.create_image(0, 0, anchor=tk.NW, image=saliency_map_tk)

            # Create the right buffert
            drawn_image = Image.new("RGB", (new_width, new_height), "white")
            draw = ImageDraw.Draw(drawn_image)

            # Prevent the saliency_map_tk from being garbage-collected prematurely
            canvas.saliency_map_tk = saliency_map_tk







            def drag(event):
                x, y = event.x, event.y
                prev_x, prev_y = canvas.prev_x, canvas.prev_y
                if prev_x is not None and prev_y is not None:
                    draw.line([prev_x, prev_y, x, y], fill=self.red_color, width=self.radius*2, joint="curve" )  # Draw line on the image buffer
                    canvas.create_line(prev_x, prev_y, x, y, fill=self.red_color, width=self.radius*2, smooth=True, splinesteps=10, capstyle='round', joinstyle='round')

                canvas.prev_x, canvas.prev_y = x, y

            canvas.bind("<B1-Motion>", lambda event: drag(event))
            canvas.prev_x, canvas.prev_y = None, None  # Initialize previous coordinates

        def close_window():
            for x in range(drawn_image.width):
                for y in range(drawn_image.height):
                    if drawn_image.getpixel((x, y)) == (255, 0, 1):  # Check if the pixel is not white
                        self.marked_pixels.add((x, y))
            print(len(self.marked_pixels))
            root.destroy()

        close_button = tk.Button(window, text="Continue", command=close_window)
        close_button.place(relx=0.5, rely=0.95, anchor=tk.CENTER)  # Place the button at the bottom center of the window

        root.mainloop()
        return self
    






    def compute_saliency_map(self, input_data, label):
        self.model.eval()  # Set the model to evaluation mode
        input_data.requires_grad = True  # Set requires_grad to True to compute gradients
        print("The label is "+ str(label))
        output = self.model(input_data)
        target = torch.tensor([label])
        loss = self.default_loss(output, target)

        # Zero gradients to clear any previous gradients
        self.model.zero_grad()

        # Backpropagate to compute gradients
        loss.backward()
        # Compute the gradients with respect to the input
        saliency_map = input_data.grad.abs().squeeze().detach().cpu().numpy()

        # Normalize the gradients
        saliency_map /= saliency_map.max()
        
        return saliency_map