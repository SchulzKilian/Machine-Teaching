import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageTk, ImageDraw, ImageOps
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch.nn.functional as F

class ActivationHook:
    def __init__(self):
        self.activations = []

    def __call__(self, module, input, output):
        self.activations.append(output)




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
        if epoch%self.threshold==0 and epoch not in self.epochs and epoch != 0:
            #return self.default_loss(inputs, targets)
            print("custom")
            self.epochs.append(epoch)
            return self.custom_loss_function(inputs, targets, self.training_dataset)
        
        else:
            print("default")
            return self.default_loss(inputs, targets)


    def slider_changed(self, value):
        radius = int(value)
        print("Radius:", radius)
        self.setradius(radius)


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
            image_np = image.squeeze().detach().numpy()  # Assuming grayscale image, and un-normalizing

            if len(image_np.shape) == 2:  # Grayscale image

                image_np = (image_np * 255).astype(np.uint8)

                image_pil = Image.fromarray(image_np, 'L')
            elif len(image_np.shape) == 3: 
                if image_np.shape[0] == 3:
                    # Convert from CxHxW to HxWxC
                    image_np = np.transpose(image_np, (1, 2, 0))
                    image_np = (image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                    image_pil = Image.fromarray(image_np, 'RGB')
                elif image_np.shape[2] == 3:
                    image_np = (image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                    image_pil = Image.fromarray(image_np)

            elif len(image_np.shape) == 4 and image_np.shape[0] == 4:  # RGBA image
                # Unnormalize the RGBA image
                image_np = (image_np.transpose(1, 2, 0) * 255).astype(np.uint8)
                # Convert the NumPy array to an RGBA PIL Image
                image_pil = Image.fromarray(image_np, 'RGBA') 
            else:
                raise ValueError("Input image must have 2 or 3 dimensions.")

            image_pil = Image.fromarray(image_np.astype(np.uint8)).convert('RGB')


            saliency_map = self.compute_saliency_map(image.unsqueeze(0), label)

            # Convert the PIL Image to a Tkinter-compatible format
            width, height = image_pil.size

            scaling_factor = 800//max(width,height)

            
            new_width = width * scaling_factor
            new_height = height * scaling_factor

            saliency_map = saliency_map.resize((new_width,new_height))

            image_pil = image_pil.resize((new_width, new_height))

            image_pil.show()


            

            # saliency_map.putalpha(128)  # Set alpha value to 128 (0.5 opacity)

            # Blend the saliency map with the input image
            blended_image = Image.alpha_composite(image_pil.convert('RGBA'), saliency_map)

            # Convert the blended image back to RGB mode
            blended_image = blended_image.convert('RGB')

            # Convert the PIL Image to a Tkinter-compatible format
            width, height = blended_image.size

            scaling_factor = 800//max(width,height)
            new_width = width * scaling_factor
            new_height = height * scaling_factor

            # Resize the image
            blended_image_tk = ImageTk.PhotoImage(blended_image.resize((new_width, new_height)))



            slider = tk.Scale(window, from_=0, to=20, orient="horizontal", command=lambda value, canvas=canvas: self.slider_changed(value))
            slider.pack(side="bottom", fill="x", padx=10, pady=10)

            # Display the saliency map on the canvas
            canvas.create_image(0, 0, anchor=tk.NW, image=blended_image_tk)



            # Create the right buffert
            drawn_image = Image.new("RGB", (new_width, new_height), "white")
            draw = ImageDraw.Draw(drawn_image)

            # Prevent the saliency_map_tk from being garbage-collected prematurely
            canvas.image = blended_image_tk



            def drag(event):
                x, y = event.x, event.y
                prev_x, prev_y = canvas.prev_x, canvas.prev_y
                if prev_x is not None and prev_y is not None:
                    draw.line([prev_x, prev_y, x, y], fill=self.red_color, width=self.radius*2, joint="curve" )  # Draw line on the image buffer
                    canvas.create_line(prev_x, prev_y, x, y, fill=self.red_color, width=self.radius*2, smooth=True, splinesteps=10, capstyle='round', joinstyle='round')

                canvas.prev_x, canvas.prev_y = x, y

            canvas.bind("<B1-Motion>", lambda event: drag(event))
            canvas.bind("<ButtonRelease-1>", canvas.prev_x, canvas.prev_y = None, None)
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
        
        # Forward pass
        outputs = self.model(input_data)  

        target = torch.zeros(outputs.size(), dtype=torch.float)
        target[0][label] = 1.0
        loss = self.default_loss(outputs, target)

        # Backpropagate to compute gradients with respect to the output
        loss.backward()
        
        # Get the gradients with respect to the input
        gradients = input_data.grad.clone().detach()

        # Set negative gradients to zero
        gradients[gradients < 0] = 0

        # Compute the importance weights based on gradients
        importance_weights = torch.mean(gradients, dim=(1, 2, 3), keepdim=True)

        # Weighted input data
        weighted_input_data = F.relu(input_data * importance_weights)

        # Normalize the weighted input data
        normalized_input = weighted_input_data / weighted_input_data.max()

        # Convert to numpy array

        saliency_map_numpy = normalized_input.squeeze().cpu().detach().numpy()
        print(str(saliency_map_numpy.shape))
        
        if len(saliency_map_numpy.shape) == 2:
            saliency_map_rgba = np.zeros((saliency_map_numpy.shape[0], saliency_map_numpy.shape[1], 4), dtype=np.uint8)
            green_intensity = (saliency_map_numpy * 255).astype(np.uint8)
            alpha_channel = np.full_like(green_intensity, 128)
            # Set alpha channel to 0 where green intensity is zero
            alpha_channel[green_intensity == 0] = 0
            # Assign green intensity and alpha channel values to RGBA image
            green_intensity = (saliency_map_numpy * 255).astype(np.uint8)
            saliency_map_rgba[:, :, 1] = green_intensity
            saliency_map_rgba[:, :, 3] = alpha_channel

        elif len(saliency_map_numpy.shape) == 3:
            saliency_map_rgba = np.zeros((saliency_map_numpy.shape[1], saliency_map_numpy.shape[2], 4), dtype=np.uint8)
            green_intensity = (saliency_map_numpy[1] * 255).astype(np.uint8)
            alpha_channel = np.full_like(green_intensity, 128)

            # Set alpha channel to 0 where green intensity is zero
            alpha_channel[green_intensity == 0] = 0
            # Repeat green_intensity and alpha_channel for each channel
            saliency_map_rgba[:, :, 1] = green_intensity
            saliency_map_rgba[:, :, 3] = alpha_channel


        # Create Pillow image
        saliency_map_pil = Image.fromarray(saliency_map_rgba, 'RGBA')
        saliency_map_pil.show()

        return saliency_map_pil

    def get_final_conv_layer(self):
        # Find the last convolutional layer in the model's architecture
        final_conv_layer = None
        modules = list(self.model.modules())  # Get all modules in the model
        for module in reversed(modules):  # Iterate over modules in reverse order
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
                final_conv_layer = module
                break  # Stop iteration after finding the first convolutional layer
        return final_conv_layer