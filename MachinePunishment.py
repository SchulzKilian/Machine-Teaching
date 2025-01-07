import torch
import torch.nn as nn
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
import matplotlib.pyplot as plt
import torch.optim as optim
from gui_draw_saliency import SaliencyMapDrawer
import numpy as np
import torch.nn.functional as F
from ChoserWindow import ChoserWindow
import random

import time
torch.autograd.set_detect_anomaly(True)



class PunisherLoss(nn.Module):
    red_color = "#FF0001"
    green_color = "#00FF01"
    def __init__(self, training_dataset, model, decide_callback, default_loss = None, optimizer = None):
        super(PunisherLoss, self).__init__()
        self.decide_callback = decide_callback
        self.loss = None
        self.marked_pixels = None
        self.saliency = None
        self.input = None
        self.validation_set = self.create_validation_set(training_dataset,100)
        self.label = None
        self.original_layers = {}
        self.model = model
        self.radius = 15
        self.training_dataset = training_dataset
        if not default_loss:
            self.default_loss = nn.CrossEntropyLoss()
        else:
            self.default_loss = default_loss
        if not optimizer:
            self.optimizer = optim.Adam(self.model.parameters(), lr = 0.01)
        else:
            self.optimizer = optimizer
        
        self.saliency_drawer = SaliencyMapDrawer()



    def item(self):
        return self.loss.item()
    

    def forward(self, inputs, targets, epoch, number):
        print(epoch)
        if self.decide_callback(epoch,number):
            return self.custom_loss_function(self.training_dataset)

        
        else:
            print("default")
            return self.default_loss(inputs, targets)

        


    def create_validation_set(self, dataset, num_samples):
        indices = torch.randperm(len(dataset))[:num_samples]
        images = torch.stack([dataset[i][0] for i in indices])
        labels = torch.tensor([dataset[i][1] for i in indices])
        return images, labels
    



    def setradius(self, radius):
        self.radius = radius



    def slider_changed(self, value):
        radius = int(value)
        self.setradius(radius)





    def zero_grads(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def getloss(self, kind="classic"):
        if kind == "classic":
            return torch.sum(self.marked_pixels*torch.clamp(self.gradients, min = -0.5, max = 0.5))
        if kind == "normalized":
            return torch.sum(self.marked_pixels*torch.clamp(self.gradients, min = -0.5, max = 0.5))/ torch.sum((self.gradients))
            # continue
        

    def backward(self):
        

        if self.marked_pixels_count + self.pos_marked_pixels_count  == 0:
            return
        old_model = self.model.state_dict()
        self.marked_pixels.requires_grad = True
        self.zero_grads()
        validation1 = self.am_I_overfitting().item()
        saliency1 = self.compute_saliency_map(self.input,self.label) 
        validation_loss= self.am_I_overfitting().item()
        current_loss = validation_loss
        real_loss = self.getloss("classic")
        loss = real_loss
        start_time = time.time()
        max_duration = 300
        positive_percentage = []
        negative_percentage = []
        validation_losses = []
        epochs = []
        epoch = 0
        self.measure_impact_pixels()
        print(f"loss is {validation_loss}")
        while current_loss < validation_loss*1.2 and (loss.item()  > real_loss.item()-abs(real_loss.item()/2) or True) and time.time() - start_time < max_duration:
            
            positive_percentage.append(torch.sum(self.positive_pixels*self.gradients).item()/torch.sum(self.gradients).item())
            negative_percentage.append(torch.sum(self.negative_pixels*self.gradients).item()/torch.sum(self.gradients).item())
            epochs.append(epoch)
            # self.optimizer.zero_grad()
            print(loss.item())
            loss.backward(retain_graph=True)
            # self.optimizer.step()
            self.adjust_weights_according_grad()
            validation_losses.append(current_loss)
            current_loss = self.am_I_overfitting().item()
            epoch +=1
            loss = self.getloss("classic")
            self.optimizer.zero_grad()
            _ = self.compute_saliency_map(self.input, self.label)
            
            
        print(f"loss is {validation_loss}")
        saliency2 = self.compute_saliency_map(self.input,self.label).show()
        self.measure_impact_pixels()
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, negative_percentage, marker='o', label='Percentage Negative')
        plt.plot(epochs, positive_percentage, marker='s', label='Percentage Positive')
        # plt.plot(epochs, validation_losses, marker='^', label='Validation')

        plt.title('Development Model')
        plt.xlabel('Epoch')
        plt.ylabel('')
        plt.legend()
        plt.grid(True)

        plt.show()
        if not current_loss < validation_loss*2:
            print("it went out for the validation loss")
        
        if not loss.item()  > real_loss.item()-abs(real_loss.item()/2):
            print("it went out for the custom loss")
            print(loss.item())
            print(real_loss.item())

        self.measure_impact_pixels()
        self.adjust_weights_according_grad()
        self.zero_grads()
        
        validation2 = self.am_I_overfitting().item()

        saliency2 = self.compute_saliency_map(self.input,self.label) 

        self.measure_impact_pixels()
        image_window = ChoserWindow(saliency1, f"Original Model, loss {validation1}", saliency2, f"Modified Model, loss {validation2}")
        blended_image = Image.blend(saliency1, saliency2, alpha=1.0)
        
        blended_image.show()
        update = image_window.run()
        if not update:
            self.model.load_state_dict(old_model)
            
        self.zero_grads()

        
    def adjust_weights_according_grad(self):
        self.optimizer.step()
    





    def am_I_overfitting(self):
        return torch.tensor(0.18)
        self.model.eval()
        outputs = self.model(self.validation_set[0])
        loss = self.default_loss(outputs,self.validation_set[1])
        self.model.train()
        return loss

    def custom_loss_function(self, training_dataset, amount=1):
        for idx in np.random.choice(len(training_dataset), size=amount, replace=False):
            image, label = training_dataset[idx]
            
            saliency_map = self.compute_saliency_map(image.unsqueeze(0), label)
            
            result = self.saliency_drawer.get_user_markings(image, saliency_map)
            
            self.marked_pixels = result["marked_pixels"]
            self.marked_pixels_count = result["neg_count"]
            self.pos_marked_pixels_count = result["pos_count"]

        return self


    def old_custom_loss_function(self, training_dataset, amount=1):
        """
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
            print(label)
            image_pil = self.process_image(image)


            self.saliency = self.compute_saliency_map(image.unsqueeze(0), label)
            self.saliency.show()

            # Convert the PIL Image to a Tkinter-compatible format
            width, height = image_pil.size
            self.width, self.height = width,height

            scaling_factor = 800//max(width,height)

            new_width = width * scaling_factor
            new_height = height * scaling_factor

            saliency_map = self.saliency.resize((new_width,new_height))

            image_pil = image_pil.resize((new_width, new_height))



            # Blend the saliency map with the input image
            blended_image = Image.alpha_composite(image_pil.convert('RGBA'), saliency_map)

            # Convert the blended image back to RGB mode
            blended_image = blended_image.convert('RGB')


            # Resize the image
            blended_image_tk = ImageTk.PhotoImage(blended_image.resize((new_width, new_height)))



            slider = tk.Scale(window, from_=0, to=80, length = 200,orient="horizontal", command=lambda value, canvas=canvas: self.slider_changed(value))
            # slider.pack_propagate(0)
            slider.pack(side="bottom",anchor="w", fill="y", padx=10, pady=10)
            slider.set(self.radius)

            # Display the saliency map on the canvas
            canvas.create_image(0, 0, anchor=tk.NW, image=blended_image_tk)



            # Create the right buffert
            drawn_image = Image.new("RGB", (new_width, new_height), "white")
            draw = ImageDraw.Draw(drawn_image)

            self.color = self.red_color



            # Add a button to switch between red and green
            self.switch_button = tk.Button(window, text="Switch Color", command=self.switch_color, bg = self.color)
            self.switch_button.pack(side="bottom")

            # Prevent the saliency_map_tk from being garbage-collected prematurely
            canvas.image = blended_image_tk
            def reset(self):
                canvas.prev_x, canvas.prev_y = None, None


            def drag(event):
                x, y = event.x, event.y
                prev_x, prev_y = getattr(canvas, 'prev_x', None), getattr(canvas, 'prev_y', None)
                if prev_x is not None and prev_y is not None:
                    draw.line([prev_x, prev_y, x, y], fill=self.color, width=self.radius*2, joint="curve" )  # Draw line on the image buffer
                    canvas.create_line(prev_x, prev_y, x, y, fill=self.color, width=self.radius*2, smooth=True, splinesteps=10, capstyle='round', joinstyle='round')

                canvas.prev_x, canvas.prev_y = x, y


            def on_enter(event):
                self.switch_button.configure(bg=self.color, activebackground=self.color)

            def on_leave(event):
                self.switch_button.configure(bg= self.color)

            self.switch_button.bind("<Enter>", on_enter)
            self.switch_button.bind("<Leave>", on_leave)
            self.switch_button.bind("<ButtonRelease-1>", on_enter)

            canvas.bind("<B1-Motion>", lambda event: drag(event))
            canvas.bind("<ButtonRelease-1>", reset)



            def close_window():
                newimage = image.clone()
                self.marked_pixels = torch.zeros((1, 3, height, width))


                self.marked_pixels_count = 0  # Counter for marked pixels
                self.pos_marked_pixels_count = 0 # Counter for encouraged pixels

                for x in range(height):
                    for y in range(width):
                        original_x = x * scaling_factor
                        original_y = y * scaling_factor
                        if drawn_image.getpixel((original_y, original_x)) == (255, 0, 1) and saliency_map.getpixel((original_y, original_x))[3] > 0:
                            r,g,b = image[0,x,y].item(), image[1,x,y].item(),image[2,x,y].item()
                            newimage[0,x,y],newimage[1,x,y], newimage[2,x,y]= r-1,g-1,b-1
                            self.marked_pixels[0,0,x,y],self.marked_pixels[0,1,x,y],self.marked_pixels[0,2,x,y] =1,1,1
                            self.marked_pixels_count += 1
                        if drawn_image.getpixel((original_y, original_x)) == (0, 255, 1) and saliency_map.getpixel((original_y, original_x))[3] > 0:
                            r,g,b = image[0,x,y].item(), image[1,x,y].item(),image[2,x,y].item()
                            newimage[0,x,y],newimage[1,x,y], newimage[2,x,y]= r-1,g-1,b-1
                            self.marked_pixels[0,0,x,y],self.marked_pixels[0,1,x,y],self.marked_pixels[0,2,x,y] = -1,-1,-1
                            self.pos_marked_pixels_count += 1

                
    

                # self.measure_impact()
                root.destroy()
                


        close_button = tk.Button(window, text="Continue", command=close_window)
        close_button.place(relx=0.5, rely=0.95, anchor=tk.CENTER)  # Place the button at the bottom center of the window
        root.mainloop()
        return self

    def switch_color(self):
        if self.color == self.red_color:
            self.color = self.green_color
        else:
            self.color = self.red_color
        self.switch_button.configure(bg=self.color)
    
    def get_color(self):
        return self.color
    
"""
        pass

    def measure_impact_pixels(self):

        if self.marked_pixels is not None:

            self.negative_pixels = torch.clamp(self.marked_pixels,min=0)
            self.positive_pixels = abs(torch.clamp(self.marked_pixels, max = 0))
            gradients = self.gradients.clone()
            gradients[gradients!=0]= 1
            try:
                print(f"The weights that contributed to the negatively marked pixels now make up {str(torch.sum(self.negative_pixels*self.gradients).item()/torch.sum(self.gradients).item())}")
                print(f"The weights that contributed to the positively marked pixels now make up {str(torch.sum(self.positive_pixels*self.gradients).item()/torch.sum(self.gradients).item())}")
            except:
                pass


        
    def compute_saliency_map(self, input_data, label):


        self.model.eval()  # Set the model to evaluation mode
        input_data.requires_grad_() # Set requires_grad to True to compute gradients
        self.label = label
        for param in self.model.parameters():
            param.requires_grad_()



        if input_data.grad is not None:
            input_data.grad.zero_()
        
 

        

        self.input = input_data

        outputs = self.model(input_data)
        target = torch.zeros(outputs.size(), dtype=torch.float)
        target[0][label] = 1.0
        self.target = target
        # outputs = self.model(input_data)
        self.loss = self.default_loss(outputs, target)



        gradients = torch.abs(torch.autograd.grad(self.loss, input_data, create_graph=True, retain_graph=True)[0])

        # gradients = torch.abs(torch.autograd.grad(self.loss, input_data, create_graph=True, retain_graph=True)[0])

        self.gradients = gradients

        for param in self.model.parameters():
            if param.grad is not None:
                continue
                param.grad.zero_()
        max_grad = gradients.max().detach()
        normalized_gradients = gradients.clone().detach() / (max_grad + 1e-8) 

        saliency_map_numpy = normalized_gradients.squeeze().cpu().detach().numpy()

        saliency_map_numpy = np.log1p(saliency_map_numpy)

        

        if len(saliency_map_numpy.shape) == 2:

            saliency_map_rgba = np.zeros((saliency_map_numpy.shape[0], saliency_map_numpy.shape[1], 4), dtype=np.uint8)
            safe_saliency_map = np.nan_to_num(saliency_map_numpy.copy(), nan=0.0, posinf=1.0, neginf=0.0)
            safe_saliency_map = np.clip(safe_saliency_map,0,1)
            for i in range(safe_saliency_map.shape[0]):
                non_zero_mask = safe_saliency_map[i] != 0
                if np.any(non_zero_mask):
                    non_zero_values = safe_saliency_map[i][non_zero_mask]
                    normalized_values = (non_zero_values - np.min(non_zero_values)) / (np.max(non_zero_values) - np.min(non_zero_values))
                    safe_saliency_map[i][non_zero_mask] = normalized_values
            # safe_saliency_map = np.clip(saliency_map_numpy, 0, 1)
            green_intensity = (safe_saliency_map * 255).astype(np.uint8)
            # print(green_intensity)
            alpha_channel = np.full_like(green_intensity, 128)
            # Set alpha channel to 0 where green intensity is zero
            alpha_channel[green_intensity == 0] = 0
            saliency_map_rgba[:, :, 1] = green_intensity
            saliency_map_rgba[:, :, 3] = alpha_channel
        elif len(saliency_map_numpy.shape) == 3:
            
            safe_saliency_map = np.nan_to_num(saliency_map_numpy, nan=0.0, posinf=1.0, neginf=0.0)

            # Clip values to the expected range (0 to 1)
            safe_saliency_map = np.clip(safe_saliency_map, 0, 1)
            safe_saliency_map = np.nan_to_num(saliency_map_numpy, nan=0.0, posinf=1.0, neginf=0.0)


            for i in range(safe_saliency_map.shape[0]):
                non_zero_mask = safe_saliency_map[i] != 0
                if np.any(non_zero_mask):
                    non_zero_values = safe_saliency_map[i][non_zero_mask]
                    normalized_values = (non_zero_values - np.min(non_zero_values)) / (np.max(non_zero_values) - np.min(non_zero_values))
                    safe_saliency_map[i][non_zero_mask] = normalized_values

            saliency_map_rgba = np.zeros((safe_saliency_map.shape[1], safe_saliency_map.shape[2], 4), dtype=np.uint8)
            
            green_intensity = (np.mean(safe_saliency_map, axis=0) * 255).astype(np.uint8)
            
            alpha_channel = np.full_like(green_intensity, 128)
            
            alpha_channel[green_intensity == 0] = 0

            saliency_map_rgba[:, :, 1] = green_intensity
            saliency_map_rgba[:, :, 3] = alpha_channel
            
        

        saliency_map_pil = Image.fromarray(saliency_map_rgba.copy(), 'RGBA')

        self.model.train()

        return saliency_map_pil
    



