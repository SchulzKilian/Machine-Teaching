import torch
import torch.nn as nn
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import enum
class Mode(enum.Enum):
    NUKE = 'nuke'
    ADJUST = 'adjust'
    LOSS = 'loss'
    SCALE_UP = 'scale_up'








class PunisherLoss(nn.Module):
    red_color = "#FF0001"
    def __init__(self, threshold: int, training_dataset, model, default_loss = None, mode = Mode.SCALE_UP):
        super(PunisherLoss, self).__init__()
        self.threshold = threshold
        self.epochs = []
        self.loss = None
        self.format = None
        self.optimizer = optim.SGD(model.parameters(),0.001)
        self.val = None
        self.real = True
        self.input = None
        self.mode = mode
        self.validation_set = self.create_validation_set(training_dataset,100)
        self.label = None
        self.last_layer_linear=False
        self.changed_activations= {}
        self.layer_factors = {}
        self.original_layers = {}
        self.model = model
        self.radius = 15
        self.training_dataset = training_dataset
        if not default_loss:
            self.default_loss = nn.CrossEntropyLoss()
        else:
            self.default_loss = default_loss
        self.activations = {}  # Dictionary to store activations for each dense layer
        self.prev_layer_weights = self.get_previous_weights()
        # Define hooks
        def register_hooks(module,name):
            module.register_forward_hook(lambda module, input, output, name=name: self.forward_hook(module, input, output, name))


        previous = False
        prev_mod = None
        for name, module in self.model.named_children():
            if previous:     
                register_hooks(prev_mod, name)
                previous = False
            if isinstance(module, nn.Linear):
                prev_mod = module
                previous = True
        if previous:
            self.last_layer_linear = True

    def create_validation_set(self, dataset, num_samples):
        indices = torch.randperm(len(dataset))[:num_samples]
        images = torch.stack([dataset[i][0] for i in indices])
        labels = torch.tensor([dataset[i][1] for i in indices])
        return images, labels
    
    def get_previous_weights(self):
        prev_weights = {}
        previous = False
        for name, module in self.model.named_children():
            if previous:
                prev_weights[name]= weights
                previous = False
            if isinstance(module, nn.Linear):
                previous = True
                weights = module.weight
        if previous:
            prev_weights["output"]=weights

            

        

        return prev_weights
            

    def item(self):
        return self.loss.item()



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
        self.setradius(radius)

    

 

    def backward(self):
        print("called the backwards function")
        newlayers = {}
        if self.changed_activations=={}:
            self.default_loss.zero_grad()
            return self.default_loss

            

            """
            this is the correct code, i just want to test something
            print("the sum of changes is ")
            print(torch.sum(difference_change).item())
            weight_value = self.prev_layer_weights[item]*difference_change.squeeze(0).unsqueeze(1)*1000
            anti_overfitting_constant = weight_value.mean()
            newlayers[item]= (weight_value-anti_overfitting_constant)
            """
        for name, layer in self.model.named_children():
            if name not in self.activations.keys():
                layer.zero_grad()
                continue
            

            difference_change=abs(self.activations[name]-self.changed_activations[name])
            difference_change[difference_change>0.001]=0
            difference_change[(difference_change > 0)] = 1
            self.layer_factors[name]=difference_change
            amount = len(difference_change[difference_change<0.001])  
            self.layer_factors[name]= difference_change.squeeze(0).unsqueeze(1)
            self.original_layers[name] = layer.weight
            weight_value = self.prev_layer_weights[name]*difference_change.squeeze(0).unsqueeze(1)
            newlayers[name]= weight_value
            weight_sum_change = torch.sum(weight_value).item()-torch.sum(self.prev_layer_weights[name]).item()
            weight_value[weight_value!=0]+= weight_sum_change/amount
            layer.zero_grad()
            # layer.data = weight_value

        self.changed_activations = {}
        self.activations = {}
        self.adjust_model(False)
        loss = self.am_I_overfitting().item()
        prev_loss = loss
        threshold = loss *1.1
        while loss<= threshold and loss <=prev_loss:
            self.adjust_model(True)
            self.train_on_image()
            self.adjust_model(False)
            prev_loss = loss
            loss = self.am_I_overfitting().item()
            
        if loss > threshold:
            print("We went out of the image training because loss was "+str(loss) + " and threshold was "+ str(threshold))
        else:
            print("We went out of the image training because loss was "+str(loss) + " and previous loss was "+ str(prev_loss))


        self.compute_saliency_map(self.input, label=self.label).show()

        
    def invert_process_image(self,image_pil):
        image_np = np.array(image_pil)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:  # RGB image

            # Reverse scaling and clipping
            image_np = np.clip(image_np, 0, 255).astype(np.float32) / 255.0
            # Reverse mean subtraction and division by standard deviation
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image_np = (image_np - mean) / std
            return torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        elif len(image_np.shape) == 2:  # Grayscale image
            image_np = (image_np / 255).astype(np.float32)
            return torch.from_numpy((image_np * 255).astype(np.uint8))
        elif len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA image
            image_np = (image_np / 255).astype(np.float32)
            return torch.from_numpy((image_np * 255).astype(np.uint8))
        else:
            raise ValueError("Input image must have 2 or 3 dimensions.")

        
    def process_image(self,image):
        
        image_np = image.squeeze().detach().numpy()


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

        return Image.fromarray(image_np.astype(np.uint8)).convert('RGB')

    def adjust_model(self,backwards):

        for name, layer in self.model.named_parameters(): 
            if name not in self.activations.keys():
                continue  
            if backwards:
                self.original_layers[name]= layer.weight
                setattr(self.model,name,layer*self.layer_factors[name])
            else:
                setattr(self.model, name ,self.original_layers[name])



    def image_difference(self,image1,image2):
        import imagehash
        hash1 = imagehash.phash(image1)
        hash2 = imagehash.phash(image2)

        # Compute hamming distance (lower is more similar)
        hamming_distance = hash1 - hash2
        return hamming_distance
    
    def train_on_image(self):

        output = self.model(self.input)
        loss = self.default_loss(output, self.target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        



    def am_I_overfitting(self):
        outputs = self.model(self.validation_set[0])
        loss = self.default_loss(outputs,self.validation_set[1])
        print("the current loss is "+str(loss.item()))
        return loss




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


            image_pil = self.process_image(image)


            saliency_map = self.compute_saliency_map(image.unsqueeze(0), label)
            saliency_map.show()

            # Convert the PIL Image to a Tkinter-compatible format
            width, height = image_pil.size

            scaling_factor = 800//max(width,height)

            new_width = width * scaling_factor
            new_height = height * scaling_factor

            saliency_map = saliency_map.resize((new_width,new_height))

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

            # Prevent the saliency_map_tk from being garbage-collected prematurely
            canvas.image = blended_image_tk
            def reset(self):
                canvas.prev_x, canvas.prev_y = None, None


            def drag(event):
                x, y = event.x, event.y
                prev_x, prev_y = getattr(canvas, 'prev_x', None), getattr(canvas, 'prev_y', None)
                if prev_x is not None and prev_y is not None:
                    draw.line([prev_x, prev_y, x, y], fill=self.red_color, width=self.radius*2, joint="curve" )  # Draw line on the image buffer
                    canvas.create_line(prev_x, prev_y, x, y, fill=self.red_color, width=self.radius*2, smooth=True, splinesteps=10, capstyle='round', joinstyle='round')

                canvas.prev_x, canvas.prev_y = x, y

            canvas.bind("<B1-Motion>", lambda event: drag(event))
            canvas.bind("<ButtonRelease-1>", reset)



            def close_window():

                marked_pixels_count = 0  # Counter for marked pixels


                for x in range(height):
                    for y in range(width):
                        original_x = x * scaling_factor
                        original_y = y * scaling_factor
                        if drawn_image.getpixel((original_y, original_x)) == (255, 0, 1) and saliency_map.getpixel((original_y, original_x))[3] > 0:
                            r,g,b = image[0,x,y].item(), image[1,x,y].item(),image[2,x,y].item()
                            image[0,x,y],image[1,x,y], image[2,x,y]= r-1,g-1,b-1
                            # Get the corresponding coordinates in the original image


                            marked_pixels_count += 1
                self.process_image(image).resize((new_width,new_height))
                newimage= image.squeeze(0)

                # drawn_image.show()
                if marked_pixels_count !=0:
                    self.real = False
                    output = self.model(newimage)
                    if self.last_layer_linear:
                        self.changed_activations["output"]=output
                    self.real = True


                root.destroy()
                


        close_button = tk.Button(window, text="Continue", command=close_window)
        close_button.place(relx=0.5, rely=0.95, anchor=tk.CENTER)  # Place the button at the bottom center of the window

        root.mainloop()

        return self
    


    # Function to handle forward pass hook
    def forward_hook(self, module, input, output,name):
        # Store the output (activation) of the module
        if self.real:
            self.activations[name] = output.clone().detach()
        else:
            self.changed_activations[name] = output.clone().detach()


    def compute_saliency_map(self, input_data, label):
        self.model.eval()  # Set the model to evaluation mode
        input_data.requires_grad = True  # Set requires_grad to True to compute gradients
        self.label = label
        # Forward pass
        outputs = self.model(input_data) 
        if self.last_layer_linear:
            self.activations["output"]=outputs
        self.input = input_data 

        target = torch.zeros(outputs.size(), dtype=torch.float)
        target[0][label] = 1.0
        self.target = target
        self.loss = self.default_loss(outputs, target)

        # Backpropagate to compute gradients with respect to the output
        self.loss.backward(retain_graph=True)
        
        # Get the gradients with respect to the input
        self.gradients = input_data.grad.clone().detach()

        self.gradients[self.gradients < 0] = 0

        # Compute the importance weights based on gradients
        importance_weights = torch.mean(self.gradients, dim=(1, 2, 3), keepdim=True)

        # Weighted input data
        weighted_input_data = F.relu(input_data * importance_weights)

        # Normalize the weighted input data
        normalized_input = weighted_input_data / weighted_input_data.max()

        # Convert to numpy array
        saliency_map_numpy = normalized_input.squeeze().cpu().detach().numpy()
        
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