import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
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
        self.marked_pixels = None
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
            return
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
    

    def distribution(self,tensor):
        plt.hist(tensor.numpy().flatten(), bins=50)
        plt.show()




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

    def get_model_params(self):
        return {name: param.clone() for name, param in self.model.state_dict().items()}
    


    def backward(self):
        for name, param in self.model.named_parameters():        
            assert self.gradients.requires_grad, "gradients dont require gradient"
            assert param.requires_grad, f"{name} parameters dont require gradient"
            assert self.gradients.grad_fn is not None, "gradients were not part of graph"
            assert param.grad_fn is not None, f"{name} parameters were not part of graph"


            second_order_grad = autograd.grad(outputs=self.gradients, inputs=param, grad_outputs=torch.ones_like(self.gradients), retain_graph=True, create_graph=True)
            print(second_order_grad)




    def old_backward(self):
        old_parameters = self.get_model_params()
        # self.compute_saliency_map(self.input,self.label).show()

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
        statesdict = self.model.state_dict()
        image = transforms.ToPILImage()(self.marked_pixels)
        image.show()
        prev_layer = None
        for name, layer in list(self.model.named_children())+[("output",None)]:
            if name not in self.activations.keys():
                layer.zero_grad()
                prev_layer = name + ".weight"
                print(prev_layer)
                continue
            

            # print(f"{name} has the shape {self.activations[name]  .shape} on the activations, {self.changed_activations[name].shape} on the changed activations as well as {self.prev_layer_weights[name].shape} for the weights")
            difference_change=abs((self.activations[name]-self.changed_activations[name]).squeeze(0).unsqueeze(1)*torch.ones_like(self.prev_layer_weights[name]))#self.prev_layer_weights[name])
            percentile = (self.marked_pixels_count*3)/self.input.numel()
            print(f"percentile is {percentile}")
            limit = torch.quantile(difference_change, percentile).item()
            # limit = 0.01
            # self.distribution(difference_change)
            # print("The limit in this case was "+str(limit)) 
            difference_change[(difference_change>limit)]=0.0
            difference_change[(difference_change > 0)] = 1.0

            num_zeros = torch.sum(difference_change == 0.0).item()

            # Find the number of 1s
            num_ones = torch.sum(difference_change == 1.0).item()

            print(f"Number of 0s: {num_zeros}")
            print(f"Number of 1s: {num_ones}")
            # self.layer_factors[name]=difference_change# * self.marked_pixels_count/(self.width*self.height)  
            # self.layer_factors[name]= difference_change.squeeze(0).unsqueeze(1)
            weight_value = self.prev_layer_weights[name]*difference_change
            # print(f"shape is {self.prev_layer_weights[name].shape} or {difference_change.shape}")
            statesdict[prev_layer]= nn.Parameter(weight_value)

            # old_stuff =getattr(self.model, prev_layer.rstrip(".weight"))
            # old_weights = setattr(old_stuff, "weight",nn.Parameter(weight_value))
            # print(old_weights)

            

            # print(weight_value)


            # assert old_weights is not weight_value
            # print(f"i am trying to change {prev_layer} by {num_zeros} zero entries")

            try:
                layer.zero_grad()
            except:
                pass
            prev_layer = name + ".weight"
        
        
        self.model.load_state_dict(statesdict)
        new_parameters = self.get_model_params()
        
        print(f"The difference between the two models is {self.calculate_parameter_change(old_params=old_parameters,new_params=new_parameters)}")

        # Check for missing and unexpected keys
        """
        if missing_keys:
            print("Missing keys in state_dict:", missing_keys)
        if unexpected_keys:
            print("Unexpected keys in state_dict:", unexpected_keys)
            """
        self.model.zero_grad()
        self.compute_saliency_map(self.input, self.label).show()
        # self.improve_image_attention()
        self.marked_pixels = None




    def improve_image_attention(self):
        # self.adjust_model(False)
        loss = self.am_I_overfitting().item()
        prev_loss = loss
        threshold = loss *1.1
        while loss<= threshold and loss <=prev_loss:
            # self.adjust_model(True)
            # self.train_on_image()
            

            # self.adjust_model(False)

            while True:
                continue
            prev_loss = loss
            loss = self.am_I_overfitting().item()
        
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
        
        image_np = image.squeeze().numpy()
        # image_np = image.squeeze().detach().numpy()


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

    def adjust_model(self, backwards):
        print("State is "+str(backwards))
        print(len(list(self.model.named_parameters())))
        if not hasattr(self, 'original_layers'):
            self.original_layers = {}
        
        # Get the state_dict of the model
        state_dict = self.model.state_dict()

        for name, param in self.model.named_children():
            if name not in self.prev_layer_weights.keys():
                print(name)
                continue
        

            
            if backwards:
                print("here it gets to")
                # Save the current parameter tensor to restore later
                self.original_layers[name] = param
                
                # Modify the parameter tensor
                modified_param = self.prev_layer_weights[name] * self.layer_factors[name]
                
                # Update the state_dict directly
                state_dict[name] = modified_param
            else:
                print("it gets to this point why the fuck does it not work!!!!")
                # Restore the original parameter tensor
                state_dict[name] = self.original_layers[name]

        
        # Load the updated state_dict back into the model
        self.model.load_state_dict(state_dict)

    def calculate_parameter_change(self, old_params, new_params):
        total_change = 0
        for key in old_params.keys():
            old_param, new_param = old_params[key],new_params[key]
            change = torch.sum(torch.abs(old_param - new_param)).item()
            total_change += change
        return total_change

    def image_difference(self,image1,image2):
        import imagehash
        hash1 = imagehash.phash(image1)
        hash2 = imagehash.phash(image2)

        # Compute hamming distance (lower is more similar)
        hamming_distance = hash1 - hash2
        return hamming_distance
    
    def train_on_image(self):
        print("WARNING")
        weight_loss = 0
        exploration_loss = 0
        for name, layer in self.model.named_parameters():
            if name in self.original_layers.keys():
                weight_loss += layer.data*self.layer_factors[name]
                
            exploration_loss -= torch.sum(layer.data)


        output = self.model(self.input)
        loss = self.default_loss(output, self.target) + weight_loss # exploration_loss*0.1
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
            self.width, self.height = width,height

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
                newimage = image.clone()
                self.marked_pixels = torch.zeros((1, height, width))

                self.marked_pixels_count = 0  # Counter for marked pixels

                for x in range(height):
                    for y in range(width):
                        original_x = x * scaling_factor
                        original_y = y * scaling_factor
                        if drawn_image.getpixel((original_y, original_x)) == (255, 0, 1) and saliency_map.getpixel((original_y, original_x))[3] > 0:
                            r,g,b = image[0,x,y].item(), image[1,x,y].item(),image[2,x,y].item()
                            newimage[0,x,y],newimage[1,x,y], newimage[2,x,y]= r-1,g-1,b-1
                            self.marked_pixels[0,x,y] = 1
                            self.marked_pixels_count += 1

                
                # self.process_image(newimage).resize((new_width,new_height)).show()
    
                newimage=newimage.unsqueeze(0)               
                # drawn_image.show()
                if self.marked_pixels_count !=0:
                    self.real = False
                    # print("for the changed image the size is "+str(newimage.size()))
                    output = self.model(newimage)
                    if self.last_layer_linear:
                        self.changed_activations["output"]=output
                    self.real = True
                self.marked_pixels.squeeze(0)
                self.measure_impact()
                root.destroy()
                


        close_button = tk.Button(window, text="Continue", command=close_window)
        close_button.place(relx=0.5, rely=0.95, anchor=tk.CENTER)  # Place the button at the bottom center of the window
        root.mainloop()
        return self
    
    def measure_impact(self):

        if self.marked_pixels != None:
            print(f"Sum of marked pixels is {torch.sum(self.marked_pixels)}")
            print(f"Sum of gradients is {torch.sum(self.gradients)}")
            try:
                print(f"The weights that contributed to the marked pixels now make up {str(torch.sum(self.marked_pixels*abs(self.gradients)).item()/torch.sum(abs(self.gradients)).item())}")
            except:
                pass
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

        if input_data.grad is not None:
            input_data.grad.zero_()
        # Forward pass
        # print("for the real image the size is "+str(input_data.size()))
        outputs = self.model(input_data) 
        outputs.required_grad = True
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

        assert input_data.grad.grad_fn is not None, "input data  were not part of graph"

        self.gradients = input_data.grad.clone()# .detach()

        assert self.gradients.grad_fn is not None, "gradients were not part of graph after cloning"

        self.gradients.requires_grad = True

        self.measure_impact()

        # self.gradients[self.gradients < 0] = 0

        # Compute the importance weights based on gradients
        importance_weights = torch.mean(self.gradients, dim=(1, 2, 3), keepdim=True)

        # Weighted input data
        weighted_input_data = F.relu(input_data * importance_weights)

        # Normalize the weighted input data
        normalized_input = weighted_input_data / weighted_input_data.max()

        # Convert to numpy array
        # saliency_map_numpy = normalized_input.squeeze().cpu().detach().numpy()
        saliency_map_numpy = normalized_input.squeeze().cpu().clone().detach().numpy()
        
        assert self.gradients.grad_fn is not None, "gradients were not part of graph after numpy operation"
        
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