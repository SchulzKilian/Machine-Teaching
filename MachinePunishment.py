import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk, ImageDraw, ImageFile
import tkinter as tk
import matplotlib.pyplot as plt
import torch.optim as optim
import imagehash
import numpy as np
import torch.nn.functional as F
import torch.func as func
from torch.func import functional_call
import random
import hashlib
import pickle
import time
import uuid


from torch import vmap


import torch.autograd as autograd
import enum
class Mode(enum.Enum):
    NUKE = 'nuke'
    ADJUST = 'adjust'
    LOSS = 'loss'
    SCALE_UP = 'scale_up'







class PunisherLoss(nn.Module):
    red_color = "#FF0001"
    green_color = "#00FF01"
    def __init__(self, threshold: int, training_dataset, model, decide_callback, default_loss = None, mode = Mode.SCALE_UP):
        super(PunisherLoss, self).__init__()
        self.threshold = threshold
        self.decide_callback = decide_callback
        self.loss = None
        self.format = None
        self.jacobian_func = func.jacrev(self.modelfunction, argnums = 1) 
        self.optimizer = optim.SGD(model.parameters(),0.001)
        self.frozen_layers = []
        self.fcall = lambda params, x: functional_call(self.model, params, x)
        self.marked_pixels = None
        self.real = True
        self.saliency = None
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

    def forward(self, inputs, targets, epoch, number):
        print(epoch)
        if self.decide_callback(epoch,number):
            return self.custom_loss_function(inputs, targets, self.training_dataset)

        
        else:
            print("default")
            return self.default_loss(inputs, targets)


    def slider_changed(self, value):
        radius = int(value)
        self.setradius(radius)

    def show_gradients(self):
        sample_size = 1000
        all_gradients = []

        for param in self.model.parameters():
            if param.grad is not None:
                all_gradients.extend(param.grad.view(-1).tolist())
        print("Ich hab so viele nullen:")
        
        print(len([i for i in all_gradients if i == 0.0]))
        print("und soviele nicht nullen")
        print(len([i for i in all_gradients if i != 0.0]))
        if len(all_gradients) > sample_size:
            pass
            all_gradients = random.sample(all_gradients, sample_size)
        # Plotting the gradients
        plt.figure(figsize=(10, 6))

        plt.scatter(range(len(all_gradients)), all_gradients, alpha=0.6, edgecolors='w', s=40)
        plt.title('Scatter Plot of Gradients')
        plt.ylabel('Gradient Value')
        plt.grid(True)
        plt.show()


    def get_model_params(self):
        return {name: param.clone() for name, param in self.model.state_dict().items()}
    
    def freeze_layers(self,layers=None, instance=None):

        if layers is not None:
            for layer in layers:
                layer_obj = getattr(self.model,layer)
                layer_obj.requires_grad = False
                self.frozen_layers.append(layer)
        
        if instance is not None:
            for name, layer in self.model.named_children():
                if isinstance(layer, instance):
                    layer.requires_grad = False
                    self.frozen_layers.append(name)


    def unfreeze_layers(self, specific_layers=None, instance=None):
        if specific_layers is None:
            for layer_name in self.frozen_layers:
                layer_obj = getattr(self.model, layer_name)
                layer_obj.requires_grad = True
            self.frozen_layers = []
        
        elif instance is not None:
            for name, layer in self.model.named_children():
                if isinstance(layer, instance):
                    layer.requires_grad = True
                    self.frozen_layers.remove(name)

        elif specific_layers is not None:
            for layer_name in specific_layers:
                layer_obj = getattr(self.model, layer_name)
                layer_obj.requires_grad = True
                self.frozen_layers.remove(name)





    def adjust_weights_according_grad(self):
        for name, param in self.model.named_parameters():
            if name.startswith("conv") and False:
                continue
            if param.grad is not None:
                torch.nn.utils.clip_grad_value_(param, clip_value=1.0)
                param.data = param.data - param.grad* 0.001
                param.grad.zero_()
    

    def zero_weights_with_non_zero_gradients(self, instance_type= None):
        for name, param in self.model.named_parameters():
            if not instance_type is None:
                if not name.startswith(instance_type):
                    continue
            print(name)
            if param.grad is None:
                continue

            if torch.isnan(param.grad).any():
                print("Gradient contains NaN values.")
                continue  # Skip this parameter
            sum1 =torch.sum(param.data)
            
            # percentile = 1-  (max(self.pos_marked_pixels_count,self.marked_pixels_count)*3)/self.input.numel()
            percentile = 0.2
            # print(f"percentile is {percentile}")
            if param.grad is not None:
                limit = torch.quantile(abs(param.grad), percentile).item()
                # print(f"limit is {limit}")
            if param.grad is not None and instance_type is None:
                param.data[abs(param.grad) > limit] = 0
                param.data[abs(param.grad) <= limit] *= 1/(1-percentile)
                # print(f"Amount of zeros before is {param.data.numel()} amount removed is {param.data[abs(param.grad) > limit].numel()}")
            
            elif param.grad is not None:
                param.data[abs(param.grad) > limit] = 0
                param.data[abs(param.grad) <= limit] *= 1/(1-percentile)
            sum2 =torch.sum(param.data)  
            # print(f"We went from {sum1} to {sum2}")
            param.requires_grad_()

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
        # while real_loss.item()
            _ = self.compute_saliency_map(self.input, self.label)
            positive_percentage.append(torch.sum(self.positive_pixels*self.gradients).item()/torch.sum(self.gradients).item())
            negative_percentage.append(torch.sum(self.negative_pixels*self.gradients).item()/torch.sum(self.gradients).item())
            epochs.append(epoch)
            loss.backward()
            self.adjust_weights_according_grad()
            validation_losses.append(current_loss)
            current_loss = self.am_I_overfitting().item()
            epoch +=1
            loss = self.getloss("classic")
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
        # self.zero_weights_with_non_zero_gradients("conv")
        # self.zero_weights_with_non_zero_gradients()
        self.adjust_weights_according_grad()
        self.zero_grads()
        
        validation2 = self.am_I_overfitting().item()

        saliency2 = self.compute_saliency_map(self.input,self.label) 
        print("important it is "+str(torch.sum(self.marked_pixels*self.gradients)))
        self.measure_impact_pixels()
        image_window = ChoserWindow(saliency1, f"Original Model, loss {validation1}", saliency2, f"Modified Model, loss {validation2}")
        blended_image = Image.blend(saliency1, saliency2, alpha=1.0)
        
        blended_image.show()
        update = image_window.run()
        print(f"update it {update}")
        if not update:
            self.model.load_state_dict(old_model)
            
        self.zero_grads()

        # self.show_gradients()
        

    def explicit_backward(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total parameters: {total_params}')

        hessian = func.jacrev(self.jacobian_func,  argnums = 0)

        # self.target.unsqueeze(0)
        
        

        # input_grad_grad = torch.autograd.grad(outputs=self.gradients, inputs=self.input, grad_outputs=torch.oneslike(self.gradients), retain_graph=True)[0]
        self.model.zero_grad()
        # self.gradients.requires_grad_()
        # weight_gradients = torch.autograd.grad(self.gradients, self.model.parameters(), grad_outputs=torch.ones_like(self.gradients),  create_graph=True)


        hessian_per_sample = vmap(hessian, in_dims=(None,0,None))(dict(self.model.named_parameters()), self.input, self.target)

        
        for name, param in self.model.named_parameters(): 
            print(f"input is {self.input.shape}")
            print(f"input is {dict(self.model.named_parameters())[name].shape}")
            print(f"parameter is {self.target.shape}")
            
            
            # hessian_matrix = hessian(dict(self.model.named_parameters()), self.input, self.target)
            # print(hessian_matrix)
            break
            continue
            assert self.gradients.requires_grad, "gradients dont require gradient"
            assert param.requires_grad, f"{name} parameters dont require gradient"
            assert self.gradients.grad_fn is not None, "gradients were not part of graph"
            assert param.grad_fn is not None, f"{name} parameters were not part of graph"



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
        self.saliency = self.compute_saliency_map(self.input, self.label)
        self.saliency.show()
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
        self.model.eval()
        outputs = self.model(self.validation_set[0])
        loss = self.default_loss(outputs,self.validation_set[1])
        # print("the current loss is "+str(loss.item()))
        self.model.train()
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
                # self.measure_impact()
                root.destroy()
                


        close_button = tk.Button(window, text="Continue", command=close_window)
        close_button.place(relx=0.5, rely=0.95, anchor=tk.CENTER)  # Place the button at the bottom center of the window
        root.mainloop()
        # print(f"Loss is {self.loss.item()} and my added loss is {torch.sum(nn.Sigmoid()(self.gradients)* self.marked_pixels)/torch.sum(self.marked_pixels)}")
        # return self.loss
        return self
        # return (torch.sum((self.gradients)* self.marked_pixels))/50 + self.loss
        # return torch.sum(abs(self.gradients)* self.marked_pixels)
    
    def switch_color(self):
        if self.color == self.red_color:
            self.color = self.green_color
        else:
            self.color = self.red_color
        self.switch_button.configure(bg=self.color)
    
    def get_color(self):
        return self.color
    
    def measure_impact(self):

        if self.marked_pixels is not None:
            # print(f"Sum of marked pixels is {torch.sum(self.marked_pixels)}")
            # print(f"Sum of gradients is {torch.sum(self.gradients)}")
            negative_pixels = torch.clamp(self.marked_pixels,min=0)
            positive_pixels = abs(torch.clamp(self.marked_pixels, max = 0))
            try:
                print(f"The weights that contributed to the negatively marked pixels now make up {str(torch.sum(negative_pixels*self.gradients).item()/torch.sum(self.gradients).item())}")
                print(f"The weights that contributed to the positively marked pixels now make up {str(torch.sum(positive_pixels*self.gradients).item()/torch.sum(self.gradients).item())}")
            except:
                pass
    def measure_impact_pixels(self):

        if self.marked_pixels is not None:
            # print(f"Sum of marked pixels is {torch.sum(self.marked_pixels)}")
            # print(f"Sum of gradients is {torch.sum(self.gradients)}")
            self.negative_pixels = torch.clamp(self.marked_pixels,min=0)
            self.positive_pixels = abs(torch.clamp(self.marked_pixels, max = 0))
            gradients = self.gradients.clone()
            gradients[gradients!=0]= 1
            try:
                print(f"The weights that contributed to the negatively marked pixels now make up {str(torch.sum(self.negative_pixels*self.gradients).item()/torch.sum(self.gradients).item())}")
                print(f"The weights that contributed to the positively marked pixels now make up {str(torch.sum(self.positive_pixels*self.gradients).item()/torch.sum(self.gradients).item())}")
            except:
                pass

    # Function to handle forward pass hook
    def forward_hook(self, module, input, output,name):
        # Store the output (activation) of the module
        if self.real:
            self.activations[name] = output.clone().detach()
        else:
            self.changed_activations[name] = output.clone().detach()

    def modelfunction(self,params ,x,target):
        output = self.fcall(params, x)
        # print(f"the shape for the fcall output is {output.shape} the real shape is {self.model(x).shape}")

        loss = self.default_loss(output, target)
        return loss
        
    def compute_saliency_map(self, input_data, label):
        # ImageFile.LOAD_TRUNCATED_IMAGES = False

        self.model.eval()  # Set the model to evaluation mode
        input_data.requires_grad_() # Set requires_grad to True to compute gradients
        self.label = label
        for param in self.model.parameters():
            param.requires_grad_()

        for name, param in self.model.named_parameters():
            if torch.all(param == 0):
                print(f"Layer {name} has all zero weights.")

        # state_dictionary = self.model.state_dict()

        if input_data.grad is not None:
            input_data.grad.zero_()
        
        # input_data_copy = input_data.clone()
        # input_data_copy.requires_grad = True

        # Forward pass
        # outputs = self.model(input_data) 
        params = dict(self.model.named_parameters())
        
        # print(input_data)

        

        self.input = input_data
        # functionalized_model = func.functionalize(self.modelfunction)
        # outputs = self.fcall(params,input_data)
        # print(outputs)
        outputs = self.model(input_data)
        target = torch.zeros(outputs.size(), dtype=torch.float)
        target[0][label] = 1.0
        self.target = target
        outputs = self.model(input_data)
        self.loss = self.default_loss(outputs, target)



        for name, param in self.model.named_parameters():
            if torch.all(param == 0):
                print(f"Layer {name} has all zero weights.")
        # assert self.loss == functionalized_model(input_data, target)
        # assert torch.equal(outputs ,self.model.forward(input_data))

                # Check if the operations are performed with torch.no_grad() context

        
        
        jacobian_x = self.jacobian_func(params,input_data, target)
        

        # here i compute the jacobian to have a backpropagatable way to get input.grad
        # jacobian = autograd.functional.jacobian(lambda x: self.default_loss(self.model.forward(x), target), input_data)


        # self.gradients = torch.abs(torch.matmul(jacobian_x, torch.ones_like(input_data)))        # print(jacobian_x)


        # Backpropagate to compute gradients with respect to the output
        # self.loss.backward(retain_graph=True)

        
        # Get the gradients with respect to the input
        
        # assert input_data_copy.grad_fn is not None, "input data  were not part of graph"

        # self.gradients = input_data.grad# .detach()


        # assert self.gradients.shape == gradients.shape

        """
        diff = torch.abs(self.gradients - gradients)

        # Compute the average difference
        avg_diff = torch.mean(diff)

        avg_val = (torch.mean(self.gradients)+torch.mean(gradients))/2

        print(f"Average difference as percentage of actual values is {avg_diff/avg_val}")"""

        # assert torch.equal(self.gradients, gradients)

        # assert self.gradients.grad_fn is not None, "gradients were not part of graph after cloning"

        # self.gradients.requires_grad = True

        # self.measure_impact()

        # self.gradients[self.gradients < 0] = 0

        # Compute the importance weights based on gradients

        # gradients = self.gradients

        gradients = torch.abs(torch.autograd.grad(self.loss, input_data, create_graph=True)[0])

        self.gradients = gradients

        # print(self.gradients)
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        max_grad = gradients.max().detach()
        normalized_gradients = gradients / (max_grad + 1e-8) 

        saliency_map_numpy = normalized_gradients.squeeze().cpu().detach().numpy()

        # Apply logarithmic scaling to enhance contrast
        saliency_map_numpy = np.log1p(saliency_map_numpy)
        saliency_map_numpy = (saliency_map_numpy / np.log1p(max_grad)) * 255
        # saliency_map_numpy = saliency_map_numpy.astype(np.uint8)
        

        # assert self.gradients.grad_fn is not None, "gradients were not part of graph after numpy operation"
        # saliency_map_numpy = (saliency_map_numpy - np.min(saliency_map_numpy)) / (np.max(saliency_map_numpy) - np.min(saliency_map_numpy))
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

            # print(safe_saliency_map)

            # Create RGBA array with shape (height, width, 4)
            saliency_map_rgba = np.zeros((safe_saliency_map.shape[1], safe_saliency_map.shape[2], 4), dtype=np.uint8)
            
            # Calculate green intensity from the second channel of the saliency map
            green_intensity = (np.mean(safe_saliency_map, axis=0) * 255).astype(np.uint8)
            
            # print(f"Value is {self.hash_value(green_intensity)}")
            # print(green_intensity)
            alpha_channel = np.full_like(green_intensity, 128)
            
            # print(green_intensity)


            # Set alpha channel to 0 where green intensity is zero
            alpha_channel[green_intensity == 0] = 0
            # Repeat green_intensity and alpha_channel for each channel
            saliency_map_rgba[:, :, 1] = green_intensity
            saliency_map_rgba[:, :, 3] = alpha_channel
            
        

        # Create Pillow image
        saliency_map_pil = Image.fromarray(saliency_map_rgba.copy(), 'RGBA')

        # phash = imagehash.phash(saliency_map_pil, hash_size=16)
        # print(f"phash is {phash} you potato")
        self.model.train()
        # print(f"hash is {hash(saliency_map_pil)}")
        return saliency_map_pil
    
    def hash_value(self, value):
        pickled = pickle.dumps(value)
        return hashlib.sha256(pickled).hexdigest()

    def get_final_conv_layer(self):
        # Find the last convolutional layer in the model's architecture
        final_conv_layer = None
        modules = list(self.model.modules())  # Get all modules in the model
        for module in reversed(modules):  # Iterate over modules in reverse order
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
                final_conv_layer = module
                break  # Stop iteration after finding the first convolutional layer
        return final_conv_layer


class ChoserWindow:
    def __init__(self, image1, image1_text, image2, image2_text):
        self.root = tk.Tk()
        self.root.title("Image Window")
        self.root.geometry("1600x900")  # Set the window size to accommodate two images side by side

        self.image1 = image1
        self.image1_text = image1_text
        self.image2 = image2
        self.image2_text = image2_text
        
        self.blended_image_tk1 = None
        self.blended_image_tk2 = None
        
        self.selection = None
        
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.canvas1 = tk.Canvas(self.frame, bg="white")
        self.canvas1.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.canvas2 = tk.Canvas(self.frame, bg="white")
        self.canvas2.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.label1 = tk.Label(self.root, text=self.image1_text, font=("Helvetica", 16))
        self.label1.pack(side=tk.LEFT, padx=10, pady=10)

        self.label2 = tk.Label(self.root, text=self.image2_text, font=("Helvetica", 16))
        self.label2.pack(side=tk.LEFT, padx=10, pady=10)

        self.root.update_idletasks()
        self.display_images()

        self.canvas1.bind("<Button-1>", lambda event: self.on_image_click(False))
        self.canvas2.bind("<Button-1>", lambda event: self.on_image_click(True))

    def display_images(self):
        width = self.root.winfo_width()
        height = self.root.winfo_height()

        # Example: assuming blended_image1 and blended_image2 are Image objects
        blended_image1 = self.blend_images(self.image1)
        blended_image2 = self.blend_images(self.image2)

        self.blended_image_tk1 = ImageTk.PhotoImage(blended_image1.resize((width // 2, height)))
        self.blended_image_tk2 = ImageTk.PhotoImage(blended_image2.resize((width // 2, height)))

        self.display_image(self.blended_image_tk1, self.canvas1)
        self.display_image(self.blended_image_tk2, self.canvas2)

    def blend_images(self, image):
        # Example function to blend an image (replace with your own logic)
        return image

    def display_image(self, image_tk, canvas):
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)

    def on_image_click(self, selection):
        self.selection = selection
        self.root.quit()
        self.root.destroy()  # Ensure the window is destroyed after quitting the main loop

    def run(self):
        self.root.mainloop()
        return self.selection