import torch
import torch.nn as nn
from PIL import Image

import torch.optim as optim

import numpy as np

from GUI import SaliencyMapDrawer, ChoserWindow, TrainingProgressPlotter

import time
torch.autograd.set_detect_anomaly(True)



class PunisherLoss(nn.Module):

    def __init__(self, training_dataset, model, decide_callback, default_loss = None, optimizer = None):
        super(PunisherLoss, self).__init__()
        self.decide_callback = decide_callback
        self.loss = None
        self.marked_pixels = None
        self.input = None
        self.start_time = time.time()
        self.max_duration = 300
        self.positive_percentage = []
        self.negative_percentage = []
        self.validation_losses = []
        self.epoch = 0
        self.validation_set = self.create_validation_set(training_dataset,100)
        self.label = None
        self.model = model
        self.training_dataset = training_dataset
        if not default_loss:
            self.default_loss = nn.CrossEntropyLoss()
        else:
            self.default_loss = default_loss
        if not optimizer:
            self.optimizer = optim.SGD(self.model.parameters(), lr = 0.01)
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

        saliency1 = self.gradients.clone().detach()


        self.measure_impact_pixels()

        while self.stop_condition():
            # Compute fresh gradients with retain_graph
            self.gradients = self.compute_saliency_gradients()
            
            # Compute loss
            loss = self.getloss("classic")
            
            # Backward with retain_graph
            loss.backward(retain_graph=True)
            
            # Update weights
            with torch.no_grad():
                self.adjust_weights_according_grad()
            
            print("one managed")
            self.optimizer.zero_grad()


            
            

        self.measure_impact_pixels()
        plotter = TrainingProgressPlotter()
        plotter.plot_percentages(list(range(self.epoch)), self.negative_percentage, self.positive_percentage)



        

        saliency2 = self.gradients.clone().detach()

        image_window = ChoserWindow(saliency1, f"Original Model, loss {self.validation_losses[0]}", self.gradients, f"Modified Model, loss {self.validation_losses[-1]}")
        blended_image = Image.blend(saliency1, self.gradients, alpha=1.0)
        
        # blended_image.show()
        update = image_window.run()
        if not update:
            self.model.load_state_dict(old_model)
            

        
    def adjust_weights_according_grad(self):
        with torch.no_grad():
            self.optimizer.step()
    




    def padjust_weights_according_grad(self):
        for name, param in self.model.named_parameters():
            if name.startswith("conv") and False:
                continue
            if param.grad is not None:
                torch.nn.utils.clip_grad_value_(param, clip_value=1.0)
                param.data = param.data - param.grad* 0.001
                param.grad.zero_()
    

    def am_I_overfitting(self):

        self.model.eval()
        outputs = self.model(self.validation_set[0])
        loss = self.default_loss(outputs,self.validation_set[1])
        self.model.train()
        return loss

    def custom_loss_function(self, training_dataset, amount=1):

        for idx in np.random.choice(len(training_dataset), size=amount, replace=False):
            image, label = training_dataset[idx]
            
            self.compute_saliency_gradients(image.unsqueeze(0), label)
            result = self.saliency_drawer.get_user_markings(image, gradients=self.gradients)
            
            self.marked_pixels = result["marked_pixels"]
            self.marked_pixels_count = result["neg_count"]
            self.pos_marked_pixels_count = result["pos_count"]

        return self



    def measure_impact_pixels(self):

        if self.marked_pixels is not None:

            self.negative_pixels = torch.clamp(self.marked_pixels,min=0)
            self.positive_pixels = abs(torch.clamp(self.marked_pixels, max = 0))
            gradients = self.gradients.clone().detach()
            gradients[gradients!=0]= 1
            try:
                print(f"The weights that contributed to the negatively marked pixels now make up {str(torch.sum(self.negative_pixels*gradients).item()/torch.sum(gradients).item())}")
                print(f"The weights that contributed to the positively marked pixels now make up {str(torch.sum(self.positive_pixels*gradients).item()/torch.sum(gradients).item())}")
            except:
                pass

    def compute_saliency_gradients(self, input_data= None, label= None):
 

        if input_data is None:
            input_data = self.input
        else:
            self.input = input_data

        if label is None:
            label = self.label
        else:
            self.label = label

        self.model.eval()
        input_data.requires_grad_()
        
        if input_data.grad is not None:
            input_data.grad.zero_()

        outputs = self.model(input_data)
        target = torch.zeros(outputs.size(), dtype=torch.float)
        target[0][label] = 1.0
        
        loss = self.default_loss(outputs, target)
        self.gradients = torch.abs(torch.autograd.grad(loss, input_data, create_graph=True, retain_graph=True)[0])
        
        
        
        self.model.train()
        return self.gradients
    


    def stop_condition(self):
        
        loss = self.am_I_overfitting()
        print(loss.item())

        # tracking progress
        self.positive_percentage.append(torch.sum(self.positive_pixels*self.gradients).item()/torch.sum(self.gradients).item())
        self.negative_percentage.append(torch.sum(self.negative_pixels*self.gradients).item()/torch.sum(self.gradients).item())
        self.validation_losses.append(loss)
        self.epoch += 1


        condition = time.time() - self.start_time < self.max_duration
        return condition

        