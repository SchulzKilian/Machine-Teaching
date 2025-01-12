import torch
import torch.nn as nn
import stop_conditions
import torch.optim as optim

import numpy as np

from GUI import SaliencyMapDrawer, ChoserWindow, TrainingProgressPlotter

import time
torch.autograd.set_detect_anomaly(True)



class PunisherLoss(nn.Module):

    def __init__(self, training_dataset, model, decide_callback, default_loss = None, optimizer = None):
        super(PunisherLoss, self).__init__()
        self.decide_callback = decide_callback
        self.validation_loss = None
        self.marked_pixels = None
        self.input = None
        self.loss = None
        self.max_duration = 8
        self.positive_percentage = []
        self.negative_percentage = []
        self.validation_losses = []
        self.epoch = 0
        self.label = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.training_dataset = training_dataset#   .to(self.device)

        self.validation_set = self.create_validation_set(training_dataset,100)
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
            self.custom_loss_function(self.training_dataset)
            return self

        
        else:
            print("default")
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            return self.default_loss(inputs, targets)

        


    def create_validation_set(self, dataset, num_samples):
        indices = torch.randperm(len(dataset))[:num_samples]
        images = torch.stack([dataset[i][0] for i in indices]).to(self.device)
        labels = torch.tensor([dataset[i][1] for i in indices]).to(self.device)
        return images, labels
    


    def zero_grads(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()


    def getloss(self, kind="normalized"):
        if kind == "classic":
            return torch.sum(self.marked_pixels*torch.clamp(self.gradients, min = -0.5, max = 0.5))
        if kind == "normalized":
            return torch.sum(self.marked_pixels*torch.clamp(self.gradients, min = -0.5, max = 0.5))/ torch.sum((self.gradients))
            # so far normalized gives the most natural saliency maps
        

    def backward(self):
        if torch.all(self.marked_pixels == 0).item():
            return
        
        old_model = self.model.state_dict()
        self.current_min_model = old_model
        self.current_min_loss = None
        
        self.marked_pixels.requires_grad = True

        saliency1 = self.gradients.clone().detach()
        self.start_time = time.time()

        self.measure_impact_pixels()

        while self.stop_condition():

            self.gradients = self.compute_saliency_gradients()
            self.loss = self.getloss()
            self.loss.backward(retain_graph=True)
            self.adjust_weights_according_grad()
            
            self.optimizer.zero_grad()


            
        print(f"Best model was {self.current_min_epoch} we will load that version")
        print(f"The loss there is {self.current_min_loss} and not like here {self.loss.item()}")
        if self.epoch != self.current_min_epoch:
            self.model.load_state_dict(self.current_min_model)


        self.measure_impact_pixels()
        plotter = TrainingProgressPlotter()
        plotter.plot_percentages(list(range(self.epoch)), self.negative_percentage, self.positive_percentage)



        

        saliency2 = self.gradients.clone().detach()

        image_window = ChoserWindow(saliency1, f"Original Model, loss {self.validation_losses[0]}", saliency2, f"Modified Model, loss {self.validation_losses[-1]}")

        update = image_window.run()
        if not update:
            self.model.load_state_dict(old_model)
            

        
    def adjust_weights_according_grad(self):
        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
    


    def am_I_overfitting(self):

        self.model.eval()
        outputs = self.model(self.validation_set[0])
        validation_loss = self.default_loss(outputs,self.validation_set[1])
        self.model.train()
        return validation_loss

    def custom_loss_function(self, training_dataset, amount=1):
        for idx in np.random.choice(len(training_dataset), size=amount, replace=False):
            image, label = training_dataset[idx]

            image = image.to(self.device)
            label = torch.tensor(label).to(self.device)
            
            self.compute_saliency_gradients(image.unsqueeze(0), label)

            cpu_gradients = self.gradients.cpu()
            result = self.saliency_drawer.get_user_markings(image.cpu(), gradients=cpu_gradients)
            
            self.marked_pixels = result["marked_pixels"].to(self.device)


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

    def compute_saliency_gradients(self, input_data=None, label=None):
        if input_data is None:
            input_data = self.input
        else:
            self.input = input_data.to(self.device)

        if label is None:
            label = self.label
        else:
            self.label = label.to(self.device)

        self.model.eval()
        input_data.requires_grad_()
        
        if input_data.grad is not None:
            input_data.grad.zero_()

        outputs = self.model(input_data)
        target = torch.zeros(outputs.size(), dtype=torch.float).to(self.device)
        target[0][label] = 1.0
        
        first_loss = self.default_loss(outputs, target)
        self.gradients = torch.abs(torch.autograd.grad(first_loss, input_data, create_graph=True, retain_graph=True)[0])
        
        self.model.train()
        return self.gradients

    

    def stop_condition(self):
        
        self.validation_loss = self.am_I_overfitting()
        print(self.validation_loss.item())
        if self.loss:
            if self.current_min_loss is None:

                self.current_min_loss = self.loss.item()
                self.current_min_epoch = 1
            else:
                if self.loss.item() < self.current_min_loss:
                    self.current_min_epoch = self.epoch
                    self.current_min_model = self.model.state_dict()
                    self.current_min_loss = self.loss.item()

        # tracking progress
        self.positive_percentage.append(torch.sum(self.positive_pixels*self.gradients).item()/torch.sum(self.gradients).item())
        self.negative_percentage.append(torch.sum(self.negative_pixels*self.gradients).item()/torch.sum(self.gradients).item())
        self.validation_losses.append(self.validation_loss)
        self.epoch += 1


        condition = time.time() - self.start_time < self.max_duration
        
        return condition and not stop_conditions.stop_for_validation(self.validation_losses) and not stop_conditions.stop_for_pixel_loss(self.positive_percentage,self.negative_percentage)

        