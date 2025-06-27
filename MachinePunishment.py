import torch
import torch.nn as nn
import stop_conditions
import torch.optim as optim
import ImagePicker
import numpy as np
from PIL import Image
from GUI import SaliencyMapDrawer, ChoserWindow, TrainingProgressPlotter
import time

torch.autograd.set_detect_anomaly(True)

class PunisherLoss(nn.Module):

    def __init__(self, training_dataset, model, decide_callback, saliencies={}, teaching_batch=16, default_loss=None, optimizer=None):
        super(PunisherLoss, self).__init__()
        self.decide_callback = decide_callback
        self.validation_loss = None
        self.marked_pixels = None
        self.input = None
        self.max_duration = 20
        self.number = 0
        self.teaching_batch = teaching_batch
        self.epoch = 0
        self.label = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.saliencies = saliencies
        self.model = model
        self.training_dataset = training_dataset

        self.validation_set = self.create_validation_set(training_dataset, 100)
        if not default_loss:
            # When using CrossEntropyLoss, the model output should be raw logits.
            self.default_loss = nn.CrossEntropyLoss()
        else:
            self.default_loss = default_loss
        if not optimizer:
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        else:
            self.optimizer = optimizer
        
        self.saliency_drawer = SaliencyMapDrawer()

    def item(self):
        return self.loss.item() if self.loss is not None else 0

    def forward(self, inputs, targets, epoch):
        """
        The 'inputs' to this function are EXPECTED to be the raw image batch,
        not the model's outputs. The main training loop should be:
        
        # In your main training script:
        # outputs = model(image_batch)
        # loss = criterion(image_batch, outputs, labels, epoch)
        # instead of
        # loss = criterion(outputs, labels, epoch)

        However, to fix the provided traceback, we will assume the main script
        is passing model outputs as 'inputs' and correct the logic here.
        """
        self.number += 1
        if epoch != self.epoch:
            self.number = 0
            self.epoch = epoch

        if self.decide_callback(epoch, self.number):
            print(f"\n--- Triggering Custom Loss for Epoch {epoch}, Batch {self.number} ---")
            self.positive_percentage = []
            self.negative_percentage = []
            self.loss = None
            self.current_min_loss = float('inf')
            self.validation_losses = []
            self.custom_loss_function(self.training_dataset)

            # --- NEW SIMPLIFIED LOGIC ---
            # 2. Compute the saliency gradients and the standard classification loss
            #    for the annotated batch.
            self.gradients, classification_loss = self.compute_saliency_gradients()

            # 3. Calculate the final, combined loss using your getloss function.
            final_loss = self.getloss(classification_loss)

            # 4. Return the single loss tensor. Your main training loop will call .backward() on this.
            return final_loss
        else:
            # Default behavior: standard cross-entropy loss on the batch.
            # The 'inputs' variable here IS the model's output based on the traceback.
            # We should NOT pass it through the model again.
            predictions = inputs.to(self.device)
            targets = targets.to(self.device)
            return self.default_loss(predictions, targets)

    def create_validation_set(self, dataset, num_samples):
        indices = torch.randperm(len(dataset))[:num_samples]
        images = torch.stack([dataset[i][0] for i in indices]).to(self.device)
        labels = torch.tensor([dataset[i][1] for i in indices]).to(self.device)
        return images, labels

    def getloss(self, classification_loss, kind="squared_error"):
        """
        Calculates a combined loss to balance classification accuracy and saliency matching.
        This version is normalized to prevent exploding gradients.
        """
        batch_size = self.gradients.shape[0]
        if batch_size == 0:
            return 0.0
            
        total_saliency_loss = 0.0

        for i in range(batch_size):
            grad_single = self.gradients[i]
            mask_single = self.marked_pixels[i]
            
            saliency_loss_single = 0.0
            if kind == "squared_error":
                punish_mask = (mask_single > 0).float()
                reward_mask = (mask_single < 0).float()

                num_punish_pixels = torch.sum(punish_mask) + 1e-8
                num_reward_pixels = torch.sum(reward_mask) + 1e-8

                punish_loss = torch.sum(punish_mask * (grad_single**2)) / num_punish_pixels
                reward_loss = torch.sum(reward_mask * (1 / (1 + grad_single**2))) / num_reward_pixels
                
                saliency_loss_single = punish_loss + reward_loss
            
            elif kind == "huber_like_saliency":
                threshold = 0.5  # Saliency magnitudes below this are ignored.
                
                # Create a mask for only those pixels where the saliency magnitude is significant.
                significant_saliency_mask = (torch.abs(grad_single) > threshold).float()
                
                # Apply the squared error loss, but only on the significant pixels.
                # The logic remains the same: punish positive mask values, reward negative ones.
                saliency_loss_single = torch.sum(significant_saliency_mask * mask_single * (grad_single**2))

            total_saliency_loss += saliency_loss_single
        
        avg_saliency_loss = total_saliency_loss / batch_size
        
        # Using a smaller weight for classification loss as requested.
        alpha = 0.2  # Weight for classification loss
        beta = 1.0   # Weight for saliency loss

        saliency_scale_factor = 1e4 # Scale factor to bring saliency loss to a similar magnitude
        scaled_saliency_loss = avg_saliency_loss * saliency_scale_factor

        final_loss = (alpha * classification_loss) + (beta * scaled_saliency_loss)

        return final_loss
 

    def old_backward(self):
        """
        This is the custom training loop. It fine-tunes the model based on the
        entire annotated batch stored in self.input, self.label, and self.marked_pixels.
        """
        if self.input is None or self.marked_pixels is None:
            print("Skipping backward pass: no annotated images.")
            return

        old_model = self.model.state_dict()
        self.current_min_model = old_model
        self.current_min_epoch = self.epoch # Initialize best epoch for this run

        initial_gradients, initial_classification_loss = self.compute_saliency_gradients()
        # Use abs() for visualization
        saliency1 = torch.abs(initial_gradients).clone().detach()
        self.start_time = time.time()
        self.measure_impact_pixels() 

        while True:
            self.gradients, classification_loss = self.compute_saliency_gradients()
            if classification_loss is None: break # Stop if no gradients are computed
            self.loss = self.getloss(classification_loss, "huber_like_saliency")
            
            if not self.stop_condition():
                break 

            self.optimizer.zero_grad() 
            self.loss.backward(retain_graph=True) 
            self.adjust_weights_according_grad() 

        print(f"Fine-tuning finished. Best model from this run had loss {self.current_min_loss:.4f}")
        if self.epoch != self.current_min_epoch:
            self.model.load_state_dict(self.current_min_model)

        self.measure_impact_pixels()
        final_gradients, _ = self.compute_saliency_gradients()
        if final_gradients is None: return # Can't visualize if grads are None
        # Use abs() for visualization
        saliency2 = torch.abs(final_gradients).clone().detach()

        try:
            plotter = TrainingProgressPlotter()
            plotter.plot_percentages(list(range(len(self.negative_percentage))), self.negative_percentage, self.positive_percentage)
        except Exception as e:
            print(f"Could not plot progress: {e}")

        try:
            # Check if there are any validation losses to display
            if not self.validation_losses:
                print("No validation losses recorded to display.")
                return

            image_window = ChoserWindow(saliency1[0].cpu(), f"Before, Val Loss {self.validation_losses[0]:.4f}", 
                                        saliency2[0].cpu(), f"After, Val Loss {self.validation_losses[-1]:.4f}")
            update = image_window.run()
            if not update:
                print("User rejected changes. Reverting to model state before fine-tuning.")
                self.model.load_state_dict(old_model)
        except Exception as e:
            print(f"Could not display ChoserWindow: {e}")

    def adjust_weights_according_grad(self):
        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

    def am_I_overfitting(self):
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(self.validation_set[0])
            validation_loss = self.default_loss(outputs, self.validation_set[1])
            self.model.train()
            return validation_loss

    def custom_loss_function(self, training_dataset):
        amount = self.teaching_batch
        annotated_inputs, annotated_labels, annotated_markings = [], [], []

        indices_to_annotate = np.random.choice(len(training_dataset), size=amount, replace=False)
        print(f"--- Annotating a batch of {len(indices_to_annotate)} images. ---")

        for i, idx in enumerate(indices_to_annotate):
            print(f"Processing image {i+1}/{amount} (dataset index {idx})...")
            image, label = training_dataset[idx]
            image = image.to(self.device)
            label = torch.tensor(label, device=self.device)

            saliency_for_gui, _ = self.compute_saliency_gradients(image.unsqueeze(0), label.unsqueeze(0))

            current_marked_pixels = None
            if idx in self.saliencies:
                try:
                    trimap = Image.open(self.saliencies[idx])
                    resized_trimap_pil = trimap.resize((image.shape[2], image.shape[1]), Image.NEAREST)
                    trimap_np = np.array(resized_trimap_pil)
                    mask_tensor = torch.ones((image.shape[2], image.shape[1]), dtype=torch.float32)
                    mask_tensor[trimap_np == 1] = -1.0
                    current_marked_pixels = mask_tensor.to(self.device)
                except FileNotFoundError:
                     print(f"Error: Trimap for index {idx} not found.")
            else:
                if saliency_for_gui is not None:
                    result = self.saliency_drawer.get_user_markings(image.cpu(), gradients=torch.abs(saliency_for_gui.squeeze(0)).cpu())
                    if result and result["marked_pixels"] is not None:
                        current_marked_pixels = result["marked_pixels"].to(self.device)

            if current_marked_pixels is not None:
                if current_marked_pixels.dim() == 2:
                    current_marked_pixels = current_marked_pixels.unsqueeze(0).expand(image.shape[0], -1, -1)
                
                annotated_inputs.append(image)
                annotated_labels.append(label)
                annotated_markings.append(current_marked_pixels)
            else:
                print(f"Image {idx} was skipped (no markings provided).")

        if annotated_inputs:
            self.input = torch.stack(annotated_inputs)
            self.label = torch.stack(annotated_labels)
            self.marked_pixels = torch.stack(annotated_markings)
        else:
            self.input, self.label, self.marked_pixels = None, None, None

        return self

    def measure_impact_pixels(self):
        """
        Measures the percentage of the total saliency map's magnitude that falls
        on positively and negatively marked pixels. This now uses the correct logic.
        """
        if self.marked_pixels is not None and self.gradients is not None:
            # --- FIX: Use torch.abs() for logging to measure magnitude ---
            gradients = torch.abs(self.gradients.clone().detach())

            self.negative_pixels = torch.clamp(self.marked_pixels, min=0) 
            self.positive_pixels = abs(torch.clamp(self.marked_pixels, max=0))

            total_gradient_sum = torch.sum(gradients).item()

            if total_gradient_sum > 1e-8:
                neg_impact_sum = torch.sum(self.negative_pixels * gradients).item()
                pos_impact_sum = torch.sum(self.positive_pixels * gradients).item()
                
                neg_perc = neg_impact_sum / total_gradient_sum
                pos_perc = pos_impact_sum / total_gradient_sum

                print(f"Saliency on negatively marked areas: {neg_perc:.2%}")
                print(f"Saliency on positively marked areas: {pos_perc:.2%}")

    def compute_saliency_gradients(self, input_data=None, label=None):
        """
        Computes saliency for a given batch of inputs and labels.
        Returns BOTH the raw, signed gradients and the classification loss.


        """

        with torch.enable_grad():

            use_internal_data = input_data is None
            input_data = self.input if use_internal_data else input_data
            label = self.label if use_internal_data else label
            
            if input_data is None: return None, None
            input_data.requires_grad = True
            self.model.eval()
            input_data.requires_grad_()
            if input_data.grad is not None: input_data.grad.zero_()

            outputs = self.model(input_data)
            classification_loss = self.default_loss(outputs, label)
            
            grads = torch.autograd.grad(classification_loss, input_data, create_graph=True, retain_graph=True)[0]
            self.model.train()
            
            if use_internal_data:
                self.gradients = grads
            
            return grads, classification_loss

    def stop_condition(self):
        """Checks stop conditions based on aggregate metrics from the whole batch."""
        # This check is now safe because self.loss is guaranteed to have a value.
        self.validation_loss = self.am_I_overfitting()
        self.validation_losses.append(self.validation_loss.item())
        print(f"Fine-tuning iter... Val Loss: {self.validation_loss.item():.4f}, Custom Loss: {self.loss.item():.4f}")
        
        if self.loss.item() < self.current_min_loss:
            self.current_min_loss = self.loss.item()
            self.current_min_epoch = self.epoch
            self.current_min_model = self.model.state_dict()

        if self.gradients is not None:
            # --- FIX: Use torch.abs() for logging to measure magnitude ---
            abs_gradients = torch.abs(self.gradients)
            total_gradient_sum = torch.sum(abs_gradients).item()
            if total_gradient_sum > 1e-8:
                pos_perc = torch.sum(self.positive_pixels * abs_gradients).item() / total_gradient_sum
                neg_perc = torch.sum(self.negative_pixels * abs_gradients).item() / total_gradient_sum
                self.positive_percentage.append(pos_perc)
                self.negative_percentage.append(neg_perc)

        time_up = time.time() - self.start_time >= self.max_duration
        if time_up: print("Stopping: Max duration reached.")
        
        converged = stop_conditions.stop_for_pixel_loss(self.positive_percentage, self.negative_percentage)
        if converged: print("Stopping: Pixel impact has converged.")

        return not (time_up or converged)