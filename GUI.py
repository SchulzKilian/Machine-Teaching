import matplotlib.pyplot as plt
import torch
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import numpy as np

class TrainingProgressPlotter:
    def __init__(self):
        self.fig = None
        self.ax = None

    def plot_percentages(self, epochs, negative_percentage, positive_percentage, validation_losses=None):
        """
        Plot the training progress showing positive and negative percentages
        
        Args:
            epochs: List of epoch numbers
            negative_percentage: List of negative percentage values
            positive_percentage: List of positive percentage values
            validation_losses: Optional list of validation loss values
        """
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, negative_percentage, marker='o', label='Percentage Negative')
        plt.plot(epochs, positive_percentage, marker='s', label='Percentage Positive')
        
        if validation_losses is not None:
            plt.plot(epochs, validation_losses, marker='^', label='Validation')

        plt.title('Development Model')
        plt.xlabel('Epoch')
        plt.ylabel('')
        plt.legend()
        plt.grid(True)
        plt.show()




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

    def create_saliency_map_image(self, gradients):
        """Converts raw gradients into a PIL Image saliency map"""
        max_grad = gradients.max().detach()
        normalized_gradients = gradients.clone().detach() / (max_grad + 1e-8)
        saliency_map_numpy = normalized_gradients.squeeze().cpu().detach().numpy()
        saliency_map_numpy = np.log1p(saliency_map_numpy)

        if len(saliency_map_numpy.shape) == 2:
            saliency_map_rgba = np.zeros((saliency_map_numpy.shape[0], saliency_map_numpy.shape[1], 4), dtype=np.uint8)
            safe_saliency_map = np.nan_to_num(saliency_map_numpy.copy(), nan=0.0, posinf=1.0, neginf=0.0)
            safe_saliency_map = np.clip(safe_saliency_map, 0, 1)
            
            green_intensity = (safe_saliency_map * 255).astype(np.uint8)
            alpha_channel = np.full_like(green_intensity, 128)
            alpha_channel[green_intensity == 0] = 0
            
            saliency_map_rgba[:, :, 1] = green_intensity
            saliency_map_rgba[:, :, 3] = alpha_channel
        else:
            safe_saliency_map = np.nan_to_num(saliency_map_numpy, nan=0.0, posinf=1.0, neginf=0.0)
            safe_saliency_map = np.clip(safe_saliency_map, 0, 1)
            
            saliency_map_rgba = np.zeros((safe_saliency_map.shape[1], safe_saliency_map.shape[2], 4), dtype=np.uint8)
            green_intensity = (np.mean(safe_saliency_map, axis=0) * 255).astype(np.uint8)
            alpha_channel = np.full_like(green_intensity, 128)
            alpha_channel[green_intensity == 0] = 0
            
            saliency_map_rgba[:, :, 1] = green_intensity
            saliency_map_rgba[:, :, 3] = alpha_channel

        return Image.fromarray(saliency_map_rgba, 'RGBA')

    def display_images(self):
        width = self.root.winfo_width()
        height = self.root.winfo_height()

        # Example: assuming blended_image1 and blended_image2 are Image objects
        self.image1 = self.create_saliency_map_image(self.image1)
        self.image2 = self.create_saliency_map_image(self.image2)


        self.blended_image_tk1 = ImageTk.PhotoImage(self.image1.resize((width // 2, height)))
        self.blended_image_tk2 = ImageTk.PhotoImage(self.image2.resize((width // 2, height)))

        self.display_image(self.blended_image_tk1, self.canvas1)
        self.display_image(self.blended_image_tk2, self.canvas2)


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
    




class SaliencyMapDrawer:
    def __init__(self):
        self.red_color = "#FF0001"
        self.green_color = "#00FF01"
        self.color = self.red_color
        self.radius = 15
        self.width = None
        self.height = None

    def slider_changed(self, value):
        self.radius = int(value)


    def create_saliency_map_image(self, gradients):
        """Converts raw gradients into a PIL Image saliency map"""
        max_grad = gradients.max().detach()
        normalized_gradients = gradients.clone().detach() / (max_grad + 1e-8)
        saliency_map_numpy = normalized_gradients.squeeze().cpu().detach().numpy()
        saliency_map_numpy = np.log1p(saliency_map_numpy)

        if len(saliency_map_numpy.shape) == 2:
            saliency_map_rgba = np.zeros((saliency_map_numpy.shape[0], saliency_map_numpy.shape[1], 4), dtype=np.uint8)
            safe_saliency_map = np.nan_to_num(saliency_map_numpy.copy(), nan=0.0, posinf=1.0, neginf=0.0)
            safe_saliency_map = np.clip(safe_saliency_map, 0, 1)
            
            green_intensity = (safe_saliency_map * 255).astype(np.uint8)
            alpha_channel = np.full_like(green_intensity, 128)
            alpha_channel[green_intensity == 0] = 0
            
            saliency_map_rgba[:, :, 1] = green_intensity
            saliency_map_rgba[:, :, 3] = alpha_channel
        else:
            safe_saliency_map = np.nan_to_num(saliency_map_numpy, nan=0.0, posinf=1.0, neginf=0.0)
            safe_saliency_map = np.clip(safe_saliency_map, 0, 1)
            
            saliency_map_rgba = np.zeros((safe_saliency_map.shape[1], safe_saliency_map.shape[2], 4), dtype=np.uint8)
            green_intensity = (np.mean(safe_saliency_map, axis=0) * 255).astype(np.uint8)
            alpha_channel = np.full_like(green_intensity, 128)
            alpha_channel[green_intensity == 0] = 0
            
            saliency_map_rgba[:, :, 1] = green_intensity
            saliency_map_rgba[:, :, 3] = alpha_channel

        return Image.fromarray(saliency_map_rgba, 'RGBA')

    def switch_color(self):
        if self.color == self.red_color:
            self.color = self.green_color
        else:
            self.color = self.red_color
        self.switch_button.configure(bg=self.color)

    
    def process_image(self,image):

        print("process image called")
        
        image_np = image.squeeze().numpy()

        if len(image_np.shape) == 2:  # Grayscale image

            image_np = (image_np * 255).astype(np.uint8)

        elif len(image_np.shape) == 3: 
            if image_np.shape[0] == 3:
                # Convert from CxHxW to HxWxC
                image_np = np.transpose(image_np, (1, 2, 0))
                image_np = (image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)

            elif image_np.shape[2] == 3:
                image_np = (image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)


        elif len(image_np.shape) == 4 and image_np.shape[0] == 4:  # RGBA image
 
            image_np = (image_np.transpose(1, 2, 0) * 255).astype(np.uint8)

        else:
            raise ValueError("Input image must have 2 or 3 dimensions.")

        return Image.fromarray(image_np.astype(np.uint8)).convert('RGB')


    def get_user_markings(self, image, gradients, scaling_factor=1):
        root = tk.Tk()
        root.title("Mark Pixels")
        
        window = tk.Toplevel(root)
        window.title("Mark Pixels")
        window.geometry("800x800")

        saliency_map = self.create_saliency_map_image(gradients)
        image_pil = self.process_image(image)

        canvas = tk.Canvas(window, bg="white")
        canvas.place(relwidth=10, relheight=10)

        width, height = image_pil.size
        self.width, self.height = width, height

        scaling_factor = 800 // max(width, height)
        new_width = width * scaling_factor
        new_height = height * scaling_factor

        saliency_resized = saliency_map.resize((new_width, new_height))
        image_resized = image_pil.resize((new_width, new_height))

        blended_image = Image.alpha_composite(image_resized.convert('RGBA'), saliency_resized)
        blended_image = blended_image.convert('RGB')
        blended_image_tk = ImageTk.PhotoImage(blended_image)

        # Create slider
        slider = tk.Scale(window, from_=0, to=80, length=200, orient="horizontal",
                         command=lambda value: self.slider_changed(value))
        slider.pack(side="bottom", anchor="w", fill="y", padx=10, pady=10)
        slider.set(self.radius)

        canvas.create_image(0, 0, anchor=tk.NW, image=blended_image_tk)

        # Create drawing buffer
        drawn_image = Image.new("RGB", (new_width, new_height), "white")
        draw = ImageDraw.Draw(drawn_image)

        # Add color switch button
        self.switch_button = tk.Button(window, text="Switch Color", 
                                     command=self.switch_color, bg=self.color)
        self.switch_button.pack(side="bottom")

        canvas.image = blended_image_tk

        def reset():
            canvas.prev_x, canvas.prev_y = None, None

        def drag(event):
            x, y = event.x, event.y
            prev_x, prev_y = getattr(canvas, 'prev_x', None), getattr(canvas, 'prev_y', None)
            if prev_x is not None and prev_y is not None:
                draw.line([prev_x, prev_y, x, y], fill=self.color, 
                         width=self.radius*2, joint="curve")
                canvas.create_line(prev_x, prev_y, x, y, fill=self.color, 
                                 width=self.radius*2, smooth=True, splinesteps=10, 
                                 capstyle='round', joinstyle='round')
            canvas.prev_x, canvas.prev_y = x, y

        # Button event handlers
        canvas.bind("<B1-Motion>", lambda event: drag(event))
        canvas.bind("<ButtonRelease-1>", lambda event: reset())

        result = {"marked_pixels": None, "pos_count": 0, "neg_count": 0}

        def close_window():
            marked_pixels = torch.zeros((1, 3, height, width))


            for x in range(height):
                for y in range(width):
                    original_x = x * scaling_factor
                    original_y = y * scaling_factor
                    pixel = drawn_image.getpixel((original_y, original_x))
                    saliency_alpha = saliency_resized.getpixel((original_y, original_x))[3]

                    if pixel == (255, 0, 1) and saliency_alpha > 0:
                        marked_pixels[0, :, x, y] = 1
                    elif pixel == (0, 255, 1) and saliency_alpha > 0:
                        marked_pixels[0, :, x, y] = -1

            result["marked_pixels"] = marked_pixels
            root.destroy()

        close_button = tk.Button(window, text="Continue", command=close_window)
        close_button.place(relx=0.5, rely=0.95, anchor=tk.CENTER)

        root.mainloop()
        return result