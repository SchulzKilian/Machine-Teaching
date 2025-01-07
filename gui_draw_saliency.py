import torch
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import numpy as np

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


    def get_user_markings(self, image, saliency_map, scaling_factor=1):
        root = tk.Tk()
        root.title("Mark Pixels")
        
        window = tk.Toplevel(root)
        window.title("Mark Pixels")
        window.geometry("800x800")

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
            pos_count = 0
            neg_count = 0

            for x in range(height):
                for y in range(width):
                    original_x = x * scaling_factor
                    original_y = y * scaling_factor
                    pixel = drawn_image.getpixel((original_y, original_x))
                    saliency_alpha = saliency_resized.getpixel((original_y, original_x))[3]

                    if pixel == (255, 0, 1) and saliency_alpha > 0:
                        marked_pixels[0, :, x, y] = 1
                        neg_count += 1
                    elif pixel == (0, 255, 1) and saliency_alpha > 0:
                        marked_pixels[0, :, x, y] = -1
                        pos_count += 1

            result["marked_pixels"] = marked_pixels
            result["pos_count"] = pos_count
            result["neg_count"] = neg_count
            root.destroy()

        close_button = tk.Button(window, text="Continue", command=close_window)
        close_button.place(relx=0.5, rely=0.95, anchor=tk.CENTER)

        root.mainloop()
        return result