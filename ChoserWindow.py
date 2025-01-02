import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
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