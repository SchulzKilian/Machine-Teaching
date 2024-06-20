
class ImageWindow:
    def __init__(self, image1, image1_text, image2, image2_text):
        self.root = tk.Tk()
        self.root.title("Image Window")
        self.root.geometry("800x900")  # Set the window size

        # Store images and their texts
        self.image1 = image1
        self.image1_text = image1_text
        self.image2 = image2
        self.image2_text = image2_text
        
        # Create canvas to display images
        self.canvas = tk.Canvas(self.root, bg="white", width=800, height=800)
        self.canvas.pack()

        # Create a label for image text
        self.label = tk.Label(self.root, text="", font=("Helvetica", 16))
        self.label.pack()

        # Display the first image initially
        self.display_image(self.image1, self.image1_text)

        # Bind mouse click to image display
        self.canvas.bind("<Button-1>", self.on_image_click)

    def display_image(self, image, text):
        self.canvas.delete("all")  # Clear the canvas
        self.tk_image = ImageTk.PhotoImage(image.resize((800, 800)))  # Resize and convert image for Tkinter
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.label.config(text=text)  # Update label with the image text

    def on_image_click(self, event):
        # Toggle between image1 and image2
        if self.label.cget("text") == self.image1_text:
            self.display_image(self.image2, self.image2_text)
        else:
            self.display_image(self.image1, self.image1_text)

    def run(self):
        self.root.mainloop()