import matplotlib as plt


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