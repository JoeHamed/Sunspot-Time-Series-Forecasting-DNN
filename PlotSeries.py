import matplotlib.pyplot as plt
import numpy as np

class PlotSeries:
    def plot_series(self, x, y, format="-", start=0, end=None,
                    title=None, xlabel=None, ylabel=None, legend=None):
        """
        Visualizes time series data

        Args:
          x (array of int) - contains values for the x-axis
          y (array of int or tuple of arrays) - contains the values for the y-axis
          format (string) - line style when plotting the graph
          label (string) - tag for the line
          start (int) - first time step to plot
          end (int) - last time step to plot
          title (string) - title of the plot
          xlabel (string) - label for the x-axis
          ylabel (string) - label for the y-axis
          legend (list of strings) - legend for the plot
        """

        # Setup dimensions of the graph figure
        plt.figure(figsize=(10, 6))

        # Check if there are more than two series to plot
        if type(y) is tuple:

            # Loop over the y elements
            for y_curr in y:
                # Plot the x and current y values
                plt.plot(x[start:end], y_curr[start:end], format)

        else:
            # Plot the x and y values
            plt.plot(x[start:end], y[start:end], format)

        # Label the x-axis
        plt.xlabel(xlabel)

        # Label the y-axis
        plt.ylabel(ylabel)

        # Set the legend
        if legend:
            plt.legend(legend)

        # Set the title
        plt.title(title)

        # Overlay a grid on the graph
        plt.grid(True)

        # Draw the graph on screen
        plt.show()

    def plot_learning_rate(self, history):
        # Define the learning rate array
        lrs = 1e-8 * (10 ** (np.arange(100) / 20))

        # Set the figure size
        plt.figure(figsize=(10, 6))

        # Set the grid
        plt.grid(True)

        # Plot the loss in log scale
        plt.semilogx(lrs, history.history["loss"])

        # Increase the tickmarks size
        plt.tick_params('both', length=10, width=1, which='both')

        # Set the plot boundaries
        plt.axis([1e-8, 1e-3, 0, 100])

        # Draw the graph on screen
        plt.show()