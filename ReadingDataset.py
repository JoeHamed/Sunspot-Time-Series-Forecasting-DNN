import csv
import numpy as np

class ReadingDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.sunspots = []
        self.time_stamps = []

    def read_dataset(self):
        """Reads the dataset and populates the sunspots and time_stamps lists"""
        with open(self.dataset_path, mode='r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)  # Skipping the column names line
            for row in reader:
                self.sunspots.append(float(row[2]))
                self.time_stamps.append(int(row[0]))

    def convert_np_array(self):
        """
        Converts the sunspots and time_stamps lists into Numpy arrays.

        Returns:
            series (np.array): Array of sunspot values.
            time (np.array): Array of corresponding time values.
        """
        series = np.array(self.sunspots)
        time = np.array(self.time_stamps)
        return series, time

    def split_dataset(self, split_time, series, time):
        """
        Splits the dataset into training and validation sets.

        Args:
            split_time (int or float): Index or proportion for the split.
            series (np.array): Sunspot data series.
            time (np.array): Corresponding time values.

        Returns:
            time_train, series_train, time_val, series_val: Sliced training and validation sets.
        """
        split_time = int(split_time)
        # Get the train set
        time_train = time[:split_time]
        series_train = series[:split_time]
        # Get the validation set
        time_val = time[split_time:]
        series_val = series[split_time:]
        return time_train, series_train, time_val, series_val