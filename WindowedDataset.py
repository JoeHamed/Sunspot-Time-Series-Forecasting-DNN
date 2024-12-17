import tensorflow as tf

class WindowedDataset:
    def __init__(self, series, window_size, batch_size, buffer_size, cache_path=None):
        """
        Initialize the WindowedDataset class.

        Args:
            series (array-like): Input time series data (list, numpy array, or tensor).
            window_size (int): The size of each window for training.
            batch_size (int): Number of windows per batch.
            buffer_size (int): Size of the shuffle buffer.
            cache_path (str, optional): Path to cache the dataset on disk. Default is None (cache in memory).
        """
        self.series = series
        self.window_size = window_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.cache_path = cache_path

    def window_dataset(self):
        """
        Transforms the time series data into a windowed TensorFlow dataset.

        Returns:
        tf.data.Dataset: A windowed, batched, shuffled, and prefetch-optimized dataset.
        """
        # Convert the input series into a TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices(self.series)

        # Create sliding windows of size (window_size + 1)
        dataset = dataset.window(self.window_size + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.window_size + 1))

        # Split each window into input (features) and label (target)
        dataset = dataset.map(lambda window: (window[:-1], window[-1]))

        # Shuffle, batch, and prefetch for performance
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.batch(self.batch_size)

        # Cache for efficiency (in memory or disk)
        if self.cache_path:
            dataset = dataset.cache(self.cache_path)  # Cache to disk
        else:
            dataset = dataset.cache()  # Cache in memory

        dataset = dataset.prefetch(1)

        return dataset

