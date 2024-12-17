import tensorflow as tf

class MainModel:
    def __init__(self, window_size):
        """
        Initialize the MainModel class.

        Args:
            window_size (int): The size of the sliding window for time series data.
        """
        self.window_size = window_size

    def build_model(self):
        """
        Builds and returns a simple Dense Neural Network model for time series forecasting.

        Returns:
            model (tf.keras.Model): The compiled Keras model.
            init_weights (list): A list of the initial weights of the model.
        """
        # Reset states generated by Keras
        tf.keras.backend.clear_session()

        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.window_size,)),
            tf.keras.layers.Dense(30, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        # Get initial weights
        init_weights = model.get_weights()
        return model, init_weights

    def tuning_learn_rate(self, model, train_set, optimizer, loss):
        """
        Tunes the learning rate using an exponential increase.

        Args:
            model (tf.keras.Model): The Keras model.
            train_set (tf.data.Dataset): Training dataset.
            optimizer (tf.keras.optimizers): Optimizer to use.
            loss (str or tf.keras.losses.Loss): Loss function for the model.
            epochs (int): Number of epochs to tune learning rate.

        Returns:
            history (tf.keras.callbacks.History): Training history object.
        """
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-8 * 10**(epoch / 20) # Exponential growth
        )
        model.compile(
            optimizer=optimizer,
            loss=loss
        )
        history = model.fit(train_set, epochs=100, callbacks=[lr_scheduler])
        return history

    def train_model(self, model, train_set, optimizer, loss, epochs):
        """
        Trains the model on the provided dataset.

        Args:
            model (tf.keras.Model): The Keras model to train.
            train_set (tf.data.Dataset): Training dataset.
            optimizer (tf.keras.optimizers): Optimizer to use.
            loss (str or tf.keras.losses.Loss): Loss function for the model.
            epochs (int): Number of training epochs.

        Returns:
            history (tf.keras.callbacks.History): Training history object.
        """
        # Reset states generated by Keras
        tf.keras.backend.clear_session()
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["mae"]
        )
        history = model.fit(train_set, epochs=epochs)
        return history

    def predict_model(self, model, series, window_size, batch_size):
        """
        Generates predictions on a time series using the trained model.

        Args:
            model (tf.keras.Model): The trained Keras model.
            series (array-like): Input time series data.
            window_size (int): Size of the sliding window.
            batch_size (int): Batch size for prediction.

        Returns:
            forecast (np.array): Predicted values for the series.
        """
        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(series)

        # Window the data but only take those with the specified size
        dataset = dataset.window(window_size, shift=1, drop_remainder=True)

        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda w: w.batch(window_size))

        # Create batches of windows
        dataset = dataset.batch(batch_size).prefetch(1)

        # Get predictions on the entire dataset
        forecast = model.predict(dataset, verbose=0)
        return forecast