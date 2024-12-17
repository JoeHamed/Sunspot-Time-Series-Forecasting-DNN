import tensorflow as tf
from PlotSeries import PlotSeries
from MainModel import MainModel
from ReadingDataset import ReadingDataset
from WindowedDataset import WindowedDataset

# Parameters
SPLIT_TIME = 3000
WINDOW_SIZE = 30
BUFFER_SIZE = 1000
BATCH_SIZE = 32

# Reading & Splitting the dataset
dataset = ReadingDataset("./data/Sunspots.csv")
dataset.read_dataset()
series, time = dataset.convert_np_array()
time_train, series_train, time_val, series_val= dataset.split_dataset(split_time=SPLIT_TIME, series=series, time=time)

# Create an instance of the PlotSeries class
plotter = PlotSeries()
# Visualise the data
plotter.plot_series(time, series, format="-", xlabel="Month", ylabel="Sunspots")

# Prepare the features and labels
# Create an instance of the WindowedDataset class
wd = WindowedDataset(series=series_train,
                     window_size=WINDOW_SIZE,
                     batch_size=BATCH_SIZE,
                     buffer_size=BUFFER_SIZE)
windowed_dataset = wd.window_dataset()

# Building the Model
# Create an instance of the MainModel class
main_model = MainModel(window_size=WINDOW_SIZE)
model, init_weights = main_model.build_model()
model.summary()

# Tuning the Learning Rate
tuning_history = main_model.tuning_learn_rate(model=model,
                                              train_set=windowed_dataset,
                                              optimizer=tf.keras.optimizers.SGD(momentum=0.9),
                                              loss=tf.keras.losses.Huber())
print(tuning_history.history.keys())

# Visualising the learning rate
plotter.plot_learning_rate(history=tuning_history)

# Adjusting the learning rate
tuned_learning_rate = float(input('New Learning Rate = '))

# Train the model with the tuned learning rate
history = main_model.train_model(model=model,
                                 train_set=windowed_dataset,
                                 optimizer=tf.keras.optimizers.SGD(learning_rate=tuned_learning_rate, momentum=0.9),
                                 loss=tf.keras.losses.Huber(),
                                 epochs=600)

# Reduce the original series
forecast_series = series[SPLIT_TIME-WINDOW_SIZE:-1]
# Use helper function to generate predictions
forecast = main_model.predict_model(model=model,
                                    series=forecast_series,
                                    window_size=WINDOW_SIZE,
                                    batch_size=BATCH_SIZE)
# Drop single dimensional axis
results = forecast.squeeze()
# Plot the results
plotter.plot_series(time_val, (series_val, results))
print(tf.keras.metrics.mae(series_val, results).numpy())
