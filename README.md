# Sunspot Time Series Forecasting

This project implements a **time series forecasting** model using TensorFlow to predict **sunspot activity**. It incorporates data visualization, dataset preparation, model building, training, and evaluation.  

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dependencies](#dependencies)  
3. [File Descriptions](#file-descriptions)  
4. [How to Run](#how-to-run)  
5. [Code Walkthrough](#code-walkthrough)  
6. [Results](#results)  
7. [Visualization](#Visualization)  
---

## Project Overview

This project forecasts sunspot activity using a neural network model. Key features include:
- Data reading and processing using sliding windows.  
- Building and tuning a **Dense Neural Network** model.  
- Visualizing data, learning rates, and predictions.  
- Performance evaluation using **Mean Absolute Error (MAE)**.  

---

## Dependencies

The project requires the following libraries:
- Python 3.8 or above  
- TensorFlow 2.x  
- NumPy  
- Matplotlib  

To install the dependencies, run:

```bash
pip install tensorflow numpy matplotlib
```
## File Descriptions
This project contains the following files:

### 1. MainScript.py
- The main script that runs the entire pipeline.
- Key tasks include dataset reading, visualization, model building, learning rate tuning, training, and evaluation.

### 2. ReadingDataset.py
- Contains the ReadingDataset class for:
  - Reading the input dataset (CSV file).
  - Converting data into NumPy arrays.
  - Splitting the dataset into training and validation sets.

### 3. WindowedDataset.py
- Contains the WindowedDataset class for:
  - Creating a sliding window dataset for training.
  - Shuffling, batching, and optimizing the dataset.
  
### 4. MainModel.py
- Contains the MainModel class for:
  - Building a simple dense neural network.
  - Tuning the learning rate using an exponential scheduler.
  - Training the model.
  - Generating predictions on unseen data.

### 5. PlotSeries.py
- Contains the PlotSeries class for:
  - Plotting time series data.
  - Visualizing learning rate vs. loss for tuning purposes.

### 6. Sunspots.csv
- The dataset containing monthly sunspot observations.

---
## How to Run
Follow these steps to set up and run the project:

1. Clone the repository
Use the following command to clone the repository:
```bash
git clone https://github.com/your-username/sunspot-forecasting.git
cd Sunspot-Time-Series-Forecasting-DNN
```
2. Install the dependencies
Ensure the required libraries are installed:
```bash
pip install tensorflow numpy matplotlib
```
3. Place the dataset
- Download the Sunspots.csv dataset.
- Place it in the `data` folder.

4. Run the project
Execute the main script:
```bash
python MainScript.py
```

---

## Code Walkthrough
1. Dataset Reading and Splitting
The `ReadingDataset` class reads the Sunspots.csv file and splits it into training and validation sets:
```bash
dataset = ReadingDataset("./data/Sunspots.csv")
dataset.read_dataset()
series, time = dataset.convert_np_array()
time_train, series_train, time_val, series_val = dataset.split_dataset(split_time=3000, series=series, time=time)
```
2. Data Visualization
The `PlotSeries` class visualizes the time series data:
```bash
plotter = PlotSeries()
plotter.plot_series(time, series, xlabel="Month", ylabel="Sunspots")
```
3. Data Preparation
The `WindowedDataset` class prepares the training data using sliding windows:
```bash
wd = WindowedDataset(series_train, window_size=30, batch_size=32, buffer_size=1000)
windowed_dataset = wd.window_dataset()
```
4. Model Building
The `MainModel` class creates a simple dense neural network:
```bash
main_model = MainModel(window_size=30)
model, _ = main_model.build_model()
model.summary()
```
5. Learning Rate Tuning
The learning rate is tuned and visualized:
```bash
tuning_history = main_model.tuning_learn_rate(model, windowed_dataset, tf.keras.optimizers.SGD(momentum=0.9), loss=tf.keras.losses.Huber())
plotter.plot_learning_rate(tuning_history)
```
6. Model Training and Prediction
Train the model and generate predictions:
```bash
history = main_model.train_model(model, windowed_dataset, tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss=tf.keras.losses.Huber(), epochs=600)
forecast = main_model.predict_model(model, series[SPLIT_TIME-30:-1], window_size=30, batch_size=32)
```
7. Results Visualization
The predictions are visualized and evaluated:
```bash
plotter.plot_series(time_val, (series_val, forecast.squeeze()))
print(tf.keras.metrics.mae(series_val, forecast.squeeze()).numpy())
```
---

## Results
- The project outputs:
1. Learning Rate Tuning Plot: Visualizes loss over different learning rates.
2. Forecast Visualization: Actual vs. predicted sunspots on the validation set.
3. Performance Metric: Mean Absolute Error (MAE).
- using a `1e-5` learning rate
- the model got an MAE of `13.765716`

---

## Visualization

- Raw Data of Sunspots
![rawdataplot](https://github.com/user-attachments/assets/b318a2af-497e-4837-923f-846c8b04e685)

- Learning Rate Tuning
![LearningRateTuning](https://github.com/user-attachments/assets/188006eb-8745-4b96-bd18-0af60b8f697c)

- Predictions Plotted
![Predictions](https://github.com/user-attachments/assets/5f10c501-1b58-4303-ab9b-a44ce13eb7c0)
