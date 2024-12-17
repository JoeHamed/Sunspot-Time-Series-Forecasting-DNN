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
7. [Acknowledgments](#acknowledgments)  

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

