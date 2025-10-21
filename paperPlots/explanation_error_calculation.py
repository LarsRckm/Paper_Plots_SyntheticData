import sys
sys.path.append(".")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib as mpl
from plot import predict_encoder_interpolation_projection_roundedInput
from useful import sliding_window
from create_data import callFunction
import numpy as np
import pandas as pd
from pathlib import Path
import random


colorcodes = {
    "y_raw": "#92c5de",
    "y_groundtruth": "#0571b0",
    "Classic": "#67001f",
    "Interpolation": "#f2f2f2",
    "Model": "#f4a582",
    "Error": "#ca0020"
}

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],  # Standard-LaTeX-Schrift
    "axes.labelsize": 20,   # Achsentitel
    # "font.size": 13,        # Gesamtschriftgröße
    "legend.fontsize": 14,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

if __name__ == "__main__":
    #settings
    x_values = np.arange(1000)
    y_start = random.uniform(-10,10)
    random_number_range = ["norm",0,5]              # ["uni", lowerRange, upperRange], ["norm", mean, std]
    splineValue = [800000,1100000]
    vocabSize = 100000
    extraToken = ["ukn"]                            # UKN Token
    widthArray = [10,100,10]                        # [max number of Interpolation Intervals, maxWidth of Interpolation Interval, min width of Interpolation Interval]
    offset = 10
    x_limit  = [0,1000]
    timeSeriesChoose = random.randint(0,4)          # choose a number between 0 and 7 (so far)
    noise_std = ["norm",0,0.15]                     # ["uni", lowerRange, upperRange], ["norm", mean, std]


    #create synthetic timeseries
    y_trend, y_trend_noise, min_value, max_value, noise_std_value = callFunction(x_values, y_start, random_number_range, splineValue, vocabSize, timeSeriesChoose, noise_std)


    fig, ax = plt.subplots(1,1)
    ax.plot(x_values, y_trend_noise, label="noise")
    ax.plot(x_values, y_trend, label="groundtruth")

    plt.show()


    #ask user for model input
    models = []
    while True:
        filepath = input("Drag model into console, press enter to continue, \npress c to create new timeseries, press s to stop: ").strip("'")
        if(filepath == ""):
            break
        elif(filepath == "s"):
            sys.exit()
        elif(filepath == "c"):
            #choose a number 4 to create periodic sum functions
            timeSeriesChoose = 4
            #create synthetic timeseries
            y_trend, y_trend_noise, min_value, max_value, noise_std_value = callFunction(x_values, y_start, random_number_range, splineValue, vocabSize, timeSeriesChoose, noise_std)
            fig, ax = plt.subplots(1,1)
            ax.plot(x_values, y_trend_noise, label="noise")
            ax.plot(x_values, y_trend, label="groundtruth")
            plt.show()
        elif(filepath in models):
            None
        else:
            models.append(filepath)
    
    assert len(models) != 0, "Please select model"
        
    ArrayPredictions = []
    ArrayPredictionsSlidingWindow = []
    ArrayError = []


    mask = np.zeros_like(x_values)

    for model_filename in models:
        #denoise timeSeries with model
        print(f"Denoising Time Series with model {model_filename}")
        prediction_encoder = predict_encoder_interpolation_projection_roundedInput(len(x_values), model_filename, y_trend_noise,min_value, max_value, vocabSize, extraToken, mask)
        prediction_encoder = prediction_encoder.squeeze()



        #plot groundtruth and prediction Data and generate error plot
        prediction = prediction_encoder.numpy()
        groundtruth = y_trend
        noisyData = y_trend_noise

        #calculate sliding window timeseries from raw prediction
        predictionSlidingWindow = sliding_window(prediction, 2)
        for i in range(5):
            predictionSlidingWindow = sliding_window(predictionSlidingWindow, 2)

        #calculate error relative to noiseStdValue

        difference = prediction - groundtruth
        difference = difference / noise_std_value

        ArrayPredictions.append(prediction)
        ArrayPredictionsSlidingWindow.append(predictionSlidingWindow)
        ArrayError.append(difference)

    #calculate global max and min values for each subplot
    array_copy = ArrayPredictions[:]
    array_copy.append(y_trend)
    array_copy.append(y_trend_noise)
    array_copy = np.concatenate(array_copy)
    minval0 = min(array_copy)
    maxval0 = max(array_copy)
    offset0 = (maxval0-minval0)*0.05

    array_copy = ArrayPredictions[:]
    array_copy = np.concatenate(array_copy)
    minval1 = min(array_copy)
    maxval1 = max(array_copy)
    offset1 = (maxval1-minval1)*0.05


    array_copy = ArrayError[:]
    array_copy = np.concatenate(array_copy)
    minval2 = min(array_copy)
    maxval2 = max(array_copy)
    offset2 = (maxval2-minval2)*0.05


    #store data in csv
    df = pd.DataFrame()
    df["y_raw_groundtruth"] = y_trend
    df["y_raw_noiseGroundtruth"] = y_trend_noise
    df["y_raw_prediction"] = ArrayPredictions[0]
    df["error"] = ArrayError[0]
    df.to_csv("data/explanation_error_calculation/data.csv")



    #plot settings
    fig, ax = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]}, figsize=(8.27, 11.69))

    for (i, axis) in enumerate(ax):
        axis: Axes
        match i:
            case 0:
                axis.margins(x=0.0)
                axis.set_ylabel("Data")
                axis.set_ylim(minval0-offset0, maxval0+offset0)
                axis.tick_params(axis='both', direction='in', length=8, width=1)
                axis.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.55)

                axis.plot(x_values, y_trend_noise, label = "Noisy Groundtruth", color=colorcodes["y_raw"])
                axis.plot(x_values, y_trend, label = "Groundtruth", color=colorcodes["y_groundtruth"])
                for j, arr in enumerate(ArrayPredictions):
                    label = models[j].split("\\")[-1].split(".")[0].split("_")[-2]
                    axis.plot(x_values, arr, label = f"Model Prediction", color=colorcodes["Model"])
                axis.legend()
            case 1:
                axis.margins(x=0.0)
                axis.set_ylabel("Data")
                axis.set_ylim(minval1-offset1, maxval1+offset1)
                axis.tick_params(axis='both', direction='in', length=8, width=1)
                axis.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.55)


                axis.fill_between(x_values, y_trend-noise_std_value, y_trend+noise_std_value, color="#1d4b62", alpha=0.3)
                axis.fill_between(x_values, y_trend-2*noise_std_value, y_trend+2*noise_std_value, color="#92c5de", alpha=0.3)
                axis.fill_between(x_values, y_trend-3*noise_std_value, y_trend+3*noise_std_value, color="#ebf5f9", alpha=0.3)
                for j, arr in enumerate(ArrayPredictions):
                    models: list[str]
                    label = models[j].split("\\")[-1].split(".")[0].split("_")[-2]
                    axis.plot(x_values, arr, label = f"Model Prediction", color= colorcodes["Model"])
                axis.legend()
            case 2:
                axis.tick_params(axis='both',which="both", direction='in', length=8, width=1)
                axis.margins(x=0.0)
                axis.set_ylabel(r"$\sigma$")
                axis.set_xlabel("Steps")
                axis.set_ylim(minval2-offset2, maxval2+offset2)


                ones = np.ones_like(x_values)
                axis.fill_between(x_values, ones, -ones, color="#1d4b62", alpha=0.3)
                axis.fill_between(x_values, 2*ones, -2*ones, color="#92c5de", alpha=0.3)
                axis.fill_between(x_values, 3*ones, -3*ones, color="#ebf5f9", alpha=0.3)
                for j, arr in enumerate(ArrayError):
                    models: list[str]
                    label = models[j].split("\\")[-1].split(".")[0].split("_")[-2]
                    axis.plot(x_values, arr, label = f"Error Model Prediction", color= colorcodes["Error"])
                axis.legend()






    plt.tight_layout()
    save_path = Path(f"pictures/")
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / f"{"explanation_error_calculation"}.{"svg"}" if save_path else f"{"explanation_error_calculation"}.{"svg"}"
    plt.gcf().savefig(file_path, format="svg", dpi=300, bbox_inches='tight')
    plt.show()