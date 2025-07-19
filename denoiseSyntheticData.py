from create_data import callFunction
from plot import predict_encoder_interpolation_projection_roundedInput
from useful import remove_parts_of_graph_encoder, sliding_window
import numpy as np
import random
import matplotlib.pyplot as plt



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
    timeSeriesChoose = random.randint(0,7)                            # choose a number between 0 and 7 (so far)
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
        filepath = input("Drag model into console, press enter to continue: ").strip("'")
        if(filepath == ""):
            break
        if(filepath in models):
            None
        else:
            models.append(filepath)
    
    assert len(models) != 0, "Please select model"
        
    ArrayGroundtruth = []
    ArrayNoiseData = []
    ArrayPredictions = []
    ArrayPredictionsSlidingWindow = []
    ArrayMaskIntervals = []
    ArrayError = []
    ArrayErrorMEAN = []
    ArrayErrorSTD = []

    mask = remove_parts_of_graph_encoder(x_values, y_trend_noise, widthArray, offset, x_limit)

    for model_filename in models:
        #denoise timeSeries with model
        print(f"Denoising Time Series with model {model_filename}")
        prediction_encoder = predict_encoder_interpolation_projection_roundedInput(len(x_values), model_filename, y_trend_noise,min_value, max_value, vocabSize, extraToken, mask)
        prediction_encoder = prediction_encoder.squeeze()
        y_masked = np.where(mask == 0, y_trend_noise, np.nan)


        #calculate missing intervals
        intervalle = []
        nan_indices = np.where(np.isnan(y_masked))[0]
        length = len(nan_indices)

        if length > 0:
            start = nan_indices[0]  # Startwert des ersten Intervalls
            
            for i in range(1, length):
                # Wenn der Abstand zum vorherigen Wert 2 oder mehr beträgt, endet das Intervall
                if nan_indices[i] - nan_indices[i - 1] >= 2:
                    intervalle.append([start, nan_indices[i - 1]])  # Aktuelles Intervall speichern
                    start = nan_indices[i]  # Neues Intervall beginnen
            
            # Letztes Intervall hinzufügen
            intervalle.append([start, nan_indices[-1]])


        #plot groundtruth and prediction Data and generate error plot
        prediction = prediction_encoder.numpy()
        groundtruth = y_trend
        noisyData = y_trend_noise

        predictionSlidingWindow = sliding_window(prediction, 2)
        for i in range(5):
            predictionSlidingWindow = sliding_window(predictionSlidingWindow, 2)

        difference = groundtruth - prediction
        difference = difference / noise_std_value
        mean = np.mean(difference)
        std = np.std(difference)

        ArrayNoiseData.append(y_trend_noise)
        ArrayGroundtruth.append(y_trend)
        ArrayPredictions.append(prediction)
        ArrayPredictionsSlidingWindow.append(predictionSlidingWindow)
        ArrayMaskIntervals.append(intervalle)
        ArrayError.append(difference)
        ArrayErrorSTD.append(std)
        ArrayErrorMEAN.append(mean)


    for i in range(len(ArrayNoiseData)):
        fig, ax = plt.subplots(2,1, sharex=True)
        errorMean = ArrayErrorMEAN[i]
        errorSTD = ArrayErrorSTD[i]
        ax[0].plot(x_values, ArrayNoiseData[i], label="Noisy Data")
        ax[0].plot(x_values, ArrayGroundtruth[i], label="Groundtruth Data")
        ax[0].plot(x_values, ArrayPredictions[i], label="Prediction")
        ax[0].plot(x_values, ArrayPredictionsSlidingWindow[i], label="Prediction Sliding Window")

        ax[1].plot(x_values, ArrayError[i], label="error")
        ax[1].plot(x_values, errorMean * np.ones(len(x_values)), label="mean", alpha=0.3)
        ax[1].plot(x_values, (errorMean + errorSTD) * np.ones(len(x_values)), label="mean + std", alpha=0.8, color= "green")
        ax[1].plot(x_values, (errorMean - errorSTD) * np.ones(len(x_values)), label="mean - std", alpha=0.8, color= "green")


        for bottom_border, upper_border in ArrayMaskIntervals[i]:
            ax[0].axvspan(bottom_border - 0.5, upper_border + 0.5, color='red', alpha=0.05)
            ax[1].axvspan(bottom_border - 0.5, upper_border + 0.5, color='red', alpha=0.05)
        
        ax[1].axhspan((errorMean - errorSTD), (errorMean + errorSTD), color='green', alpha=0.05)

        ax[0].legend()
        ax[1].legend()

        ax[0].set_title(f"{models[i]}")
    
    plt.show()


