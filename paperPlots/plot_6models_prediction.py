import sys
sys.path.append(".")

import matplotlib.pyplot as plt
import matplotlib as mpl
from plot import predict_encoder_interpolation_projection_roundedInput
from useful import sliding_window
from create_data import callFunction
import pandas as pd
import numpy as np
from pathlib import Path
import random
import json
import string
import pywt
from scipy.signal import butter, filtfilt

def butter_lowpass(cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def wavelet_denoise(
        x,
        wavelet="db4",          # Wavelet-Familie / -Ordnung (z.B. "db4", "sym8", "coif5", "bior4.4")
        level=None,             # Zerlegungstiefe; None = automatisch (max. sinnvoll)
        threshold_rule="universal",  # 'universal' (VisuShrink), 'sure' (SURE), 'bayes' (BayesShrink)
        threshold_scale=1.0,    # Faktor für die Schwelle (z.B. 0.8..1.2 Feintuning)
        mode="soft",            # 'soft' oder 'hard' Thresholding
        signal_extension="symmetric",# Randbehandlung: 'symmetric','periodization','reflect','zero', ...
        use_swt=False           # True: Stationary Wavelet Transform (übersetzungsinvariant)
    ):
        """
        Gibt das geglättete Signal zurück und ein Dict mit Metadaten.
        """
        # Hilfsfunktionen
        def noise_sigma_from_coeffs(detail_coeffs):
            # Rauschschätzer (MAD) aus feinster Detail-Ebene
            d1 = detail_coeffs[-1]
            sigma = np.median(np.abs(d1 - np.median(d1))) / 0.6745
            return sigma

        def calc_threshold(sigma, n, rule):
            if rule == "universal":         # VisuShrink
                return sigma * np.sqrt(2*np.log(n))
            elif rule == "sure":            # Stein's Unbiased Risk Estimate (PyWavelets hat Hilfen, hier simple Approx.)
                # Vereinfachte SURE-Variante: starte bei universal und skaliere
                return sigma * np.sqrt(2*np.log(n)) * 0.8
            elif rule == "bayes":
                # BayesShrink (vereinfachte Heuristik): sigma^2 / std(detail)
                return None  # pro Ebene separat weiter unten
            else:
                raise ValueError("threshold_rule must be 'universal', 'sure', or 'bayes'.")

        # Zerlegung
        if use_swt:
            coeffs = pywt.swt(x, wavelet=wavelet, level=level or pywt.swt_max_level(len(x)), start_level=0, trim_approx=True)
            # coeffs: Liste [(cA_L, cD_L), ..., (cA_1, cD_1)]
            sigma = noise_sigma_from_coeffs([cD for (_, cD) in coeffs])
            n = len(x)
            T_global = calc_threshold(sigma, n, threshold_rule)
            denoised_coeffs = []
            for (cA, cD) in coeffs:
                if threshold_rule == "bayes":
                    sigma_d = np.std(cD)
                    T = (sigma**2) / (sigma_d + 1e-12)
                else:
                    T = T_global
                T = threshold_scale * T if T is not None else None
                cD_th = pywt.threshold(cD, T, mode=mode) if T is not None else cD
                denoised_coeffs.append((cA, cD_th))
            x_hat = pywt.iswt(denoised_coeffs, wavelet=wavelet)
        else:
            max_level = pywt.dwt_max_level(data_len=len(x), filter_len=pywt.Wavelet(wavelet).dec_len)
            use_level = min(level or max_level, max_level)
            coeffs = pywt.wavedec(x, wavelet=wavelet, mode=signal_extension, level=use_level)
            cA, cDs = coeffs[0], coeffs[1:]
            sigma = noise_sigma_from_coeffs(cDs)
            n = len(x)
            T_global = calc_threshold(sigma, n, threshold_rule)

            new_cDs = []
            for cD in cDs:
                if threshold_rule == "bayes":
                    sigma_d = np.std(cD)
                    T = (sigma**2) / (sigma_d + 1e-12)
                else:
                    T = T_global
                T = threshold_scale * T if T is not None else None
                cD_th = pywt.threshold(cD, T, mode=mode) if T is not None else cD
                new_cDs.append(cD_th)

            x_hat = pywt.waverec([cA] + new_cDs, wavelet=wavelet, mode=signal_extension)

        return x_hat[:len(x)], {
            "wavelet": wavelet,
            "level": level,
            "threshold_rule": threshold_rule,
            "threshold_scale": threshold_scale,
            "mode": mode,
            "signal_extension": signal_extension,
            "use_swt": use_swt
        }




mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],  # Standard-LaTeX-Schrift
    "axes.labelsize": 12,   # Achsentitel
    "font.size": 10,        # Gesamtschriftgröße
    "legend.fontsize": 9,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


modelName_timeseries = {
    "AllInOne": [0,1,2,3,4],
    "Discontinuous": [5,6],
    "Periodic":[3],
    "PeriodicSum": [4]
}


colorcodes = {
    "y_raw": "#92c5de",
    "y_groundtruth": "#0571b0",
    "Classic": "#67001f",
    "Interpolation": "#f2f2f2",
    "Model": "#f4a582",
    "Error": "#ca0020"
}




if __name__ == "__main__":
    #read in models
    #based on model, create different suitable timeseries for user to choose
    #make user confirm time series and calculate prediction/noise suppression
    #store groundtruth, noisy groundtruth, prediction, error  in arrays

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
    noise_std = ["norm",0,0.15]                     # ["uni", lowerRange, upperRange], ["norm", mean, std]


    #store models being used
    models = []


    #create dataframe to store timeseries
    df_groundtruth = pd.DataFrame()
    df_noisyGroundtruth = pd.DataFrame()
    df_rawPrediction = pd.DataFrame()
    df_slidingWindowPrediction = pd.DataFrame()
    df_errorRaw = pd.DataFrame()
    df_errorSlidingWindow = pd.DataFrame()
    # json_interpolation = {}


    #create arrays to store results in
    arrayGroundtruth = []
    arrayNoisyGroundtruth = []
    arrayRawPrediction = []
    arraySlidingWindowPrediction = []
    arrayErrorRaw = []
    arrayErrorSlidingWindow = []
    # arrayInterpolation = []


    timeSeriesChoose = 4

    #create synthetic timeseries
    y_trend, y_trend_noise, min_value, max_value, noise_std_value = callFunction(x_values, y_start, random_number_range, splineValue, vocabSize, timeSeriesChoose, noise_std)
    fig, ax = plt.subplots(1,1)
    ax.plot(x_values, y_trend_noise, label="noise")
    ax.plot(x_values, y_trend, label="groundtruth")
    plt.show()

    while True:
        user_confirmation = input("Press enter to create new timeseries, input y to take the timeseries: ")
        if(user_confirmation == ""):
            #create synthetic timeseries
            timeSeriesChoose = 4
            y_trend, y_trend_noise, min_value, max_value, noise_std_value = callFunction(x_values, y_start, random_number_range, splineValue, vocabSize, timeSeriesChoose, noise_std)
            fig, ax = plt.subplots(1,1)
            ax.plot(x_values, y_trend_noise, label="noise")
            ax.plot(x_values, y_trend, label="groundtruth")
            plt.show()
        elif(user_confirmation == "y"):
            #create mask
            mask = np.zeros_like(y_trend)

            #apply mask to noisy groundtruth
            y_masked = np.where(mask == 0, y_trend_noise, np.nan)

            #calculate missing intervals
            # intervalle = []
            # nan_indices = np.where(np.isnan(y_masked))[0]
            # length = len(nan_indices)
            break


    new_old_userinput = input("Enter n to create new timeseries, press o to use old ones: ")
    if new_old_userinput == "n":
        while len(models)<4:
            model_input = input("Please drag and drop model here: ")
            if(model_input == ""):
                break
            if(model_input.split(".")[-1] == "pt" and model_input not in models):
                #create timeseries until user confirms

                #store model
                models.append(model_input)

                #extract model from model_input
                model_name = model_input.split(".")[0].split("_")[-2]
                # timeSeriesChoose = 4

                # #create synthetic timeseries
                # y_trend, y_trend_noise, min_value, max_value, noise_std_value = callFunction(x_values, y_start, random_number_range, splineValue, vocabSize, timeSeriesChoose, noise_std)
                # fig, ax = plt.subplots(1,1)
                # ax.plot(x_values, y_trend_noise, label="noise")
                # ax.plot(x_values, y_trend, label="groundtruth")
                # plt.show()

                # while True:
                #     user_confirmation = input("Press enter to create new timeseries, input y to take the timeseries: ")
                #     if(user_confirmation == ""):
                #         #create synthetic timeseries
                #         timeSeriesChoose = 4
                #         y_trend, y_trend_noise, min_value, max_value, noise_std_value = callFunction(x_values, y_start, random_number_range, splineValue, vocabSize, timeSeriesChoose, noise_std)
                #         fig, ax = plt.subplots(1,1)
                #         ax.plot(x_values, y_trend_noise, label="noise")
                #         ax.plot(x_values, y_trend, label="groundtruth")
                #         plt.show()
                #     elif(user_confirmation == "y"):
                        # #create mask
                        # mask = remove_parts_of_graph_encoder(x_values, y_trend_noise, widthArray, offset, x_limit)

                        # #apply mask to noisy groundtruth
                        # y_masked = np.where(mask == 0, y_trend_noise, np.nan)

                        # #calculate missing intervals
                        # intervalle = []
                        # nan_indices = np.where(np.isnan(y_masked))[0]
                        # length = len(nan_indices)

                # if length > 0:
                #     start = nan_indices[0]  # Startwert des ersten Intervalls
                    
                #     for i in range(1, length):
                #         # Wenn der Abstand zum vorherigen Wert 2 oder mehr beträgt, endet das Intervall
                #         if nan_indices[i] - nan_indices[i - 1] >= 2:
                #             intervalle.append([int(start), int(nan_indices[i - 1])])  # Aktuelles Intervall speichern
                #             start = nan_indices[i]  # Neues Intervall beginnen
                    
                #     # Letztes Intervall hinzufügen
                #     intervalle.append([int(start), int(nan_indices[-1])])
                
                #make prediction
                prediction_encoder = predict_encoder_interpolation_projection_roundedInput(len(x_values), model_input, y_trend_noise,min_value, max_value, vocabSize, extraToken, mask)
                prediction_encoder = prediction_encoder.squeeze()
                prediction_encoder = prediction_encoder.numpy()
                
                #create sliding window prediction
                predictionSlidingWindow = sliding_window(prediction_encoder, 2)
                for i in range(5):
                    predictionSlidingWindow = sliding_window(predictionSlidingWindow, 2)
            
                #create error
                #error raw prediction
                differencRaw = (prediction_encoder - y_trend)/noise_std_value
                #error sliding window prediction
                differenceSlidingWindow = (predictionSlidingWindow - y_trend)/noise_std_value
                
                #store data in arrays
                arrayGroundtruth.append(y_trend)
                arrayNoisyGroundtruth.append(y_masked)
                arrayRawPrediction.append(prediction_encoder)
                arraySlidingWindowPrediction.append(predictionSlidingWindow)
                arrayErrorRaw.append(differencRaw)
                arrayErrorSlidingWindow.append(differenceSlidingWindow)
                # arrayInterpolation.append(intervalle)

        #put here wavelet and fourier
        #wavelet
        prediction_encoder, _ = wavelet_denoise(
            y_trend_noise,
            wavelet="db6",
            level=None,            # automatisch
            threshold_rule="universal",
            threshold_scale=1.0,
            mode="soft",
            signal_extension="symmetric",
            use_swt=False          # True ausprobieren für SWT (meist glatter, aber langsamer)
            )
        #create sliding window prediction
        predictionSlidingWindow = sliding_window(prediction_encoder, 2)
        for i in range(5):
            predictionSlidingWindow = sliding_window(predictionSlidingWindow, 2)

        #error raw prediction
        differencRaw = (prediction_encoder - y_trend)/noise_std_value

        #error sliding window prediction
        differenceSlidingWindow = (predictionSlidingWindow - y_trend)/noise_std_value

        #store data in arrays
        arrayGroundtruth.append(y_trend)
        arrayNoisyGroundtruth.append(y_masked)
        arrayRawPrediction.append(prediction_encoder)
        arraySlidingWindowPrediction.append(predictionSlidingWindow)
        arrayErrorRaw.append(differencRaw)
        arrayErrorSlidingWindow.append(differenceSlidingWindow)



        #------------------------------------------------------------------------------------
        #fourier
        fs = 500  # Abtastrate [Hz]
        cutoff = 10  # Cutoff-Frequenz (Hz)
        prediction_encoder = lowpass_filter(y_trend_noise, cutoff, fs)

        #create sliding window prediction
        predictionSlidingWindow = sliding_window(prediction_encoder, 2)
        for i in range(5):
            predictionSlidingWindow = sliding_window(predictionSlidingWindow, 2)

        #error raw prediction
        differencRaw = (prediction_encoder - y_trend)/noise_std_value

        #error sliding window prediction
        differenceSlidingWindow = (predictionSlidingWindow - y_trend)/noise_std_value

        #store data in arrays
        arrayGroundtruth.append(y_trend)
        arrayNoisyGroundtruth.append(y_masked)
        arrayRawPrediction.append(prediction_encoder)
        arraySlidingWindowPrediction.append(predictionSlidingWindow)
        arrayErrorRaw.append(differencRaw)
        arrayErrorSlidingWindow.append(differenceSlidingWindow)


        #store values in dataframes
        for i in range(len(arrayGroundtruth)):
            df_groundtruth[f"{i}"] = arrayGroundtruth[i]
            df_noisyGroundtruth[f"{i}"] = arrayNoisyGroundtruth[i]
            df_rawPrediction[f"{i}"] = arrayRawPrediction[i]
            df_slidingWindowPrediction[f"{i}"] = arrayErrorSlidingWindow[i]
            df_errorRaw[f"{i}"] = arrayErrorRaw[i]
            df_errorSlidingWindow[f"{i}"] = arrayErrorSlidingWindow[i]
            # json_interpolation[f"{i}"] = arrayInterpolation[i]
        

        df_groundtruth.to_csv("data/explanation_6models_sameTimeSeries/df_groundtruth_singleTimeSeries.csv", index=False)
        df_noisyGroundtruth.to_csv("data/explanation_6models_sameTimeSeries/df_noisyGroundtruth_singleTimeSeries.csv", index=False)
        df_rawPrediction.to_csv("data/explanation_6models_sameTimeSeries/df_rawPrediction_singleTimeSeries.csv", index=False)
        df_slidingWindowPrediction.to_csv("data/explanation_6models_sameTimeSeries/df_slidingWindowPrediction_singleTimeSeries.csv", index=False)
        df_errorRaw.to_csv("data/explanation_6models_sameTimeSeries/df_errorRaw_singleTimeSeries.csv", index=False)
        df_errorSlidingWindow.to_csv("data/explanation_6models_sameTimeSeries/df_errorSlidingWindow_singleTimeSeries.csv", index=False)
        # with open("data/explanation_6models_sameTimeSeries/json_interpolation_singleTimeSeries.json", "w") as f:
        #     json.dump(json_interpolation, f, indent=4)

    elif new_old_userinput == "o":
        df_groundtruth = pd.read_csv("data/explanation_6models_sameTimeSeries/df_groundtruth_singleTimeSeries.csv")
        df_noisyGroundtruth = pd.read_csv("data/explanation_6models_sameTimeSeries/df_noisyGroundtruth_singleTimeSeries.csv")
        df_rawPrediction = pd.read_csv("data/explanation_6models_sameTimeSeries/df_rawPrediction_singleTimeSeries.csv")
        df_slidingWindowPrediction = pd.read_csv("data/explanation_6models_sameTimeSeries/df_slidingWindowPrediction_singleTimeSeries.csv")
        df_errorRaw = pd.read_csv("data/explanation_6models_sameTimeSeries/df_errorRaw_singleTimeSeries.csv")
        df_errorSlidingWindow = pd.read_csv("data/explanation_6models_sameTimeSeries/df_errorSlidingWindow_singleTimeSeries.csv")
        # with open("data/explanation_6models_sameTimeSeries/json_interpolation_singleTimeSeries.json", "r") as f:
        #     json_interpolation = json.load(f)

        for i in range(df_groundtruth.shape[1]):
            arrayGroundtruth.append(df_groundtruth.iloc[:,i].to_numpy())
            arrayNoisyGroundtruth.append(df_noisyGroundtruth.iloc[:,i].to_numpy())
            arrayRawPrediction.append(df_rawPrediction.iloc[:,i].to_numpy())
            arraySlidingWindowPrediction.append(df_slidingWindowPrediction.iloc[:,i].to_numpy())
            arrayErrorRaw.append(df_errorRaw.iloc[:,i].to_numpy())
            arrayErrorSlidingWindow.append(df_errorSlidingWindow.iloc[:,i].to_numpy())
            # arrayInterpolation.append(json_interpolation[f"{i}"])
    else:
        sys.exit()



    #strings for alphabetic order
    labels = list(string.ascii_lowercase)

    #plot all timeseries into 2,2 subplot
    fig = plt.figure(figsize=(10, 14))
    
    # Haupt-Grid: 2x2
    outer_grid = fig.add_gridspec(3, 2, wspace=0.25, hspace=0.3)
    count = 0

    for i in range(3):
        for j in range(2):
            # Für jedes Feld im 2x2-Grid ein 2x1-Subgrid erstellen
            # model_name = models[count].split(".")[0].split("_")[-2]
            ones = np.ones_like(x_values)
            error = arrayErrorRaw[count]
            inner_grid = outer_grid[i, j].subgridspec(2, 1, hspace=0.05)
            ax1 = fig.add_subplot(inner_grid[0])
            ax2 = fig.add_subplot(inner_grid[1])
            ax1.set_xmargin(0)
            ax2.set_xmargin(0)
            ax2.set_xlabel(f"Steps\n({labels[count]})")
            ax1.tick_params(axis='both',direction='in', length=6, width=1)
            ax2.tick_params(axis='both',direction='in', length=6, width=1)
            ax1.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.55)
            ax2.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.55)
            if j == 0:
                ax1.set_ylabel("Data")
                ax2.set_ylabel(r"$\sigma$")
            
            ax1.plot(x_values, arrayNoisyGroundtruth[count], color=colorcodes["y_raw"])
            ax1.plot(x_values, arrayGroundtruth[count], color=colorcodes["y_groundtruth"])
            ax1.plot(x_values, arrayRawPrediction[count], color=colorcodes["Model"])
            ax1.set_xticklabels([])

            # ax2.fill_between(x_values, ones, -ones, color="navy", alpha=0.3)
            # ax2.fill_between(x_values, 2*ones, -2*ones, color="royalblue", alpha=0.3)
            # ax2.fill_between(x_values, 3*ones, -3*ones, color="lightsteelblue", alpha=0.3)
            ax2.plot(x_values, error, color=colorcodes["Error"])

            maxerror = max(error)
            minerror = min(error)
            offset = (maxerror - minerror) * 0.1
            ax2.set_ylim(minerror-offset, maxerror+offset)

            # for bottom_border, upper_border in arrayInterpolation[count]:
            #     ax1.axvspan(bottom_border - 0.5, upper_border + 0.5, color=colorcodes["Interpolation"], alpha=1)
            #     ax2.axvspan(bottom_border - 0.5, upper_border + 0.5, color=colorcodes["Interpolation"], alpha=1)


            count += 1
    


    plt.tight_layout()
    save_path = Path(f"pictures/")
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / f"{"application_6_models_prediction"}.{"svg"}" if save_path else f"{"application_6_models_prediction"}.{"svg"}"
    plt.gcf().savefig(file_path, format="svg", dpi=300, bbox_inches='tight')
    plt.show()