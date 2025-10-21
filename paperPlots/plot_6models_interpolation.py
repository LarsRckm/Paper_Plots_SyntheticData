import sys
sys.path.append(".")

import matplotlib.pyplot as plt
import matplotlib as mpl
from plot import predict_encoder_interpolation_projection_roundedInput
from useful import remove_parts_of_graph_encoder, sliding_window
from create_data import callFunction
import pandas as pd
import numpy as np
from pathlib import Path
import random
import json
import string
from scipy.interpolate import PchipInterpolator
import statsmodels.api as sm
from scipy.signal import savgol_filter

def smooth_segments(s: pd.Series,
                    method: str = "savgol",
                    window: int = 7,      # Anzahl Punkte (ungerade) für SavGol/rolling
                    polyorder: int = 2,   # SavGol Polynomgrad
                    loess_frac: float = 0.2,
                    ewma_alpha: float = 0.2):
    """
    Glättet NUR innerhalb zusammenhängender nicht-NaN-Segmente.
    Gibt eine Serie mit NaNs an den ursprünglichen Lücken zurück.
    """
    s = s.copy()
    isna = s.isna()
    groups = (isna.ne(isna.shift())).cumsum()  # Segment-IDs
    out = pd.Series(index=s.index, dtype=float)

    for gid, seg in s.groupby(groups):
        if seg.isna().all():  # Das ist eine NaN-Lücke -> überspringen
            continue
        y = seg.values.astype(float)

        if method == "rolling_mean":
            y_hat = pd.Series(y, index=seg.index).rolling(window, min_periods=1, center=True).mean().values
        elif method == "rolling_median":
            y_hat = pd.Series(y, index=seg.index).rolling(window, min_periods=1, center=True).median().values
        elif method == "ewma":
            y_hat = pd.Series(y, index=seg.index).ewm(alpha=ewma_alpha, adjust=False).mean().values
        elif method == "loess":
            x = np.arange(len(y))
            y_hat = sm.nonparametric.lowess(y, x, frac=loess_frac, return_sorted=False)
        elif method == "savgol":
            win = window if window % 2 == 1 else window + 1
            if len(y) >= win:
                y_hat = savgol_filter(y, window_length=win, polyorder=min(polyorder, win-1), mode="interp")
            else:
                # Fallback: kleine Segmente -> Median-Rolling
                y_hat = pd.Series(y, index=seg.index).rolling(min(len(y), 3), min_periods=1, center=True).median().values
        elif method == "none":
            y_hat = y
        else:
            raise ValueError("Unbekannte Glättungsmethode")

        out.loc[seg.index] = y_hat

    return out

def interpolate_gaps(s: pd.Series,
                     method: str = "pchip",
                     max_gap: int | None = None,
                     datetime_aware: bool = True):
    """
    Interpoliert NUR NaN-Lücken. Optional: nur Lücken bis max_gap Länge.
    method: 'time' (bei DatetimeIndex), 'pchip', 'spline3', 'linear', 'polynomial3', 'kalman'
    """
    s = s.copy()

    # Optional: nur kleine/mittlere Lücken interpolieren
    if max_gap is not None:
        isna = s.isna()
        grp = (isna.ne(isna.shift())).cumsum()
        gap_lens = isna.groupby(grp).transform('sum')
        mask_large_gaps = isna & (gap_lens > max_gap)
    else:
        mask_large_gaps = pd.Series(False, index=s.index)

    if method == "time":
        if not isinstance(s.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise ValueError("method='time' benötigt DatetimeIndex.")
        filled = s.interpolate(method="time", limit_area="inside")
    elif method == "linear":
        filled = s.interpolate(method="linear", limit_area="inside")
    elif method == "pchip":
        # PCHIP via SciPy
        x = np.arange(len(s))
        ok = ~s.isna()
        f = PchipInterpolator(x[ok], s[ok])
        filled = pd.Series(f(x), index=s.index)
        # Ränder (Extrapolation) auf Original lassen, falls dort NaN:
        filled[~ok & ((~ok).cummax() & (~ok)[::-1].cummax()[::-1])] = np.nan
    elif method == "spline3":
        filled = s.interpolate(method="spline", order=3, limit_area="inside")
    elif method == "polynomial3":
        filled = s.interpolate(method="polynomial", order=3, limit_area="inside")
    elif method == "kalman":
        # sehr einfache Zustandsraum-Variante (lokaler Trend)
        mod = sm.tsa.UnobservedComponents(s, level='local linear trend')
        res = mod.fit(disp=False)
        filled = pd.Series(res.predict(), index=s.index)
    else:
        raise ValueError("Unbekannte Interpolationsmethode")

    # Große Lücken bewusst offen lassen
    filled[mask_large_gaps] = np.nan
    return filled

def smooth_then_interpolate(s: pd.Series,
                            smooth_method="savgol",
                            interp_method="pchip",
                            smooth_window=7,
                            max_gap=None):
    """
    1) innerhalb sichtbarer Segmente glätten
    2) NaN-Lücken interpolieren (kontrolliert)
    """
    s_sm = smooth_segments(s, method=smooth_method, window=smooth_window)
    out = interpolate_gaps(s_sm, method=interp_method, max_gap=max_gap)
    return out

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
    noise_std = ["norm",0,0.001]                     # ["uni", lowerRange, upperRange], ["norm", mean, std]


    #store models being used
    models = []


    #create dataframe to store timeseries
    df_groundtruth = pd.DataFrame()
    df_noisyGroundtruth = pd.DataFrame()
    df_rawPrediction = pd.DataFrame()
    df_slidingWindowPrediction = pd.DataFrame()
    df_errorRaw = pd.DataFrame()
    df_errorSlidingWindow = pd.DataFrame()
    json_interpolation = {}


    #create arrays to store results in
    arrayGroundtruth = []
    arrayNoisyGroundtruth = []
    arrayRawPrediction = []
    arraySlidingWindowPrediction = []
    arrayErrorRaw = []
    arrayErrorSlidingWindow = []
    arrayInterpolation = []


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
            mask = remove_parts_of_graph_encoder(np.arange(1000), y_trend_noise,[10,100,10],10, [0,1000])

            #apply mask to noisy groundtruth
            y_masked = np.where(mask == 0, y_trend_noise, np.nan)
            y_trend_noise = np.where(mask==1, 0, y_trend_noise)

            #calculate missing intervals
            intervalle = []
            nan_indices = np.where(np.isnan(y_masked))[0]
            length = len(nan_indices)

            if length > 0:
                start = nan_indices[0]  # Startwert des ersten Intervalls
                
                for i in range(1, length):
                    # Wenn der Abstand zum vorherigen Wert 2 oder mehr beträgt, endet das Intervall
                    if nan_indices[i] - nan_indices[i - 1] >= 2:
                        intervalle.append([int(start), int(nan_indices[i - 1])])  # Aktuelles Intervall speichern
                        start = nan_indices[i]  # Neues Intervall beginnen
                
                # Letztes Intervall hinzufügen
                intervalle.append([int(start), int(nan_indices[-1])])
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
                arrayInterpolation.append(intervalle)

        #preparing y_noise for classic interpolation
        y_trend_noise[mask == 1] = np.nan
        y_trend_noise = pd.Series(y_trend_noise)


        #----------------linear interpolation--------------------------

        prediction_encoder = smooth_then_interpolate(y_trend_noise,
                                        smooth_method="loess",   # oder 'loess', 'rolling_median', 'ewma', 'none', savgol
                                        interp_method="linear",    # oder 'time','spline3','linear','polynomial3','kalman', 'pchip
                                        smooth_window=9,
                                        max_gap=1000)
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
        arrayInterpolation.append(intervalle)



        #------------------polynomial interpolation-----------------------------------------
        prediction_encoder = smooth_then_interpolate(y_trend_noise,
                                        smooth_method="loess",   # oder 'loess', 'rolling_median', 'ewma', 'none', savgol
                                        interp_method="polynomial3",    # oder 'time','spline3','linear','polynomial3','kalman', 'pchip
                                        smooth_window=9,
                                        max_gap=1000)

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
        arrayInterpolation.append(intervalle)



        #store values in dataframes
        for i in range(len(arrayGroundtruth)):
            df_groundtruth[f"{i}"] = arrayGroundtruth[i]
            df_noisyGroundtruth[f"{i}"] = arrayNoisyGroundtruth[i]
            df_rawPrediction[f"{i}"] = arrayRawPrediction[i]
            df_slidingWindowPrediction[f"{i}"] = arrayErrorSlidingWindow[i]
            df_errorRaw[f"{i}"] = arrayErrorRaw[i]
            df_errorSlidingWindow[f"{i}"] = arrayErrorSlidingWindow[i]
            json_interpolation[f"{i}"] = arrayInterpolation[i]
        

        df_groundtruth.to_csv("data/explanation_6models_interpolation/df_groundtruth_singleTimeSeries.csv", index=False)
        df_noisyGroundtruth.to_csv("data/explanation_6models_interpolation/df_noisyGroundtruth_singleTimeSeries.csv", index=False)
        df_rawPrediction.to_csv("data/explanation_6models_interpolation/df_rawPrediction_singleTimeSeries.csv", index=False)
        df_slidingWindowPrediction.to_csv("data/explanation_6models_interpolation/df_slidingWindowPrediction_singleTimeSeries.csv", index=False)
        df_errorRaw.to_csv("data/explanation_6models_interpolation/df_errorRaw_singleTimeSeries.csv", index=False)
        df_errorSlidingWindow.to_csv("data/explanation_6models_interpolation/df_errorSlidingWindow_singleTimeSeries.csv", index=False)
        with open("data/explanation_6models_interpolation/json_interpolation_singleTimeSeries.json", "w") as f:
            json.dump(json_interpolation, f, indent=4)

    elif new_old_userinput == "o":
        df_groundtruth = pd.read_csv("data/explanation_6models_interpolation/df_groundtruth_singleTimeSeries.csv")
        df_noisyGroundtruth = pd.read_csv("data/explanation_6models_interpolation/df_noisyGroundtruth_singleTimeSeries.csv")
        df_rawPrediction = pd.read_csv("data/explanation_6models_interpolation/df_rawPrediction_singleTimeSeries.csv")
        df_slidingWindowPrediction = pd.read_csv("data/explanation_6models_interpolation/df_slidingWindowPrediction_singleTimeSeries.csv")
        df_errorRaw = pd.read_csv("data/explanation_6models_interpolation/df_errorRaw_singleTimeSeries.csv")
        df_errorSlidingWindow = pd.read_csv("data/explanation_6models_interpolation/df_errorSlidingWindow_singleTimeSeries.csv")
        with open("data/explanation_6models_interpolation/json_interpolation_singleTimeSeries.json", "r") as f:
            json_interpolation = json.load(f)

        for i in range(df_groundtruth.shape[1]):
            arrayGroundtruth.append(df_groundtruth.iloc[:,i].to_numpy())
            arrayNoisyGroundtruth.append(df_noisyGroundtruth.iloc[:,i].to_numpy())
            arrayRawPrediction.append(df_rawPrediction.iloc[:,i].to_numpy())
            arraySlidingWindowPrediction.append(df_slidingWindowPrediction.iloc[:,i].to_numpy())
            arrayErrorRaw.append(df_errorRaw.iloc[:,i].to_numpy())
            arrayErrorSlidingWindow.append(df_errorSlidingWindow.iloc[:,i].to_numpy())
            arrayInterpolation.append(json_interpolation[f"{i}"])
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

            for bottom_border, upper_border in arrayInterpolation[count]:
                ax1.axvspan(bottom_border - 0.5, upper_border + 0.5, color=colorcodes["Interpolation"], alpha=1)
                ax2.axvspan(bottom_border - 0.5, upper_border + 0.5, color=colorcodes["Interpolation"], alpha=1)


            count += 1
    


    plt.tight_layout()
    save_path = Path(f"pictures/")
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / f"{"application_6_models_interpolation"}.{"svg"}" if save_path else f"{"application_4_models_singleTimeSeries"}.{"svg"}"
    plt.gcf().savefig(file_path, format="svg", dpi=300, bbox_inches='tight')
    plt.show()