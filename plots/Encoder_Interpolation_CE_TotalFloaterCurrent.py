import sys
sys.path.append(".")
from plot import plot_multiple_pred_with_names, denoise_floaterCurrent_encoder_interpolation_CE,plot_multiple_pred_with_names_error_bar_single
import pandas as pd
import numpy as np
import torch
import os
import math


def get_config():
    return {
        "number_x_values": 1000,                            #time series length
        "extra_tokens": ["ukn"],                            #additional tokens next to normal int index
        "vocab_size": 100000,                               #number of equally distanced pieces of number range [0,1], encoder interpolation CE 
        "remove_parts": True,                               #whether or not to remove arbitrary parts of the time series (corresponding to the entry "width_array")
        "spline_value": [800000,1100000],                                  #spline value for smooth time series shape, higher value -> more smooth
        "x_lim": [0,1000],                                  #range of x-array for removing parts of time series
        "y_lim": [10,10000],                                  #range of y-values in which time series is bounded
        "noise_std": [0,10],                             #[noise_std_min, noise_std_max], noise_std is normal distributed or uniform distributed itself, specify in create_data.py in generate_noisy_data
        "dropout": 0.1,                                     #value for dropout layer in model
        "train_count":10000,                                    #number of training examples per epoch
        "val_count": 100,                                     #number of validation examples per epoch
        "random_number_range": ["norm",0,5],          #y-values are created randomly with normal distribution and given mean_value and std_value
        "offset": 10,                                       #offset to the lowest and highest border of y_values
        "width_array_encoder": [10,100,10],                  #[min_width, max_width, max_count_width]
        "batch_size": 10,                                    #train_count / batch_size = number_iteration_per_epoch
        "num_epochs": 400,                                  #max number of training epochs
        "lr": 10**-4,                                       #learning rate for Adam optimizer
        "d_model": 512,
        "model_folder": "weights",                          #folder name to store models in
        "model_basename": "Encoder_Interpolation_CE_Periodic_",
        "preload": "latest",                                #whether or not to start training with the latest trained version
        "experiment_name": "runs/tmodel"
    }

def list_files_in_directory(directory_path):
    try:
        # List all files and directories in the given directory
        entries = os.listdir(directory_path)
        
        # Filter out only files, ignoring directories
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
        
        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def plot_floater_current_CE():
    config = get_config()
    files = list_files_in_directory("jmh_data/")            #read all files from directory
    total_df = pd.DataFrame(columns=["Time", "Current"])    #Dataframe to store content of individual files


    #iteratively concatenate content into one Dataframe
    for csv_file in files:
        df = pd.read_csv(f"jmh_data/{csv_file}")
        firstTimestamp = df.iloc[1:,[0,3]]
        firstTimestamp["Time"] = pd.to_datetime(firstTimestamp["Time"], format="%Y-%m-%d %H:%M:%S.%f")
        firstTimestamp["Current"] = firstTimestamp["Current"].apply(lambda x: float(x))
        total_df = pd.concat([total_df, firstTimestamp],axis=0)
        
    #sort by time
    total_df = total_df.sort_values(by="Time", ascending=True)
    dflength = total_df.shape[0]
    dflength = math.floor(dflength / 1000)

    # dflength = 1000
    # print(f"Abtastung: {dflength}")


    #introduce sampling rate to reduce data points 
    #(in this case of floating currents, there is a testing point every 10 seconds)
    # dflength = 200
    current = total_df.iloc[0::dflength,1]
    # current = current[:1000]
    
    #calculate mean over sampled data points
    #mean_border_factor indicating the limit from which higher values should be interpolated
    mean_current = current.mean()
    mean_border_factor = 1.5
    shape = current.shape[0]
    mean_array = mean_current * np.ones((shape))
    mean_array_changed = mean_border_factor * mean_current * np.ones((shape))

    #every value higher than mean_border_factor * mean_current gets masked out (equivalent to mask value 1)
    mask = torch.tensor(current.to_numpy()).type(torch.float32).apply_(lambda x: 1.0 if x >= mean_border_factor*mean_current else 0.0).type(torch.int16).numpy()

    y_masked = np.where(mask == 0, current, np.nan)

    final_df = pd.DataFrame()
    final_df.loc[:,"current"] = current.to_numpy()
    final_df.loc[:,"mean_1_comma_2"] = mean_array_changed
    final_df.loc[:,"mask"] = mask
    final_df.to_csv("total_df.csv", index=False)

    # #create prediction
    prediction_encoder_LOWORDER,prediction_encoder_sliding_LOWORDER, encoderinput_removed_tensor = denoise_floaterCurrent_encoder_interpolation_CE("weights/Encoder_Interpolation_CE_LowOrder_300.pt",1000, final_df["current"].to_numpy(), config,final_df["mask"])

    prediction_encoder_HIGHORDER,prediction_encoder_sliding_HIGHORDER, encoderinput_removed_tensor = denoise_floaterCurrent_encoder_interpolation_CE("weights/Encoder_Interpolation_CE_HighOrder_300.pt",1000, final_df["current"].to_numpy(), config,final_df["mask"])

    # prediction_encoder_PERIODIC,prediction_encoder_sliding_PERIODIC, encoderinput_removed_tensor = denoise_floaterCurrent_encoder_interpolation_CE("weights/Encoder_Interpolation_CE_Periodic_300.pt",1000, final_df["current"].to_numpy(), config,final_df["mask"])

    prediction_encoder_ALLINONE,prediction_encoder_sliding_ALLINONE, encoderinput_removed_tensor = denoise_floaterCurrent_encoder_interpolation_CE("weights/Encoder_Interpolation_CE_AllInOne_2400.pt",1000, final_df["current"].to_numpy(), config,final_df["mask"])

    prediction_encoder_PERIODICSUM,prediction_encoder_sliding_PERIODICSUM, encoderinput_removed_tensor = denoise_floaterCurrent_encoder_interpolation_CE("weights/Encoder_Interpolation_CE_PeriodicSum_300.pt",1000, final_df["current"].to_numpy(), config,final_df["mask"])

    x_values = np.arange(shape)
    result = []

    #calculate vertical color painting for interpolation data
    intervalle = []
    nan_indices = np.where(np.isnan(y_masked))[0]
    length = len(nan_indices)

    if length > 0:
        start = nan_indices[0]  # Startwert des ersten Intervalls
        
        for i in range(1, length):
            # Wenn der Abstand zum vorherigen Wert 2 oder mehr beträgt, endet das Intervall
            if nan_indices[i] - nan_indices[i - 1] >= 2:
                intervalle.append((start, nan_indices[i - 1]))  # Aktuelles Intervall speichern
                start = nan_indices[i]  # Neues Intervall beginnen
        
        # Letztes Intervall hinzufügen
        intervalle.append((start, nan_indices[-1]))
    # result.append([[current, "Noisy Data (Current)"],[encoderinput_removed_tensor, "Noisy Data (Interpolation)"],[mean_array_changed, "current mean border"], [prediction_encoder_LOWORDER, "Prediction (LO)"],[prediction_encoder_HIGHORDER, "Prediction (HO)"],[prediction_encoder_PERIODIC, "Prediction (P)"],[prediction_encoder_ALLINONE, "Prediction (C)"]])

    # result.append([[current, "Noisy Data (Current)"],[encoderinput_removed_tensor, "Noisy Data (Interpolation)"],[mean_array_changed, "current mean border"], [prediction_encoder_sliding_LOWORDER, "Prediction (LO) (Sliding Window)"],[prediction_encoder_sliding_HIGHORDER, "Prediction (HO) (Sliding Window)"],[prediction_encoder_sliding_PERIODIC, "Prediction (P) (Sliding Window)"],[prediction_encoder_sliding_ALLINONE, "Prediction (C) (Sliding Window)"]])
    # plot_multiple_pred_with_names(x_values, result, 0.33, "Floating Current BALd_Lifun_183 22.05.2023-14.03.2024", f"Time Steps ({dflength*10} s / Timestep)", "Current (A)")

    result.append([
        [current, "Noisy Data", "blue"],
                   [y_masked, "Interpolation Data", "cornflowerblue"],
                    [mean_array_changed, "Interpolation Border", "gold"],
                    [prediction_encoder_sliding_LOWORDER, "LO", "purple"],
                    [prediction_encoder_sliding_HIGHORDER, "HO", "mediumvioletred"],
                    [prediction_encoder_sliding_PERIODICSUM, "PS", "darkturquoise"],
                    [prediction_encoder_sliding_ALLINONE, "C", "yellowgreen"]
                    ])
        
    titles = ["CET-ZIR Modell"]
    labels = ["A"]

    # titles = ["Prediction CET-TSIR Model"]
    # labels = ["dV/dQ"]
        
    plot_multiple_pred_with_names_error_bar_single(x_values, result, titles, labels, intervalle)


plot_floater_current_CE()