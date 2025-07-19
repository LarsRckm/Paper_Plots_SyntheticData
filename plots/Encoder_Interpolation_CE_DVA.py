import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from plot import plot_multiple_pred_with_names, denoise_floaterCurrent_encoder_interpolation_CE, plot_multiple_pred_with_names_error_bar_single
import pandas as pd
import numpy as np
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



if __name__ == "__main__":
    config = get_config()
    files = list_files_in_directory("DVA/")

    for file in files:
        # data = np.load("DVA/dva_checkup000.npy")
        data = np.load(f"DVA/{file}")
        data = pd.DataFrame(data)
        data = data.iloc[1500:]
        data = data.iloc[:-500]
        sampling_rate = math.floor(len(data)/1000)
        # sampling_rate = 5
        data = data.iloc[0::sampling_rate, 0]
        data = data[:1000]
        # data = data.iloc[250:]
        length = len(data)
        x_values = np.arange(length)

        # fig, ax = plt.subplots(1,1)
        # ax.plot(x_values, data)
        # plt.show()

        
        mask = np.zeros(shape=length,dtype=np.float32)

        final_df = pd.DataFrame()
        final_df.loc[:,"current"] = data
        final_df.loc[:,"mean_1_comma_2"] = data
        final_df.loc[:,"mask"] = mask
        final_df.to_csv("total_df.csv", index=False)

        #create prediction
        prediction_encoder_LOWORDER,prediction_encoder_sliding_LOWORDER, encoderinput_removed_tensor = denoise_floaterCurrent_encoder_interpolation_CE("weights/Encoder_Interpolation_CE_LowOrder_300.pt",1000, final_df["current"].to_numpy(), config,final_df["mask"])


        # df = pd.DataFrame(prediction_encoder_sliding_LOWORDER)
        # df.to_csv("DVAGroundTruth.csv", index=False)

        prediction_encoder_HIGHORDER,prediction_encoder_sliding_HIGHORDER, encoderinput_removed_tensor = denoise_floaterCurrent_encoder_interpolation_CE("weights/Encoder_Interpolation_CE_HighOrder_300.pt",1000, final_df["current"].to_numpy(), config,final_df["mask"])

        prediction_encoder_PERIODIC,prediction_encoder_sliding_PERIODIC, encoderinput_removed_tensor = denoise_floaterCurrent_encoder_interpolation_CE("weights/Encoder_Interpolation_CE_Periodic_300.pt",1000, final_df["current"].to_numpy(), config,final_df["mask"])

        prediction_encoder_ALLINONE,prediction_encoder_sliding_ALLINONE, encoderinput_removed_tensor = denoise_floaterCurrent_encoder_interpolation_CE("weights/Encoder_Interpolation_CE_AllInOne_2400.pt",1000, final_df["current"].to_numpy(), config,final_df["mask"])

        prediction_encoder_sliding_PERIODICSUM,prediction_encoder_sliding_PERIODICSUM, encoderinput_removed_tensor = denoise_floaterCurrent_encoder_interpolation_CE("weights/Encoder_Interpolation_CE_PeriodicSum_2000.pt",1000, final_df["current"].to_numpy(), config,final_df["mask"])


        # x_values = np.arange(shape)
        result = []
        # result.append([[data, "Noisy Data"], [prediction_encoder_LOWORDER, "Prediction (LO)"], [prediction_encoder_HIGHORDER, "Prediction (HO)"], [prediction_encoder_PERIODIC, "Prediction (P)"], [prediction_encoder_ALLINONE, "Prediction (C)"]])
        
        
        # result.append([[data, "Noisy Data"], [prediction_encoder_sliding_LOWORDER, "Prediction (LO) (Sliding Window)"], [prediction_encoder_sliding_HIGHORDER, "Prediction (HO) (Sliding Window)"], [prediction_encoder_sliding_PERIODIC, "Prediction (P) (Sliding Window)"], [prediction_encoder_sliding_ALLINONE, "Prediction (C) (Sliding Window)"]])

        # result.append([[data, "Noisy Data"], [prediction_encoder_sliding_LOWORDER, "Prediction (LO) (Sliding Window)"]])
        # plot_multiple_pred_with_names(x_values, result, 0.33, "DVA Data", "Time Steps", "dV / dQ")


        result.append([[data, "Noisy Data", "blue", 7],
                    [prediction_encoder_sliding_LOWORDER, "LO", "lightpink", 5],
                    [prediction_encoder_sliding_HIGHORDER, "HO", "mediumvioletred", 5],
                    [prediction_encoder_sliding_PERIODICSUM, "PS", "darkturquoise", 5],
                    [prediction_encoder_sliding_ALLINONE, "C", "yellowgreen", 5]
                    ])
        
        titles = ["CET-ZIR Model Predictions"]
        labels = ["dV/dQ"]

        # titles = ["Prediction CET-TSIR Model"]
        # labels = ["dV/dQ"]
        intervalle = []


        plot_multiple_pred_with_names_error_bar_single(x_values, result, titles, labels, intervalle)