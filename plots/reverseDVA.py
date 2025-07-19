import sys
sys.path.append(".")

import matplotlib.pyplot as plt
from plot import plot_multiple_pred_with_names_error_bar_area, denoise_floaterCurrent_encoder_interpolation_CE, plot_multiple_pred_with_names_multiple_error_bar, plot_multiple_pred_with_names_error_bar
import pandas as pd
import numpy as np
from random import normalvariate, randint, uniform
from useful import calc_exp
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


def createNoisyData(y_trend):
    x_values = y_trend.shape[0]
    # noise_std_value = abs(normalvariate(0,(max(y_trend)-min(y_trend))*0.01))
    noise_std_value = (max(y_trend)-min(y_trend))*0.1
    y_noise = np.random.normal(0, noise_std_value, x_values)

    y_trend_noise = y_trend + y_noise

    # exp = calc_exp(smallest_number=(1/vocab_size))
    #rounding max to ceil and min to floor to be able to display values properly
    # max_value = math.ceil(max(max(y_trend_noise), max(y_trend))*(10**exp))/(10**exp)
    # min_value = math.floor(min(min(y_trend_noise), min(y_trend))*(10**exp))/(10**exp)

    return y_trend_noise, noise_std_value


if __name__ == "__main__":
    vocab_size = 100000
    config = get_config()

    #lese gespeicherte GroundTruth ein und füge Rauschen hinzu
    y_trend = pd.read_csv("DVAGroundTruth.csv").squeeze().to_numpy()
    y_trend = y_trend[:1000]
    y_trend_len = y_trend.shape[0]
    x_values = np.arange(0,y_trend_len)
    mask = np.zeros(shape=y_trend_len,dtype=np.float32)
    y_trend_noise, noise_std_value = createNoisyData(y_trend)

    #PERIODIC
    #------------------------------------------------------------------------------------------------------------------------------
    # prediction_encoder_PERIODIC,prediction_encoder_sliding_PERIODIC, encoderinput_removed_tensor = denoise_floaterCurrent_encoder_interpolation_CE("weights/Encoder_Interpolation_CE_Periodic_300.pt",1000, y_trend_noise, config,mask)

    # differencePERIODIC = prediction_encoder_PERIODIC-y_trend
    # differencePERIODIC = differencePERIODIC/noise_std_value

    #PERIODIC SUM
    #------------------------------------------------------------------------------------------------------------------------------
    prediction_encoder_PERIODICSUM,prediction_encoder_sliding_PERIODICSUM, encoderinput_removed_tensor = denoise_floaterCurrent_encoder_interpolation_CE("weights/Encoder_Interpolation_CE_PeriodicSum_300.pt",1000, y_trend_noise, config,mask)

    differencePERIODICSUM = prediction_encoder_PERIODICSUM-y_trend
    differencePERIODICSUM = differencePERIODICSUM/noise_std_value


    #LOW ORDER
    #------------------------------------------------------------------------------------------------------------------------------
    prediction_encoder_LOWORDER,prediction_encoder_sliding_LOWORDER, encoderinput_removed_tensor = denoise_floaterCurrent_encoder_interpolation_CE("weights/Encoder_Interpolation_CE_LowOrder_300.pt",1000, y_trend_noise, config,mask)

    differenceLOWORDER = prediction_encoder_LOWORDER-y_trend
    differenceLOWORDER = differenceLOWORDER/noise_std_value


    #HIGH ORDER
    #------------------------------------------------------------------------------------------------------------------------------
    prediction_encoder_HIGHORDER,prediction_encoder_sliding_HIGHORDER, encoderinput_removed_tensor = denoise_floaterCurrent_encoder_interpolation_CE("weights/Encoder_Interpolation_CE_HighOrder_300.pt",1000, y_trend_noise, config,mask)

    differenceHIGHORDER = prediction_encoder_HIGHORDER-y_trend
    differenceHIGHORDER = differenceHIGHORDER/noise_std_value


    #ALL IN ONE (COMBINED)
    #------------------------------------------------------------------------------------------------------------------------------
    prediction_encoder_ALLINONE,prediction_encoder_sliding_ALLINONE, encoderinput_removed_tensor = denoise_floaterCurrent_encoder_interpolation_CE("weights/Encoder_Interpolation_CE_AllInOne_2400.pt",1000, y_trend_noise, config,mask)

    differenceALLINONE = prediction_encoder_ALLINONE-y_trend
    differenceALLINONE = differenceALLINONE/noise_std_value



    #calculate upper and lower noise standard deviation plots
    upper_deviation = y_trend + noise_std_value
    lower_deviation = y_trend - noise_std_value

    upper_deviation_2 = y_trend + 2*noise_std_value
    lower_deviation_2 = y_trend - 2*noise_std_value

    upper_deviation_3 = y_trend + 3*noise_std_value
    lower_deviation_3 = y_trend - 3*noise_std_value







    result = []
    #adding the first subplots data
    result.append([[y_trend_noise, "Noisy Data", "blue"],
                    [y_trend, "Ground Truth", "orange"], 
                    [prediction_encoder_LOWORDER, "LO", "purple"],
                    [prediction_encoder_HIGHORDER, "HO", "mediumvioletred"],
                    [prediction_encoder_PERIODICSUM, "PS", "darkturquoise"],
                    [prediction_encoder_ALLINONE, "C", "yellowgreen"]])
    #adding the second subplots data
    result.append([[differenceHIGHORDER, "relativer Fehler bei Sigma=1", "mediumvioletred"],
                   [differencePERIODICSUM, "relativer Fehler bei Sigma=1", "darkturquoise"],
                   [differenceALLINONE, "relativer Fehler bei Sigma=1", "yellowgreen"],
                   [differenceLOWORDER, "relativer Fehler bei Sigma=1", "purple"],
                   [-1, 1, "navy"],
                   [-2, 2, "royalblue"],
                   [-3, 3, "lightsteelblue"]])


    titles = ["Vorhersagen CET-ZIR Modell", "Relativer Fehler bezogen auf die Störgrößenstandardabweichung"]
    labels = ["dV/dQ", "\u03C3"]

    plot_multiple_pred_with_names_error_bar(x_values, result, titles, labels)




    
    # #first subplot
    # result.append([[y_trend_noise, "Noisy Data", "blue"], 
    #                [y_trend, "GroundTruth", "orange"], 
    #                [prediction_encoder_LOWORDER, "LO", "fuchsia"],[prediction_encoder_HIGHORDER, "HO", "darkviolet"],[prediction_encoder_PERIODICSUM, "PS", "cyan"],[prediction_encoder_ALLINONE, "C", "lawngreen"]])
    # #adding  the second subplots data
    # result.append([
    #         [y_trend, "GroundTruth", "orange"], 
    #         [prediction_encoder_LOWORDER, "Prediction", "fuchsia"],
    #         [prediction_encoder_HIGHORDER, "Prediction", "darkviolet"],
    #         [prediction_encoder_PERIODICSUM, "Prediction", "cyan"],
    #         [prediction_encoder_ALLINONE, "Prediction", "lawngreen"],
    #         [[lower_deviation, "Lower Standard Deviation Border", "navy"], [upper_deviation, "Upper Standard Deviation Border", "navy"]],
    #         [[lower_deviation_2, "Lower Standard Deviation Border", "royalblue"], [upper_deviation_2, "Upper Standard Deviation Border", "royalblue"]],
    #         [[lower_deviation_3, "Lower Standard Deviation Border", "lightsteelblue"], [upper_deviation_3, "Upper Standard Deviation Border", "lightsteelblue"]],
    #     ])
    # #adding the third subplots data
    # result.append([[differenceLOWORDER, "relativer Fehler bei Sigma=1", "fuchsia"],
    #                [differenceHIGHORDER, "relativer Fehler bei Sigma=1", "darkviolet"],[differencePERIODICSUM, "relativer Fehler bei Sigma=1", "cyan"],
    #                [differenceALLINONE, "relativer Fehler bei Sigma=1", "lawngreen"],])
    
    # titles = ["Vorhersagen CET-ZIR Modell", "Darstellung der Störgrößenstandardabweichung bezogen auf die GroundTruth", "Relativer Fehler bezogen auf die Störgrößenstandardabweichung"]
    # labels = ["dV/dQ","dV/dQ","sigma"]

    # plot_multiple_pred_with_names_error_bar_area(x_values, result, titles, labels)
    # result.append([
    #     [y_trend_noise, "Noisy Data"],
    #       [y_trend, "GroundTruth"],
    #         [prediction_encoder_LOWORDER,differenceLOWORDER, "LO", "Error (LO)"],
    #           [prediction_encoder_HIGHORDER,differenceHIGHORDER, "HO", "Error (HO)"],
    #           [prediction_encoder_PERIODIC,differencePERIODIC, "P", "Error (P)"],
    #           [prediction_encoder_PERIODICSUM,differencePERIODICSUM, "PS", "Error (PS)"],
    #           [prediction_encoder_ALLINONE,differenceALLINONE, "C", "Error (C)"]])

    # plot_multiple_pred_with_names_multiple_error_bar(x_values, result, noise_std_value, "CET-ZIR Modell")


