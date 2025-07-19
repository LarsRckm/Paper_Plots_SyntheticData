import sys
sys.path.append(".")
import numpy as np

from create_data import createTimeSeries_PS, remove_parts_of_graph_encoder
from plot import predict_encoder_interpolation_projection_roundedInput, plot_multiple_pred_with_names_error_bar_area, plot_multiple_pred_with_names_error_bar, plot_multiple_pred_with_names_error_bar_interpolationPainting



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



def plot_encoder_interpolation_CE():
    #create artificial timeseries with arbitrary interpolation intervals
    config = get_config()
    seq_len = 1000
    x_values = np.arange(0,seq_len)
    y_start = np.random.uniform(config["y_lim"][0]+1,config["y_lim"][1]-1)
    y_values_spline, y_noise_spline,min_value_spline, max_value_spline, noise_std = createTimeSeries_PS(x_values, y_start, config['random_number_range'], config["spline_value"], config["vocab_size"])
    mask = remove_parts_of_graph_encoder(x_values,y_noise_spline, config["width_array_encoder"], config["offset"], config["x_lim"])


    #different models for model encoder interpolation CE at different training levels
    model_filename = "weights/Encoder_Interpolation_CE_PeriodicSum_2000.pt"
    
    #create prediction
    prediction_encoder, encoder_removed = predict_encoder_interpolation_projection_roundedInput(seq_len, model_filename, y_noise_spline,min_value_spline, max_value_spline, config, mask)
    prediction_encoder = prediction_encoder.squeeze()
    y_masked = np.where(mask == 0, y_noise_spline, np.nan)




    #calculate error plot
    difference_std = prediction_encoder-y_values_spline
    difference_std = difference_std/noise_std

    difference_2std = prediction_encoder-y_values_spline
    difference_2std = difference_2std/(2*noise_std)

    difference_3std = prediction_encoder-y_values_spline
    difference_3std = difference_3std/(3*noise_std)

    #calculate upper and lower noise standard deviation plots
    upper_deviation = y_values_spline + noise_std
    lower_deviation = y_values_spline - noise_std

    upper_deviation_2 = y_values_spline + 2*noise_std
    lower_deviation_2 = y_values_spline - 2*noise_std

    upper_deviation_3 = y_values_spline + 3*noise_std
    lower_deviation_3 = y_values_spline - 3*noise_std



    #plot results
    result = []



    # noise_proportion = noise_std / (max(y_values_spline)-min(y_values_spline))
    # if (noise_proportion >= 0.1):
    #     # adding the first subplots data
    #     result.append([[y_masked, "Noisy Data", "blue"], [y_values_spline, "Ground Truth", "orange"], [prediction_encoder.detach().numpy(), "Prediction", "green"]])
    #     #adding  the second subplots data
    #     result.append([
    #             [y_values_spline, "GroundTruth", "orange"], 
    #             [prediction_encoder.detach().numpy(), "Prediction", "green"], 
    #             [[lower_deviation, "Lower Standard Deviation Border", "navy"], [upper_deviation, "Upper Standard Deviation Border", "navy"]],
    #             [[lower_deviation_2, "Lower Standard Deviation Border", "royalblue"], [upper_deviation_2, "Upper Standard Deviation Border", "royalblue"]],
    #             [[lower_deviation_3, "Lower Standard Deviation Border", "lightsteelblue"], [upper_deviation_3, "Upper Standard Deviation Border", "lightsteelblue"]],
    #         ]) 
    #     #adding the third subplots data
    #     result.append([[difference_std, "relativer Fehler bei Sigma=1", "navy"]])

    #     #creating title list
    #     titles = ["CET-ZIR Modell", "Darstellung der Störgrößenstandardabweichung bezogen auf die Ground Truth", "Relativer Fehler bezogen auf die Störgrößenstandardabweichung"]

    #     labels = ["synthetische\nDaten","synthetische\nDaten","\u03C3"]

    #     plot_multiple_pred_with_names_error_bar_area(x_values, result, titles, labels)


    # #calculate vertical color painting for interpolation data
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
        

    
    #adding the first subplots data
    result.append([[y_masked, "Noisy Data", "blue"], [y_values_spline, "Ground Truth", "orange"], [prediction_encoder.detach().numpy(), "Prediction", "green"]])
    #adding the second subplots data
    result.append([[difference_std, "relativer Fehler bei Sigma=1", "navy"],[-1, 1, "navy"],[-2, 2, "royalblue"],[-3, 3, "lightsteelblue"]])


    titles = ["CET-TSIR Model", "Relative Error Rate vs. Noise Standard Deviation"]
    labels = ["synthetic\nData", "\u03C3"]

    # titles = ["CET-ZIR Modell", "Relativer Fehler bezogen auf die Störgrößenstandardabweichung"]
    # labels = ["synthetische\nDaten", "\u03C3"]

    # # plot_multiple_pred_with_names_error_bar(x_values, result, titles, labels)
    plot_multiple_pred_with_names_error_bar_interpolationPainting(x_values, result, titles, labels, intervalle)




        




if __name__ == "__main__":
    for i in range(20):
        plot_encoder_interpolation_CE()