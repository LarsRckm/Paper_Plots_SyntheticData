import sys
sys.path.append(".")
import numpy as np

from create_data import callFunctionPaper, remove_parts_of_graph_encoder
from plot import predict_encoder_interpolation_projection_roundedInput, plot_multiple_pred_with_names_error_bar_area, plot_multiple_pred_with_names_error_bar, plot_multiple_pred_with_names_error_bar_interpolationPainting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path
from random import randint


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



# def plot_encoder_interpolation_CE(count):
#     #create artificial timeseries with arbitrary interpolation intervals
#     config = get_config()
#     seq_len = 1000
#     x_values = np.arange(0,seq_len)
#     y_start = np.random.uniform(config["y_lim"][0]+1,config["y_lim"][1]-1)
#     y_values_spline, y_noise_spline,min_value_spline, max_value_spline, noise_std = callFunction(x_values, y_start, config['random_number_range'], config["spline_value"], config["vocab_size"])
#     mask = remove_parts_of_graph_encoder(x_values,y_noise_spline, config["width_array_encoder"], config["offset"], config["x_lim"])


#     #different models for model encoder interpolation CE at different training levels
#     # model_filename = "weights/Encoder_Interpolation_CE_PeriodicSum_2000.pt"
#     # model_filename = "weights/Encoder_Interpolation_CE_Periodic_300.pt"
#     model_filename = "weights/Encoder_Interpolation_CE_LowOrder_300.pt"
#     # model_filename = "weights/Encoder_Interpolation_CE_HighOrder_300.pt"
#     # model_filename = "weights/Encoder_Interpolation_CE_AllInOne_2400.pt"
#     # model_filename = "weights/Encoder_Interpolation_CE_Discontinuous_1200.pt"
    
#     #create prediction
#     prediction_encoder, encoder_removed = predict_encoder_interpolation_projection_roundedInput(seq_len, model_filename, y_noise_spline,min_value_spline, max_value_spline, config, mask)
#     prediction_encoder = prediction_encoder.squeeze()
#     y_masked = np.where(mask == 0, y_noise_spline, np.nan)




#     #calculate error plot
#     difference_std = prediction_encoder-y_values_spline
#     difference_std = difference_std/noise_std

#     difference_2std = prediction_encoder-y_values_spline
#     difference_2std = difference_2std/(2*noise_std)

#     difference_3std = prediction_encoder-y_values_spline
#     difference_3std = difference_3std/(3*noise_std)

#     # #plot results
#     result = []


#     # # #calculate vertical color painting for interpolation data
#     intervalle = []
#     nan_indices = np.where(np.isnan(y_masked))[0]
#     length = len(nan_indices)

#     if length > 0:
#         start = nan_indices[0]  # Startwert des ersten Intervalls
        
#         for i in range(1, length):
#             # Wenn der Abstand zum vorherigen Wert 2 oder mehr beträgt, endet das Intervall
#             if nan_indices[i] - nan_indices[i - 1] >= 2:
#                 intervalle.append((start, nan_indices[i - 1]))  # Aktuelles Intervall speichern
#                 start = nan_indices[i]  # Neues Intervall beginnen
        
#         # Letztes Intervall hinzufügen
#         intervalle.append((start, nan_indices[-1]))
        

    
#     #adding the first subplots data
#     result.append([[y_masked, "Noisy Data", "blue"], [y_values_spline, "Ground Truth", "orange"], [prediction_encoder.detach().numpy(), "Prediction", "green"]])
#     #adding the second subplots data
#     result.append([[difference_std, "relativer Fehler bei Sigma=1", "navy"],[-1, 1, "navy"],[-2, 2, "royalblue"],[-3, 3, "lightsteelblue"]])


#     # titles = ["CET-TSIR Model", "Relative Error Rate vs. Noise Standard Deviation"]
#     # labels = ["synthetic\nData", "\u03C3"]

#     # titles = ["CET-ZIR Modell", "Relativer Fehler bezogen auf die Störgrößenstandardabweichung"]
#     # titles = ["", ""]
#     # labels = ["synthetische\nDaten", "\u03C3"]

#     titles = ["CET-TSIR Model", "Relative error rate relating to noise std value"]
#     labels = ["synthetic\nData", "\u03C3"]

    # # plot_multiple_pred_with_names_error_bar(x_values, result, titles, labels)
    # 
    # plot_multiple_pred_with_names_error_bar_interpolationPainting(x_values, result, titles, labels, intervalle, f"Prediction_{count}", "LowOrder")




def multipleSubplots():
    model_filename = ["weights/Encoder_Interpolation_CE_PeriodicSum_2000.pt", 
                    #   "weights/Encoder_Interpolation_CE_Periodic_300.pt", 
                    #   "weights/Encoder_Interpolation_CE_LowOrder_300.pt", 
                    #   "weights/Encoder_Interpolation_CE_HighOrder_300.pt", 
                      "weights/Encoder_Interpolation_CE_AllInOne_2400.pt", 
                      "weights/Encoder_Interpolation_CE_Discontinuous_1200.pt"
                      ]

    modelFilename_TimeseriesGenerator = {
        "_PeriodicSum_": [5],
        "_Periodic_": [3],
        "_LowOrder_": [0,1,2],
        "_HighOrder_": [4],
        "_AllInOne_": [0,1,2,3,4,5,6,7],
        "_Discontinuous_": [6,7]
    }

    colorNoisyData = "blue"
    colorGroundTruth = "orange"
    colorPrediction = "green"
    lineWidth = 2
    folderName = "MultipleSubplots"
    title = "MultipleSubplots"
    fmt = "svg"

    resultFirstPlot = []
    resultSecondPlot = []
    intervals = []
    labels = []
    rangePlots = [[-1, 1, "navy"],[-2, 2, "royalblue"],[-3, 3, "lightsteelblue"]]
    labels = ["synthetic\nData", "\u03C3"]
    #jede einzelne model_filename über eine for schleife durchgehen
    for model in model_filename:
        for key, value in modelFilename_TimeseriesGenerator.items():
            if key in model:
                #variables
                arr_len = len(value)
                random_int = randint(0,arr_len-1)
                config = get_config()
                seq_len = 1000
                x_values = np.arange(0,seq_len)

                
                #create timeSeries
                y_start = np.random.uniform(config["y_lim"][0]+1,config["y_lim"][1]-1)
                y_values_spline, y_noise_spline,min_value_spline, max_value_spline, noise_std = callFunctionPaper(x_values, y_start, config['random_number_range'], config["spline_value"], config["vocab_size"], value[random_int])
                mask = remove_parts_of_graph_encoder(x_values,y_noise_spline, config["width_array_encoder"], config["offset"], config["x_lim"])

                #create prediction
                prediction_encoder, encoder_removed = predict_encoder_interpolation_projection_roundedInput(seq_len, model, y_noise_spline,min_value_spline, max_value_spline, config, mask)
                prediction_encoder = prediction_encoder.squeeze()
                y_masked = np.where(mask == 0, y_noise_spline, np.nan)


                #calculate error plot
                difference_std = prediction_encoder-y_values_spline
                difference_std = difference_std/noise_std

                # # #calculate vertical color painting for interpolation data
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


                resultFirstPlot.append([[y_masked, "Noisy Data", colorNoisyData], [y_values_spline, "Ground Truth", colorGroundTruth], [prediction_encoder.detach().numpy(), "Prediction", colorPrediction]])
                #adding the second subplots data
                resultSecondPlot.append([[difference_std, "relativer Fehler bei Sigma=1", "navy"]])

                intervals.append(intervalle)





    a4_width_inch = 8.27  #DIN A4 Breite in Zoll für volle Breite
    a4_height_inch = 11  #DIN A4 Höhe in Zoll für volle Breite
    
    latex_font_size_pt = 10 # Schriftgröße in LaTeX-Punkten (z. B. 10pt, 12pt)
    points_to_inches = 1 / 72 # Umrechnung von Punkt in Zoll (1 pt = 1/72 Zoll)
    labelSizeY = latex_font_size_pt
    labelSizeX = latex_font_size_pt -3
    labelSizeLegend = latex_font_size_pt

    num_plots = len(model_filename)
    n_rows = num_plots//2 if num_plots%2 == 0 else (num_plots//2)+1

    # Hauptfigur und äußeres GridSpec mit Abständen
    fig = plt.figure(figsize=(a4_width_inch, a4_height_inch))
    outer_grid = GridSpec(n_rows, 2, figure=fig)  # Abstände festlegen


    for idx in range(num_plots):
        # Inneres GridSpec für jeden Haupt-Subplot
        inner_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[idx // 2, idx % 2])  # 2x1 Inneres Layout

        # Erster innerer Subplot
        ax1 = fig.add_subplot(inner_grid[0, 0])
        # Zweiter innerer Subplot
        ax2 = fig.add_subplot(inner_grid[1, 0])
        #plot first subplot data
        for yValues, label, color in resultFirstPlot[idx]:
            ax1.plot(x_values, yValues, label=f"{label}", linewidth=lineWidth, color=color)
        #plot second subplot data
        for yValues, label, color in resultSecondPlot[idx]:
            maxValue = max(yValues)
            minValue = min(yValues)
            ax2.plot(x_values, yValues, label=f"{label}", linewidth=lineWidth, color=color)
        #display border to illustrate difference between prediction and groundTruth
        for lowerBorder, upperBorder, color in rangePlots:
            ax2.fill_between(x_values, lowerBorder, upperBorder, color=color, alpha=0.1)
        #display interpolation intervals in both subplots
        for bottom_border, upper_border in intervals[idx]:
            ax1.axvspan(bottom_border - 0.5, upper_border + 0.5, color='red', alpha=0.1)
            ax2.axvspan(bottom_border - 0.5, upper_border + 0.5, color='red', alpha=0.1)
        

        
        ax1.set_ylabel(labels[0],fontsize=labelSizeY)
        ax2.set_ylabel(labels[1],fontsize=labelSizeY)
        ax2.set_xlabel("Time Steps",fontsize=labelSizeX)
        ax1.margins(x=0.0)
        ax2.margins(x=0.0)
        offset = (maxValue-minValue)*0.05
        ax2.set_ylim(minValue-offset, maxValue+offset)
        ax1.tick_params(axis='x', which='both', bottom=False, left=True, labelbottom=False, labelleft=False)
        ax2.tick_params(axis='x', which='both', bottom=True, left=True, labelbottom=False, labelleft=False)
    

    plt.tight_layout()
    save_path = Path(f"save_figure/{folderName}/")
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / f"{title}.{fmt}" if save_path else f"{title}.{fmt}"
    plt.gcf().savefig(file_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.show()
        



if __name__ == "__main__":
    for i in range(20):
        # plot_encoder_interpolation_CE(i)
        multipleSubplots()