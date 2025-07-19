import sys
sys.path.append(".")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from useful import round_numbers_individually
from create_data import generate_noisy_data_slope
import torch
from pathlib import Path

#perform min max scaling
def performMinMaxScaling(yValues, minValue, divTerm):
    MinMaxResult = torch.tensor((yValues-minValue)/divTerm,dtype=torch.float32)[:]
    return MinMaxResult

def performRoundingProcess(vocabSize, yValues):
    RoundingResult = round_numbers_individually(vocabSize,yValues)[:]
    return RoundingResult



if __name__ == "__main__":
    #read in data via csv file (groundTruth, Noise)
    try:
        for i in range(10):
            #settings for data generation
            x_values = np.arange(0,1000)
            y_start = 50
            lower_slope = -500.0
            upper_slope = 500.0
            vocab_size = 5
            lowerNoiseStdUniformShare = 0.1
            upperNoiseStdUniformShare = 0.15
            noise = ["uni", lowerNoiseStdUniformShare, upperNoiseStdUniformShare]
            #erstelle Kurve mit generate_noisy_data_slope
            y_spline, y_noise_spline,min_value, max_value, noise_std = generate_noisy_data_slope(x_values,y_start,lower_slope, upper_slope, vocab_size, noise)
            divTerm = max_value-min_value

            plt.plot(x_values, y_noise_spline, y_spline)
            plt.show()

            # Benutzerabfrage zum Abbrechen
            abbruch = input("Zum Abbrechen 'q' eingeben, sonst Enter drücken: ")
            if abbruch.lower() == 'q':
                #plot results in one plot
                fig, (ax1, ax2) = plt.subplots(2,1, figsize=(11, 5))
                #perform min max scaling with Groundtruth and noisy data
                ySplineMinMax = performMinMaxScaling(y_spline, min_value, divTerm)
                yNoiseMinMax = performMinMaxScaling(y_noise_spline, min_value, divTerm)
                NoiseCopy = np.array(yNoiseMinMax)
                SplineCopy = np.array(ySplineMinMax)
                

                #perform rounding with Groundtruth und noisy data
                ySplineRounded = performRoundingProcess(vocab_size, ySplineMinMax)
                yNoiseRounded = performRoundingProcess(vocab_size, yNoiseMinMax)

                borders = [np.ones(1000) * 0.1,np.ones(1000) * 0.3,np.ones(1000) * 0.5,np.ones(1000) * 0.7,np.ones(1000) * 0.9]
                for border in borders:
                    ax1.plot(x_values, border, alpha=0.15, color="black",linewidth=1)
                    ax2.plot(x_values, border, alpha=0.15, color="black",linewidth=1)


                # ax1.plot(x_values, yNoiseRounded,label="Rounding(NoiseData)",color="blue")
                # ax1.plot(x_values, NoiseCopy,label= "MinMax(NoiseData)", color="blue",linestyle='--',alpha=0.5)
                
                # ax2.plot(x_values, ySplineRounded, label="Rounding(Groundtruth)", color="red")
                # ax2.plot(x_values, SplineCopy, label="MinMax(GroundTruth)", color="red",alpha=0.5, linestyle='--')


                ax1.plot(x_values, NoiseCopy,label= "MinMax(NoiseData)", color="blue", linewidth=2, alpha=0.5)
                ax1.plot(x_values, SplineCopy, label="MinMax(GroundTruth)", color="orange",linewidth=2)
                
                ax2.plot(x_values, yNoiseRounded,label="Rounding(NoiseData)",color="blue",linewidth=2, alpha=0.5)
                ax2.plot(x_values, ySplineRounded, label="Rounding(Groundtruth)", color="orange",linewidth=2)
                

                

                fmt = "svg"
                folderName = "RoundingProcess"
                title = "Rounding"
                latex_font_size_pt = 12 # Schriftgröße in LaTeX-Punkten (z. B. 10pt, 12pt)
                labelSizeY = latex_font_size_pt
                labelSizeX = latex_font_size_pt -3
                labelSizeLegend = latex_font_size_pt


                ax1.set_ylabel("Min Max Scaling",fontsize=labelSizeY,fontweight='bold')
                ax2.set_ylabel("Discretization",fontsize=labelSizeY,fontweight='bold')
                ax2.set_xlabel("Time Steps",fontsize=labelSizeX)
                ax1.margins(x=0.0)
                ax2.margins(x=0.0)
                offset = (divTerm)*0.05
                ax1.tick_params(axis='x', which='both', bottom=False, left=True, labelbottom=False, labelleft=False)
                ax2.tick_params(axis='x', which='both', bottom=False, left=True, labelbottom=False, labelleft=False)
                ax1.legend(prop={'weight': 'bold'},fontsize=14)
                ax2.legend(prop={'weight': 'bold'},fontsize=14)

                plt.tight_layout()
                save_path = Path(f"save_figure/{folderName}/")
                save_path.mkdir(parents=True, exist_ok=True)
                file_path = save_path / f"{title}.{fmt}" if save_path else f"{title}.{fmt}"
                plt.gcf().savefig(file_path, format=fmt, dpi=300, bbox_inches='tight')

                plt.show()


                break

    except KeyboardInterrupt:
        print("Abbruch per Tastenkombination (Strg+C).")


    #mark specific rounding region to emphasize process (vocabSize = 6)