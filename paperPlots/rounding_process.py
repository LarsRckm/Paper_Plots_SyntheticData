import sys
sys.path.append(".")
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from useful import round_numbers_individually
from create_data import generate_noisy_data_periodic


mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],  # Standard-LaTeX-Schrift
    "axes.labelsize": 20,   # Achsentitel
    # "font.size": 15,        # Gesamtschriftgröße
    "legend.fontsize": 15,
    # "xtick.labelsize": 9,
    # "ytick.labelsize": 9,
})

colorcodes = {
    "y_raw": "#92c5de",
    "y_groundtruth": "#0571b0",
    "Classic": "#67001f",
    "Interpolation": "#cccccc",
    "Model": "#f4a582",
    "Error": "#ca0020"
}

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
            vocab_size = 5
            lowerNoiseStdUniformShare = 0.05
            upperNoiseStdUniformShare = 0.15
            noise = ["uni", lowerNoiseStdUniformShare, upperNoiseStdUniformShare]
            y_spline, y_noise_spline,min_value, max_value, noise_std = generate_noisy_data_periodic(x_values, vocab_size, noise)
            divTerm = max_value-min_value

            plt.plot(x_values, y_noise_spline, y_spline)
            plt.show()

            # Benutzerabfrage zum Abbrechen
            abbruch = input("Zum Abbrechen 'q' eingeben, sonst Enter drücken: ")
            if abbruch.lower() == 'q':
                #plot results in one plot
                fig, (ax1, ax2) = plt.subplots(2,1, figsize=(11,5))
                ax1: Axes
                ax2: Axes
                #perform min max scaling with Groundtruth and noisy data
                ySplineMinMax = performMinMaxScaling(y_spline, min_value, divTerm)
                yNoiseMinMax = performMinMaxScaling(y_noise_spline, min_value, divTerm)
                NoiseCopy = np.array(yNoiseMinMax)
                SplineCopy = np.array(ySplineMinMax)
                

                #perform rounding with Groundtruth und noisy data
                ySplineRounded = performRoundingProcess(vocab_size, ySplineMinMax)
                yNoiseRounded = performRoundingProcess(vocab_size, yNoiseMinMax)

                #save data to folder
                df = pd.DataFrame()
                df["y_raw_noise"] = NoiseCopy
                df["y_raw_groundtruth"] = SplineCopy
                df["y_discrete_noise"] = yNoiseRounded
                df["y_discrete_groundtruth"] = ySplineRounded
                df.to_csv("data/rounding_process/data.csv", index=False)

                #plotting definitions
                alpha_auxiliary = 0.15
                color_auxiliary = "black"
                linewidth_auxiliary = 1

                linewidth_GT_N = 2
                alpha_GT_N = 1
                linewidth_GT = 3

                fmt = "svg"
                title = "RoundingProcess"

                borders = [np.ones(1000) * 0.1,np.ones(1000) * 0.3,np.ones(1000) * 0.5,np.ones(1000) * 0.7,np.ones(1000) * 0.9]
                for border in borders:
                    ax1.plot(x_values, border, alpha=alpha_auxiliary, color=color_auxiliary,linewidth=linewidth_auxiliary)
                    ax2.plot(x_values, border, alpha=alpha_auxiliary, color=color_auxiliary,linewidth=linewidth_auxiliary)


                ax1.plot(x_values, NoiseCopy,label= "MinMax(NoiseData)", color=colorcodes["y_raw"], linewidth=linewidth_GT_N, alpha=alpha_GT_N)
                ax1.plot(x_values, SplineCopy, label="MinMax(GroundTruth)", color=colorcodes["y_groundtruth"],linewidth=linewidth_GT)
                
                ax2.plot(x_values, yNoiseRounded,label="Rounding(NoiseData)",color=colorcodes["y_raw"],linewidth=linewidth_GT_N, alpha=alpha_GT_N)
                ax2.plot(x_values, ySplineRounded, label="Rounding(Groundtruth)", color=colorcodes["y_groundtruth"],linewidth=linewidth_GT)
                


                ax1.set_ylabel("Min Max Scaling",fontweight='bold')
                ax2.set_ylabel("Discretization",fontweight='bold')
                ax2.set_xlabel("Steps")
                ax1.margins(x=0.0)
                ax2.margins(x=0.0)
                offset = (divTerm)*0.05
                ax1.tick_params(axis='x', which='both', bottom=True, left=True, labelbottom=False, labelleft=False)
                ax2.tick_params(axis='x', which='both', bottom=True, left=True, labelbottom=False, labelleft=False)
                
                ax1.tick_params(axis='both', labelsize=15,direction='in', length=8, width=1)
                ax2.tick_params(axis='both', labelsize=15, direction='in', length=8, width=1)

                ax1.legend(prop={'weight': 'bold'},loc='upper right')
                ax2.legend(prop={'weight': 'bold'},loc='upper right')

                plt.tight_layout()
                save_path = Path(f"pictures/")
                save_path.mkdir(parents=True, exist_ok=True)
                file_path = save_path / f"{title}.{fmt}" if save_path else f"{title}.{fmt}"
                plt.gcf().savefig(file_path, format=fmt, dpi=300, bbox_inches='tight')

                plt.show()


                break

    except KeyboardInterrupt:
        print("Abbruch per Tastenkombination (Strg+C).")

