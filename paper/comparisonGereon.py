import sys
sys.path.append(".")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle
from plot import denoise_floaterCurrent_encoder_interpolation_CE_forRawDVA
import dill



class ExcelFile:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = pd.ExcelFile(filepath)
        self.current = []
        self.voltage = []
        self.charge = []
        self.index = []
        self.status = [] #CC_DChg, CC_Chg
        self.negativCurrent = []
        self.positiveCurrent = []

    def storeEveryValue(self):
        print(f"Reading Excel File: {self.filepath}")
        for arbeitsmappe in self.data.sheet_names:
            if "Detail" in arbeitsmappe:
                print(f"Reading Arbeitsmappe: {arbeitsmappe}")
                data = self.data.parse(arbeitsmappe)
                voltage = data["Voltage(V)"].to_numpy()
                charge = data["CapaCity(Ah)"].to_numpy()
                current = data["Cur(A)"].to_numpy()
                index = data["Record Index"].to_numpy()
                status = data["Status"].to_numpy()
                self.current.extend(current)
                self.voltage.extend(voltage)
                self.charge.extend(charge)
                self.index.extend(index)
                self.status.extend(status)
        
        print("Storing Values in Arrays")
        self.voltage = np.array(self.voltage)
        self.current = np.array(self.current)
        self.charge = np.array(self.charge)
        self.index = np.array(self.index)
        self.status = np.array(self.status)
    

    def splitByChargingState(self):
        indicesCHG = np.where(self.status == "CC_Chg", 1, 0)
        indicesDCHG = np.where(self.status == "CC_DChg", 1, 0)

        voltageCHG = self.voltage[indicesCHG == 1]
        currentCHG = self.current[indicesCHG == 1]
        chargeCHG = self.charge[indicesCHG == 1]
        indexCHG = self.index[indicesCHG == 1]
        indicesTransistionsCHG = np.concatenate((np.array([0]), np.where(np.diff(indexCHG) > 1)[0]))
        for i,index in enumerate(indicesTransistionsCHG):
            if(index == indicesTransistionsCHG[-1]):
                self.positiveCurrent.append({"voltage" : voltageCHG[index:], "current": currentCHG[index:], "charge": chargeCHG[index:]})
            else:
                nextIndex = indicesTransistionsCHG[i+1]
                self.positiveCurrent.append({"voltage" : voltageCHG[index:nextIndex], "current": currentCHG[index:nextIndex], "charge": chargeCHG[index:nextIndex]})

        voltageDCHG = self.voltage[indicesDCHG == 1]
        currentDCHG = self.current[indicesDCHG == 1]
        chargeDCHG = self.charge[indicesDCHG == 1]
        indexDCHG = self.index[indicesDCHG == 1]
        indicesTransistionsDCHG = np.concatenate((np.array([0]),np.where(np.diff(indexDCHG) > 1)[0]))
        for i,index in enumerate(indicesTransistionsDCHG):
            if(index == indicesTransistionsDCHG[-1]):
                self.negativCurrent.append({"voltage" : voltageDCHG[index:], "current": currentDCHG[index:], "charge": chargeDCHG[index:]})
            elif(index == 0):
                nextIndex = indicesTransistionsDCHG[i+1]
                self.negativCurrent.append({"voltage" : voltageDCHG[:nextIndex+1], "current": currentDCHG[:nextIndex+1], "charge": chargeDCHG[:nextIndex+1]})
            else:
                nextIndex = indicesTransistionsDCHG[i+1]
                self.negativCurrent.append({"voltage" : voltageDCHG[index:nextIndex+1], "current": currentDCHG[index:nextIndex+1], "charge": chargeDCHG[index:nextIndex+1]})

def sliding_window(spline_array, window_size):
    length_spline_array = len(spline_array)
    result = np.array([])
    for index, number in enumerate(spline_array, start=0):
        if index == 0:
            result = np.concatenate((result,[spline_array[index]]), axis = 0)
        elif index == length_spline_array-1:
            result = np.concatenate((result,[spline_array[index]]), axis = 0)
        elif index < window_size:#index < 200
            window_size_copy = index
            spline_array_lower_window = spline_array[:index]
            middle_value = spline_array[index]
            spline_array_upper_window = spline_array[index+1:index+1+window_size_copy]
            arr = np.concatenate((spline_array_lower_window,[middle_value],spline_array_upper_window), axis = 0)
            length = len(arr)
            sum = np.sum(arr)
            new_value = sum/length
            result = np.concatenate((result,[new_value]), axis = 0)
        elif index+window_size > length_spline_array-1:
            window_size_copy = length_spline_array - 1 - index 
            spline_array_lower_window = spline_array[index-window_size_copy:index]
            middle_value = spline_array[index]
            spline_array_upper_window = spline_array[index+1:length_spline_array]
            arr = np.concatenate((spline_array_lower_window,[middle_value],spline_array_upper_window), axis = 0)
            length = len(arr)
            sum = np.sum(arr)
            new_value = sum/length
            result = np.concatenate((result,[new_value]), axis = 0)
        else:
            spline_array_lower_window = spline_array[index-window_size:index]
            middle_value = spline_array[index]
            spline_array_upper_window = spline_array[index+1:index+1+window_size]
            arr = np.concatenate((spline_array_lower_window,[middle_value],spline_array_upper_window), axis = 0)
            length = len(arr)
            sum = np.sum(arr)
            new_value = sum/length
            result = np.concatenate((result,[new_value]), axis = 0)
    
    return result

def calcDVA(voltage, charge):
        dva = []
        dva.append(0)
        for index in range(1, len(voltage)):
            difference_voltage = voltage[index]-voltage[index-1]
            difference_charge = charge[index]- charge[index-1]
            dva_value = difference_voltage/difference_charge
                
            if math.isinf(dva_value):
                dva.append(0)
            else:
                dva.append(dva_value)
        
        return np.array(dva)

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

if __name__ == "__main__":
    filepath = "/Users/larsreckmann/Desktop/nmc900/gst_900_LB_001_Cu1.xls"
    excel = ExcelFile(filepath)
    excel.storeEveryValue()
    excel.splitByChargingState()

    #create first plot to show raw voltage, current and charge data
    fig, ax = plt.subplots(1,1)
    fig.suptitle("RAW Data")


    #extract first discharge
    firstDischarge = excel.negativCurrent[0]
    firstDischargeVoltage = firstDischarge["voltage"]
    firstDischargeCurrent = firstDischarge["current"]
    firstDischargeCharge = firstDischarge["charge"]

    #plot raw Data
    assert len(firstDischargeVoltage) == len(firstDischargeCurrent) == len(firstDischargeCharge), "Alle Arrays müssen die gleiche Länge haben"
    x_values = np.arange(len(firstDischargeVoltage))
    ax.plot(x_values, firstDischargeVoltage, label="voltage")
    ax.plot(x_values, firstDischargeCurrent, label="current")
    ax.plot(x_values, firstDischargeCharge, label="charge")
    ax.legend()

    
    #create second plot for sliding window voltage plot
    fig, ax = plt.subplots(2,1,constrained_layout=True, sharex=True)
    
    #read in Gereon csv voltage Data
    # filepath = "/Users/larsreckmann/Desktop/GereonComparison/Interpolierte_Spannung_Zeit.csv"
    filepath = "/Users/larsreckmann/Desktop/GereonComparison/Interpolierte_daten.csv"
    gereonData = pd.read_csv(filepath, sep=",")
    # GereonVoltage = gereonData.loc[gereonData["Var3"] == 4, "Var2"].to_numpy()
    GereonVoltage = gereonData["Var2"].to_numpy()

    print(f"LENGTH GEREON: {len(GereonVoltage)}")

    # assert len(firstDischargeVoltage) == len(GereonVoltage), f"Gereon Voltage ({len(GereonVoltage)});meine Voltage ({len(firstDischargeVoltage)})"
    #sampling signals
    # seq_len = len(firstDischargeVoltage)/5
    # sampling_rate = math.floor(seq_len/1000)
    sampling_rate = math.floor(len(firstDischargeVoltage)/len(GereonVoltage))
    firstDischargeVoltage = firstDischargeVoltage[0::sampling_rate]
    firstDischargeCharge = firstDischargeCharge[0::sampling_rate]
    firstDischargeCurrent = firstDischargeCurrent[0::sampling_rate]
    # GereonVoltage = GereonVoltage[0::sampling_rate]

    cutOff = math.floor(len(firstDischargeVoltage)/1000)*1000
    firstDischargeVoltage = firstDischargeVoltage[:cutOff]
    firstDischargeCharge = firstDischargeCharge[:cutOff]
    firstDischargeCurrent = firstDischargeCurrent[:cutOff]
    GereonVoltage = GereonVoltage[:cutOff]

    #model parameters
    windowsize = 2
    windowIterations = 20
    config = get_config()
    mask = np.zeros(shape=len(firstDischargeVoltage),dtype=np.int16)
    model_AIO = "/Users/larsreckmann/Desktop/UNI/Bachelorarbeit/CodePflegen/weights/Encoder_Interpolation_CE_AllInOne_2400.pt"
    
    #model application
    #model all in one
    firstdischargeVoltageDenoised_AIO, firstdischargeVoltageDenoisedSlidingWindow_AIO, _ = denoise_floaterCurrent_encoder_interpolation_CE_forRawDVA(model_AIO, 1000, firstDischargeVoltage, config, windowsize, windowIterations, mask)

    #calculate error GereonVoltage - RawVoltage
    differenceGereonRaw = GereonVoltage - firstDischargeVoltage
    meanValueGereonRaw = np.mean(differenceGereonRaw)
    stdValueGereonRaw = np.std(differenceGereonRaw)

    #calculate error PredictionVoltageSlidingWindow - RawVoltage
    differenceModelRaw = firstdischargeVoltageDenoisedSlidingWindow_AIO - firstDischargeVoltage
    meanValueModelRaw = np.mean(differenceModelRaw)
    stdValueModelRaw = np.std(differenceModelRaw)


    #plotting
    x_values = np.arange(len(firstDischargeVoltage))
    #plot Raw Voltage, Gereon Voltage, Model Prediction Sliding Window
    ax[0].plot(x_values, firstDischargeVoltage,"--","red", label="Voltage Raw")
    ax[0].plot(x_values, GereonVoltage,"blue", label="Voltage Classic Fitting")
    ax[0].plot(x_values, firstdischargeVoltageDenoisedSlidingWindow_AIO,"blue", label="Voltage CET-TSIR",alpha=0.5)
    ax[0].set_title("Classic Fitting vs CET-TSIR")
    ax[0].legend()


    x_values = np.arange(len(differenceGereonRaw))
    ax[1].plot(x_values, differenceGereonRaw,"blue", label="Classic Fitting")
    ax[1].plot(x_values, differenceModelRaw,"blue", label="CET-TSIR")
    # ax[1].set_title("f"Gereon: Mean = {meanValueGereonRaw}; std = {stdValueGereonRaw}\nModel: Mean = {meanValueModelRaw}; std = {stdValueModelRaw}"")
    ax[1].set_title("Error Plot")
    ax[1].legend()


    fig.savefig("/Users/larsreckmann/Desktop/UNI/Bachelorarbeit/CodePflegen/paper/saveFIG/plot.png",dpi=300, bbox_inches='tight')
    fig.savefig("/Users/larsreckmann/Desktop/UNI/Bachelorarbeit/CodePflegen/paper/saveFIG/plot.svg",dpi=300, bbox_inches='tight')
    # with open("/Users/larsreckmann/Desktop/UNI/Bachelorarbeit/CodePflegen/paper/saveFIG/plot.pkl", "wb") as f:
    #     pickle.dump(fig, f)


    with open("/Users/larsreckmann/Desktop/UNI/Bachelorarbeit/CodePflegen/paper/saveFIG/plot.pkl", 'wb') as file:
          dill.dump(fig, file, protocol=dill.HIGHEST_PROTOCOL)



    plt.show()














    #plot raw predictions for comparison
    # ax[0].plot(x_values, firstDischargeVoltage, label="Voltage Raw")
    # ax[0].plot(x_values, firstdischargeVoltageDenoised_PS, label="Voltage Denoised PS")
    # ax[0].plot(x_values, firstdischargeVoltageDenoised_LO, label="Voltage Denoised LO")
    # ax[0].plot(x_values, firstdischargeVoltageDenoised_AIO, label="Voltage Denoised AIO")

    #plot sliding window predictions for comparison
    # ax[1].plot(x_values, firstDischargeVoltage, label="Voltage Raw")
    # ax[1].plot(x_values, firstdischargeVoltageDenoisedSlidingWindow_PS, label="Voltage Denoised Sliding Window PS")
    # ax[1].plot(x_values, firstdischargeVoltageDenoisedSlidingWindow_LO, label="Voltage Denoised Sliding Window LO")
    # ax[1].plot(x_values, firstdischargeVoltageDenoisedSlidingWindow_AIO, label="Voltage Denoised Sliding Window AIO")












    #create third plot for dva
    # fig, ax = plt.subplots(2,1,constrained_layout=True)
    # fig.suptitle("DVA Data (from Raw Voltage Prediction/from sliding window Voltage Prediction)")

    #DVA Calculation
    # dvaRaw = calcDVA(firstDischargeVoltage, firstDischargeCharge)
    # dvaDenoised_PS = calcDVA(firstdischargeVoltageDenoised_PS, firstDischargeCharge)
    # dvaDenoised_LO = calcDVA(firstdischargeVoltageDenoised_LO, firstDischargeCharge)
    # dvaDenoised_AIO = calcDVA(firstdischargeVoltageDenoised_AIO, firstDischargeCharge)
    # dvaGereon = calcDVA(GereonVoltage, firstDischargeCharge)

    # dvaDenoisedSlidingWindow_PS = calcDVA(firstdischargeVoltageDenoisedSlidingWindow_PS, firstDischargeCharge)
    # dvaDenoisedSlidingWindow_LO = calcDVA(firstdischargeVoltageDenoisedSlidingWindow_LO, firstDischargeCharge)
    # dvaDenoisedSlidingWindow_AIO = calcDVA(firstdischargeVoltageDenoisedSlidingWindow_AIO, firstDischargeCharge)

    # ax[0].plot(x_values, dvaRaw, label=f"DVA Raw")
    # ax[0].plot(x_values, dvaDenoised_PS, label=f"DVA Denoised PS")
    # ax[0].plot(x_values, dvaDenoised_LO, label=f"DVA Denoised LO")
    # ax[0].plot(x_values, dvaDenoised_AIO, label=f"DVA Denoised AIO",alpha=0.5)
    # ax[0].plot(x_values, dvaGereon, label=f"DVA Gereon",alpha=0.5)
    # ax[0].legend()

    # ax[1].plot(x_values, dvaRaw, label=f"DVA Raw")
    # ax[1].plot(x_values, dvaDenoisedSlidingWindow_PS, label=f"DVA Denoised Sliding Window PS")
    # ax[1].plot(x_values, dvaDenoisedSlidingWindow_LO, label=f"DVA Denoised Sliding Window LO")
    # ax[1].plot(x_values, dvaDenoisedSlidingWindow_AIO, label=f"DVA Denoised Sliding Window AIO",alpha=0.5)
    # ax[1].plot(x_values, dvaGereon, label=f"DVA Gereon",alpha=0.5)
    # ax[1].legend()


    # plt.legend()
    # plt.show()
