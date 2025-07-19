import sys
sys.path.append(".")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import glob
import os
from plot import denoise_floaterCurrent_encoder_interpolation_CE_forRawDVA

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

class ExcelFile:
    def __init__(self, filename:str):
        self.filename = filename.split("/")[-1].split(".xls")[0]
        self.data = pd.ExcelFile(filename)
        self.current = []
        self.voltage = []
        self.charge = []
        self.index = []
        self.positiveCurrent = [] #charging DVA
        self.negativeCurrent = [] #discharging DVA
    

    #Funktion, die jede einzelne arbeitsmappe ausliest
    def readEveryFile(self):
        print(f"Reading Excel worksheets: {self.filename}")
        for arbeitsmappe in self.data.sheet_names:
            if "Detail" in arbeitsmappe:
                print(f"{arbeitsmappe}")
                data = self.data.parse(arbeitsmappe)
                voltage = data["Voltage(V)"].to_numpy()
                charge = data["CapaCity(Ah)"].to_numpy()
                current = data["Cur(A)"].to_numpy()
                index = data["Record Index"].to_numpy()
                self.current.extend(current)
                self.voltage.extend(voltage)
                self.charge.extend(charge)
                self.index.extend(index)
        
        self.voltage = np.array(self.voltage)
        self.current = np.array(self.current)
        self.charge = np.array(self.charge)
        self.index = np.array(self.index)

    def splitByCurrentPos(self):
        indices_posCur = np.where(np.array(self.current) > 0)[0]
        if(len(indices_posCur) != 0):
            posVoltage = self.voltage[indices_posCur]
            posCharge = self.charge[indices_posCur]
            indices_posCur_diff = np.diff(indices_posCur)
            indices_posCur_uebergang = np.append(np.array([0]), np.where(indices_posCur_diff > 1)[0])
            length = len(indices_posCur_uebergang)-1


            for i,uebergang in enumerate(indices_posCur_uebergang):
                if(i != length):
                    voltage = posVoltage[uebergang:indices_posCur_uebergang[i+1]]
                    charge = posCharge[uebergang:indices_posCur_uebergang[i+1]]
                    if(len(voltage) >= 1000 and len(charge) >= 1000):
                        self.positiveCurrent.append({
                            "voltage": voltage,
                            "charge": charge,
                        })
                else:
                    voltage = posVoltage[uebergang:]
                    charge = posCharge[uebergang:]
                    if(len(voltage) >= 1000 and len(charge) >= 1000):
                        self.positiveCurrent.append({
                            "voltage": voltage,
                            "charge": charge,
                        })
            
            print(f"Total amount of Charging Intervals: {len(self.positiveCurrent)}")

    def splitByCurrentNeg(self):
        indices_negCur = np.where(np.array(self.current) < 0)[0]
        if(len(indices_negCur) != 0):
            negVoltage = self.voltage[indices_negCur]
            negCharge = self.charge[indices_negCur]
            indices_negCur_diff = np.diff(indices_negCur)
            indices_negCur_uebergang = np.append(np.array([0]), np.where(indices_negCur_diff > 1)[0])
            length = len(indices_negCur_uebergang)-1

            for i,uebergang in enumerate(indices_negCur_uebergang):
                if(i != length):
                    voltage = negVoltage[uebergang:indices_negCur_uebergang[i+1]]
                    charge = negCharge[uebergang:indices_negCur_uebergang[i+1]]
                    if(len(voltage) >= 1000 and len(charge) >= 1000):
                        self.negativeCurrent.append({
                            "voltage": voltage,
                            "charge": charge
                        })
                else:
                    voltage = negVoltage[uebergang:]
                    charge = negCharge[uebergang:]
                    self.negativeCurrent.append({
                            "voltage": voltage,
                            "charge": charge
                        })
            print(f"Total amount of Discharging Intervals: {len(self.negativeCurrent)}")

    def plotCurrentPos(self):
        for voltage, charge in self.positiveCurrent:
            _,axis = plt.subplots(1,1)
            x_values = np.arange(len(voltage))

            axis.plot(x_values, voltage, label="voltage")
            axis.plot(x_values, charge, label="charge")

            axis.legend(loc='best')
            plt.title("Positive Current")
            plt.show()

    def plotCurrentNeg(self):
        for voltage, charge in self.negativeCurrent:
            _,axis = plt.subplots(1,1)
            x_values = np.arange(len(voltage))

            axis.plot(x_values, voltage, label="voltage")
            axis.plot(x_values, charge, label="charge")

            axis.legend(loc='best')
            plt.title("Negative Current")
            plt.show()
    
    def calcDVA(self, voltage, charge):
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

    def denoiseVoltagePositive(self, model_filename, voltageWindowSize, voltageWindowCount, PredictionWindowSize, PredictionWindowCount):
        config = get_config()
        for i, dictionary in enumerate(self.positiveCurrent):
            print(f"Processing Charging Interval: {i+1}/{len(self.positiveCurrent)}")
            voltage = dictionary["voltage"]
            charge = dictionary["charge"]
            seq_len = len(voltage)
            sampling_rate = math.floor(seq_len/1000)
            voltage = voltage[0::sampling_rate]
            charge = charge[0::sampling_rate]
            voltage = voltage[:1000]
            charge = charge[:1000]
            mask = np.zeros(shape=seq_len,dtype=np.int16)

            zeros = np.zeros(shape=970,dtype=np.int16)  #Länge der einzelnen Intervalle für die Maske noch anpassen
            ones = np.ones(shape=30,dtype=np.int16) #Länge der einzelnen Intervalle für die Maske noch anpassen
            mask = np.concatenate((ones, zeros))

            voltagePrediction_slidingFunction = sliding_window(voltage, voltageWindowSize)
            for j in range(voltageWindowCount):
                voltagePrediction_slidingFunction = sliding_window(voltagePrediction_slidingFunction, 4)
            
            # dva = self.calcDVA(voltagePrediction, charge)
            # dva_sliding = self.calcDVA(voltagePrediction_sliding, charge)     
            # dva_original = self.calcDVA(voltage, charge)
            dva_slidingWindowFunction = self.calcDVA(voltagePrediction_slidingFunction, charge)

            dvaDenoised, dvaDenoised_sliding, _ = denoise_floaterCurrent_encoder_interpolation_CE_forRawDVA(model_filename, 1000, dva_slidingWindowFunction, config, PredictionWindowSize, PredictionWindowCount,mask)

            dictionary["voltageSlidingWindow"] = voltagePrediction_slidingFunction
            dictionary["dvaSlidingWindow"] = dva_slidingWindowFunction
            dictionary["dvaPrediction"] = dvaDenoised
            dictionary["dvaPredictionSlidingWindow"] = dvaDenoised_sliding
        
    def denoiseVoltageNegative(self, model_filename, voltageWindowSize, voltageWindowCount, PredictionWindowSize, PredictionWindowCount):
        config = get_config()
        for i, dictionary in enumerate(self.negativeCurrent):
            print(f"Processing Charging Interval: {i+1}/{len(self.negativeCurrent)}")
            voltage = dictionary["voltage"]
            charge = dictionary["charge"]
            seq_len = len(voltage)
            sampling_rate = math.floor(seq_len/1000)
            voltage = voltage[0::sampling_rate]
            charge = charge[0::sampling_rate]
            voltage = voltage[:1000]
            charge = charge[:1000]
            mask = np.zeros(shape=seq_len,dtype=np.int16)

            zeros = np.zeros(shape=970,dtype=np.int16)  #Länge der einzelnen Intervalle für die Maske noch anpassen
            ones = np.ones(shape=30,dtype=np.int16) #Länge der einzelnen Intervalle für die Maske noch anpassen
            mask = np.concatenate((ones, zeros))

            voltagePrediction_slidingFunction = sliding_window(voltage, voltageWindowSize)
            for j in range(voltageWindowCount):
                voltagePrediction_slidingFunction = sliding_window(voltagePrediction_slidingFunction, voltageWindowSize)
            
            # dva = self.calcDVA(voltagePrediction, charge)
            # dva_sliding = self.calcDVA(voltagePrediction_sliding, charge)     
            # dva_original = self.calcDVA(voltage, charge)
            dva_slidingWindowFunction = self.calcDVA(voltagePrediction_slidingFunction, charge)

            dvaDenoised, dvaDenoised_sliding, _ = denoise_floaterCurrent_encoder_interpolation_CE_forRawDVA(model_filename, 1000, dva_slidingWindowFunction, config, PredictionWindowSize, PredictionWindowCount,mask)

            dictionary["voltage"] = voltage
            dictionary["charge"] = charge
            dictionary["voltageSlidingWindow"] = voltagePrediction_slidingFunction
            dictionary["dvaSlidingWindow"] = dva_slidingWindowFunction
            dictionary["dvaPrediction"] = dvaDenoised
            dictionary["dvaPredictionSlidingWindow"] = dvaDenoised_sliding

            break

    def save_csv_picture_positive(self, folder):
        #in dieser Funktion lediglich ein Dataframe erstellen mit "dvaPredictionSlidingWindow" und die dazugehörige Grafik als png abspeichern
        for i, dictionary in enumerate(self.positiveCurrent):
            #DVA Sliding Window in Dataframe umgewandelt
            dva_slidingWindowFunction = dictionary["dvaSlidingWindow"]
            dvaDenoised = dictionary["dvaPrediction"]
            dvaDenoised_sliding = dictionary["dvaPredictionSlidingWindow"]
            df = pd.DataFrame(dvaDenoised)


            #plotting DVA mit sliding Window
            x = np.arange(len(dvaDenoised[50:]))
            plt.figure()
            plt.plot(x, dva_slidingWindowFunction[50:], label="DVA (based on sliding Window voltage)")
            plt.plot(x, dvaDenoised[50:], label="DVA Prediction")
            plt.plot(x, dvaDenoised_sliding[50:], label="DVA Prediction Sliding Window")
            plt.title(f"{self.filename}")
            plt.xlabel("Time Steps")
            plt.ylabel("DVA")
            plt.legend()

            folderPathTotal = os.path.join(folder, self.filename, "positiveCurrent", f"{i}")

            if not os.path.exists(folderPathTotal):
                os.makedirs(folderPathTotal)
            
            df.to_csv(os.path.join(folderPathTotal, "Daten.csv"), index = False)
            plt.savefig(os.path.join(folderPathTotal, "DVA.png"), dpi=300, bbox_inches='tight')
     
    def save_csv_picture_negative(self, folder):
        #in dieser Funktion lediglich ein Dataframe erstellen mit "dvaPredictionSlidingWindow" und die dazugehörige Grafik als png abspeichern
        for i, dictionary in enumerate(self.negativeCurrent):
            #DVA Sliding Window in Dataframe umgewandelt
            dva_slidingWindowFunction = dictionary["dvaSlidingWindow"]
            dvaDenoised = dictionary["dvaPrediction"]
            dvaDenoised_sliding = dictionary["dvaPredictionSlidingWindow"]
            voltage = dictionary["voltage"]
            charge = dictionary["charge"]
            df = pd.DataFrame(dvaDenoised)


            #plotting DVA mit sliding Window
            x = np.arange(len(dvaDenoised[50:]))
            voltage = voltage[50:]
            charge = charge[50:]
            plt.figure()
            # plt.plot(x, dva_slidingWindowFunction[50:], label="DVA (based on sliding Window voltage)")
            # plt.plot(x, dvaDenoised[50:], label="DVA Prediction")
            # plt.plot(x, dvaDenoised_sliding[50:], label="DVA Prediction Sliding Window")
            plt.plot(voltage, dvaDenoised_sliding[50:], label="DVA Prediction Sliding Window")
            
            plt.title(f"{self.filename}")
            plt.xlabel("Charge")
            plt.ylabel("DVA")
            plt.legend()

            folderPathTotal = os.path.join(folder, self.filename, "negativeCurrent", f"{i}")

            if not os.path.exists(folderPathTotal):
                os.makedirs(folderPathTotal)
            
            df.to_csv(os.path.join(folderPathTotal, "Daten.csv"), index = False)
            plt.savefig(os.path.join(folderPathTotal, "DVA.png"), dpi=300, bbox_inches='tight')
            break

class CSVFile:
    def __init__(self, filename):
        self.data = pd.read_csv(filename, sep=",")
        self.current = []
        self.voltage = []
        self.charge = []
        self.index = []
        self.positiveCurrent = [] #charging DVA
        self.negativeCurrent = [] #discharging DVA


#Funktion, die jede einzelne arbeitsmappe ausliest
    def readEveryFile(self):
        voltage = self.data["Potential (V)"].to_numpy()
        charge = self.data["Capacity (Ah)"].to_numpy()
        current = self.data["Current (A)"].to_numpy()
        # index = self.data["Record Index"].to_numpy()
        self.current.extend(current)
        self.voltage.extend(voltage)
        self.charge.extend(charge)
        # self.index.extend(index)
        
        self.voltage = np.array(self.voltage)
        self.current = np.array(self.current)
        self.charge = np.array(self.charge)
        # self.index = np.array(self.index)
                
                
                #calculate the differential voltage analysis by calculating the difference between voltage with reference to charge
                # dva = []
                # dva.append(0)
                # assert len(voltage) == len(charge), "Spannung und Ladung haben nicht die gleiche Größe"
                # for index in range(1, len(voltage)):
                #     difference_voltage = voltage[index]-voltage[index-1]
                #     difference_charge = charge[index]- charge[index-1]
                #     dva_value = difference_voltage/difference_charge
                #     if math.isinf(dva_value):
                #         dva.append(0)
                #     else:
                #         dva.append(dva_value)
                # self.Voltage_Charge_DVA.append([voltage, charge, dva, current])

    def splitByCurrentPos(self):
        indices_posCur = np.where(np.array(self.current) > 0)[0]
        if(len(indices_posCur) != 0):
            posVoltage = self.voltage[indices_posCur]
            posCharge = self.charge[indices_posCur]
            # posIndex = self.index[indices_posCur]
            indices_posCur_diff = np.diff(indices_posCur)
            indices_posCur_uebergang = np.append(np.array([0]), np.where(indices_posCur_diff > 1)[0])
            length = len(indices_posCur_uebergang)-1


            for i,uebergang in enumerate(indices_posCur_uebergang):
                if(i != length):
                    voltage = posVoltage[uebergang:indices_posCur_uebergang[i+1]]
                    charge = posCharge[uebergang:indices_posCur_uebergang[i+1]]
                    if(len(voltage) >= 1000 and len(charge) >= 1000):
                        self.positiveCurrent.append([voltage, charge])
                else:
                    voltage = posVoltage[uebergang:]
                    charge = posCharge[uebergang:]
                    if(len(voltage) >= 1000 and len(charge) >= 1000):
                        self.positiveCurrent.append([voltage, charge])
    
    def splitByCurrentNeg(self):
        indices_negCur = np.where(np.array(self.current) < 0)[0]
        if(len(indices_negCur) != 0):
            negVoltage = self.voltage[indices_negCur]
            negCharge = self.charge[indices_negCur]
            indices_negCur_diff = np.diff(indices_negCur)
            indices_negCur_uebergang = np.append(np.array([0]), np.where(indices_negCur_diff > 1)[0])
            length = len(indices_negCur_uebergang)-1

            for i,uebergang in enumerate(indices_negCur_uebergang):
                if(i != length):
                    voltage = negVoltage[uebergang:uebergang+1]
                    charge = negCharge[uebergang:uebergang+1]
                    self.negativeCurrent.append([voltage, charge])
                else:
                    voltage = negVoltage[uebergang:]
                    charge = negCharge[uebergang:]
                    self.negativeCurrent.append([voltage, charge])
    
    def plotCurrentPos(self):
        for voltage, charge in self.positiveCurrent:
            _,axis = plt.subplots(1,1)
            x_values = np.arange(len(voltage))

            axis.plot(x_values, voltage, label="voltage")
            axis.plot(x_values, charge, label="charge")

            axis.legend(loc='best')
            plt.title("Positive Current")
            plt.show()


    def plotCurrentNeg(self):
        for voltage, charge in self.negativeCurrent:
            _,axis = plt.subplots(1,1)
            x_values = np.arange(len(voltage))

            axis.plot(x_values, voltage, label="voltage")
            axis.plot(x_values, charge, label="charge")

            axis.legend(loc='best')
            plt.title("Negative Current")
            plt.show()
    
    def calcDVA(self, voltage, charge):
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
        
        return dva

    def denoiseVoltage(self, voltage_charge_arr, model_filename):
        for voltage, charge in voltage_charge_arr:
            config = get_config()
            seq_len = len(voltage)
            sampling_rate = math.floor(seq_len/1000)
            voltage = voltage[0::sampling_rate]
            charge = charge[0::sampling_rate]
            voltage = voltage[:1000]
            charge = charge[:1000]
            mask = np.zeros(shape=seq_len,dtype=np.int16)
            voltagePrediction, voltagePrediction_sliding, _ = denoise_floaterCurrent_encoder_interpolation_CE_forRawDVA(model_filename, 1000, voltage, config, 2, 10,mask)
            voltagePrediction_slidingFunction = sliding_window(voltage, 2)
            for i in range(30):
                voltagePrediction_slidingFunction = sliding_window(voltagePrediction_slidingFunction, 2)


            dva = self.calcDVA(voltagePrediction, charge)
            dva_sliding = self.calcDVA(voltagePrediction_sliding, charge)
            dva_original = self.calcDVA(voltage, charge)
            dva_slidingWindowFunction = self.calcDVA(voltagePrediction_slidingFunction, charge)
            
            _,axis = plt.subplots(1,1)
            x_values = np.arange(len(voltage))

            axis.plot(x_values, voltage, label="Voltage")
            # axis.plot(x_values, charge, label="Charge")
            # axis.plot(x_values, voltagePrediction, label="Voltage (Raw Prediction)")
            axis.plot(x_values, voltagePrediction_sliding, label="Voltage (Smooth Prediction)")
            axis.plot(x_values, dva_original, label="DVA (Original)")
            axis.plot(x_values, dva_slidingWindowFunction, label="DVA (SlidingWindow Function)")
            # axis.plot(x_values, dva, label="DVA (Raw Prediction)")
            axis.plot(x_values, dva_sliding, label="DVA (Smooth Prediction)")

            axis.legend(loc='best')
            plt.title(f"{model_filename}")
            plt.show()

        

def readCSV():
    csv_file = CSVFile("/Users/larsreckmann/Downloads/dva_gut/gst_900_oa_011.csv")
    csv_file.readEveryFile()
    csv_file.splitByCurrentPos()
    csv_file.denoiseVoltage(csv_file.positiveCurrent, "/Users/larsreckmann/Desktop/UNI/Bachelorarbeit/CodePflegen/weights/Encoder_Interpolation_CE_PeriodicSum_2000.pt")

def readExcel():
    folderPath = "/Users/larsreckmann/Downloads/nmc900"
    xls_files = glob.glob(f"{folderPath}/*.xls")
    predictionWindowSize = 2
    predictionWindowCount = 5
    voltageWindowSize = 2
    voltageWindowCount = 3

    for filepath in xls_files:
        file = ExcelFile(filepath)
        file.readEveryFile()
        file.splitByCurrentPos()
        file.denoiseVoltagePositive("/Users/larsreckmann/Desktop/UNI/Bachelorarbeit/CodePflegen/weights/Encoder_Interpolation_CE_PeriodicSum_2000.pt", voltageWindowSize, voltageWindowCount, predictionWindowSize, predictionWindowCount)
        file.splitByCurrentNeg()
        file.denoiseVoltageNegative("/Users/larsreckmann/Desktop/UNI/Bachelorarbeit/CodePflegen/weights/Encoder_Interpolation_CE_PeriodicSum_2000.pt", voltageWindowSize, voltageWindowCount, predictionWindowSize, predictionWindowCount)
        file.save_csv_picture_positive("/Users/larsreckmann/Desktop/resultsPeriodicSum2")
        file.save_csv_picture_negative("/Users/larsreckmann/Desktop/resultsPeriodicSum2")


def comparisonGereon(filename:str):
    folderPath = "/Users/larsreckmann/Desktop/nmc900"
    xls_filepath = glob.glob(f"{folderPath}/{filename}.xls")
    predictionWindowSize = 2
    predictionWindowCount = 5
    voltageWindowSize = 2
    voltageWindowCount = 3
    file = ExcelFile(xls_filepath[0])
    file.readEveryFile()
    file.splitByCurrentNeg()
    file.denoiseVoltageNegative("/Users/larsreckmann/Desktop/UNI/Bachelorarbeit/CodePflegen/weights/Encoder_Interpolation_CE_PeriodicSum_2000.pt", voltageWindowSize, voltageWindowCount, predictionWindowSize, predictionWindowCount)
    file.save_csv_picture_negative("/Users/larsreckmann/Desktop/comparison")



if __name__ == "__main__":
    # readExcel()
    comparisonGereon("gst_900_LB_001_Cu1")
    # readCSV()


#Idee:
#automatisiert alle Excel Dateien auslesen und für jede Ladekurve DVA Daten erstellen und dann in einem Ordner sowohl die Plots, als auch die Daten an sich abspeichern
#