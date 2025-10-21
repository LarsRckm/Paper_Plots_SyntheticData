import matplotlib.pyplot as plt
import numpy as np
import torch
from model import build_encoder_interpolation_uknToken_projection
from useful import value_to_index_dict, index_to_value_dict, round_numbers_individually, calc_exp, round_with_exp, greedy_decode_timeSeries_paper_projection, sliding_window
from pathlib import Path
import math


#needed
def predict_encoder_interpolation_projection_roundedInput(seq_length: int, model_filename, y_noisy_spline,min_value_spline, max_value_spline, vocab_size, vocab_extra_tokens, mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using device:", device)

    v2i = value_to_index_dict(vocab_size_numbers=vocab_size, vocab_extra_tokens=vocab_extra_tokens)
    i2v = index_to_value_dict(vocab_size_numbers=vocab_size, vocab_extra_tokens=vocab_extra_tokens)
    exp = calc_exp(smallest_number=1/vocab_size)

    model = build_encoder_interpolation_uknToken_projection(len(v2i), seq_length)


    print(f'Preloading model {model_filename}')
    state = torch.load(model_filename, map_location=torch.device('cpu'))
    model.load_state_dict(state['model_state_dict'])
    model.eval()    


    with torch.no_grad():
        #min-max scaling
        div_term = max_value_spline-min_value_spline
        encoder_input_scaled = torch.tensor((y_noisy_spline-min_value_spline)/div_term,dtype=torch.float32)[:].to(device)
        
        #masking       
        mask_indices = np.where(mask == 1)[0] # mask == 1 --> remove
        
        #round scaled input timeseries taking into account vocab_size
        encoder_input_discrete = round_numbers_individually(vocab_size,encoder_input_scaled)[:]

        #replace discrete values with indices
        encoder_input_discrete[mask_indices] = v2i["ukn"] #replace interpolation points with the index-equivalent of the "ukn" token
        encoder_input_index = encoder_input_discrete.apply_(lambda x: x if x>vocab_size else v2i[f"{round_with_exp(x, exp)}"]).type(torch.long) #map discrete values to indices

        #create prediction
        model_out = greedy_decode_timeSeries_paper_projection(model, encoder_input_index.unsqueeze(0))

    #map indices to discrete values
    model_out = model_out.type(torch.float32).apply_(lambda x: 0 if int(x)>vocab_size else i2v[f"{int(x)}"])
    
    #denormalize discrete values with stores values of min-max scaling
    model_out = (model_out*div_term)+min_value_spline


    return model_out



#needed
def denoise_floaterCurrent_encoder_interpolation_CE(latest_encoderModel_filename, seq_len, current, config, mask=None):
    print("function called")
    shape_current = current.shape[0]

    prediction_encoder = torch.tensor([])
    encoder_input_removed_tensor = torch.tensor([])

    i=0
    exp = calc_exp(smallest_number=1/config["vocab_size"])
    predict_len = 800
    ueberlappen_len = 1000-predict_len


    while i < shape_current:
        #first prediction includes first 1000 data points
        if i == 0:
            lower_border = i
            upper_border = i+1000

            #assert upper_border not larger than total timeseries length
            if upper_border > shape_current:
                upper_border = shape_current

            if type(mask) != None:
                mask_copy = mask[lower_border:upper_border]
                mask_indices_in = np.where(mask_copy == 0)[0]   #values which are not masked out
                mask_indices_out = np.where(mask_copy == 1)[0]  #values which are masked out

            current_copy = current[lower_border:upper_border]
            current_copy_min_max = current_copy[mask_indices_in] #extract all values that are not masked out for minimum and maximum calculation
            
            #calculate global min and max values
            max_value_current = math.ceil(max((current_copy_min_max))*(10**exp))/(10**exp)
            min_value_current = math.floor(min((current_copy_min_max))*(10**exp))/(10**exp)
            print(f"min:{min(current_copy)}")
            print(f"max:{max(current_copy)}")    

            #create prediction
            prediction, encoder_input_removed = predict_encoder_interpolation_roundedInput_CE_floatCurrent(seq_len,latest_encoderModel_filename,current_copy,min_value_current, max_value_current,mask_indices_out, config)
            prediction = prediction.squeeze()

            #append results to existing torch tensors
            prediction_encoder = torch.cat((prediction_encoder,prediction),0)
            encoder_input_removed_tensor = torch.cat((encoder_input_removed_tensor, encoder_input_removed), 0)

            i += 1000
        
        #following predictions use 500 data points of the previous prediction and 500 new data points
        #the rear part of the prediction (last 500 data points of prediction) is appended to prediction array
        else:
            lower_border = i
            upper_border = i+predict_len
            prediction = prediction_encoder[-ueberlappen_len:]                                  #past 500 values of total prediction array so far
            current_copy = current[lower_border:upper_border]                       #next 500 values of noisy data after end of prediction
            current_copy = torch.cat((prediction, torch.tensor(current_copy)), 0)   #connect previous tensors to one 1000 values tensor

            #assert upper_border not larger than total timeseries length
            if upper_border > shape_current:
                upper_border = shape_current

            if type(mask) != None:
                mask_previous_prediction = np.zeros(ueberlappen_len, dtype=int) #all 500 values from previous prediction are taken into account for next prediction -> mask index = 0
                mask_copy = mask[lower_border:upper_border]         #take mask values for the next 500 values
                mask_copy = np.concatenate((mask_previous_prediction, mask_copy), 0) #connect the masking values
                mask_indices_in = np.where(mask_copy == 0)[0]   #values which are not masked out
                mask_indices_out = np.where(mask_copy == 1)[0]  #values which are masked out

            current_copy_min_max = current_copy[mask_indices_in]    #extract all values that are not masked out for minimum and maximum calculation

            #calculate global min and max values
            max_value_current = math.ceil(max((current_copy_min_max))*(10**exp))/(10**exp)
            min_value_current = math.floor(min((current_copy_min_max))*(10**exp))/(10**exp)
            print(f"min:{min(current_copy)}")
            print(f"max:{max(current_copy)}")    

            #create prediction
            prediction, encoder_input_removed = predict_encoder_interpolation_roundedInput_CE_floatCurrent(seq_len,latest_encoderModel_filename,current_copy,min_value_current, max_value_current,mask_indices_out, config)
            prediction = prediction.squeeze()

            #append results to existing torch tensors
            prediction_encoder = torch.cat((prediction_encoder,prediction[ueberlappen_len:]),0)
            encoder_input_removed_tensor = torch.cat((encoder_input_removed_tensor, encoder_input_removed[ueberlappen_len:]), 0)

            i += predict_len



    prediction_encoder = prediction_encoder.detach().numpy()
    prediction_encoder_sliding = sliding_window(prediction_encoder, 2) 
    #apply two-side sliding window
    # prediction_encoder_sliding = sliding_window(prediction_encoder_sliding, 4000)   #apply two-side sliding window again to smooth more
    # print("in Sliding")
    for i in range(5):
        prediction_encoder_sliding = sliding_window(prediction_encoder_sliding, 2)

    return prediction_encoder,prediction_encoder_sliding, encoder_input_removed_tensor


#needed
def predict_encoder_interpolation_roundedInput_CE_floatCurrent(seq_length: int, model_filename, y_noisy_spline,min_value_spline, max_value_spline, mask_indices,config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using device:", device)
    vocab_size = config["vocab_size"]
    vocab_extra_tokens = config["extra_tokens"]

    v2i = value_to_index_dict(vocab_size_numbers=vocab_size, vocab_extra_tokens=vocab_extra_tokens)
    i2v = index_to_value_dict(vocab_size_numbers=vocab_size, vocab_extra_tokens=vocab_extra_tokens)
    exp = calc_exp(smallest_number=1/vocab_size)

    #build model
    model = build_encoder_interpolation_uknToken_projection(len(v2i), seq_length)
    # print(f'Preloading model {model_filename}')
    print("\tLoading Model")
    state = torch.load(model_filename, map_location=torch.device('cpu'))
    model.load_state_dict(state['model_state_dict'])
    model.eval()    


    with torch.no_grad():
        #min-max scaling
        div_term = max_value_spline-min_value_spline
        encoder_input_removed = torch.tensor((y_noisy_spline-min_value_spline)/div_term,dtype=torch.float32)[:].to(device)

        #round normalized values to discrete values
        encoder_input_removed = round_numbers_individually(vocab_size,encoder_input_removed)[:]

        #map discrete values to indices and map masked values to index equivalent of "ukn"-token
        encoder_input_removed[mask_indices] = v2i["ukn"]
        encoder_input_removed = encoder_input_removed.apply_(lambda x: x if x>vocab_size else v2i[f"{round_with_exp(x, exp)}"]).type(torch.long)

        #prediction
        model_out = greedy_decode_timeSeries_paper_projection(model, encoder_input_removed.unsqueeze(0))


    #map indices to discrete values
    model_out = model_out.type(torch.float32).apply_(lambda x: 0 if int(x)>vocab_size else i2v[f"{int(x)}"])
    encoder_input_removed = encoder_input_removed.type(torch.float32).apply_(lambda x: 0 if int(x)>vocab_size else i2v[f"{int(x)}"])

    #denormalize discrete values
    model_out = (model_out*div_term)+min_value_spline
    encoder_input_removed = (encoder_input_removed*div_term)+min_value_spline


    return model_out, encoder_input_removed
