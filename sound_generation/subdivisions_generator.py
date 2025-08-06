import math
import numpy as np
import soundfile as sf
import random
from config import get_audio_data,paths
from squeleton_generator import squeleton_generator,in_bpm,in_maxsubd,in_num_cycles,in_skeleton,in_num_cycles

# this code is part of track 3 
# this will call the generate squeleton function
# and will put random hits based on the user input for each of the following parameters

# even subdivision percentage and the ones he already have for the squeleton part

even_subdivisions_percentage = 0.5

# in_skeleton = [(1, "D"), (1, "T1"), (1, "OTA"), (1, "RA")]




# def subdivisions_generatorold(y, maxsubd,squeleton_samples_indices,beat_length_in_samples):
#     print(f"Beat length in samples {beat_length_in_samples}")
#     for (start,end) in squeleton_samples_indices:
#         number_of_available_maxsubdiv = maxsubd 
#         index_of_current_slot_in_samples = start
#         print("")
#         print(f"start ={start}, end={end}")
#         while number_of_available_maxsubdiv > 0 :
#             subdiv_choosen = random_even(maxsubd)
#             hit_choosen = random.choice(list(paths.keys()))
#             print(f"Index of current slot: {index_of_current_slot_in_samples}")
#             print(f"The choosen hit is {hit_choosen}, and subdiv is {subdiv_choosen}")
#             if maxsubd//subdiv_choosen > number_of_available_maxsubdiv:
#                 break
#             number_of_available_maxsubdiv -= maxsubd//subdiv_choosen
#             print(f"the available divisions: {number_of_available_maxsubdiv} ")
#             duration = beat_length_in_samples // subdiv_choosen
#             print(f"Duration of choosen hit in samples: {duration} ")
            
#             if index_of_current_slot_in_samples +duration<=end:
#                 hit_data = get_audio_data(hit_choosen)[:duration]
#                 y[
#                     index_of_current_slot_in_samples:
#                     index_of_current_slot_in_samples +duration
#                 ] += hit_data
#                 index_of_current_slot_in_samples += duration
#         print("")

#     sf.write("even_testing.wav",y,samplerate=48000)
#     return y

def random_number(n):
    return  random.randint(1, n,2)
    


def subdivisions_generator(y, maxsubd,squeleton_samples_indices,beat_length_in_samples):
    subdivisions_y = np.zeros(len(y))
    index_of_current_slot_samples = 0

    while index_of_current_slot_samples < len(subdivisions_y):
        remaining = len(subdivisions_y) - index_of_current_slot_samples
        subdiv_choosen = random_number(maxsubd)
        hit_choosen = random.choice(list(paths.keys()))
        duration_of_choosen_hit_in_samples = beat_length_in_samples // subdiv_choosen
        hit_y = get_audio_data(hit_choosen)

        if duration_of_choosen_hit_in_samples < len(hit_y):
            add_len = duration_of_choosen_hit_in_samples
        else:
            add_len = len(hit_y)
        remaining = len(subdivisions_y) - index_of_current_slot_samples
        
        add_len = min(add_len, remaining)
        if add_len <= 0:
            break

        subdivisions_y[
            index_of_current_slot_samples : index_of_current_slot_samples + add_len
        ] += hit_y[:add_len]

        index_of_current_slot_samples += add_len
    
    for (start,end) in squeleton_samples_indices:
        subdivisions_y[start:end]=np.zeros(end-start)
    y+=subdivisions_y
    sf.write("even_testing_wih_s.wav",y,samplerate=48000)
    return y


squeleton_y,beat_length_in_samples,skeleton_length,squeleton_samples_indices= squeleton_generator(in_bpm, in_skeleton, in_maxsubd, in_num_cycles, "test.wav")
subdivisions_generator(y=squeleton_y,squeleton_samples_indices=squeleton_samples_indices,beat_length_in_samples=beat_length_in_samples,maxsubd=in_maxsubd)