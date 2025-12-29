"""This script takes the derbukka file format and turns it into a
sequence of tokens thatt will be taken as input by the model"""

from utils import load_file, save_json
import os

folder = "./fixed"


def get_arrays(content):
    contents = content.split("\n")
    global_tempo = float(contents[0])
    tempos = [float(x) for x in contents[1].split()]
    skeleton = contents[2].split()
    variations = contents[3].split()
    return global_tempo, tempos, skeleton, variations


def get_sequence(file_path, index):
    data = load_file(file_path=file_path)
    global_tempo, tempos, skeleton, variations = get_arrays(content=data)
    sequence = []
    i = 0
    j = 0
    k = 0
    count = 0
    value = 0
    current_subd = 0
    previous_tempo = global_tempo
    while i != len(skeleton):
        if skeleton[i].split("_")[0] == "DELAY":
            count = float(skeleton[i].split("_")[1])
            i += 1
        else:
            sequence.append(
                skeleton[i].split("_")[1]
                + "_0_"
                + str(tempos[k] - previous_tempo)
                + "_3_"
                + skeleton[i + 1][-1]
            )
            i += 2
        while count != 0 and j != len(variations):
            if variations[j].split("_")[0] == "SUBD":
                value = int(variations[j].split("_")[1])
                current_subd = variations[j].split("_")[1]
                previous_tempo = tempos[k]
                if k < len(tempos) - 1:
                    k += 1
                    current_delta = str(tempos[k] - previous_tempo)
                else:
                    current_delta = str(tempos[k] - previous_tempo)
                j += 1
            else:
                sequence.append(
                    variations[j].split("_")[1]
                    + "_"
                    + current_subd
                    + "_"
                    + current_delta
                    + "_"
                    + variations[j + 1].split("_")[1]
                    + "_0"
                )
                j += 2
                count -= 1 / value
    sequences = {"sequence": sequence}
    save_json(output_json_path=f"./corpus/sequence_{index}.json", data=sequences)


for i, filename in enumerate(os.listdir(folder)):
    filepath = os.path.join(folder, filename)
    if os.path.isfile(filepath):
        get_sequence(file_path=filepath, index=i)
