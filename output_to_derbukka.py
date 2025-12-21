"""This script takes the ouput of the model 
and turns it into a derbukka file"""

import json
def load_json(directory):
    with open(directory, 'r') as file:
        data = json.load(file)
    return data["sequence"]

def write_derbukka(file_name,string):
    with open(file_name, "w") as file:
        file.write(string)

def to_derbukka(directory,initial_tempo):
    sequence=load_json(directory)
    initial_tempo_str=str(initial_tempo)
    count=0
    value=0
    tempos_str=""
    skeleton_str=""
    variations_str=""
    for i in range(len(sequence)):
        if sequence[i].split("_")[0]=="TEMPO":
            tempos_str+=str(initial_tempo+float(sequence[i].split("_")[1]))
            tempos_str+=" "
        elif sequence[i].split("_")[0]=="SUBD":
            variations_str+=sequence[i]
            variations_str+=" "
            value=1/int(sequence[i].split("_")[1])
        elif sequence[i].split("_")[0]=="AMP":
            variations_str+=sequence[i-1]+" "+sequence[i]+" "
            count+=value
        elif sequence[i].split("_")[0]=="DEV":
            skeleton_str+="DELAY_"+str(int(count) if count%1==0 else count)+" "+sequence[i-1]+" "+sequence[i]+" "
            count=0
        else:
            continue
    write_derbukka("output.txt", initial_tempo_str+"\n"+tempos_str[:-1]+"\n"+skeleton_str[:-1]+"\n"+variations_str[:-1])

to_derbukka("sequence.json", 110.0)






