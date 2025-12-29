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
    last_delay=0
    tempos_str=""
    skeleton_str=""
    variations_str=""
    previous_tempo=initial_tempo
    if (sequence[0].split("_")[1]!="0"):
        variations_str+="SUBD_"+sequence[0].split("_")[1]+" "
    else:
        variations_str+="SUBD_"+sequence[1].split("_")[1]+" "

    for i in range(len(sequence)):
        components=sequence[i].split("_")
        if components[1]!="0":
            value=int(components[1])
            count+=1/value
            if count%1==0:
                variations_str+="HIT_"+components[0]+" "
                variations_str+="AMP_"+components[3]+" "
                variations_str+="SUBD_"+components[1]+" "
                tempos_str+=str(float(previous_tempo+float(components[2])))+" "
            else:
                variations_str+="HIT_"+components[0]+" "
                variations_str+="AMP_"+components[3]+" "
        else:
            skeleton_str+="DELAY_"+str(count-last_delay)+" "
            last_delay=count
            skeleton_str+="HIT_"+components[0]+" "
            skeleton_str+="DEV_"+components[-1]+" "

    write_derbukka("output.txt", initial_tempo_str+"\n"+tempos_str[:-1]+"\n"+skeleton_str[:-1]+"\n"+variations_str[:-1])

to_derbukka("sequence.json", 110.0)






