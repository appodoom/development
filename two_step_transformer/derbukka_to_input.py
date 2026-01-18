"""This script takes the derbukka file format and turns it into a 
sequence of tokens thatt will be taken as input by the model"""

import json

def save_json(file_name, data):
    with open(file_name, "w", encoding="utf_8") as f:
        json.dump(data, f, indent=2)
def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content

def get_arrays(content):
    contents=content.split("\n")
    global_tempo=float(contents[0])
    tempos=[float(x) for x in contents[1].split()]
    skeleton=contents[2].split()
    variations=contents[3].split()
    return global_tempo, tempos, skeleton, variations

def get_sequence(file_path):
    data=load_file(file_path=file_path)
    global_tempo, tempos, skeleton, variations= get_arrays(content=data)
    sequence=[]
    i=0
    j=0
    k=0
    count=0
    value=0
    beat_value=0
    previous_tempo=global_tempo
    while (i!=len(skeleton)):
        if skeleton[i].split("_")[0]=="DELAY":
            count=float(skeleton[i].split("_")[1])
            i+=1
        else:
            sequence.append(skeleton[i])
            sequence.append(skeleton[i+1])
            i+=2
        while (count!=0 and j!=len(variations)):
            if variations[j].split("_")[0]=="SUBD":
                value=int(variations[j].split("_")[1])
                previous_tempo=tempos[k]
                if k<len(tempos)-1:
                    k+=1
                    sequence.append("TEMPO_"+str(tempos[k]-previous_tempo))
                else:
                    sequence.append("TEMPO_"+str(tempos[k]-previous_tempo))
                sequence.append(variations[j])
                j+=1
            else:
                sequence.append(variations[j])
                sequence.append(variations[j+1])
                j+=2
                count-=1/value
    sequences={"sequence":sequence}
    save_json(data=sequences, file_name="sequence_khara.json")
get_sequence("khara.txt")




    



