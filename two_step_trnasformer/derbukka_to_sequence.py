"""This script takes the derbukka file format and turns it into a 
sequence of hits where each hit is represented by 
(hit_type, hit_subd, delta_tempo, amplitude, deviation)"""

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
    sequence=[]
    data=load_file(file_path=file_path)
    global_tempo, tempos, skeleton, variations= get_arrays(content=data)
    i=0
    j=0
    k=0
    count=0
    value=0
    while (i!=len(skeleton)):
        if skeleton[i].split("_")[0]=="DELAY":
            count=int(skeleton[i].split("_")[1])
            i+=1
        else:
            sequence.append((skeleton[i],"SUBD_0",tempos[k]-global_tempo,"AMP_1.5",skeleton[i+1]))
            i+=2
        while (count!=0 and j!=len(variations)):
            if variations[j].split("_")[0]=="SUBD":
                value=int(variations[j].split("_")[1])
                j+=1
            else:
                sequence.append((variations[j],"SUBD_"+str(value), tempos[k]-global_tempo, variations[j+1],"DEV_0"))
                j+=2
                count-=1/value
                if count%1==0:
                    k+=1
    sequence_skeleton=[]
    sequence_variations=[]
    for hit in sequence:
        if hit[1]=="SUBD_0":
            sequence_skeleton.append(hit)
        else:
            sequence_variations.append(hit)
    sequences={"skeleton":sequence_skeleton, "variations": sequence_variations, "sequence":sequence, "global_tempo":global_tempo}
    save_json(data=sequences, file_name="sequence.json")
get_sequence("sample.txt")




    



