with open("output.txt", "r", encoding="utf-8") as file:
        output=file.read()
with open("sample.txt", "r", encoding="utf-8") as file:
        sample=file.read()
i=0
while i<len(output) and i<len(sample):
    if sample[i]!=output[i]:
           print(i)
           print(sample[i-10:i+10]+"\n"+output[i-10:i+10])
    i+=1