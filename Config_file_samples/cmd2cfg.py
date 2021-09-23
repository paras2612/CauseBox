import os

#use sample_cmd as input format
inname = "sample_cmd.txt"
with open(inname, "r") as f:
    cmd = f.readline()
    cmdList = cmd.split(" ")
    del cmdList[0:3] #python main.py modelname
    i = 0
    keys = list()
    values = list()
    for param in cmdList:
        if i%2 == 0:
            param = param.replace("--", "")
            key = param
            keys.append(key)
        else:
            value = param
            values.append(value)
        i+=1

outname = "CFRNET_cfg.txt"
if len(keys) != len(values):
    assert "Key value pair not matching"
    exit(0)

with open(outname, "w") as f:
    for key, value in zip(keys, values):
        if key == "dataset":
            continue
        f.write(key+" "+value+"\n")

print("File converted to cfg format 'key value'")
os.remove(inname)