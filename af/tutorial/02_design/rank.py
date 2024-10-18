#!/usr/bin/python3

import re
import os
import shutil
import matplotlib.pyplot as plt

# Update the regex pattern to include loss, i_ptm, and i_con values
pattern = re.compile(r"(\d+) models \[\d+\].*?loss (\d+\.\d+).*?i_con_1 (\d+\.\d+).*?i_con_2 (\d+\.\d+).*?plddt (\d+\.\d+).*?i_ptm (\d+\.\d+)")

# Initialize a list to store tuples of (model number, loss, i_con, plddt, i_ptm)
model_data = []

# Read the log file and extract the relevant information
with open('out.log', 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            model_number = int(match.group(1)) - 1
            loss_value = float(match.group(2))
            i_con_value_1 = float(match.group(3))
            i_con_value_2 = float(match.group(4))
            plddt_value = float(match.group(5))
            i_ptm_value = float(match.group(6))
            model_data.append((model_number, loss_value, i_con_value_1, i_con_value_2, plddt_value, i_ptm_value))

# Sort the models by their plddt values
sorted_models = sorted(model_data, key=lambda x: x[4], reverse=True)


for model in sorted_models[:20]:
    line = f"Model {model[0]}: loss {model[1]} i_con_1 {model[2]} i_con_2 {model[3]} plddt {model[4]} i_ptm {model[5]}"
    print(line)

model_num = [el[0] for el in model_data]
loss = [el[1] for el in model_data]




fig,ax = plt.subplots()

ax.plot(model_num, loss)
ax.set_xlabel('Model Num.')
ax.set_ylabel('Loss')
plt.show() 
