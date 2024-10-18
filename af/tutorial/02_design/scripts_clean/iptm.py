#!/usr/bin/python3


import re
import os
import shutil


# Update the regex pattern to include loss, i_ptm, and i_con values
pattern = re.compile(r"(\d+) models \[\d+\].*?loss (\d+\.\d+).*?i_con_1 (\d+\.\d+).*?i_con_2 (\d+\.\d+).*?plddt (\d+\.\d+).*?i_ptm (\d+\.\d+).*?com_loss (\d+\.\d+)")

# Initialize a list to store tuples of (model number, loss, i_con, plddt, i_ptm)
model_data = []

# Read the log file and extract the relevant information
with open('out.log', 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match and float(match.group(5)) > 0.75:
            model_number = int(match.group(1)) - 1
            loss_value = float(match.group(2))
            i_con_value_1 = float(match.group(3))
            i_con_value_2 = float(match.group(4))
            plddt_value = float(match.group(5))
            i_ptm_value = float(match.group(6))
            model_data.append((model_number, loss_value, i_con_value_1, i_con_value_2, plddt_value, i_ptm_value))

# Sort the models by their plddt values
sorted_models = sorted(model_data, key=lambda x: x[4], reverse=True)


# Path to the steps directory and the destination directory
steps_directory = 'trajectory'
destination_directory = 'ranked_top'

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Copy the corresponding pdb files to the destination directory
for model in sorted_models[:20]:
    model_number = model[0] 
    pdb_filename = f'model_step_{model_number}.pdb'
    
    # Find the pdb file in the steps directory (assuming there's only one match)
    for root, dirs, files in os.walk(steps_directory):
        for file in files:
            if file.startswith(f'model_step_{model_number}') and file.endswith('.pdb'):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_directory, file)
                
                # Copy the pdb file to the destination directory
                shutil.copy2(source_path, destination_path)
                break

# Display the ranked models and write to ranked.txt
with open('ranked.txt', 'w') as ranked_file:
    for model in sorted_models[:20]:
        line = f"Model {model[0]}: loss {model[1]} i_con_1 {model[2]} i_con_2 {model[3]} plddt {model[4]} i_ptm {model[5]}"
        print(line)
        ranked_file.write(line + '\n')

