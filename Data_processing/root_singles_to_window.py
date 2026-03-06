'''
    This code takes all the root files in the Data folder and creates windows. 
    It saves the data as torch graph objects
''' 

import torch
import uproot 
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data
import opengate as gate
import glob
import os

"""
    CONFIGURATION:

    input a folder path where all the .ROOT files live. 
    input a folder pash where you want all the saved pytorch graph objects to go

"""

INPUT_FOLDER_PATH = rf"C:\Users\Krisps\PET_ML_RandomsCorrection\Data\ROOT"
OUTPUT_FOLDER_PATH = rf"C:\Users\Krisps\PET_ML_RandomsCorrection\Data\WINDOWED"

def root_to_window(input_file, save_path, window_ns=10):
    
    #Loading data from ROOT file
    tree = uproot.open(input_file)["photopeak"]
    raw_data = tree.arrays([
        "GlobalTime",
        "TotalEnergyDeposit",
        "PostPosition_X",
        "PostPosition_Y",
        "PostPosition_Z",
        "EventID"], library="np")


    #Sort the data by GlobalTime:
    #First create the correct index order
    sorted_indices = np.argsort(raw_data["GlobalTime"])
    #Then sort the data using the sorted indices
    sorted_time = raw_data["GlobalTime"][sorted_indices]
    sorted_energy = raw_data["TotalEnergyDeposit"][sorted_indices]
    sorted_x = raw_data["PostPosition_X"][sorted_indices]
    sorted_y = raw_data["PostPosition_Y"][sorted_indices]
    sorted_z = raw_data["PostPosition_Z"][sorted_indices]
    sorted_event = raw_data["EventID"][sorted_indices]

    #Settings for the windowing:
    window_width = window_ns * gate.g4_units.ns 
    start_ptr = 0
    total_events = len(sorted_time)

    #List to hold the windowed data
    dataset = []

    print("Iterating through the windows ... ")
    progress_bar = tqdm(total=total_events)

    #Iterate through each window:
    while start_ptr < total_events:

        end_time = sorted_time[start_ptr] + window_width
        end_ptr = np.searchsorted(sorted_time, end_time, side="right")

        if end_ptr - start_ptr >= 2:

            windowed_time = sorted_time[start_ptr: end_ptr]
            windowed_energy = sorted_energy[start_ptr: end_ptr]
            windowed_x = sorted_x[start_ptr: end_ptr]
            windowed_y = sorted_y[start_ptr: end_ptr]
            windowed_z = sorted_z[start_ptr: end_ptr]
            windowed_pos = np.stack([windowed_x, windowed_y, windowed_z], axis=1)
            windowed_event = sorted_event[start_ptr: end_ptr]

            # Normalize the time
            normalized_window_time = (windowed_time - windowed_time[0]) / window_width

            # Node feature matricies
            x = torch.tensor(np.column_stack([windowed_energy, windowed_pos, normalized_window_time]), dtype=torch.float)

            num_nodes = end_ptr - start_ptr

            adj = torch.combinations(torch.arange(num_nodes), r=2).t()
            # example for 3 nodes 0,1,2:
            # [[0, 1],
            #  [0, 2],
            #  [0, 3], 
            #  [1, 2],
            # ...

            id_i = windowed_event[adj[0]]
            id_j = windowed_event[adj[1]]

            y = torch.tensor((id_i == id_j), dtype=torch.float)

            graph = Data(x=x, edge_index=adj, y=y)
            dataset.append(graph)

        new_start_ptr = max((end_ptr-start_ptr) // 2, start_ptr + 1)

        progress_bar.update(new_start_ptr - start_ptr)

        start_ptr = new_start_ptr

    if save_path:
        torch.save(dataset, save_path)
        print(f"Saved {len(dataset)} windows to {save_path}")

    return dataset

def main():

    root_files = glob.glob(INPUT_FOLDER_PATH + "/*.root")

    for root_file in root_files:
        
        print(f"Processing {os.path.basename(root_file)[:-5]}")

        filename = os.path.basename(root_file)[:-5]

        output_file = OUTPUT_FOLDER_PATH + "/" + filename

        root_to_window(root_file, output_file, window_ns=15)

if __name__ == "__main__":

    main()


