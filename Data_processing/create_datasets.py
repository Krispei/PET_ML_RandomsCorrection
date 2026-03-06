'''
    This file creates the training and validation datasets 
'''
import torch
import os 
import glob

WINDOWED_DATA_FILE = rf"C:\Users\Krisps\PET_ML_RandomsCorrection\Data\WINDOWED"

def retrieve_data(file_path):
    pass

def merge_and_mix_data(data):
    pass

def main():
    
    # Load all datasets into one dataset and mix them

    file_names = glob.glob(WINDOWED_DATA_FILE + "/*")

    data = []

    for file in file_names:

        new_data = retrieve_data(file)
        data.extend(new_data)

    merge_and_mix_data

    # Split into training and validation.
    
    # save 
    
    pass

if __name__ == "__main__":
    main()