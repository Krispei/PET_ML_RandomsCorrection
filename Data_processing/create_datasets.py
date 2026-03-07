'''
    This file creates the training and validation datasets 
'''
import torch
import os 
import glob
import random

WINDOWED_DATA_FILE = rf"C:\Users\Krisps\PET_ML_RandomsCorrection\Data\WINDOWED"

def retrieve_data(file_path):
    
    obj = torch.load(file_path, mmap=True, map_location="cpu", weights_only=False)

    return obj


def merge_and_mix_data(data):
    
    random.shuffle(data)

def split_dataset(data, ratios):

    n = len(data)

    train_split = int(n * ratios[0])
    validation_split = int(n * ratios[1])
    test_split = int(n * ratios[2])

    train_dataset = data[:train_split]
    valdiation_dataset = data[train_split:validation_split]
    test_dataset = data[validation_split:]

    return train_dataset, valdiation_dataset, test_dataset


def main():
    
    random.seed(123123)

    # Load all datasets into one dataset and mix them

    file_names = glob.glob(WINDOWED_DATA_FILE + "/*250k")

    data = []

    for file in file_names:
        
        print(f"Retrieving {file}")

        new_data = retrieve_data(file)
        print(len(new_data))
        data.append(new_data)

        del new_data

    print(f"found {len(data)} files")


    max_data_length = min(len(lst) for lst in data)

    print(f"windows drawn from each file: {max_data_length}")
    # Make sure that theres the same amount of windows from each file
    even_dataset = []
    
    for lst in data:
        even_dataset.extend(lst[:max_data_length])

    del data

    merge_and_mix_data(even_dataset)

    # Split into training and validation.
    train, validation, test = split_dataset(even_dataset, (0.7,0.85,1))
    
    del even_dataset
    # save 
    torch.save(train, "training_dataset")
    torch.save(validation, "validation_dataset")
    torch.save(test, "testing_dataset")


if __name__ == "__main__":
    main()