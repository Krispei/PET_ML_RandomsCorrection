import glob
import torch

path = rf"C:\Users\Krisps\PET_ML_RandomsCorrection\Data\WINDOWED"

files = glob.glob(path + "/*")

for file in files:

    print(f"Loading {file}")

    obj = torch.load(file, mmap=True, map_location="cpu", weights_only=False)

    obj = obj[:250000]

    torch.save(obj, file + "_250k")
