import uproot
import pandas as pd

# 1. Open the ROOT file
file = uproot.open('test_singles.root')

# 2. Access the specific data table (TTree)
# You can usually ignore the ';1' or include it explicitly
tree = file['photopeak'] 

# 3. See what columns (branches) are available inside
print("Available data columns:")
print(tree.keys())

# 4. Convert the ROOT tree into a Pandas DataFrame for easy analysis
# This loads all the arrays (Energy, Time, EventID, etc.) into a table
df = tree.arrays(library="pd")

# 5. Display the first few rows of your actual Singles data
print("\nFirst 5 Singles recorded:")
print(df.head())