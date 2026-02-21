import uproot
import pandas as pd
import numpy as np

def sort_coincidences(input_file, window_ns=4.8):
    # 1. Load Singles
    tree = uproot.open(input_file)["photopeak"]
    df = tree.arrays(library="pd")
    
    # Sort by time (essential for sliding window)
    df = df.sort_values('GlobalTime').reset_index(drop=True)
    
    coincidences = []
    num_singles = len(df)
    window_s = window_ns * 1e-9 # Convert ns to seconds (Geant4 base unit)

    print(f"Sorting {num_singles} singles...")

    # 2. Sliding Window Loop
    for i in range(num_singles):
        j = i + 1
        # Look ahead as long as hits are within the time window
        while j < num_singles and (df.iloc[j]['GlobalTime'] - df.iloc[i]['GlobalTime']) < window_s:
            
            # Spatial filter: Ignore hits in the same crystal (cross-talk)
            if df.iloc[i]['PreStepUniqueVolumeID'] == df.iloc[j]['PreStepUniqueVolumeID']:
                j += 1
                continue

            # Create List-Mode record
            pair = {
                't1': df.iloc[i]['GlobalTime'],
                't2': df.iloc[j]['GlobalTime'],
                'dt': df.iloc[j]['GlobalTime'] - df.iloc[i]['GlobalTime'],
                'e1': df.iloc[i]['TotalEnergyDeposit'],
                'e2': df.iloc[j]['TotalEnergyDeposit'],
                'vol1': df.iloc[i]['PreStepUniqueVolumeID'],
                'vol2': df.iloc[j]['PreStepUniqueVolumeID'],
                'event1': df.iloc[i]['EventID'],
                'event2': df.iloc[j]['EventID'],
                # GROUND TRUTH LABEL: 1 for True, 0 for Random
                'is_true': 1 if df.iloc[i]['EventID'] == df.iloc[j]['EventID'] else 0
            }
            coincidences.append(pair)
            j += 1

    # 3. Save to List-Mode CSV (or another ROOT file)
    coinc_df = pd.DataFrame(coincidences)
    coinc_df.to_csv("coincidence_listmode.csv", index=False)
    print(f"Done! Found {len(coinc_df)} coincidences.")
    return coinc_df

# Run it
coinc_data = sort_coincidences("test_singles.root")