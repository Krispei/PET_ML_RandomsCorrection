import numpy as np
from numba import njit
import opengate as gate

@njit
def coincidence_kernel(times, vols, energies, xs, ys, zs, events, window):

    n = len(times)

    max_pairs = n * 5
    out = np.zeros((max_pairs, 11))

    count = 0
    j = 0

    for i in range(n):

        while j < n and times[j] - times[i] < window:
            j += 1

        for k in range(i+1, j):

            if vols[i] == vols[k]:
                continue

            # Detector 1
            out[count,0] = xs[i]
            out[count,1] = ys[i]
            out[count,2] = zs[i]
            out[count,3] = energies[i]
            out[count,4] = times[i]
            # Detector 2
            out[count,5] = xs[k]
            out[count,6] = ys[k]
            out[count,7] = zs[k]
            out[count,8] = energies[k]
            out[count,9] = times[k]

            out[count,10] = 1 if events[i]==events[k] else 0

            count += 1

    return out[:count]

import uproot
import pandas as pd

def sort_coincidences(input_file, window_ns=4):

    tree = uproot.open(input_file)["photopeak"]

    df = tree.arrays([
        "GlobalTime",
        "PreStepUniqueVolumeID",
        "TotalEnergyDeposit",
        "EventID",
        "PostPosition_X",
        "PostPosition_Y",
        "PostPosition_Z",
    ], library="pd")
 
    df = df.sort_values("GlobalTime") # SORTS BY TIME
    
    print(f"Found {len(df['GlobalTime'])} singles!")


    df["DetectorID"], detector_map = pd.factorize(
        df["PreStepUniqueVolumeID"]
    )


    times = df["GlobalTime"].to_numpy(np.float64)
    x = df["PostPosition_X"].to_numpy(np.float32)
    y = df["PostPosition_Y"].to_numpy(np.float32)
    z = df["PostPosition_Z"].to_numpy(np.float32)
    vols = df["DetectorID"].to_numpy(np.int32)
    energies = df["TotalEnergyDeposit"].to_numpy(np.float32)
    events = df["EventID"].to_numpy(np.int32)

    print("Running compiled coincidence sorter...")

    coinc = coincidence_kernel(
        times,
        vols,
        energies,
        x, y, z,
        events,
        window_ns
    )

    columns = [
        'X1', 'Y1', 'Z1', 'E1', 'T1', 'X2', 'Y2', 'Z2', 'E2', 'T2', 'is_true'
    ]

    coinc_df = pd.DataFrame(coinc, columns=columns)
    coinc_df.to_csv("coincidence_listmode.csv", index=False)

    print(f"Done! Found {len(coinc_df)} coincidences.")

    n_true = coinc_df['is_true'].sum()       # 1s → true events
    n_false = len(coinc_df) - n_true        # 0s → false events


    print(f"Number of true coincidences: {n_true}")
    print(f"Number of random coincidences: {n_false}")
    print(f"Randoms percentage : {round(n_false/len(coinc_df)*100, 2)}%")

    total_time_s = times[-1] / 1e9
    singles_count_rate = round( (len(df['GlobalTime']) / total_time_s) / 1000, 2)
    coincidence_count_rate = round( (len(coinc_df) / total_time_s ) / 1000, 2)

    print(f"Singles count rate : {singles_count_rate} kcps")
    print(f"Coincidence count rate : {coincidence_count_rate} kcps")


sort_coincidences("test_f18_singles.root")
