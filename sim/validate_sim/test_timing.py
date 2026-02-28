import uproot
import numpy as np

tree = uproot.open("0.02s_20004_F18_100MBq_0_0_0.root")["photopeak"]
t = tree["GlobalTime"].array(library="np")
e = tree["TotalEnergyDeposit"].array(library="np")

print(np.min(t), np.max(t))

print(np.mean(np.diff(np.sort(t))))

'''
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# Plot a 'zoom' to see the interval between events
plt.hist(e, bins=100, color='salmon', edgecolor='black')
plt.xlabel("Energies")
plt.ylabel("Counts")

plt.tight_layout()
plt.show()
'''