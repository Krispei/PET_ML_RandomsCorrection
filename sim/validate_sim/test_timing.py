import uproot
import numpy as np

tree = uproot.open("0.02s_20004_F18_100MBq_0_0_0.root")["photopeak"]
t = tree["GlobalTime"].array(library="np")
e = tree["TotalEnergyDeposit"].array(library="np")

print(np.min(t), np.max(t))
print(np.mean(np.diff(np.sort(t))))

