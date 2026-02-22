import uproot
import numpy as np

tree = uproot.open("test_singles.root")["photopeak"]
t = tree["GlobalTime"].array(library="np")

t = np.sort(t)

print(t[:100])

print(np.min(t), np.max(t))


print(np.min(np.diff(np.sort(t))))