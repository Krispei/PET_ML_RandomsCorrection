import uproot
import matplotlib.pyplot as plt
import pandas as pd

file = uproot.open(rf"C:\Users\Krisps\PET_ML_RandomsCorrection\Data\temp\0.1s_20010_Mn52_80MBq_45_0_0.root")
tree = file["photopeak"]
df = tree.arrays(library="pd")

plt.figure(figsize=(10, 6))
plt.scatter(df['PostPosition_Z'], df['PostPosition_Y'], 
            c=df['TotalEnergyDeposit'], cmap='viridis', s=1, alpha=0.5)
plt.colorbar(label='Energy Deposit (MeV)')
plt.xlabel('Axial Position (Z) [mm]')
plt.ylabel('Transverse Position (Y) [mm]')
plt.title('Detector Hit Map')
plt.show()

# 3. Visualize the Energy Spectrum
plt.figure(figsize=(8, 5))
plt.hist(df['TotalEnergyDeposit'], bins=100, color='blue', edgecolor='black')
plt.xlabel('Energy (MeV)')
plt.ylabel('Counts')
plt.title('Energy Spectrum (Photopeak Validation)')
plt.axvline(0.511, color='red', linestyle='--') # Theoretical 511 keV
plt.show()