import itertools
import random
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import qmc
import pandas as pd

grammages = [150, 200, 250, 300]  # in gsm
material_types = ['Virgin', 'Recycled', 'Mixed']
flute_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'K']
# Define thicknesses for each flute type according to DIN 55468-1
flute_thicknesses = {
    'A': [x * 0.1 for x in range(40, 50)] ,  # Example thickness range for A-flute
    'B': [x * 0.1 for x in range(22, 31)],  # Example thickness range for B-flute
    'C': [x * 0.1 for x in range(31, 40)],  # Thickness range for C-flute
    'D': [x * 0.1 for x in range(19, 22)],  # Example thickness range for D-flute
    'E': [x * 0.1 for x in range(10, 19)],  # Example thickness range for E-flute
    'F': [x * 0.1 for x in range(6, 10)],  # Example thickness range for F-flute
    'G': [x * 0.1 for x in range(2, 6)],  # Example thickness for G-flute
    'K': [x * 0.1 for x in range(50, 100)]   # Example thickness for K-flute
}

wellenteilung = {
    'A': [0.1 + x * 0.1 for x in range(79, 100)],  # Example thickness range for A-flute
    'B': [0.1 + x * 0.1 for x in range(48, 65)],  # Example thickness range for B-flute
    'C': [0.1 + x * 0.1 for x in range(65, 79)],  # Thickness range for C-flute
    'D': [0.1 + x * 0.1 for x in range(35, 48)],  # Example thickness range for D-flute
    'E': [0.1 + x * 0.1 for x in range(26, 35)],  # Example thickness range for E-flute
    'F': [0.1 + x * 0.1 for x in range(18, 26)],  # Example thickness range for F-flute
    'G': [0.1 + x * 0.1 for x in range(2, 18)],  # Example thickness for G-flute
    'K': [0.1 + x * 0.1 for x in range(100, 1000)]  # Example thickness for K-flute
}

# Encode each category
encoded_grammages = {v: k for k, v in enumerate(grammages)}
encoded_materials = {v: k for k, v in enumerate(material_types)}
encoded_flutes = {v: k for k, v in enumerate(flute_types)}

# Maximum lengths of the lists for thickness and wellenteilung
max_thickness_length = max([len(v) for v in flute_thicknesses.values()])
max_wellenteilung_length = max([len(v) for v in wellenteilung.values()])

# Set up the LHS sampler
d = 5
n = 50
sampler = qmc.LatinHypercube(d=d, seed=0)
sample = sampler.random(n=n)

# Decode the samples and scale indices for thickness and wellenteilung
decoded_samples = []
for s in sample:
    grammage = grammages[int(np.floor(s[0] * len(grammages)))]
    material = material_types[int(np.floor(s[1] * len(material_types)))]
    flute = flute_types[int(np.floor(s[2] * len(flute_types)))]
    thickness_index = int(np.floor(s[3] * len(flute_thicknesses[flute])))
    wellenteilung_index = int(np.floor(s[4] * len(wellenteilung[flute])))

    thickness = round(flute_thicknesses[flute][thickness_index % len(flute_thicknesses[flute])], 1)
    wellen = round(wellenteilung[flute][wellenteilung_index % len(wellenteilung[flute])], 1)

    decoded_samples.append((grammage, material, flute, thickness, wellen))

# Convert to DataFrame
df = pd.DataFrame(decoded_samples, columns=['Grammage', 'Material Type', 'Flute Type', 'Thickness', 'Wellenteilung'])

# Print and save the DataFrame
print(df.head())
file_path = 'corrugated_board_combinations.csv'
df.to_csv(file_path, index=False)

# Return file path
file_path