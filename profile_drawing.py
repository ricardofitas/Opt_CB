import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# LaTeX and bold settings
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['axes.linewidth'] = 2  # Bold axis lines
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
# Load data
csv_path = 'C:\\Users\\ricar\\OneDrive\\Documentos\\Spyder\\Optimization_CB\\Prob_prod_fix\\EPSO0_v5\\output.csv'  # Update this path
data = pd.read_csv(csv_path)
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values

# Special points
min_x_idx = np.argmin(x)
min_y_idx = np.argmin(y)

log_x = np.log10(x)
log_y = np.log10(y)
norm_log_x = (log_x - np.min(log_x)) / (np.max(log_x) - np.min(log_x))
norm_log_y = (log_y - np.min(log_y)) / (np.max(log_y) - np.min(log_y))
norms = np.sqrt(norm_log_x**2 + norm_log_y**2)
min_norm_idx = np.argmin(norms)

# Plot
fig, ax = plt.subplots(figsize=(12, 9))

# Scatter with larger points
ax.scatter(x, y, c='gray', edgecolor='black', s=500, zorder=1)
ax.scatter(x[[min_x_idx, min_y_idx, min_norm_idx]],
           y[[min_x_idx, min_y_idx, min_norm_idx]],
           c='red', edgecolor='black', s=1000, zorder=2)

# Bold axis labels
ax.set_xlabel(r'\textbf{Inertia$^{-1}$ [m$^3 \times 10^{-6}$]}', fontsize=28)
ax.set_ylabel(r'\textbf{Profile Area [mm$^2$]}', fontsize=28)

# Remove grid, keep bold axes
ax.grid(False)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

plt.tight_layout()
plt.show()