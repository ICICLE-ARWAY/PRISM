import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read CPU data
cpu_data = pd.read_csv('./graphs/pt-cpu-times.csv', header=None, names=['cores', 'time'])
cpu_data = cpu_data.iloc[1:]  # Remove the first row (number of datapoints)

# Read GPU data
gpu_data = pd.read_csv('./graphs/pt-gpu-times.csv', header=None, names=['time'])
gpu_data = gpu_data.iloc[1:]  # Remove the first row (number of datapoints)

# Prepare data for plotting
cpu_groups = cpu_data.groupby('cores')['time']
cpu_data_list = [group for _, group in cpu_groups]
labels = sorted(cpu_groups.groups.keys()) + ['GPU']
data = cpu_data_list + [gpu_data['time']]

# Create the plot
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# Create box plots
bp = plt.boxplot(data, labels=labels, patch_artist=True)

# Customize colors
colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Set y-axis to log scale
plt.yscale('log')

# Set y-axis limits and ticks
plt.ylim(0.1, 10000)  # Ensure lower limit is above 0 to avoid log(0) issues and set upper limit higher for better spread
major_ticks = [0.1, 1, 10, 100, 1000, 10000]
minor_ticks = [i for i in np.arange(0.1, 1, 0.1)] + \
              [i for i in np.arange(1, 10, 1)] + \
              [i for i in np.arange(10, 100, 10)] + \
              [i for i in np.arange(100, 1000, 100)] + \
              [i for i in np.arange(1000, 10000, 1000)]

plt.yticks(major_ticks)
plt.gca().yaxis.set_minor_locator(plt.FixedLocator(minor_ticks))
plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())

# Adjust tick labels
plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter())
plt.gca().yaxis.set_tick_params(which='minor', size=0)
plt.gca().yaxis.set_tick_params(which='major', size=5)

# Add labels and title
plt.xlabel('Number of CPU Cores / GPU')
plt.ylabel('Time (seconds) - Log Scale')
plt.title('Benchmark: CPU vs GPU Performance')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add grid
plt.grid(True, which="major", ls="-", alpha=0.2)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
