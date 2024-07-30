import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cpu_data = pd.read_csv('./graphs/pt-cpu-times.csv', header=None, names=['cores', 'time'])
cpu_data = cpu_data.iloc[1:]  # Remove the first row (number of datapoints)

gpu_data = pd.read_csv('./graphs/pt-gpu-times.csv', header=None, names=['time'])
gpu_data = gpu_data.iloc[1:]  # Remove the first row (number of datapoints)

cpu_groups = cpu_data.groupby('cores')['time']
cpu_data_list = [group for _, group in cpu_groups]
labels = sorted(cpu_groups.groups.keys()) + ['GPU']
data = cpu_data_list + [gpu_data['time']]

plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

bp = plt.boxplot(data, labels=labels, patch_artist=True)

colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)



plt.ylim(1, 10000)  
major_ticks = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
minor_ticks = np.logspace(0, 2, 9)

plt.yticks(major_ticks)
plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())

plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter())
plt.gca().yaxis.set_tick_params(which='minor', size=0)
plt.gca().yaxis.set_tick_params(which='major', size=5)

plt.xlabel('Number of CPU Cores / GPU')
plt.ylabel('Time (seconds)')
plt.title('Pytorch Benchmark: CPU vs GPU Performance')

plt.xticks(rotation=45)

plt.grid(True, which="major", ls="-", alpha=0.2)

plt.tight_layout()
plt.show()
