import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def plot_results(data, title, labels):
    keys = ['map', 'map_50', 'map_75']
    values = [[d[key] for d in data] for key in keys]

    # Calculate the number of dictionaries and the width for each group of bars
    num_dicts = len(data)
    bar_width = 0.2

    # Create a list of indices for the x-axis ticks
    indices = np.arange(num_dicts)

    # Plot the histogram
    fig, ax = plt.subplots()
    for i, value in enumerate(values):
        ax.bar(indices + i * bar_width, value, width=bar_width, label=keys[i])
        for j, v in enumerate(value):
            ax.text(indices[j] + i * bar_width, v, str(v), ha='center', va='bottom')

    ax.set_xlabel('Run')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(indices + (bar_width * (len(keys) - 1)) / 2)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

data = [
    { 'map': (0.5457), 'map_50': (0.7604), 'map_75': (0.6261), },
    {'map': (0.5872), 'map_50': (0.7736), 'map_75': (0.6598),},

    {'map': (0.5461), 'map_50': (0.7622), 'map_75': (0.6204),},
    {'map': (0.6385), 'map_50': (0.8212), 'map_75': (0.7241),}
]

#'map': (0.5457), 'map_50': (0.7604), 'map_75': (0.6261), 
#
title = 'BBox mAp'
labels = ['v1_5layer','v2_5layer', 'v1_3layer', 'v2_3layer']
plot_results(data,title,labels)