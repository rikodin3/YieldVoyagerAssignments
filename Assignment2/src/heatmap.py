# heatmap.py

import matplotlib.pyplot as plt
import numpy as np
import os

def save_heatmap(scores, labels, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 1))
    plt.imshow(np.array([scores]), cmap="viridis", aspect="auto")
    plt.yticks([])
    plt.xticks(range(len(scores)), labels, rotation=90, fontsize=6)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
