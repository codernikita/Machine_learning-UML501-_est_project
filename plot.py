import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
csv_path = "slr\model\keypoint.csv"
df = pd.read_csv(csv_path, header=None)

# Columns: first is label, rest are keypoints
labels = df[0].values
keypoints = df.iloc[:, 1:].values

# Function to plot a hand
def plot_hand(kp, ax, title=''):
    kp = np.array(kp).reshape(-1, 2)
    ax.scatter(kp[:, 0], -kp[:, 1])
    for i in range(len(kp)):
        ax.text(kp[i,0], -kp[i,1], str(i))
    ax.set_title(title)
    ax.axis('equal')

# Plot some examples per confusing letters
confusing_letters = [14,14]  # e.g., S, T, M, N
fig, axs = plt.subplots(1, len(confusing_letters), figsize=(15,5))
for i, lbl in enumerate(confusing_letters):
    sample = keypoints[labels==lbl][0]
    plot_hand(sample, axs[i], f'Label {lbl}')
plt.show()
