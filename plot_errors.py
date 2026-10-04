import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Prepare data extracted from the report
data = {
    "x_crop": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4,4,4,4],
    "y_crop": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2,3,4,1,2,3,4],
    "u_diff": [
        20.043318,
        28.189070,
        30.860099,
        22.609339,
        31.261282,
        41.289576,
        33.426995,
        23.258447,
        32.722672,
        40.747777,
    ],
    "v_diff": [
        11.134208,
        17.297496,
        24.731619,
        30.028619,
        12.190807,
        36.057931,
        45.584014,
        44.422203,
        23.666198,
        28.480486,
    ],
    "avg_diff": [
        15.588763,
        22.743283,
        27.795859,
        26.318979,
        21.726044,
        38.673753,
        39.505504,
        33.840325,
        28.194435,
        34.614132,
    ],
}

df = pd.DataFrame(data)

# 2. Pivot the DataFrame to form a 2D matrix (x_crop vs y_crop)
heatmap_matrix = df.pivot(
    index="y_crop", columns="x_crop", values="avg_diff"
)

# 3. Plot the Heatmap Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_matrix,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    cbar_kws={"label": "Average Parameter Error (%)"},
    linewidths=0.5,
)

plt.title("STRidge Crop Parameter Error Heatmap Matrix (t_crop = 1800)", fontsize=13, pad=12)
plt.xlabel("X Crop", fontsize=11)
plt.ylabel("Y Crop", fontsize=11)
plt.gca().invert_yaxis()  # Invert y-axis so y=1 starts at the bottom

plt.tight_layout()
plt.show()