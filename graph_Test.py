import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

aa = pd.read_csv("CSI_ampt.csv", header=None)
def fillna_with_previous_values(s):
    non_nan_values = s[s.notna()].values
    # Gets the location of the missing value
    nan_indices = s.index[s.isna()]
    # Calculate the number of elements to fill
    n_fill = len(nan_indices)
    # Count the number of repetitions required
    n_repeat = int(np.ceil(n_fill / len(non_nan_values)))
    # Generate the fill value
    fill_values = np.tile(non_nan_values, n_repeat)[:n_fill]
    # Fill missing value
    s.iloc[nan_indices] = fill_values
    return s
aa=aa.apply(fillna_with_previous_values,axis=1)
CSI_train = aa.values.astype('float32')
CSI_train=CSI_train/np.max(CSI_train)

plt.plot(CSI_train[0][0:1000])
plt.show()
