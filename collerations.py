import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('EPL_form_elo_API.csv', header=0)
corr = data.corr()
print(corr.to_csv('correlations.csv'))
print(corr.to_string())

# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0, len(data.columns), 1)
# ax.set_xticks(ticks)
# plt.xticks(rotation=90)
# ax.set_yticks(ticks)
# ax.set_xticklabels(data.columns)
# ax.set_yticklabels(data.columns)
# plt.savefig('correlations.png', dpi=300)
# plt.show()
