import math, sys, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

# ------ format ------
# plt.rcParams['text.usetex'] = True
sns.set_theme(
    context='paper', style='whitegrid', palette='bright',
    font='sans-serif', font_scale=1.4, color_codes=True, rc=None,
)
sns.set_style({'grid.linestyle': '--'})
# mpl.rc('font', family='serif', serif='Times New Roman')


# ------ init ------
ss = [
    8, 16, 32,
    64, 128, 256,
    512, 1024, 2048
]
tran_loss_dlt1 = [
    0.0882, 0.0898, 0.0855,
    0.0883, 0.0853, 0.0879,
    0.0867, 0.0866, 0.0865
]
test_loss_dlt1 = [
    0.0978, 0.0995, 0.0923,
    0.0875, 0.0856, 0.0870,
    0.0860, 0.0860, 0.0860
]
tran_col_dlt1 = [
    0, 1, 0,
    2, 1, 15,
    2, 12, 23
]
test_col_dlt1 = [
    234, 308, 143,
    42, 25, 28,
    12, 20, 25
]

tran_loss_dlt2 = [
    0.0941, 0.0945, 0.0889,
    0.0919, 0.0868, 0.0867,
    0.0905, 0.0889, 0.0889
]
test_loss_dlt2 = [
    0.1084, 0.0988, 0.0919,
    0.0929, 0.0871, 0.0864,
    0.0902, 0.0889, 0.0889
]
tran_col_dlt2 = [
    0, 0, 0,
    3, 2, 3,
    20, 0, 0,
]
test_col_dlt2 = [
    301, 54, 117,
    68, 25, 16,
    12, 0, 0,
]

# 10_12_13_57_29 (8), 10_11_12_15_58, 10_12_12_23_39, 10_12_12_24_39, 10_12_12_47_30, 10_12_12_48_26

ub_dict = {
    'delta': [0.1] * len(ss) + [0.2] * len(ss),
    'number of training rollouts': ss + ss,
    'av train loss': tran_loss_dlt1 + tran_loss_dlt2,
    'av test loss': test_loss_dlt1 + test_loss_dlt2,
    'total train num collisions': tran_col_dlt1 + tran_col_dlt2,
    'total test num collisions': test_col_dlt1 + test_col_dlt2,
    'ub on ub': [None] * 2 * len(ss)
}

# compute ub on ub
for row_num in range(2*len(ss)):
    s = ub_dict['number of training rollouts'][row_num]
    delta = ub_dict['delta'][row_num]
    lambda_star = (8*s*math.log(1/delta))**0.5
    gamma = lambda_star/8/s + 2/lambda_star*math.log(1/delta)
    ub_dict['ub on ub'][row_num] = ub_dict['av train loss'][row_num] + gamma

df = pd.DataFrame(ub_dict)

# ------------------
# ------ PLOT ------
# ------------------
# only used to create a proper FacetGrid
g = sns.catplot(
    data=df, x='number of training rollouts', y='ub on ub', #col='delta',
    hue='delta', kind='box', height=4, aspect=1.5,
    sharey=False, palette='hls', legend=False
)

# mark upper bounds
g.map_dataframe(
    sns.boxenplot, x='number of training rollouts', y='ub on ub',
    hue='delta', linewidth=2, linecolor='k', alpha = 0.6,
    legend=False, # dodge=True,
)

# mark sampled controllers performance
g.map_dataframe(
    sns.stripplot, x='number of training rollouts', y='av test loss',
    hue='delta', palette='hls', alpha=0.9, dodge=True
)

# add legend for the upper bound
custom_line = [Line2D([0], [0], color='k', lw=2)]

# ------ legends and titles ------
ax = g.axes[0,0]
# add vertical lines between groups
[ax.axvline(x+.5,color='k', alpha=0.2) for x in ax.get_xticks()]

# change xtick labels to integer without the leading 0
labels = [item.get_text() for item in ax.get_xticklabels()]
ax.set_xticks(ax.get_xticks(), [str(int(float(label))) for label in labels])

# axis labels
ax.set_xlabel(r'Number of training sequences ($s$)')
ax.set_ylabel(r'True cost ($\mathcal{L}$)')

# legend
handles, labels = ax.get_legend_handles_labels()
handles = handles+custom_line
labels = labels + ['Upper bound']
l = plt.legend(
    handles, labels, bbox_to_anchor=(0.62, 0.98),
    loc=2, borderaxespad=0.
)
# ---------------------------------
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
filename = os.path.join(save_path, 'ub.pdf')
plt.savefig(filename)
plt.show()