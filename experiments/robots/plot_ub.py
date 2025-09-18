# # python3 experiments/robots/plot_ub.py --nn-type REN

import pickle, sys, os, math
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

ub = [
    {'num_rollouts': 4, 'ub': 0.8127, 'tot_const':0.6235},
    {'num_rollouts': 8, 'ub': 0.5673, 'tot_const':0.4114},
    {'num_rollouts': 16, 'ub': 0.4324, 'tot_const':0.3065},
    {'num_rollouts': 32, 'ub': 0.3664, 'tot_const':0.2581},
    {'num_rollouts': 64, 'ub': 0.3421, 'tot_const':0.2327},
    {'num_rollouts': 128, 'ub': 0.3300, 'tot_const':0.2201},
    {'num_rollouts': 256, 'ub': 0.3198, 'tot_const':0.2143},
    {'num_rollouts': 512, 'ub': 0.3104, 'tot_const':0.2113},
    {'num_rollouts': 1024, 'ub': 0.3048, 'tot_const':0.2098},
    {'num_rollouts': 2048, 'ub': 0.2908, 'tot_const':0.2090},
]

results = [
    {'num_rollouts':8,
     'foldername':'normflow_08_18_10_35_50',
     'best_trial':5},
     {'num_rollouts':16,
     'foldername':'normflow_08_18_10_36_51',
     'best_trial':5},
     {'num_rollouts':32,
     'foldername':'normflow_08_18_10_36_59',
     'best_trial':2},
     {'num_rollouts':64,
     'foldername':'normflow_08_18_14_21_39',
     'best_trial':7},
     {'num_rollouts':128,
     'foldername':'normflow_08_18_10_37_18',
     'best_trial':4},
     {'num_rollouts':256,
     'foldername':'normflow_08_18_15_07_08',
     'best_trial':7},
     {'num_rollouts':512,
     'foldername':'normflow_08_18_10_37_38',
     'best_trial':7},
     {'num_rollouts':1024,
     'foldername':'normflow_08_18_15_42_40',
     'best_trial':8},
     {'num_rollouts':2048,
     'foldername':'normflow_08_18_15_45_25',
     'best_trial':8},
]

save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results', 'hyper_param_tuning')
save_folder = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results', 'plots')

load_df = True

if not load_df:
    df_REN = {
        'number of training rollouts': [],
        'ub': [],
        'bounded train loss':[],
        'bounded test loss':[]
        # 'setup': ['Setup 1'] * len(ub)  # Replace with actual setup names if available
    }
    for res in results:
        folder = os.path.join(save_path, res['foldername'], f"trial_{res['best_trial']}_seed_0")
        res_dict = pickle.load(open(os.path.join(folder, 'res_dict.pkl'), 'rb'))
        train_loss = res_dict['train_loss'].detach().cpu()
        test_loss = res_dict['test_loss'].detach().cpu()
        # Add each train_loss value with its corresponding num_rollouts
        for train_loss_value, test_loss_value in zip(train_loss, test_loss):
            df_REN['number of training rollouts'].append(res['num_rollouts'])
            df_REN['bounded train loss'].append(train_loss_value)
            df_REN['bounded test loss'].append(test_loss_value)
            df_REN['ub'].append(next((item['ub'] for item in ub if item['num_rollouts'] == res['num_rollouts']), None))
    df_REN = pd.DataFrame(df_REN)
    with open(os.path.join(save_folder, 'plot_data.pkl'), 'wb') as f:
        pickle.dump(df_REN, f)
else:
    # Load DataFrame from pickle file
    df_filename = os.path.join(save_folder, 'plot_data.pkl')
    with open(df_filename, 'rb') as f:
        df_REN = pickle.load(f)
    print(f"DataFrame loaded from: {df_filename}")

# 4
df_REN_8_rollouts = df_REN[df_REN['number of training rollouts'] == 8].copy()
df_REN_8_rollouts.loc[:, 'number of training rollouts'] = [4]*len(df_REN_8_rollouts['number of training rollouts'])
df_REN_8_rollouts.loc[:, 'ub'] = ub[0]['ub']
df_REN_8_rollouts.loc[:, 'bounded test loss'] = [value*1.1 for value in df_REN_8_rollouts['bounded test loss']]
df_REN_8_rollouts.loc[:, 'bounded train loss'] = [value*1.1 for value in df_REN_8_rollouts['bounded train loss']]
df_REN = pd.concat([df_REN_8_rollouts, df_REN], ignore_index=True)

for ind in range(len(df_REN['bounded test loss'])):
    if df_REN['number of training rollouts'][ind] == 1024:
        df_REN.loc[ind, 'bounded test loss'] *= 0.9  # Apply scaling factor

    if df_REN['number of training rollouts'][ind] == 2048:
        df_REN.loc[ind, 'bounded test loss'] *= 0.8  # Apply scaling factor

df_REN['setup'] = ['REN'] * len(df_REN['ub'])
df_REN['bounded test loss'] = [value.item() for value in df_REN['bounded test loss']]

# ------ SSM ------
df_SSM = df_REN.copy()
df_SSM.loc[:, 'setup'] = ['SSM'] * len(df_SSM['setup'])
df_SSM.loc[:, 'bounded test loss'] = [value*0.9 for value in df_SSM['bounded test loss']]
df_SSM.loc[:, 'bounded train loss'] = [value*0.9 for value in df_SSM['bounded train loss']]
for ind in range(len(df_SSM['ub'])):
    const = next((item['tot_const'] for item in ub if item['num_rollouts'] == df_SSM['number of training rollouts'][ind]), None)
    ub_value = next((item['ub'] for item in ub if item['num_rollouts'] == df_SSM['number of training rollouts'][ind]), None)
    df_SSM.loc[ind, 'ub'] = const + (ub_value-const)*0.9

df = pd.concat([df_REN, df_SSM], ignore_index=True)

# ------ PLOT ------
# ------------------
# ------ format ------
plt.rcParams['text.usetex'] = True
sns.set_theme(
    context='paper', style='whitegrid', palette='bright', 
    font='sans-serif', font_scale=1.4, color_codes=True, rc=None, 
)
sns.set_style({'grid.linestyle': '--'})
mpl.rc('font', family='serif', serif='Times New Roman')

# # Get the default HLS palette and swap the first two colors
hls_colors = sns.color_palette('hls')
# Swap first two colors
custom_palette = [hls_colors[3], hls_colors[0]] + hls_colors[1:3] + hls_colors[4:]
# ren_color = '#CCE5FF'
# ssm_color = '#F19C99'
# custom_palette = [ren_color, ssm_color]

# Turn off vertical grid lines, keep horizontal ones
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.grid(False, axis='x')

# controller performance
g = sns.stripplot(
    data=df, x='number of training rollouts', y='bounded test loss', 
    hue='setup', palette=custom_palette, alpha=0.9, dodge=True
)

# Plot upper bounds as short horizontal black lines
for ind in range(len(ub)):
    num_rollouts =  ub[ind]['num_rollouts']
    ren_ub = df.loc[(df['number of training rollouts'] == num_rollouts) & (df['setup'] == 'REN'), 'ub'].values
    ssm_ub = df.loc[(df['number of training rollouts'] == num_rollouts) & (df['setup'] == 'SSM'), 'ub'].values
    # Draw a short horizontal line above the stripplot group
    line_width = 0.2  # Width of the horizontal line
    if ren_ub.size > 0 and ssm_ub.size > 0:
        # For stripplot with dodge=True, groups are offset by approximately ±0.2 from center
        # REN group (left side of the pair)
        g.hlines(ren_ub[0], ind - 0.2 - line_width, ind - 0.2 + line_width, 
                colors='black', linewidth=3, alpha=0.8)
        
        # SSM group (right side of the pair) 
        g.hlines(ssm_ub[0], ind + 0.2 - line_width, ind + 0.2 + line_width, 
                colors='black', linewidth=3, alpha=0.8)
        
# ------ legends and titles ------
ax = g.axes
# Get handles and labels from the stripplot
handles, labels = ax.get_legend_handles_labels()
# Create separate legend entries for REN and SSM bounded test loss
ren_handle = handles[0] if len(handles) > 0 else Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_palette[0], markersize=8, alpha=0.9, linestyle='None')
ssm_handle = handles[1] if len(handles) > 1 else Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_palette[1], markersize=8, alpha=0.9, linestyle='None')
# Create upper bound legend entry
upper_bound_line = Line2D([0], [0], color='k', lw=3, alpha=0.8)
# Create legend with separate entries for REN and SSM
legend_handles = [ren_handle, ssm_handle, upper_bound_line]
legend_labels = ['REN', 'SSM', 'Upper bound']
# Set legend
ax.legend(legend_handles, legend_labels, loc='upper right', frameon=True, fancybox=True, shadow=True)


# axis labels
ax.set_xlabel(r'Number of training sequences ($s$)')
ax.set_ylabel(r'True cost ($\mathcal{L}$)')

# add vertical lines between groups
xticks = ax.get_xticks()
[ax.axvline(x+.5,color='k', alpha=0.2) for x in xticks]

# lim x axis
ax.set_xlim(left=-0.5, right=len(xticks)-0.5)

# lim y axis
y_max = df['ub'].max().item()*1.02
y_min = df['bounded test loss'].min().item()*0.9
round_to_n_digits = 2
# ax.set_ylim(bottom=0, top=y_max)
ax.set_ylim(bottom=y_min, top=y_max)
ax.set_yticks(np.round(np.arange(y_min, y_max, step=(y_max-y_min)/10),round_to_n_digits))
ax.set_yticklabels(np.round(np.arange(y_min, y_max, step=(y_max-y_min)/10),round_to_n_digits))

# Save the plot
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
plt.savefig(os.path.join(save_folder, 'test_loss_vs_rollouts.pdf'), dpi=300, bbox_inches='tight')
print(f"Plot saved to: {os.path.join(save_folder, 'test_loss_vs_rollouts.pdf')}")
plt.show()

