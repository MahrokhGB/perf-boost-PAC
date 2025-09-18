import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------ plot test cost vs training time
res = [
    {'method':'empirical', 'test cost':0.0921, 'training time':3083, 'NN':'REN'},
    {'method':'SVGD - 1', 'test cost':0.0881, 'training time':3348, 'NN':'REN'},
    {'method':'SVGD - 5', 'test cost':0.0731, 'training time':16680, 'NN':'REN'},
    {'method':'normalizing flows', 'test cost':0.0744, 'training time':5495, 'NN':'REN'},
    {'method':'empirical', 'test cost':0.0904, 'training time':4200, 'NN':'SSM'},
    {'method':'SVGD - 1', 'test cost':0.0882, 'training time':4600, 'NN':'SSM'},
    {'method':'SVGD - 5', 'test cost':0.0727, 'training time':22516, 'NN':'SSM'},
    {'method':'normalizing flows', 'test cost':0.0761, 'training time':6214, 'NN':'SSM'},
]

# ------ format ------
plt.rcParams['text.usetex'] = True
sns.set_theme(
    context='paper', style='whitegrid', palette='bright', 
    font='sans-serif', font_scale=1.8, color_codes=True, rc=None, 
)
sns.set_style({'grid.linestyle': '--'})
mpl.rc('font', family='serif', serif='Times New Roman')

# pastel
# ren_color = '#CCE5FF'
# ssm_color = '#F19C99'
# bright
ren_color = sns.color_palette('hls')[3]
ssm_color = sns.color_palette('hls')[0]

# Create scatter plot for test cost vs training time
fig, ax = plt.subplots(figsize=(8, 6))

# Separate data by NN type
ren_data = [entry for entry in res if entry['NN'] == 'REN']
ssm_data = [entry for entry in res if entry['NN'] == 'SSM']

# Define colors and markers for each method
method_styles = {
    'empirical': {'marker': 'o', 'size': 80},
    'SVGD - 1': {'marker': 's', 'size': 80}, 
    'SVGD - 5': {'marker': '^', 'size': 80},
    'normalizing flows': {'marker': 'D', 'size': 80}
}

# Plot REN data
for entry in ren_data:
    ax.scatter(entry['training time'], entry['test cost'], 
              c=ren_color, 
              marker=method_styles[entry['method']]['marker'],
              s=method_styles[entry['method']]['size'],
              alpha=0.8,
              edgecolors='black',
              linewidth=1)

# Plot SSM data  
for entry in ssm_data:
    ax.scatter(entry['training time'], entry['test cost'],
              c=ssm_color,
              marker=method_styles[entry['method']]['marker'], 
              s=method_styles[entry['method']]['size'],
              alpha=0.9,
              edgecolors='black', 
              linewidth=1)

# add method names as labels
methods = ['empirical', 'SVGD - 1', 'SVGD - 5', 'normalizing flows']

for i, method in enumerate(methods):
    # Get REN and SSM data for this method
    ren_point = next(entry for entry in ren_data if entry['method'] == method)
    ssm_point = next(entry for entry in ssm_data if entry['method'] == method)
    
    # Calculate loc
    max_x = max(ren_point['training time'], ssm_point['training time'])
    center_y = (ren_point['test cost'] + ssm_point['test cost']) / 2

    if not method=='SVGD - 5':
        # Add text label for REN
        ax.text(max_x+2100, center_y, method, ha='center', va='center')
    else:
        ax.text(max_x-2500, center_y, method, ha='center', va='center')


# Create simple legend for NN types only
from matplotlib.lines import Line2D
# Get handles and labels from the stripplot
handles, labels = ax.get_legend_handles_labels()
# Create separate legend entries for REN and SSM bounded test loss
ren_handle = handles[0] if len(handles) > 0 else Line2D([0], [0], marker='o', color='w', markerfacecolor=ren_color, markersize=8, alpha=0.9, linestyle='None')
ssm_handle = handles[1] if len(handles) > 1 else Line2D([0], [0], marker='o', color='w', markerfacecolor=ssm_color, markersize=8, alpha=0.9, linestyle='None')
# Create upper bound legend entry
upper_bound_line = Line2D([0], [0], color='k', lw=3, alpha=0.8)
# Create legend with separate entries for REN and SSM
legend_handles = [ren_handle, ssm_handle, upper_bound_line]
legend_labels = ['REN', 'SSM']

# legend_elements = [
#     Line2D([0], [0], marker='o', color='w', markerfacecolor=ren_color, 
#            markersize=10, label='REN', markeredgecolor='w', markeredgewidth=1),
#     Line2D([0], [0], marker='o', color='w', markerfacecolor=ssm_color,
#            markersize=10, label='SSM', markeredgecolor='w', markeredgewidth=1)
# ]

ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right', frameon=True, fancybox=True, shadow=False)

# Set labels and title
ax.set_xlabel('Training time (seconds)')
ax.set_ylabel('Test cost')

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle='--')

# Adjust layout
plt.tight_layout()

# Save the plot
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
save_folder = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results', 'plots')
filename_scatter = os.path.join(save_folder, 'test_cost_vs_training_time.pdf')
plt.savefig(filename_scatter, dpi=300, bbox_inches='tight')
plt.show()
plt.close()