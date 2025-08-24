# # python3 experiments/robots/plot_ub.py --nn-type REN

import pickle, sys, os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, BASE_DIR)

# from utils.assistive_functions import WrapLogger
# from arg_parser import argument_parser
# from experiments.robots.run_normflow import train_normflow

# # norm flow
# # use lambda star and tune nominal prior std for best performance, then grid search over N_p for tightest bound 
# # python3 Simulations/perf-boost-PAC/experiments/robots/hyper_param_opt.py --optuna-training-method normflow --num-rollouts 2048 --batch-size 256 --cont-type PerfBoost --epochs 5000 --log-epoch 50 --early-stopping True --nominal-prior True --base-is-prior True --flow-activation tanh --delta 0.1 --lr 5e-4 --random-seed 0
# res_normflow = [
#     {
#         'num_rollouts':8,
#         'nominal_prior_std_scale': 473.1147454034813,
#         'Bounded train loss': 0.0838, 
#         'original train loss': None,
#         'train num collisions': 0,
#         'bounded test loss': 0.0909, 
#         'original test loss': None, 
#         'test num collisions': 58
#     },
#     {
#         'num_rollouts':16,
#         'nominal_prior_std_scale': 420.88108056283505,
#         'Bounded train loss': 0.0801, 
#         'original train loss': None,
#         'train num collisions': 0,
#         'bounded test loss': 0.0854, 
#         'original test loss': None, 
#         'test num collisions': 4
#     },
#     {
#         'num_rollouts':32,
#         'nominal_prior_std_scale': 58.93082020462471,
#         'Bounded train loss': 0.0823, 
#         'original train loss': None,
#         'train num collisions': 0,
#         'bounded test loss': 0.0841, 
#         'original test loss': None, 
#         'test num collisions': 11
#     },
#     {
#         'num_rollouts':64,
#         'nominal_prior_std_scale': 55.299916543094554,
#         'Bounded train loss': 0.0840, 
#         'original train loss': None,
#         'train num collisions': 0,
#         'bounded test loss': 0.0856, 
#         'original test loss': None, 
#         'test num collisions': 40
#     },
#     {
#         'num_rollouts':128,
#         'nominal_prior_std_scale': 80.12092549898787,
#         'Bounded train loss': 0.0823, 
#         'original train loss': None,
#         'train num collisions': 0,
#         'bounded test loss': 0.0822, 
#         'original test loss': None, 
#         'test num collisions': 0
#     },
#     {
#         'num_rollouts':256,
#         'nominal_prior_std_scale': 170.39857841504897,
#         'Bounded train loss': 0.0830, 
#         'original train loss': None,
#         'train num collisions': 2,
#         'bounded test loss': 0.0835, 
#         'original test loss': None, 
#         'test num collisions': 10
#     },
#     {
#         'num_rollouts':512,
#         'nominal_prior_std_scale': 99.42019860940434,
#         'Bounded train loss': 0.0825, 
#         'original train loss': None,
#         'train num collisions': 1,
#         'bounded test loss': 0.0827, 
#         'original test loss': None, 
#         'test num collisions': 3  
#     },
#     {
#         'num_rollouts':1024,
#         'nominal_prior_std_scale': 5.657081525352462,
#         'Bounded train loss': 0.0864, 
#         'original train loss': None,
#         'train num collisions': 41,
#         'bounded test loss': 0.0861, 
#         'original test loss': None, 
#         'test num collisions': 35
#     },
#     {
#         'num_rollouts':2048,
#         'nominal_prior_std_scale': 284.5670370215786,
#         'Bounded train loss': None, 
#         'original train loss': None,
#         'train num collisions': None,
#         'bounded test loss': 0.0826, 
#         'original test loss': None, 
#         'test num collisions': 3
#     },
# ]


# # ----- default experiment arguments -----
# args = argument_parser()
# args.random_seed = 0
# args.batch_size = None # will be set later
# args.nn_type = 'REN'
# args.nominal_prior = True
# args.base_is_prior = True
# args.flow_activation = 'tanh'
# args.nominal_prior_std_scale = None # will be set later
# args.cont_type = 'PerfBoost'
# args.lr = 5e-4
# args.delta = 0.1
# args.log_epoch = 50
# args.epochs = 5000

# save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
# setup_name ='internal' + str(args.dim_internal)
# if args.nn_type == 'REN':
#     setup_name += '_nl' + str(args.dim_nl)
# elif args.nn_type == 'SSM':
#     setup_name += '_middle' + str(args.dim_middle) + '_scaffolding' + str(args.dim_scaffolding)

# res = res_normflow[int(math.log(args.num_rollouts/8, 2))]  
# args.batch_size = min(res['num_rollouts'], 256)
# args.nominal_prior_std_scale = res['nominal_prior_std_scale']
#     # ----- SET UP LOGGER -----
# save_folder = os.path.join(save_path, 'normflow', args.nn_type, setup_name, args.cont_type+'_'+str(res['num_rollouts']))
# os.makedirs(save_folder)
# logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
# logger = logging.getLogger('ren_controller_')
# logger.setLevel(logging.DEBUG)
# logger = WrapLogger(logger)
# logger.info(f"Num rollouts: {res['num_rollouts']}, Bounded test loss: {res['bounded test loss']}, Test num collisions: {res['test num collisions']}")

# # train 
# res_dict, filename_save, nfm = train_normflow(args, logger, save_folder)

ub = [
    {'num_rollouts': 4, 'ub': 0.8127, 'tot_const':None},
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

load_df = False

if not load_df:
    df = {
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
            df['number of training rollouts'].append(res['num_rollouts'])
            df['bounded train loss'].append(train_loss_value)
            df['bounded test loss'].append(test_loss_value)
            df['ub'].append(next((item['ub'] for item in ub if item['num_rollouts'] == res['num_rollouts']), None))
    df = pd.DataFrame(df)
    with open(os.path.join(save_folder, 'plot_data.pkl'), 'wb') as f:
        pickle.dump(df, f)
else:
    # Load DataFrame from pickle file
    df_filename = os.path.join(save_folder, 'plot_data.pkl')
    with open(df_filename, 'rb') as f:
        df = pickle.load(f)
    print(f"DataFrame loaded from: {df_filename}")


# # 
# for ind in range(len(plot_data)):
#     if plot_data[ind]['num_rollouts'] in [1024, 2048]:
#         plot_data[ind]['test_loss'] *= 0.95  # Apply scaling factor




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

# only used to create a proper FacetGrid
g = sns.catplot(
    data=df, x='number of training rollouts', y='ub', #col='disturbance type',
    # hue='setup', 
    kind='box', height=4, aspect=1.5,
    sharey=False, palette='hls', legend=False
)

# mark upper bounds
g.map_dataframe(
    sns.boxenplot, x='number of training rollouts', y='ub', 
    linewidth=2, linecolor='k', alpha = 0.6,
    # hue='setup', 
    legend=False, # dodge=True, 
)

# mark sample-based upper bounds
# g.map_dataframe(
#     sns.boxenplot, x='number of training rollouts', y='ub_sb', 
#     linewidth=2, linecolor='blue', alpha = 0.6,
#     hue='setup', legend=False, # dodge=True, 
# )

# mark sampled controllers performance
g.map_dataframe(
    sns.stripplot, x='number of training rollouts', y='bounded test loss', 
    hue='setup', palette='hls', alpha=0.9, dodge=True
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
filename = os.path.join(file_path, 'ub.pdf')
plt.savefig(filename)
plt.show()


# -------------------------------
# -------------------------------
# -------------------------------
# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(num_rollouts_list, test_loss_list, alpha=0.6, s=30)

# Draw horizontal lines for upper bounds
for ub_entry in ub:
    num_rollouts = ub_entry['num_rollouts']
    ub_value = ub_entry['ub']
    
    # # Draw a short horizontal line at the ub value
    # line_width = num_rollouts * 0.3  # Adjust line width based on rollouts
    # plt.hlines(ub_value, num_rollouts - line_width, num_rollouts + line_width, 
    #            colors='red', linewidth=3, alpha=0.8)


# Set x-axis to log scale if desired (since rollouts are powers of 2)
plt.xscale('log', base=2)
plt.xticks([8, 16, 32, 64, 128, 256, 512, 1024, 2048], 
           ['8', '16', '32', '64', '128', '256', '512', '1024', '2048'])

plt.xlabel('Number of Rollouts', fontsize=12)
plt.ylabel('Bounded Test Loss', fontsize=12)
plt.title('Test Loss vs Number of Rollouts', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
plt.savefig(os.path.join(save_folder, 'test_loss_vs_rollouts.pdf'), dpi=300, bbox_inches='tight')
plt.show()

# Save plot_data as pickle file
if not load_df:
    plot_data_filename = os.path.join(save_folder, 'plot_data.pkl')
    with open(plot_data_filename, 'wb') as f:
        pickle.dump(plot_data, f)
    print(f"Plot data saved to: {plot_data_filename}")


