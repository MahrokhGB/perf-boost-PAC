# SVGD - 1 particle - tuned nominal_prior_std_scale
# python3 Simulations/perf-boost-PAC/experiments/robots/hyper_param_opt.py --optuna-training-method SVGD --num-rollouts 8 --batch-size 8 --cont-type PerfBoost --epochs 5000 --log-epoch 50 --early-stopping True --nominal-prior True --delta 0.1 --num-particles 1 --lr 5e-4 --random-seed 0
res_SVGD = [
     {
        'num_rollouts':8,
        'nominal_prior_std_scale':60.857980020634635,
        'Bounded train loss':0.0859, 
        'original train loss':22.0320,
        'train num collisions':22,
        'bounded test loss':0.1004, 
        'original test loss':25.79, 
        'test num collisions':26 
    },
     {
        'num_rollouts':16,
        'nominal_prior_std_scale':50.0,
        'Bounded train loss':0.0877, 
        'original train loss':22.4867,
        'train num collisions':22,
        'bounded test loss':0.0953, 
        'original test loss':24.45, 
        'test num collisions':24 
    },
     {
        'num_rollouts':32,
        'nominal_prior_std_scale':39.546884104715836,
        'Bounded train loss':0.0862, 
        'original train loss':22.1045,
        'train num collisions':22,
        'bounded test loss':0.0884, 
        'original test loss':22.67, 
        'test num collisions':23 
    },
    {
        'num_rollouts':64,
        'nominal_prior_std_scale':68.47230140710549,
        'Bounded train loss':0.0852, 
        'original train loss':21.8508,
        'train num collisions':22,
        'bounded test loss':0.0854, 
        'original test loss':21.91, 
        'test num collisions':22 
    },
    {
        'num_rollouts':128,
        'nominal_prior_std_scale':36.47167368646971,
        'Bounded train loss':0.0849, 
        'original train loss':21.7670,
        'train num collisions':22,
        'bounded test loss':0.0851, 
        'original test loss':21.81, 
        'test num collisions':22 
    },
    {
        'num_rollouts':256,
        'nominal_prior_std_scale':37.309927890602495,
        'Bounded train loss': 0.0854, 
        'original train loss':21.8938,
        'train num collisions': 22,
        'bounded test loss': 0.0856, 
        'original test loss': 21.95, 
        'test num collisions': 22
    },
     {
        'num_rollouts':512,
        'nominal_prior_std_scale':41.659096312544804,
        'Bounded train loss':0.0837, 
        'original train loss':21.4585,
        'train num collisions':21,
        'bounded test loss':0.0839, 
        'original test loss':21.51, 
        'test num collisions':22 
    },
     {
        'num_rollouts':1024,
        'nominal_prior_std_scale':52.999635957798034,
        'Bounded train loss':0.0846, 
        'original train loss':21.6844,
        'train num collisions':22,
        'bounded test loss':0.0844, 
        'original test loss':21.64, 
        'test num collisions': 22
    },
     {
        'num_rollouts':2048,
        'nominal_prior_std_scale':50.0,
        'Bounded train loss':0.0839, 
        'original train loss':21.5082,
        'train num collisions':22,
        'bounded test loss':0.0837, 
        'original test loss':21.46, 
        'test num collisions':21 
    },
]

# -----------------------------------------------------
# -----------------------------------------------------
# nominal
# --------- SSM ---------
# python3 Simulations/perf-boost-PAC/experiments/robots/run_emp.py --nominal-exp True --num-rollouts 1 --cont-type PerfBoost --epochs 5000 --log-epoch 50 --n-logs-no-change 10 --nn-type SSM --rmin 0.78 --lr 2e-4 --random-seed 500 

# -----------------------------------------------------
# SVGD - 1 particle - 32 rollouts - using the best nominal_prior_std_scale - change seed
# --------- REN ---------
# python3 Simulations/perf-boost-PAC/experiments/robots/run_SVGD.py --num-rollouts 32 --batch-size 32 --cont-type PerfBoost --epochs 5000 --log-epoch 50 --early-stopping True --nominal-prior True --delta 0.1 --num-particles 1 --lr 5e-4 --nominal-prior-std-scale 39.546884104715836 --random-seed 0
# seeds: 500, 0, 5, 412, 719
# run times are not trustable, b.c. GPU was usually used 100%
res_SVGD_REN_32_rollouts = {
    'num_rollouts': 32,
    'Bounded train loss': [0.0842, 0.0862, 0.0859, 0.0834, 0.0856],
    'original train loss': [21.5872, 22.1045, 22.0370, 21.3935, 21.9623],
    'train num collisions': [22, 22, 22, 21, 22],
    'bounded test loss': [0.0870, 0.0884, 0.0891, 0.0860, 0.0901],
    'original test loss': [22.31, 22.67, 22.86, 22.07, 23.12],
    'test num collisions': [22, 23, 23, 22, 23],
    'training time (s)': [3306.53, 1579.64, 3215.69, 4180.51, 4456.48]
}
# --------- SSM ---------
# python3 Simulations/perf-boost-PAC/experiments/robots/run_SVGD.py --num-rollouts 32 --batch-size 32 --cont-type PerfBoost --epochs 5000 --log-epoch 50 --early-stopping True --nominal-prior True --delta 0.1 --num-particles 1 --lr 5e-4 --nominal-prior-std-scale 39.546884104715836 --nn-type SSM --rmin 0.7 --random-seed 0
res_SVGD_SSM_32_rollouts = {}

# -----------------------------------------------------
# empirical - 32 rollouts
# --------- REN ---------
# python3 Simulations/perf-boost-PAC/experiments/robots/run_emp.py --num-rollouts 32 --batch-size 32 --cont-type PerfBoost --nn-type REN --epochs 5000 --log-epoch 50 --early-stopping True --lr 5e-4 --random-seed 500
res_emp_REN_32_rollouts = {
    'num_rollouts': 32,
    'Bounded train loss':[0.0843, 0.0823, 0.0857, 0.0825, 0.0983],
    'original train loss':[21.6213, 21.4655, 21.9823, 21.1467, 25.2184],
    'train num collisions':[2, 0, 0, 0, 0], 
    'bounded test loss':[0.0877, 0.0841, 0.0920, 0.0857, 0.1110],
    'original test loss':[22.5028, 22.1920, 23.6150, 21.9774, 28.5421],
    'test num collisions':[43, 11, 166, 14, 425],
    'training time (s)': [2765, 3845, 2959, 4314,1533]
}
# --------- SSM ---------
# tuned lr and rmin = 2e-4, 0.87
# python3 Simulations/perf-boost-PAC/experiments/robots/run_emp.py --num-rollouts 32 --batch-size 32 --cont-type PerfBoost --nn-type SSM --epochs 5000 --log-epoch 50 --early-stopping True --lr 2e-4 --78 --random-seed 500
res_emp_SSM_32_rollouts = {
    'num_rollouts': 32,
    'Bounded train loss': [0.0880, 0.0866, 0.0887, 0.2258, 0.1773],
    'original train loss': [22.5642, 22.2063, 22.7617, 59.1617, 46.8513],
    'train num collisions': [1, 0, 1, 254, 163],
    'bounded test loss': [0.0927, 0.0896, 0.0922, 0.2883, 0.2574],
    'original test loss': [23.7970, 22.9972, 23.6498, 84.4291, 92.5614],
    'test num collisions': [87, 88, 153, 4153, 3775],
    # 'training time (s)': [4185, 4146, 7964]  
}

# small SSM with ~600 params
# res_emp_SSM_32_rollouts = {
#     'num_rollouts': 32,
#     'Bounded train loss': [0.0846, 0.0873, 0.0866, 0.0830, 0.0868],
#     'original train loss': [21.6978, 22.3925, 22.2209, 21.2816, 22.2589],
#     'train num collisions': [0, 0, 0, 0, 0], 
#     'bounded test loss': [0.0884, 0.0917, 0.0933, 0.0853, 0.0933],
#     'original test loss': [22.6619, 23.5147, 23.9575, 21.8727, 23.9513],
#     'test num collisions': [61, 81, 182, 16, 183]
# }


# tune r_min
# python3 Simulations/perf-boost-PAC/experiments/robots/hyper_param_opt.py --optuna-training-method empirical --num-rollouts 32 --batch-size 32 --cont-type PerfBoost --nn-type SSM --epochs 5000 --log-epoch 50 --early-stopping True --lr 5e-4 --random-seed 500

# -----------------------------------------------------
# normflow - 32 rollouts
# --------- REN ---------
# python3 Simulations/perf-boost-PAC/experiments/robots/run_normflow.py --num-rollouts 32 --batch-size 32 --cont-type PerfBoost --epochs 5000 --log-epoch 50 --lr 5e-4 --base-is-prior True --nominal-prior True --nominal-prior-std-scale 58.93082020462471 --flow-activation tanh --delta 0.1 --random-seed 500 
res_normflow_REN_32_rollouts = {
    'num_rollouts': 32,
    'Bounded train loss':[0.0835, 0.0823, 0.0874, 0.0833, 0.0865],
    'original train loss': None,
    'train num collisions':[0, 0, 0, 0, 0],
    'bounded test loss': [0.0849, 0.0841, 0.0907, 0.0845, 0.0879],
    'original test loss':None,
    'test num collisions':[4, 11, 81, 5, 22],
    'training time (s)': [2958, 6259, 2355, 2990, 1914]
}
# --------- SSM ---------
# python3 Simulations/perf-boost-PAC/experiments/robots/run_normflow.py --num-rollouts 32 --log-epoch 50 --lr 5e-4 --base-is-prior True --nominal-prior True --nominal-prior-std-scale 58.93082020462471 --flow-activation tanh --nn-type SSM --rmin 0.78 --random-seed 500 
res_normflow_SSM_32_rollouts = {}

# -----------------------------------------------------
# -----------------------------------------------------
# -----------------------------------------------------
# norm flow
# use lambda star and tune nominal prior std for best performance, then grid search over N_p for tightest bound 
# python3 Simulations/perf-boost-PAC/experiments/robots/hyper_param_opt.py --optuna-training-method normflow --nn-type REN --log-epoch 50 --nominal-prior True --base-is-prior True --flow-activation tanh --lr 5e-4 --num-rollouts 2048 
res_normflow = [
    {
        'num_rollouts':8,
        'nominal_prior_std_scale': 473.1147454034813,
        'Bounded train loss': 0.0838, 
        'original train loss': None,
        'train num collisions': 0,
        'bounded test loss': 0.0909, 
        'original test loss': None, 
        'test num collisions': 58
    },
    {
        'num_rollouts':16,
        'nominal_prior_std_scale': 420.88108056283505,
        'Bounded train loss': 0.0801, 
        'original train loss': None,
        'train num collisions': 0,
        'bounded test loss': 0.0854, 
        'original test loss': None, 
        'test num collisions': 4
    },
    {
        'num_rollouts':32,
        'nominal_prior_std_scale': 58.93082020462471,
        'Bounded train loss': 0.0823, 
        'original train loss': None,
        'train num collisions': 0,
        'bounded test loss': 0.0841, 
        'original test loss': None, 
        'test num collisions': 11
    },
    {
        'num_rollouts':64,
        'nominal_prior_std_scale': 55.299916543094554,
        'Bounded train loss': 0.0840, 
        'original train loss': None,
        'train num collisions': 0,
        'bounded test loss': 0.0856, 
        'original test loss': None, 
        'test num collisions': 40
    },
    {
        'num_rollouts':128,
        'nominal_prior_std_scale': 80.12092549898787,
        'Bounded train loss': 0.0823, 
        'original train loss': None,
        'train num collisions': 0,
        'bounded test loss': 0.0822, 
        'original test loss': None, 
        'test num collisions': 0
    },
    {
        'num_rollouts':256,
        'nominal_prior_std_scale': 170.39857841504897,
        'Bounded train loss': 0.0830, 
        'original train loss': None,
        'train num collisions': 2,
        'bounded test loss': 0.0835, 
        'original test loss': None, 
        'test num collisions': 10
    },
    {
        'num_rollouts':512,
        'nominal_prior_std_scale': 99.42019860940434,
        'Bounded train loss': 0.0825, 
        'original train loss': None,
        'train num collisions': 1,
        'bounded test loss': 0.0827, 
        'original test loss': None, 
        'test num collisions': 3  
    },
    {
        'num_rollouts':1024,
        'nominal_prior_std_scale': 5.657081525352462,
        'Bounded train loss': 0.0864, 
        'original train loss': None,
        'train num collisions': 41,
        'bounded test loss': 0.0861, 
        'original test loss': None, 
        'test num collisions': 35
    },
    {
        'num_rollouts':2048,
        'nominal_prior_std_scale': 284.5670370215786,
        'Bounded train loss': None, 
        'original train loss': None,
        'train num collisions': None,
        'bounded test loss': 0.0826, 
        'original test loss': None, 
        'test num collisions': 3
    },
]



# -----------------------------------------------------
# -----------------------------------------------------
# -----------------------------------------------------
# norm flow
# tune lambda, nominal prior std, and N_p for tightest bound




# upper bound
# REN. select Gibbs lambda to have a small constant in the upper bound, then tune nominal prior std and N_p for tightest bound
# python3 Simulations/perf-boost-PAC/experiments/robots/ub_ub/twostep.py --num-rollouts 32 --batch-size 32 --cont-type PerfBoost --epochs 5000 --log-epoch 50 --lr 5e-4 --base-is-prior True --nominal-prior True --nominal-prior-std-scale 58.93082020462471 --flow-activation tanh --delta 0.1 --nn-type REN --random-seed 500 --gibbs-lambda 13.2455










import matplotlib.pyplot as plt, sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
print('BASE_DIR', BASE_DIR)
from conf_int_calculate import get_conf_interval

if __name__=='__main__':
    
    method = 'SVGD'  # Change to 'normflow' if you want to plot normflow results
    if method == 'SVGD':
        data = res_SVGD
    elif method == 'normflow':
        data = res_normflow 
    # elif method == 'emp':
    #     data = res_emp
    else:
        raise ValueError("Unknown method. Use 'emp', 'SVGD' or 'normflow'.")
    

    # Extract data for plotting
    num_rollouts = [entry['num_rollouts'] for entry in data]
    bounded_train_loss = [entry['Bounded train loss'] for entry in data]
    bounded_test_loss = [entry['bounded test loss'] for entry in data]

    # Create the plot
    fig, axs = plt.subplots()

    axs.plot(num_rollouts, bounded_train_loss, label='Bounded Train Loss', marker='o')
    axs.plot(num_rollouts, bounded_test_loss, label='Bounded Test Loss', marker='s')

    # Add labels, title, and legend
    axs.set_xlabel('Number of Rollouts')
    axs.set_ylabel('Loss')
    axs.set_title('Loss vs. Number of Rollouts')
    axs.legend()

    # Show the plot
    plt.show()



    # plot the confidence intervals for SVGD with 32 rollouts
    methods = ['emp']#['SVGD', 'normflow', 'emp']
    nn_type = 'SSM'
    for method in methods:
        if method == 'SVGD':
            res_32 = res_SVGD_REN_32_rollouts if nn_type == 'REN' else res_SVGD_SSM_32_rollouts
        elif method == 'normflow':
            res_32 = res_normflow_REN_32_rollouts if nn_type == 'REN' else res_normflow_SSM_32_rollouts
        elif method == 'emp':
            res_32 = res_emp_REN_32_rollouts if nn_type == 'REN' else res_emp_SSM_32_rollouts

        print("\n" + "-"*65)
        print(f"\nConfidence Intervals for {method} with 32 Rollouts:")
        print("\nBounded Train Loss Confidence Interval:")
        print(res_32['Bounded train loss'])
        get_conf_interval(res_32['Bounded train loss'], confidence=95, n_digits=4)
        
        if not res_32['original train loss'] is None:
            print("\nOriginal Train Loss Confidence Interval:")
            get_conf_interval(res_32['original train loss'], confidence=95, n_digits=4)

        print("\nTrain Collisions Percentage Confidence Interval:")
        get_conf_interval(
            [i/500*100 for i in res_32['train num collisions']], 
            confidence=95, n_digits=4
        )

        print("\nBounded Test Loss Confidence Interval:")
        get_conf_interval(res_32['bounded test loss'], confidence=95, n_digits=4)

        if not res_32['original test loss'] is None:
            print("\nOriginal Test Loss Confidence Interval:")
            get_conf_interval(res_32['original test loss'], confidence=95, n_digits=4)

        print("\nTest Collisions Percentage Confidence Interval:")
        get_conf_interval(
            [i/500*100 for i in res_32['test num collisions']], 
            confidence=95, n_digits=4
        )
