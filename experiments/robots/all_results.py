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
# SVGD - 1 particle - 32 rollouts - using the best nominal_prior_std_scale - change seed
# --------- REN ---------
# python3 Simulations/perf-boost-PAC/experiments/robots/run_SVGD.py --num-rollouts 32 --batch-size 32 --cont-type PerfBoost --epochs 5000 --log-epoch 50 --early-stopping True --nominal-prior True --delta 0.1 --num-particles 1 --lr 5e-4 --nominal-prior-std-scale 39.546884104715836 --random-seed 0
# seeds: 500, 0, 5, 412, 719
# run times are not trustable, b.c. GPU was usually used 100%
res_SVGD_32_rollouts = {
    'num_rollouts': 32,
    'Bounded train loss': [0.0842, 0.0862, 0.0859, 0.0834, 0.0856],
    'original train loss': [21.5872, 22.1045, 22.0370, 21.3935, 21.9623],
    'train num collisions': [22, 22, 22, 21, 22],
    'bounded test loss': [0.0870, 0.0884, 0.0891, 0.0860, 0.0901],
    'original test loss': [22.31, 22.67, 22.86, 22.07, 23.12],
    'test num collisions': [22, 23, 23, 22, 23]
}
# --------- SSM ---------
# python3 Simulations/perf-boost-PAC/experiments/robots/run_SVGD.py --num-rollouts 32 --batch-size 32 --cont-type PerfBoost --epochs 5000 --log-epoch 50 --early-stopping True --nominal-prior True --delta 0.1 --num-particles 1 --lr 5e-4 --nominal-prior-std-scale 39.546884104715836 --nn-type SSM --rmin 0.7 --random-seed 0

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
    'test num collisions':[43, 11, 166, 14, 425]
}
# --------- SSM ---------
# tuned lr and rmin = 2e-4, 0.87
# python3 Simulations/perf-boost-PAC/experiments/robots/run_emp.py --num-rollouts 32 --batch-size 32 --cont-type PerfBoost --nn-type SSM --epochs 5000 --log-epoch 50 --early-stopping True --lr 2e-4 --rmin 0.87 --random-seed 500
res_emp_SSM_32_rollouts = {
    'num_rollouts': 32,
    'Bounded train loss': [0.0840, 0.1259, 0.0901, 0.0897, 0.0909],
    'original train loss': [21.5434, 32.4463, 23.1136, 23.0180, 23.3257],
    'train num collisions': [0, 47, 0, 1, 0], 
    'bounded test loss': [0.0858, 0.1598, 0.0947, 0.0935, 0.0990],
    'original test loss': [22.0066, 43.1995, 24.2990, 23.9999, 25.4091],
    'test num collisions': [6, 1012, 82, 115, 161]
}

# tune r_min
# python3 Simulations/perf-boost-PAC/experiments/robots/hyper_param_opt.py --optuna-training-method empirical --num-rollouts 32 --batch-size 32 --cont-type PerfBoost --nn-type SSM --epochs 5000 --log-epoch 50 --early-stopping True --lr 5e-4 --random-seed 500

# -----------------------------------------------------
# normflow - 32 rollouts
# --------- REN ---------
# python3 Simulations/perf-boost-PAC/experiments/robots/run_normflow.py --num-rollouts 32 --batch-size 32 --cont-type PerfBoost --epochs 5000 --log-epoch 50 --lr 5e-4 --base-is-prior True --nominal-prior True --nominal-prior-std-scale 58.93082020462471 --flow-activation tanh --delta 0.1 --random-seed 500 
res_normflow_32_rollouts = {
    'num_rollouts': 32,
    'Bounded train loss':[0.0835, 0.0823, 0.0874, 0.0833, 0.0865],
    'original train loss': None,
    'train num collisions':[0, 0, 0, 0, 0],
    'bounded test loss': [0.0849, 0.0841, 0.0907, 0.0845, 0.0879],
    'original test loss':None,
    'test num collisions':[4, 11, 81, 5, 22]
}
# --------- SSM ---------
# python3 Simulations/perf-boost-PAC/experiments/robots/run_normflow.py --num-rollouts 32 --batch-size 32 --cont-type PerfBoost --epochs 5000 --log-epoch 50 --lr 5e-4 --base-is-prior True --nominal-prior True --nominal-prior-std-scale 58.93082020462471 --flow-activation tanh --delta 0.1 --nn-type SSM --random-seed 500 

# -----------------------------------------------------
# -----------------------------------------------------
# -----------------------------------------------------
# norm flow
# use lambda star and tune nominal prior std for best performance, then grid search over N_p for tightest bound 
# python3 Simulations/perf-boost-PAC/experiments/robots/hyper_param_opt.py --optuna-training-method normflow --num-rollouts 2048 --batch-size 256 --cont-type PerfBoost --epochs 5000 --log-epoch 50 --early-stopping True --nominal-prior True --base-is-prior True --flow-activation tanh --delta 0.1 --lr 5e-4 --random-seed 0
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
    methods = ['SVGD', 'normflow', 'emp']
    for method in methods:
        if method == 'SVGD':
            res_32 = res_SVGD_32_rollouts
        elif method == 'normflow':
            res_32 = res_normflow_32_rollouts
        elif method == 'emp':
            res_32 = res_emp_32_rollouts

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
