import numpy as np

def get_conf_interval(array, confidence=95, n_digits=2, format='plus_minus'):
    array = np.array(array).flatten()
    num_samples = len(array)
    sample_mean = np.mean(array)
    sample_var = ((array - sample_mean)**2).sum() / (num_samples - 1)
    sample_std = sample_var**0.5
    if confidence==95:
        z = 1.96
    elif confidence==99:
        z = 2.576
    else:
        raise NotImplementedError
    if format=='plus_minus':
        print('Confidence interval: %f +/- %f' % (
            np.around(sample_mean, decimals=n_digits), 
            np.around(z*sample_std/num_samples**0.5, decimals=n_digits)
        ))
    else:
        l_bound = sample_mean - z*sample_std/num_samples**0.5
        u_bound = sample_mean + z*sample_std/num_samples**0.5
        l_bound = np.around(l_bound, decimals=n_digits)
        u_bound = np.around(u_bound, decimals=n_digits)
        print('Confidence interval: [%f, %f]' % (l_bound, u_bound))

# Example
array = np.array([12, 146, 19, 6, 42])
get_conf_interval(array, confidence=95)
