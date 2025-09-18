import os
import torch

def define_prior(args, training_param_names, save_path, logger):
    """
    Define the prior distribution for the controller parameters based on the arguments provided.
    This function sets up the prior distribution for the SVGD algorithm based on whether
    data-dependent priors or nominal controllers are used.
    
    Args:
        args: Argument parser containing various settings.
        training_param_names: List of parameter names to define priors for.
        save_path: Path to save the prior information.
        logger: Logger object for logging information.

    Returns:
        prior_dict: Dictionary containing prior distribution parameters.
    """

    # Initialize prior dictionary
    prior_dict = {}
    if args.cont_type in ['Affine', 'NN']:
        training_param_names = ['weight', 'bias']
        prior_dict = {
            'type':'Gaussian', 'type_w':'Gaussian',
            'type_b':'Gaussian_biased',
            'weight_loc':0, 'weight_scale':1,
            'bias_loc':0, 'bias_scale':5,
        }
    else:
        if args.data_dep_prior:
            if args.dim_nl==8 and args.dim_internal==8:
                if args.num_rollouts_prior==5:
                    filename_load = os.path.join(save_path, 'empirical', 'pretrained', 'trained_controller.pt')
                    res_dict_loaded = torch.load(filename_load)
        if args.nominal_prior:
            res_dict_loaded = []
            # define setup name
            setup_name ='internal' + str(args.dim_internal)
            if args.nn_type == 'REN':
                setup_name += '_nl' + str(args.dim_nl)
            elif args.nn_type == 'SSM':
                setup_name += '_middle' + str(args.dim_middle) + '_scaffolding' + str(args.dim_scaffolding)
            # load nominal controllers
            for _, dirs, _ in os.walk(os.path.join(save_path, 'nominal', args.nn_type, setup_name)):
                for dir in dirs:
                    filename_load = os.path.join(save_path, 'nominal', args.nn_type, setup_name, dir, 'trained_controller.pt')
                    if os.path.isfile(filename_load):
                        tmp_dict = torch.load(filename_load)
                        if args.nn_type=='SSM':
                            all_keys = list(tmp_dict.keys())
                            # remove emme from the beginning of dict keys
                            for key in all_keys:
                                tmp_dict[key[5:]] = tmp_dict.pop(key)
                        res_dict_loaded.append(tmp_dict)
            # check if any nominal controllers were loaded 
            if len(res_dict_loaded) == 0:
                raise ValueError("No nominal controllers found in the specified directory, "+str(os.path.join(save_path, 'nominal', args.nn_type, setup_name)))
            logger.info('[INFO] Loaded '+str(len(res_dict_loaded))+' nominal controllers.')
        prior_dict = {'type':'Gaussian'} 
        
        for name in training_param_names:
            if args.data_dep_prior:
                prior_dict[name+'_loc'] = res_dict_loaded[name]
                prior_dict[name+'_scale'] = args.prior_std
            elif args.nominal_prior:
                logger.info('[INFO] Prior distribution is the distribution over nominal controllers, with std scaled by %.4f.' % args.nominal_prior_std_scale)
                if args.nn_type=='REN':
                    if list(res_dict_loaded[0].keys())[0].startswith('emme.'):   # for compatibility with old saved models. the if condition should be removed in future versions.
                        vals = torch.stack([res['emme.'+name] for res in res_dict_loaded], dim=0)
                else:
                    vals = torch.stack([res[name] for res in res_dict_loaded], dim=0)
                # val and std computed elementwise. same shape as the training param
                prior_dict[name+'_loc'] = vals.mean(dim=0)  
                prior_dict[name+'_scale'] = vals.std(dim=0, correction=1) * args.nominal_prior_std_scale
            else:
                prior_dict[name+'_loc'] = 0
                prior_dict[name+'_scale'] = args.prior_std
    return prior_dict