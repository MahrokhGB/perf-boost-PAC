def get_filename(delta, num_rollouts, act, learn_base, prior_std):

    FILE_NAME = None

    if delta==0.01 and learn_base and act=='tanh' and prior_std==7:
        if num_rollouts==8:
            FILE_NAME = 'PerfBoost_10_21_09_58_37'
        elif num_rollouts==16:
            FILE_NAME = 'PerfBoost_10_21_09_57_40'
        elif num_rollouts==32:
            FILE_NAME = 'PerfBoost_10_21_09_54_20'
        elif num_rollouts==64:
            FILE_NAME = 'PerfBoost_10_21_09_59_53'
        elif num_rollouts==128:
            FILE_NAME = 'PerfBoost_10_21_10_00_19'
        elif num_rollouts==256:
            FILE_NAME = 'PerfBoost_10_21_10_09_31'
        elif num_rollouts==512:
            FILE_NAME = 'PerfBoost_10_21_10_27_43'
        elif num_rollouts==1024:
            FILE_NAME = 'PerfBoost_10_21_12_22_37'
        elif num_rollouts==2048:
            FILE_NAME = 'PerfBoost_10_21_12_23_47'

    if delta==0.01 and learn_base and act=='tanh' and prior_std==3:
        if num_rollouts==256:
            FILE_NAME = 'PerfBoost_11_01_15_12_43'

    if delta==0.01 and learn_base and act=='leaky_relu':
        if prior_std==7:
            if num_rollouts==8:
                FILE_NAME = 'PerfBoost_10_22_14_29_52'
            elif num_rollouts==16:
                FILE_NAME = 'PerfBoost_10_22_14_29_01'
            elif num_rollouts==32:
                FILE_NAME = 'PerfBoost_10_22_13_41_33'
            elif num_rollouts==64:
                FILE_NAME = 'PerfBoost_10_22_13_53_49'
            elif num_rollouts==128:
                FILE_NAME = 'PerfBoost_10_22_13_54_40'
            elif num_rollouts==256:
                FILE_NAME = 'PerfBoost_10_22_14_01_34'
            elif num_rollouts==512:
                FILE_NAME = ''
            elif num_rollouts==1024:
                FILE_NAME = ''
            elif num_rollouts==2048:
                FILE_NAME = 'PerfBoost_10_22_14_49_41'
        elif prior_std==1:
            if num_rollouts==8:
                FILE_NAME = 'PerfBoost_10_31_10_53_43'
            elif num_rollouts==16:
                FILE_NAME = 'PerfBoost_10_31_10_53_17'
            elif num_rollouts==32:
                FILE_NAME = 'PerfBoost_10_31_10_52_40'
            elif num_rollouts==64:
                FILE_NAME = 'PerfBoost_10_31_10_51_43'
            elif num_rollouts==128:
                FILE_NAME = 'PerfBoost_10_31_10_48_53'
            elif num_rollouts==256:
                FILE_NAME = 'PerfBoost_10_30_15_09_02'
            elif num_rollouts==512:
                FILE_NAME = ''
            elif num_rollouts==1024:
                FILE_NAME = ''
            elif num_rollouts==2048:
                FILE_NAME = ''
            # if num_rollouts==256:
            #     FILE_NAME = 'PerfBoost_10_30_15_09_02'
        elif prior_std==3:
            if num_rollouts==256:
                FILE_NAME = 'PerfBoost_10_29_15_22_24'


    if delta==0.01 and (not learn_base) and act=='leaky_relu':
        if num_rollouts==32:
            if prior_std==7:
                FILE_NAME = 'PerfBoost_10_27_14_33_01'


    # ------ delta = 0.1 ------
    if delta==0.1 and learn_base and act=='leaky_relu':
        if prior_std==7:
            if num_rollouts==256:
                FILE_NAME = 'PerfBoost_10_29_13_19_31'

    if FILE_NAME is None:
        print('[Err] File not found for ', delta, num_rollouts, act, learn_base, prior_std)
    return FILE_NAME


# PerfBoost_10_27_13_40_31