import sys, os, logging, torch, time
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from config import device
from inference_algs.normflow_assist import eval_norm_flow
from arg_parser import argument_parser, print_args
from plants import RobotsSystem, RobotsDataset
from utils.plot_functions import *
from controllers import PerfBoostController, AffineController, NNController
from loss_functions import RobotsLossMultiBatch
from utils.assistive_functions import WrapLogger
from inference_algs.distributions import GibbsPosterior

# NEW
import math
from tqdm import tqdm
from inference_algs.normflow_assist.mynf import NormalizingFlow
import normflows as nf
from inference_algs.normflow_assist import GibbsWrapperNF

from config import device

TRAIN_METHOD = 'normflow'
cont_type = 'PerfBoost'

# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'saved_results')
save_folder = os.path.join(save_path, TRAIN_METHOD, cont_type+'_'+now)

# load trained nfm
filename_load = os.path.join(save_path, 'normflow', 'PerfBoost_10_11_14_28_09', 'final_nfm')
res_dict_loaded = torch.load(filename_load, map_location=torch.device('cpu'))
print(res_dict_loaded.keys())