import torch, os, pickle
from torch.utils.data import Dataset

from config import BASE_DIR
from utils.assistive_functions import to_tensor

class CostumDataset(Dataset):
    '''
    Generates and saves large train and test datasets.
    Use "get_data" to get a subset of the saved dataset with given sizes.
    '''
    def __init__(self, random_seed, horizon, exp_name, file_name):
        self.random_seed = random_seed
        self.horizon = horizon
        self._data = None
        torch.manual_seed(self.random_seed)

        # file name and path
        file_path = os.path.join(BASE_DIR, 'experiments', exp_name, 'saved_results')
        path_exist = os.path.exists(file_path)
        if not path_exist:
            os.makedirs(file_path)
        self.file_name = os.path.join(file_path, file_name)

    def _generate_data(self, num_samples):
        '''
        Complete the template to extend the "CostumDataset" class
        '''
        data = None
        assert data.shape[0]==num_samples
        return data

    def get_data(self, num_train_samples=8192, num_test_samples=8192):
        '''
        Main function to get train and test datasets. No need to modify.
        '''
        self._load_data()
        # generate data if not enough samples are saved
        if num_train_samples+num_test_samples>self._data['train_data_full'].shape[0]:
            self._generate_data(num_samples=num_train_samples+num_test_samples)
        train_data = self._data['train_data_full'][0:num_train_samples, :, :]
        test_data = self._data['test_data'][0:num_test_samples, :, :]
        return train_data, test_data

    # ---- save and load functions. no need to modify ----
    def _save_data(self):
        '''
        No need to change. Generates and saves train and test datasets.
        '''
        train_data_full = self._generate_data(8192)
        test_data = self._generate_data(8192)
        # save
        filehandler = open(self.file_name, 'wb')
        pickle.dump({'train_data_full': train_data_full.detach().cpu(),
                     'test_data': test_data.detach().cpu()},
                    filehandler)
        filehandler.close()

    def _load_data(self):
        '''
        No need to change.
        Loads the data. If data doesn't exist, saves it.
        '''
        # check if data exists
        if not os.path.isfile(self.file_name):
            self._save_data()
        # load data
        filehandler = open(self.file_name, 'rb')
        self._data = pickle.load(filehandler)
        filehandler.close()
        # convert numpy to tensor
        self._data = to_tensor(self._data)

    def __len__(self):
        return self._data['train_data_full'].shape[0]

    def __getitem__(self, idx):
        return self._data['train_data_full'][idx, :, :]