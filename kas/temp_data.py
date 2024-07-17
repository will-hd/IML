import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class Temperatures(Dataset):
    def __init__(self, split='train', root='data/temperatures', knowledge_type='desc'):
        region = 'AK'
        self.data = pd.read_csv(f'{root}/data.csv')
        self.splits = pd.read_csv(f'{root}/train_test_val_splits.csv')
        if knowledge_type == 'desc':
            self.knowledge_df = pd.read_csv(f'{root}/gpt_temperature_descriptions.csv')    
        else:
            self.knowledge_df = pd.read_csv(f'{root}/2021-2022_{region}_knowledge.csv')
        
        
        self.knowledge_type = knowledge_type
        if knowledge_type == 'min_max':
            self.knowledge_input_dim = 2
        elif knowledge_type == 'min_max_month':
            self.knowledge_input_dim = 3
        elif knowledge_type == 'desc':
            self.knowledge_input_dim = None
        else:
            raise NotImplementedError

        self.split = split
        if self.split == 'train':
            dates = self.splits[self.splits.split == 'train'].LST_DATE
        elif self.split == 'val' or self.split == 'valid':
            dates = self.splits[self.splits.split == 'val'].LST_DATE
        elif self.split == 'test':
            dates = self.splits[self.splits.split == 'test'].LST_DATE
        
        self.data = self.data[self.data.LST_DATE.isin(dates)]
        #self.data = self.data.drop('LST_DATE', axis=1)
        #self.knowledge_df = self.knowledge_df[self.knowledge_df.LST_DATE.isin(dates)]
        #self.knowledge_df = self.knowledge_df.drop('LST_DATE', axis=1)

        self.dim_x = 1
        self.dim_y = 1

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        y = self.data.iloc[idx, 1:].values
        lst_date = self.data.iloc[idx, 0]   
        k_idx = self.knowledge_df[self.knowledge_df.LST_DATE == lst_date].index[0]
        x = np.linspace(-2, 2, len(y))
        if self.knowledge_type == 'min_max':
            knowledge = self.knowledge_df[['min', 'max']].iloc[k_idx, :].values
            knowledge = torch.tensor(knowledge, dtype=torch.float32)
        elif self.knowledge_type == 'min_max_month':
            knowledge = self.knowledge_df[['min', 'max', 'month']].iloc[k_idx, :].values
            knowledge = torch.tensor(knowledge, dtype=torch.float32)
        elif self.knowledge_type == 'desc':
            knowledge = self.knowledge_df.iloc[k_idx, :].description
        else:
            raise NotImplementedError
        

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        return x, y, knowledge