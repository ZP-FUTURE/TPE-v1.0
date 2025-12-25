import os
import glob
import re
import pandas as pd
import numpy as np 
import torch
from torch.utils.data import Dataset

class PendulumDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not self.files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
            
        print(f"Dataset found {len(self.files)} files. Parsing regex...")
        self.regex_k1 = re.compile(r"K1=(\d+\.\d+)")
        self.regex_k2 = re.compile(r"K2=(\d+\.\d+)")
        self.regex_L  = re.compile(r"L=(\d+\.\d+)")
        self.regex_m  = re.compile(r"m=(\d+\.\d+)")
        self.regex_n  = re.compile(r"N=(\d+\.\d+)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        filename = os.path.basename(filepath)
        k1, k2, L, m, n_val = 0.0, 0.0, 1.0, 1.0, 0.0
        
        try:
            if match := self.regex_k1.search(filename): k1 = float(match.group(1))
            if match := self.regex_k2.search(filename): k2 = float(match.group(1))
            if match := self.regex_L.search(filename):  L  = float(match.group(1))
            if match := self.regex_m.search(filename):  m  = float(match.group(1))
            if match := self.regex_n.search(filename):  n_val = float(match.group(1))
        except Exception as e:
            print(f"Warning: Regex extraction failed for {filename}: {e}")
        label = torch.tensor([k1, k2, L, m, n_val], dtype=torch.float32)
        try:
            df = pd.read_csv(filepath)
            if 'Angle_rad' in df.columns:
                seq = df['Angle_rad'].values
            else:
                seq = df.iloc[:, 1].values
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            seq = np.zeros(400, dtype=np.float32) 
        seq_tensor = torch.tensor(seq.copy(), dtype=torch.float32).unsqueeze(-1)
        
        return seq_tensor, label