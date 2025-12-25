import os
import random
import numpy as np
import torch
import logging
from pathlib import Path
import re
import glob
import matplotlib.pyplot as plt

def set_seed(seed=42, deterministic=True):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"🔒 Random seed set to {seed}")

def get_logger(log_file=None):
    logger = logging.getLogger('PendulumProject')
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        path = Path(f"{path}{sep}{n}{suffix}")
    
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
        
    return str(path)

def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {
        'black': '\033[30m',  'red': '\033[31m', 'green': '\033[32m', 'yellow': '\033[33m',
        'blue': '\033[34m', 'magenta': '\033[35m', 'cyan': '\033[36m', 'white': '\033[37m',
        'bright_black': '\033[90m', 'bright_red': '\033[91m', 'bright_green': '\033[92m',
        'bright_yellow': '\033[93m', 'bright_blue': '\033[94m', 'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m', 'bright_white': '\033[97m',
        'end': '\033[0m', 'bold': '\033[1m', 'underline': '\033[4m'}
    
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def smooth_curve(points, weight=0.6):
    if len(points) == 0:
        return np.array([])
    last = points[0]
    smoothed = []
    for point in points:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def plot_training_metrics(history, save_dir):
    keys = list(history.keys())
    n = len(keys)
    
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten() if n > 1 else [axes]
    
    if n > 0:
        first_key = keys[0]
        epochs = range(1, len(history[first_key]) + 1)
    
        for i, key in enumerate(keys):
            ax = axes[i]
            data = np.array(history[key])
            
            ax.plot(epochs, data, marker='.', markersize=6, linewidth=1.5, 
                    label='results', color='#1f77b4', alpha=0.5)
            
            if len(data) > 5:
                smooth_data = smooth_curve(data, weight=0.8)
                ax.plot(epochs, smooth_data, linestyle='--', linewidth=2, 
                        label='smooth', color='#ff7f0e')
            
            ax.set_title(key, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2)
            
            if i == 0:
                ax.legend()
                
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'results.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 训练图表已保存至: {save_path}")