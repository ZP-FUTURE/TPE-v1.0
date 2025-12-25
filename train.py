import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

from models.transformer import PendulumTransformer
from data.dataset import PendulumDataset
from utils.general import set_seed, increment_path, get_logger, colorstr, plot_training_metrics

def load_config(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train(cfg_path="config/params.yaml"):
    cfg = load_config(cfg_path)
    train_cfg = cfg['train']
    gen_cfg = cfg['generation']
    model_cfg = cfg['model']

    set_seed(42)

    save_dir = increment_path(Path("runs/train") / "exp", mkdir=True)
    weights_dir = Path(save_dir) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(os.path.join(save_dir, 'train.log'))
    logger.info(colorstr('green', 'bold', f'🚀 Start Training...'))
    logger.info(f"📁 Save directory: {save_dir}")

    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else "cpu")
    logger.info(f"🔌 Using device: {device}")
    history = {
        'train/loss': [],
        'train/k1_mse': [], 'train/k2_mse': [], 
        'train/L_mse': [],  'train/m_mse': [], 'train/n_mse': [],
        'val/loss': [],
        'val/k1_mse': [],   'val/k2_mse': [], 
        'val/L_mse': [],    'val/m_mse': [],   'val/n_mse': [],
        'lr': []
    }

    data_dir = gen_cfg['save_dir']
    try:
        dataset = PendulumDataset(data_dir)
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return

    val_size = int(len(dataset) * train_cfg['val_split'])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    logger.info(f"📊 Data: Total {len(dataset)} | Train {train_size} | Val {val_size}")

    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    if model_cfg['output_dim'] != 5:
        logger.warning(f"⚠️ Config output_dim is {model_cfg['output_dim']}, but expected 5 for [k1,k2,L,m,n]. Overriding to 5.")
        model_cfg['output_dim'] = 5

    model = PendulumTransformer(model_cfg).to(device)
    logger.info(f"🧠 Model Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=train_cfg['lr']*10, 
                                              steps_per_epoch=len(train_loader), epochs=train_cfg['epochs'])

    best_loss = float('inf')
    
    for epoch in range(train_cfg['epochs']):

        model.train()

        m_log = {k: 0.0 for k in ['loss', 'k1', 'k2', 'L', 'm', 'n']}
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{train_cfg['epochs']}", unit="b")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) 
            
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                # targets顺序: k1, k2, L, m, n
                m_log['loss'] += loss.item()
                m_log['k1'] += nn.functional.mse_loss(outputs[:,0], targets[:,0]).item()
                m_log['k2'] += nn.functional.mse_loss(outputs[:,1], targets[:,1]).item()
                m_log['L']  += nn.functional.mse_loss(outputs[:,2], targets[:,2]).item()
                m_log['m']  += nn.functional.mse_loss(outputs[:,3], targets[:,3]).item()
                m_log['n']  += nn.functional.mse_loss(outputs[:,4], targets[:,4]).item()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        steps = len(train_loader)
        history['train/loss'].append(m_log['loss']/steps)
        history['train/k1_mse'].append(m_log['k1']/steps)
        history['train/k2_mse'].append(m_log['k2']/steps)
        history['train/L_mse'].append(m_log['L']/steps)
        history['train/m_mse'].append(m_log['m']/steps)
        history['train/n_mse'].append(m_log['n']/steps)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        model.eval()
        v_log = {k: 0.0 for k in ['loss', 'k1', 'k2', 'L', 'm', 'n']}
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                v_log['loss'] += loss.item()
                v_log['k1'] += nn.functional.mse_loss(outputs[:,0], targets[:,0]).item()
                v_log['k2'] += nn.functional.mse_loss(outputs[:,1], targets[:,1]).item()
                v_log['L']  += nn.functional.mse_loss(outputs[:,2], targets[:,2]).item()
                v_log['m']  += nn.functional.mse_loss(outputs[:,3], targets[:,3]).item()
                v_log['n']  += nn.functional.mse_loss(outputs[:,4], targets[:,4]).item()
        val_steps = len(val_loader)
        avg_val_loss = v_log['loss'] / val_steps
        
        history['val/loss'].append(avg_val_loss)
        history['val/k1_mse'].append(v_log['k1']/val_steps)
        history['val/k2_mse'].append(v_log['k2']/val_steps)
        history['val/L_mse'].append(v_log['L']/val_steps)
        history['val/m_mse'].append(v_log['m']/val_steps)
        history['val/n_mse'].append(v_log['n']/val_steps)

        logger.info(f"Ep {epoch+1} Val | Loss: {avg_val_loss:.5f} | "
                    f"K1: {history['val/k1_mse'][-1]:.4f} | "
                    f"L: {history['val/L_mse'][-1]:.4f} | "
                    f"m: {history['val/m_mse'][-1]:.4f}")
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), weights_dir / "best.pt")
        
        torch.save(model.state_dict(), weights_dir / "last.pt")
    plot_training_metrics(history, save_dir)