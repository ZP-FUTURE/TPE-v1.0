import os
import numpy as np
import pandas as pd
import yaml
from scipy.integrate import odeint
from tqdm import tqdm
import itertools

def load_config(path="config/params.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def pendulum_derivs(state, t, k1, k2, m, g, L):
    theta, omega = state
    m = max(m, 1e-4) 
    
    damping_force = (k1 / m) * omega + (k2 / m) * omega * np.abs(omega)
    domega_dt = -(g / L) * np.sin(theta) - damping_force
    return [omega, domega_dt]

def generate_data():
    cfg = load_config()
    phy = cfg['physics']
    gen = cfg['generation']
    if not os.path.exists(gen['save_dir']):
        os.makedirs(gen['save_dir'])
    
    print(f"📂 数据将保存至: {gen['save_dir']}")
    k1s = np.round(np.arange(gen['k1_range'][0], gen['k1_range'][1] + 1e-5, gen['k1_step']), 2)
    k2s = np.round(np.arange(gen['k2_range'][0], gen['k2_range'][1] + 1e-5, gen['k2_step']), 2)
    Ls  = np.round(np.arange(gen['L_range'][0],  gen['L_range'][1]  + 1e-5, gen['L_step']), 2)
    ms  = np.round(np.arange(gen['m_range'][0],  gen['m_range'][1]  + 1e-5, gen['m_step']), 2)
    ns  = np.round(np.arange(gen['noise_range'][0], gen['noise_range'][1] + 1e-5, gen['noise_step']), 3)
    param_combinations = list(itertools.product(k1s, k2s, Ls, ms))
    total_files = len(param_combinations) * len(ns)
    
    print(f"🚀 参数网格: K1({len(k1s)}) x K2({len(k2s)}) x L({len(Ls)}) x m({len(ms)})")
    print(f"🔊 噪声等级: {len(ns)}")
    print(f"📦 预计生成总文件数: {total_files}")
    t_values = np.arange(0.0, phy['t_max'], phy['dt'])
    initial_state = [np.radians(phy['theta0_deg']), 0.0]
    for (k1, k2, L, m) in tqdm(param_combinations, desc="Physics Sim"):
        sol = odeint(pendulum_derivs, initial_state, t_values, 
                     args=(k1, k2, m, phy['g'], L))
        theta_clean = sol[:, 0]
        
        for n_val in ns:
            if n_val > 1e-6:
                noise = np.random.normal(0, n_val, size=theta_clean.shape)
                theta_final = theta_clean + noise
            else:
                theta_final = theta_clean
            fname = (f"K1={k1:.2f}_K2={k2:.2f}_"
                     f"L={L:.2f}_m={m:.2f}_"
                     f"N={n_val:.3f}.csv")
            
            save_path = os.path.join(gen['save_dir'], fname)

            df = pd.DataFrame({
                'Time': t_values,
                'Angle_rad': theta_final.astype(np.float32)
            })
            df.to_csv(save_path, index=False)

if __name__ == "__main__":
    generate_data()