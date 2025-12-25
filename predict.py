import torch
import yaml
import glob
import random
import pandas as pd
import os
import re  
from models.transformer import PendulumTransformer

def predict():
    with open("config/params.yaml", 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else "cpu")
    
    model = PendulumTransformer(cfg['model']).to(device)
    model_path = "runs/train/exp7/weights/best.pt" 
    try:
        model.load_state_dict(torch.load("runs/train/exp9/weights/best.pt"))
    except RuntimeError as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 原因可能是：权重文件的输出维度是2，但代码里设置的是3。")
        print("👉 请确保你加载的是【重新训练】后的 best.pt 文件。")
        return
    model.eval()
    data_dir = cfg['generation']['save_dir']
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        print("❌ 没有找到测试数据文件。")
        return
    test_file = random.choice(files)
    
    print(f"Testing on file: {os.path.basename(test_file)}")

    df = pd.read_csv(test_file)
    seq = df['Angle_rad'].values if 'Angle_rad' in df else df.iloc[:, 1].values
    
    input_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        pred = model(input_tensor).cpu().numpy()[0]
    
    pred_k1 = pred[0]
    pred_k2 = pred[1]
    pred_n  = pred[2]  
    
    print("-" * 30)
    print(f"Predicted K1: {pred_k1:.4f}")
    print(f"Predicted K2: {pred_k2:.4f}")
    print(f"Predicted N:  {pred_n:.4f}") 
    print("-" * 30)
    
    r1 = re.search(r"K1=(\d+\.\d+)", test_file)
    r2 = re.search(r"K2=(\d+\.\d+)", test_file)
    r3 = re.search(r"N=(\d+\.\d+)", test_file)  
    
    true_k1 = float(r1.group(1)) if r1 else 0.0
    true_k2 = float(r2.group(1)) if r2 else 0.0
    true_n  = float(r3.group(1)) if r3 else 0.0 
    
    print(f"True K1:      {true_k1:.4f}")
    print(f"True K2:      {true_k2:.4f}")
    print(f"True N:       {true_n:.4f}")      
    print("-" * 30)
    print(f"Error K1:     {abs(true_k1 - pred_k1):.4f}")
    print(f"Error K2:     {abs(true_k2 - pred_k2):.4f}")
    print(f"Error N:      {abs(true_n  - pred_n):.4f}") 

if __name__ == "__main__":
    predict()