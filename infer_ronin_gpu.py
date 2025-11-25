import os
import json
import torch
import numpy as np
import pandas as pd

from source.model_temporal import TCNSeqNetwork  # ← 正确导入路径!!!

# ============================================================
# 路径配置
# ============================================================
CHKPT = r"D:\Code\IMU_mag\pdr\ronin-master\model\Pretrained_Models\ronin_tcn\ronin_tcn\checkpoints\ronin_tcn_checkpoint.pt"
CONFIG_JSON = r"D:\Code\IMU_mag\pdr\ronin-master\model\Pretrained_Models\ronin_tcn\ronin_tcn\config.json"
CSV = r"D:\Code\IMU_mag\data\SPunder\mag_with_ref_acc_gyro_data_common_time.csv"
OUT = r"D:\Code\IMU_mag\pdr\ronin-master\model\Pretrained_Models\ronin_tcn\ronin_tcn\checkpoints\ptronin_pred_traj.txt"

# ============================================================
# 设备
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 加载模型
# ============================================================
def load_model():
    with open(CONFIG_JSON, "r") as f:
        cfg = json.load(f)

    channels = cfg["channels"]
    kernel_size = cfg["kernel_size"]
    dropout = cfg["kwargs"]["dropout"]

    model = TCNSeqNetwork(
        input_channel=6,
        output_channel=2,
        kernel_size=kernel_size,
        layer_channels=channels,
        dropout=dropout
    ).to(device)

    ckpt = torch.load(CHKPT, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("RoNIN TCN model loaded.")
    return model, cfg

# ============================================================
# CSV 输入
# ============================================================
def load_my_csv(path):
    df = pd.read_csv(path)

    acce_x = df["acc_X"].values
    acce_y = df["acc_Y"].values
    acce_z = df["acc_Z"].values + 9.81

    gyro_x = df["gyro_X"].values
    gyro_y = df["gyro_Y"].values
    gyro_z = df["gyro_Z"].values

    imu = np.stack([acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z], axis=1)
    return imu.astype(np.float32)


# ============================================================
# 推理（整段输入）
# ============================================================
def infer_full_sequence(model, imu):
    x = torch.from_numpy(imu).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x)
    return y.squeeze(0).cpu().numpy()

import numpy as np
import matplotlib.pyplot as plt





# ============================================================
# 主函数
# ============================================================
def main():
    model, cfg = load_model()
    imu = load_my_csv(CSV)
    preds = infer_full_sequence(model, imu)
    np.savetxt(OUT, preds, fmt="%.6f")
    print("Saved:", OUT)

    PRED = r"D:\Code\IMU_mag\pdr\ronin-master\model\Pretrained_Models\ronin_tcn\ronin_tcn\checkpoints\ptronin_pred_traj.txt"

    pred = np.loadtxt(PRED)  # [N, 2]

    print("pred mean:", pred.mean(axis=0))
    print("pred std:", pred.std(axis=0))
    print("pred min:", pred.min(axis=0))
    print("pred max:", pred.max(axis=0))

    plt.figure()
    plt.plot(pred[:, 0], label="vx_pred")
    plt.plot(pred[:, 1], label="vy_pred")
    plt.legend(); plt.title("RoNIN raw outputs over time"); plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
