import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 你的文件路径
PRED = r"D:\Code\IMU_mag\pdr\ronin-master\model\Pretrained_Models\ronin_tcn\ronin_tcn\checkpoints\ptronin_pred_traj.txt"
CSV  = r"D:\Code\IMU_mag\data\SPunder\mag_with_ref_acc_gyro_data_common_time.csv"

# 1) 加载 RoNIN 预测
pred = np.loadtxt(PRED)   # shape = [N, 2]

# 累加轨迹
traj = np.cumsum(pred, axis=0)   # [N, 2]

# 2) 加载真值轨迹（RS10）
df = pd.read_csv(CSV)

lats = df["ref_Latitude"].values
lons = df["ref_Longitude"].values

# 使用简单平面投影：相对第1个点
lat0, lon0 = lats[0], lons[0]
R = 6378137
gt_x = (lons - lon0) * np.cos(np.deg2rad(lat0)) * (np.pi/180) * R
gt_y = (lats - lat0) * (np.pi/180) * R

# 如果真值长度比预测长 → 对齐
min_len = min(len(traj), len(gt_x))
traj = traj[:min_len]
gt_x = gt_x[:min_len]
gt_y = gt_y[:min_len]

# 3) 绘图
plt.figure(figsize=(6,6))
plt.plot(traj[:,0], traj[:,1], label="RoNIN Predicted Trajectory")
plt.plot(gt_x, gt_y, label="RS10 Ground Truth")
plt.legend()
plt.title("Predicted vs Ground Truth Trajectory")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.axis("equal")
plt.grid(True)
plt.show()
