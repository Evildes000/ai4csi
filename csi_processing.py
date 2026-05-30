import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 定义网络类
class SimpleDNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleDNN, self).__init__()
        # 定义层
        self.fc1 = nn.Linear(input_dim, 64) # 全连接层 1
        self.fc2 = nn.Linear(64, 32)         # 全连接层 2
        self.output_layer = nn.Linear(32, 2)       # 输出层
        self.relu = nn.ReLU()                # 激活函数
        # self.sigmoid = nn.Sigmoid()          # 二分类激活

    def forward(self, x):
        # 定义前向传播过程
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.sigmoid(self.output(x))
        x = self.output_layer(x)
        return x
    

path_50cm = "D:/work/wireless sensing/0.5m.mat"
path_70cm = "D:/work/wireless sensing/0.7m.mat"
path_90cm = "D:/work/wireless sensing/0.9m.mat"


def train_test(path_50cm: str,  path_70cm: str, path_90cm: str):
    """
    Args:
        path_50cm : path to the sci data measured at 50 cm
        path_70cm : path to the sci data measured at 70 cm
        path_90cm : path to the sci data measured at 90 cm
    disciption:
        This function train and test the CNN model with givn path to data
    """
    # load the mat form data
    data_50cm = scipy.io.loadmat(path_50cm)
    data_70cm = scipy.io.loadmat(path_70cm)
    data_90cm = scipy.io.loadmat(path_90cm)
    print(f"type of element in data_50cm: {data_50cm['csiRawValid'][1,1]}")

    # rane and angle
    label_50cm = [0.5,0] 
    label_70cm = [0.7,0]
    label_90cm = [0.9,0]

    # extract raw csi data
    csi_50 = torch.from_numpy(data_50cm['csiRawValid'])
    csi_70 = torch.from_numpy(data_70cm['csiRawValid'])
    csi_90 = torch.from_numpy(data_90cm['csiRawValid'])

    print(f"type of element in csi_50: {csi_50[1,1]}")
    # print(f"type of csi is: {csi_50}")
    # print(csi_50)

    real_part = torch.real(csi_50).float()
    imag_part = torch.imag(csi_50).float()
    csi_50 = torch.cat([real_part, imag_part], dim=-1)

    real_part = torch.real(csi_70).float()
    imag_part = torch.imag(csi_70).float()
    csi_70 = torch.cat([real_part, imag_part], dim=-1)

    real_part = torch.real(csi_90).float()
    imag_part = torch.imag(csi_90).float()
    csi_90 = torch.cat([real_part, imag_part], dim=-1)

    # print(f"type of data_50 is {type(csi_50)}")
    # print(f"type of data_70 is {type(csi_70)}")
    # print(f"type of data_90 is {type(csi_90)}")

    num_features = csi_50.shape[1]

    num_packets_50 = csi_50.shape[0]
    num_packets_70 = csi_70.shape[0]
    num_packets_90 = csi_90.shape[0]

    label_50 = torch.tensor([0.5,0],dtype=torch.float)
    label_70 = torch.tensor([0.7,0],dtype=torch.float)
    label_90 = torch.tensor([0.9,0],dtype=torch.float)

    # 在第 0 维（行）重复 1000 次，在第 1 维（列）重复 1 次
    label_50 = label_50.repeat(num_packets_50, 1)
    label_70 = label_70.repeat(num_packets_70, 1)
    label_90 = label_90.repeat(num_packets_90, 1)



    # 合并所有数据
    csi = torch.cat([csi_50, csi_70, csi_90], dim=0)
    label = torch.cat([label_50, label_70, label_90], dim=0)

    # 对CSI数据进行归一化处理
    csi_mean = csi.mean(dim=0)
    csi_std = csi.std(dim=0)
    csi = (csi - csi_mean) / (csi_std + 1e-8)

    # 将数据分为训练集和数据集
    X_train, X_test, y_train, y_test = train_test_split(
        csi, label, test_size=0.2, random_state=42, shuffle=True
    )

    # 2. 实例化模型
    model = SimpleDNN(num_features)

    # 3. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 150
    for epoch in range(num_epochs):
        model.train() # 设置为训练模式
        # 前向传播
        outputs = model(X_train) # 一次性投入矩阵，模型会自动按行处理
        
        # 计算损失函数
        loss = criterion(outputs, y_train)
        
        # --- 反向传播与优化 ---
        optimizer.zero_grad()   # 清空上一轮梯度
        loss.backward()        # 计算梯度
        optimizer.step()        # 更新权重参数
        
        # 每 1 轮打印一次进度
        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


    # 评估模式
    model.eval() 

    # 禁用计算梯度
    with torch.no_grad():
        # 得到模型在测试集上的预测结果
        predictions = model(X_test)
        
        # 计算测试集上的总 MSE 损失
        val_loss = criterion(predictions, y_test)
        print(f"\n[验证结果] 测试集平均 MSE Loss: {val_loss.item():.6f}")

        # 3. 将数据转回 NumPy 方便分析或绘图
        pred_np = predictions.numpy()
        true_np = y_test.numpy()

        # 4. 计算具体的物理意义（例如距离误差）
        # 假设标签的第一列是距离 (0.5, 0.7, 0.9)
        dist_error = np.abs(pred_np[:, 0] - true_np[:, 0])
        mean_dist_error = np.mean(dist_error)
        # print(f"[距离统计] 平均距离误差: {mean_dist_error:.4f} 米")

        # 5. 打印前 10 条对比看看情况
        print("\n前 10 条对比 (预测值 vs 真实值):")
        for i in range(len(dist_error)):
            print(f"样本 {i+1}: 预测 {pred_np[i, 0]:.3f}m, 真实 {true_np[i, 0]:.3f}m")
        
        print(f"[距离统计] 平均距离误差: {mean_dist_error:.4f} 米")


    # 构造一个 DataFrame 方便绘图
    import pandas as pd
    df_res = pd.DataFrame({
        'True Distance/m': true_np[:, 0].astype(str), # 转成字符串作为分类
        'Error/m': np.abs(pred_np[:, 0] - true_np[:, 0])
    })

    sns.boxplot(x='True Distance/m', y='Error/m', data=df_res)
    plt.title('Error Distribution at Different Distances')
    # plt.xlabel("True dist/m")
    # plt.ylabel("Error/m")
    plt.grid()
    plt.show()


def exploit_csi(path_50cm: str,  path_70cm:str, path_90cm:str):
    """
    Arg:
        path_50cm : path to the sci data measured at 50 cm
        path_70cm : path to the sci data measured at 70 cm
        path_90cm : path to the sci data measured at 90 cm
    disciption:
        This function plots the amplitude and phase of the raw csi to exploit channel characteristic
        of channel and phase shift

    """

    data_50cm = scipy.io.loadmat(path_50cm)['csiRawValid']
    data_70cm = scipy.io.loadmat(path_70cm)['csiRawValid']
    data_90cm = scipy.io.loadmat(path_90cm)['csiRawValid']

    amp_csi_50 = np.abs(data_50cm)
    phase_csi_50 = np.unwrap(np.angle(data_50cm), axis=1)

    amp_csi_70 = np.abs(data_50cm)
    phase_csi_70 = np.unwrap(np.angle(data_70cm), axis=1)

    amp_csi_90 = np.abs(data_90cm)
    phase_csi_90 = np.unwrap(np.angle(data_90cm), axis=1)

    cubcarrier = np.concatenate((np.arange(-26, 0), np.arange(1, 27)))
    

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    # axes 是一个 3x2 的数组
    axes[0, 0].plot(cubcarrier, amp_csi_50[1,:])
    axes[0, 0].set_title("amp_csi_50")

    axes[0, 1].plot(cubcarrier, phase_csi_50[1,:])
    axes[0, 1].set_title("phase_csi_50")
    
    axes[1, 0].plot(cubcarrier, amp_csi_70[1,:])
    axes[1, 0].set_title("amp_csi_70")

    axes[1, 1].plot(cubcarrier, phase_csi_70[1,:])
    axes[1, 1].set_title("phase_csi_70")

    axes[2, 0].plot(cubcarrier, amp_csi_90[1,:])
    axes[2, 0].set_title("amp_csi_90")

    axes[2, 1].plot(cubcarrier, phase_csi_90[1,:])
    axes[2, 1].set_title("phase_csi_90")

    # 自动调整间距
    plt.tight_layout()

    plt.show()







if __name__ == "__main__":
    path_50cm = "D:/work/wireless sensing/0.5m.mat"
    path_70cm = "D:/work/wireless sensing/0.7m.mat"
    path_90cm = "D:/work/wireless sensing/0.9m.mat"
    # multi-hop sidelink sensing and communication 

    # 尝试先进行相位解缠绕，再训练模型
    train_test(path_50cm, path_70cm, path_90cm)
    # exploit_csi(path_50cm, path_70cm, path_90cm)