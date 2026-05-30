import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing


def cnn_model_building(csi, label_list, train: bool):
    """
    Args:
        csi: CSI data
        label_list:  Label of each csi data
        train: True if you want train and build model. False if you just want to use trained model
    """

    class CSICNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.features = nn.Sequential(

                # (B,2,64)

                nn.Conv1d(
                    in_channels=2,
                    out_channels=16,
                    kernel_size=3,
                    padding=1
                ),
                nn.BatchNorm1d(16),
                nn.ReLU(),

                # (B,16,64)

                nn.MaxPool1d(2),

                # (B,16,32)

                nn.Conv1d(
                    16,
                    32,
                    kernel_size=3,
                    padding=1
                ),
                nn.BatchNorm1d(32),
                nn.ReLU(),

                # (B,32,32)

                nn.MaxPool1d(2),

                # (B,32,16)

                nn.Conv1d(
                    32,
                    64,
                    kernel_size=3,
                    padding=1
                ),
                nn.BatchNorm1d(64),
                nn.ReLU(),

                # (B,64,16)

                nn.AdaptiveAvgPool1d(8)

                # (B,64,8)
            )

            self.regressor = nn.Sequential(
                nn.Linear(512,128),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(128,64),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(64,32),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(32,1)
            )

        def forward(self,x):

            x = self.features(x)

            # (B,64,8)
            x = torch.flatten(x,1)
            # x = x.squeeze(-1)

            # (B,512)

            x = self.regressor(x)

            return x
        
    num_features = csi.shape[1]

    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    label_list = label_list.unsqueeze(1)

    # 将数据分为训练集和数据集
    X_train, X_test, y_train, y_test = train_test_split(
        csi, label_list, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"shape of X_train: {X_train.size()}. shape of y_train: {y_train.size()}")
    print(f"shape of X_test: {X_test.size()}. shape of y_test: {y_test.size()}")

    # 创建 Dataset
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # 设置 batch size
    train_batch_size = 256
    test_batch_size = 64

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    # 定义损失函数和优化器
    criterion = nn.MSELoss()

    if train == True:

        # 实例化模型
        model = CSICNN().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )

        num_epochs = 1000

        mae_list = [] # contain mean absolute error of each epoch
        loss_list = [] # contain means loss of each epoch
        rmse_list = [] # contain mean rmse of each epoch
        print(f"In training:...\n")

        for epoch in range(num_epochs):
            # switch the model to training state 
            model.train()

            epoch_loss = 0.0
            epoch_rmse = 0.0
            epoch_mae = 0.0

            # 将数据按批量喂给模型
            for batch_x, batch_y in train_loader:
                # 这里的batch_x是（256，2，64），缺少channel的维度
                # print(f"size of batch_x is: {batch_x.size()}")
                # 加入channel维度数据

                # conv1d不需要添加新维度
                # batch_x = batch_x.unsqueeze(1)

                # 放到 GPU
                batch_x = batch_x.to(device).float()
            
                # batch_y = batch_y.to(device).long(), 分类问题用long， 回归问题用float 
                batch_y = batch_y.to(device).float()

                # 前向传播
                outputs = model(batch_x)
                # print(f"dim of outputs is: {outputs.size()}")

                # 计算绝该批量数据绝对误差的均值，损失函数和均方误差
                mae = torch.mean(torch.abs(outputs - batch_y))
            
                # loss
                loss = criterion(outputs, batch_y)
                rmse = torch.sqrt(loss)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_mae += mae.item()
                epoch_loss += loss.item()
                epoch_rmse += rmse.item()

            # 平均 loss, rmse, mae
            avg_loss = epoch_loss / len(train_loader)
            avg_rmse = epoch_rmse / len(train_loader)
            avg_mae = epoch_mae / len(train_loader)
            
            loss_list.append(avg_loss)
            mae_list.append(avg_mae)
            rmse_list.append(avg_rmse)

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Loss: {avg_loss:.4f} "
                f"Accuracy: {mae:.2f}"
            )
        

        # 保存模型的check_point
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, 'csi_checkpoint_cnn.pth')
            # =========================================================
        # Plot Training Curve

        plt.figure(figsize=(10,8))
        plt.subplot(3,1,1)
        plt.plot(loss_list)
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.grid()
        plt.title("Training MSE")

        # ---------------------------------------------------------
        plt.subplot(3,1,2)
        plt.plot(rmse_list)
        plt.xlabel("Epoch")
        plt.ylabel("RMSE (m)")
        plt.grid()
        plt.title("Training RMSE")
        # ---------------------------------------------------------

        plt.subplot(3,1,3)
        plt.plot(mae_list)
        plt.xlabel("Epoch")
        plt.ylabel("MAE (m)")
        plt.grid()
        plt.title("Training MAE")
        plt.tight_layout()
        plt.show()

    # 如果已有训练好的模型，则加载该模型
    else:
        checkpoint = torch.load(
            "csi_checkpoint.pth",
            map_location=device   
        )
        model = CSICNN()
        model.load_state_dict(
            checkpoint['model_state_dict']
        )
        # model = torch.load("csi_checkpoint.pth")
        model.to(device)
        model.eval() 
    
    # 测试阶段的损损失指，均方误差和绝对值误差
    total_loss = 0.0
    total_rmse = 0.0
    total_mae = 0.0
    
    # 存放预测值和标签
    predictions_all = []
    labels_all = []

    # correct = 0
    print(f"In testing...\n")
    with torch.no_grad():

        for batch_x, batch_y in test_loader:

            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            
            # Conv1不需要添加新维度 
            # batch_x = batch_x.unsqueeze( 1)
            predictions = model(batch_x)

            loss = criterion(predictions, batch_y)

            rmse = torch.sqrt(loss)

            mae = torch.mean(torch.abs(predictions - batch_y))

            total_loss += loss.item()

            total_rmse += rmse.item()

            total_mae += mae.item()

            # ---------------------------------------------
            # Save predictions

            # predictions_all.append(predictions.squeeze(1).cpu())
            # labels_all.append(batch_y.squeeze(1).cpu())
            predictions_all.append(predictions.cpu())
            labels_all.append(batch_y.cpu())


    avg_test_loss = total_loss / len(test_loader)
    avg_test_rmse = total_rmse / len(test_loader)
    avg_test_mae = total_mae / len(test_loader)

    
    print(f"Test MSE  : {avg_test_loss:.4f}")
    print(f"Test RMSE : {avg_test_rmse:.4f} m")
    print(f"Test MAE  : {avg_test_mae:.4f} m")


    # Visualize Prediction
    predictions_all = torch.cat(predictions_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    predictions_all = predictions_all.numpy().flatten()
    labels_all = labels_all.numpy().flatten()

    plt.figure(figsize=(8,8))
    plt.scatter(labels_all, predictions_all, alpha=0.5, label = "predictions")
    plt.plot(
        [labels_all.min(), labels_all.max()],
        [labels_all.min(), labels_all.max()],
        'r',
        label = "baseline"
    )

    plt.xlabel("True Distance (m)")
    plt.ylabel("Predicted Distance (m)")
    plt.title("Prediction Result")
    plt.legend()
    plt.grid()
    plt.show()




def cnn_input_genertor(raw_csi_data:np.ndarray)->np.ndarray:
    """
    Args:
        raw_csi_data : raw csi matrix. Each 2000* 64 submatrix indicates csi measured at a 
                       specific point
    
    Return:
        A new matrix with size (num_pakets, 2, 64)
    
    Disc:
        This function extracts amplitude and unwrapped phase of csi from a packet, and combine the amplitude 
        and unwrappd phase into a new 2*64 matrix
    """

    # amplitude and unwrapped phase of all packets
    buffer = []
    num_packets = raw_csi_data.shape[0]
    # min_max_scaler = preprocessing.MinMaxScaler()

    for i in range(num_packets):
        amplitude = np.abs(raw_csi_data[i,:])
        amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min())

        unwrapped_phase = np.unwrap(np.angle(raw_csi_data[i,:]))
        unwrapped_phase = (unwrapped_phase - unwrapped_phase.min()) / (unwrapped_phase.max() - unwrapped_phase.min())
        buffer.append(np.stack([amplitude, unwrapped_phase], axis = 0))
        # buffer_amp = np.concatenate( (buffer_amp ,np.abs(raw_csi_data[i,:])), dim = 0 ) 
        # buffer_phase = np.concatenate( (buffer_phase, unwrapped_csi), dim = 0)
    
    # buffer_amp = np.vstack(buffer_amp)
    # buffer_phase = np.vstack(buffer_phase)  
    buffer = np.array(buffer)
    print(f"shape of new matrix is: {buffer.shape}")
    
    return buffer 



if __name__ == "__main__":

    # 加载mat文件
    csi_rawdata_path = "D:/work/wireless sensing/CSI_DataSet.mat"
    data = scipy.io.loadmat(csi_rawdata_path)['CSI_Dataset']

    # 存储每个位置对应的CSI数据矩阵的名称
    name_data_arr = []
    # 在500米范围内，从1m开始，每隔10米
    num_locations = 20
    hop = 2
    for i in range(num_locations):
        name_data_arr.append('distance_' + str(i*hop+1))

    print(f"type of raw_data: {np.shape(data)}")

    csi_tensor = torch.empty(0)
    # label_list = range(1, 500, 10)
    # print(f"size of label_list: {len(label_list)}")

    # contain the raw csi data of each packet
    raw_data_buffer  = []  # 使用普通列表来收集csi数据
    label_list = []        # 存放每个csi对应的标签

    for i in range(num_locations):
        raw_data = data[name_data_arr[i]]
        # 对每一行的csi进行相位解缠绕
        # unwrapped_phase = np.unwrap(np.angle(raw_data[0,0]), axis=1)
        raw_data_buffer.append(raw_data[0,0])  # 将数据追加到列表中

        # 循环结束后，一次性垂直堆叠成最终的二维数组
        # 将numpy矩阵转化为tensor
        # temp_data = torch.from_numpy(unwrapped_phase)
        # temp_tensor = temp_data.float()
        # real_part = torch.real(temp_data).float()
        # imag_part = torch.imag(temp_data).float()

        # temp_tensor = torch.cat([real_part, imag_part], dim=-1)
        # csi_tensor = torch.cat([csi_tensor, temp_tensor],dim = 0)
        
        # 对每一组CSI都生成一个标签
        temp_label = [i*hop+1] * 2000
        label_list.extend(temp_label)
    
    raw_data_buffer = np.vstack(raw_data_buffer)

    # print(f"shape of csi_tensor: {csi_tensor.shape}")
    print(f"shape of label_list: {len(label_list)}")
    print(f"shape of raw csi matrix isL {raw_data_buffer.shape}")

    # print(f"label_list is: {label_list}")
    # print(f"shape of csi_tensor (before reshape): {csi_tensor.shape}")

    label_list = np.array(label_list)
    label_list = torch.tensor(label_list,dtype=torch.float)
    input_to_cnn = cnn_input_genertor(raw_data_buffer)

    input_to_cnn = torch.from_numpy(input_to_cnn)


    
    # 对CSI数据进行归一化处理
    # csi_mean = csi_tensor.mean(dim=0)
    # csi_std = csi_tensor.std(dim=0)
    # csi = (csi_tensor - csi_mean) / (csi_std + 1e-8)

    # csi_visulizer(raw_data_buffer, 2000, 50)
    cnn_model_building(input_to_cnn, label_list, True)