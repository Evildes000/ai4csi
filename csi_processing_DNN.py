import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader


def model_building(csi, label_list, train: bool):
    """
    Args:
        csi: CSI data
        label_list:  Label of each csi data
        train: True if you want train and build model. False if you want use trained model
    """
    # 1. 定义网络类
    class SimpleDNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleDNN, self).__init__()
            # 定义层
            self.fc1 = nn.Linear(input_dim, 512)         # input_dim is number of subcarriers
            self.fc2 = nn.Linear(512, 256)     
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 64)
            self.fc5 = nn.Linear(64, 32)
            self.fc6 = nn.Linear(32, 16)
            self.output_layer = nn.Linear(16, 1)         # output is predicton of distance
            # 本问题为分类问题，最终输出为50个位置的概率
            # self.output_layer = nn.Linear(128, 50)       # 输出层

            self.relu = nn.ReLU()                        # 激活函数
            # self.sigmod = nn.Sigmoid()
            self.dropout = nn.Dropout(0.3)
            # self.sigmoid = nn.Sigmoid()                # 二分类激活

        def forward(self, x):
            # 定义前向传播过程
            x = self.relu(self.fc1(x))
            x = self.dropout(x)

            x = self.relu(self.fc2(x))
            x = self.dropout(x)

            x = self.relu(self.fc3(x))
            # x = self.sigmoid(self.output(x))
            x = self.dropout(x)

            x = self.relu(self.fc4(x))
            x = self.dropout(x)

            x = self.relu(self.fc5(x))
            x = self.dropout(x)

            x = self.relu(self.fc6(x))
            x = self.dropout(x)

            x = self.output_layer(x)

            # x = self.sigmod(x)  # 将输出归一化，防止输出结果震动太大导致模型训练不稳定
            return x
    

    # 模型第一层的输入个数 
    num_features = csi.shape[1]
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 将数据分为训练集和数据集
    X_train, X_test, y_train, y_test = train_test_split(
        csi, label_list, test_size=0.2, random_state=42, shuffle=True
    )

    # y 要变成 [N,1]
    y_train = y_train.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

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
    # 3. 定义损失函数和优化器
    criterion = nn.MSELoss()

    if train == True:

        # 2. 实例化模型
        model = SimpleDNN(num_features).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        # model = CSICNN(input_size=num_features, num_classes=500).to(device)


        # 对于分类问题，不需要手动one-hot 编码label，而且分类问题的标签不是真实的数值，而是从0开始的index
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=0.0001)

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

                # 计算accuracy
                # dim 表示想要消失的维度。dim = 1 表示列维度消失，意思是寻找每一行中最大元素的索引值
                # pred = torch.argmax(outputs, dim=1)
                # if epoch % 10 == 0:
                    # print(f"size of batch_x: {batch_x.size()}\n")
                    # print(f"size of batch_y: {batch_y.size()}\n")
                    # print(f"size outputs: {outputs.size()}\n")

                    # print(f"predicted value :{pred[0:9]}\n")
                    # print(f"label: {batch_y[0:9]}\n")

                # correct += (pred == batch_y).sum().item()
                # total += batch_y.size(0)
                epoch_mae += mae.item()
                epoch_loss += loss.item()
                epoch_rmse += rmse.item()

            # 平均 loss
            avg_loss = epoch_loss / len(train_loader)
            avg_rmse = epoch_rmse / len(train_loader)
            avg_mae = epoch_mae / len(train_loader)
            
            loss_list.append(avg_loss)
            # accuracy = correct / total
            mae_list.append(avg_mae)
            rmse_list.append(avg_rmse)


            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Loss: {avg_loss:.4f} "
                f"Accuracy: {mae:.2f}"
            )


            # if epoch % 10 == 0:
            #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

        
        # 保存模型的check_point
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, 'csi_checkpoint.pth')
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

        """
        plt.subplot(2,1,1)
        plt.plot(range(num_epochs), mae_list)
        plt.xlabel("epochs")
        plt.ylabel("accurancy/%")
        plt.grid()
        plt.title("accurancy of epochs")

        plt.subplot(2,1,2)
        plt.plot(range(num_epochs), loss_list)
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.grid()
        plt.title("Loss of epochs")
        """

    # 如果已有训练好的模型，则加载该模型
    else:
        checkpoint = torch.load(
            "csi_checkpoint.pth",
            map_location=device   
        )
        model = SimpleDNN(num_features)
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

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)

            loss = criterion(predictions, batch_y)

            rmse = torch.sqrt(loss)

            mae = torch.mean(torch.abs(predictions - batch_y))

            total_loss += loss.item()

            total_rmse += rmse.item()

            total_mae += mae.item()

            # ---------------------------------------------
            # Save predictions

            predictions_all.append(predictions.squeeze(1).cpu())
            labels_all.append(batch_y.squeeze(1).cpu())


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

    #accuracy = correct / total
    #print(f"Accuracy: {accuracy:.4f}")

    # 拼接
    # predictions = torch.cat(all_preds, dim=0)
    # y_test = torch.cat(all_labels, dim=0)

    # MSE
    # val_loss = criterion(predictions, y_test)

    # print(f"\n[验证结果] 测试集平均 MSE Loss: {val_loss.item():.6f}")


    # print(type(data['CSI_Dataset']['distance_1']))


def csi_visulizer(raw_csi_data:np.ndarray, stride: int, num_label: int):
    """
    Args:
        raw_csi_data : raw csi matrix. Each 2000* 64 submatrix indicates csi measured at a 
                       specific point
        label_list:    array of all points at which csi been measured
    Disc:
        This function visualizes phase and ampltiude of measured csi alongside distance
    """

    # Take out first csi data of each packet measured at a point
    buffer_amp =  []
    buffer_phase = []
    print(f"type of element in csi_data: {type(raw_csi_data[23,20])}")
    for i in range(num_label):
        unwrapped_csi = np.unwrap(np.angle(raw_csi_data[i*stride,:]))
        buffer_amp.append(np.abs(raw_csi_data[i*stride,:]))
        buffer_phase.append(unwrapped_csi)
        # buffer_amp = np.concatenate( (buffer_amp ,np.abs(raw_csi_data[i,:])), dim = 0 ) 
        # buffer_phase = np.concatenate( (buffer_phase, unwrapped_csi), dim = 0)
    
    buffer_amp = np.vstack(buffer_amp)
    buffer_phase = np.vstack(buffer_phase)  
    print(f"shape of buffer_amp: {np.shape(buffer_amp)}")
    print(f"shape of buffer_phase: {np.shape(buffer_phase)}")

    # unwrapped_csi = np.unwrap(np.angle(raw_csi_data))

    plt.figure(0)
    # 画前十个csi的幅值
    plt.plot(buffer_amp[0:10,:].T, marker='o', linestyle='-')
    plt.xlabel("subcarriers")
    plt.ylabel("amplitude")
    plt.grid()
    plt.legend([f"range {i}m" for i in range(1, 40, 2 )]) # 添加图例
    plt.title("Amplitude of subcarriers")
    plt.show()

    plt.figure(1)
    # 画前十个csi的相位
    plt.plot(buffer_phase[0:10,:].T, marker='o', linestyle='-')
    plt.xlabel("subcarriers")
    plt.ylabel("phase")
    plt.grid()
    plt.legend([f"range {i}m" for i in range(1, 40, 2)])
    plt.title("Phase of subcarriers")
    plt.show()


if __name__ == "__main__":

    # 加载mat文件
    csi_rawdata_path = "D:/work/wireless sensing/CSI_DataSet.mat"
    data = scipy.io.loadmat(csi_rawdata_path)['CSI_Dataset']


    # 每个位置对应的CSI数据矩阵的名称
    name_data_arr = []
    # 在500米范围内，从1m开始，每隔10米
    for i in range(20):
        name_data_arr.append('distance_' + str(i*2+1))

    print(f"type of raw_data: {np.shape(data)}")

    csi_tensor = torch.empty(0)
    # label_list = range(1, 500, 10)
    # print(f"size of label_list: {len(label_list)}")

    # contain the raw csi data of each packet
    raw_data_buffer  = []  # 使用普通列表来收集数据
    label_list = []
    for i in range(20):
        raw_data = data[name_data_arr[i]]
        # 对每一行的csi进行相位解缠绕
        unwrapped_phase = np.unwrap(np.angle(raw_data[0,0]), axis=1)
        raw_data_buffer.append( raw_data[0,0])  # 将数据追加到列表中

        # 循环结束后，一次性垂直堆叠成最终的二维数组
        # 将numpy矩阵转化为tensor
        temp_data = torch.from_numpy(unwrapped_phase)
        temp_tensor = temp_data.float()
        # real_part = torch.real(temp_data).float()
        # imag_part = torch.imag(temp_data).float()

        # temp_tensor = torch.cat([real_part, imag_part], dim=-1)
        csi_tensor = torch.cat([csi_tensor, temp_tensor],dim = 0)
        
        # 对每一组CSI都生成一个标签
        temp_label = [i+1] * 2000
        label_list.extend(temp_label)
    
    raw_data_buffer = np.vstack(raw_data_buffer)
    # 将标签归一化
    # label_list = label_list / 500

    print(f"shape of csi_tensor: {csi_tensor.shape}")
    print(f"shape of label_list: {len(label_list)}")
    # print(f"label_list is: {label_list}")
    # print(f"shape of csi_tensor (before reshape): {csi_tensor.shape}")

    label_list = np.array(label_list)
    label_list = torch.tensor(label_list,dtype=torch.long)
    
    # 对CSI数据进行归一化处理
    # csi_mean = csi_tensor.mean(dim=0)
    # csi_std = csi_tensor.std(dim=0)
    # csi = (csi_tensor - csi_mean) / (csi_std + 1e-8)

    csi_visulizer(raw_data_buffer, 2000, 20)
    # model_building(csi_tensor,label_list, False)


    # 用卷积神经网络训练



