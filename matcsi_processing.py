import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader

# 1. 定义网络类
class SimpleDNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleDNN, self).__init__()
        # 定义层
        self.fc1 = nn.Linear(input_dim, 512) # 全连接层 1
        self.fc2 = nn.Linear(512, 256)         # 全连接层 2
        self.fc3 = nn.Linear(256, 128)
        # 本问题为分类问题，最终输出为500个位置的概率
        self.output_layer = nn.Linear(128, 500)       # 输出层

        self.relu = nn.ReLU()                # 激活函数
        self.sigmod = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)
        # self.sigmoid = nn.Sigmoid()          # 二分类激活

    def forward(self, x):
        # 定义前向传播过程
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.relu(self.fc3(x))
        # x = self.sigmoid(self.output(x))
        x = self.output_layer(x)

        # x = self.sigmod(x)  # 将输出归一化，防止输出结果震动太大导致模型训练不稳定
        return x


# repeat_times = 2000
# dist_arr = np.array([])
# for i in range(500):
#    temp_arr = np.ones(repeat_times) * (i+1)
#    dist_arr = np.concatenate((dist_arr, temp_arr))

csi_rawdata_path = "D:/work/wireless sensing/CSI_DataSet.mat"
data = scipy.io.loadmat(csi_rawdata_path)['CSI_Dataset']


# 每个位置对应的存贮CSI数据矩阵的名称
name_data_arr = []
for i in range(500):
    name_data_arr.append('distance_' + str(i+1))


print(f"type of raw_data: {np.shape(data)}")

csi_tensor = torch.empty(0)
label_list = []
for i in range(500):
    raw_data = data[name_data_arr[i]]

    # 将numpy矩阵转化为tensor
    temp_data = torch.from_numpy(raw_data[0,0])
    real_part = torch.real(temp_data).float()
    imag_part = torch.imag(temp_data).float()
    temp_tensor = torch.cat([real_part, imag_part], dim=-1)
    csi_tensor = torch.cat([csi_tensor, temp_tensor],dim = 0)
    
    # 对每一组CSI都生成一个标签
    temp_label = [i] * 2000
    label_list.extend(temp_label)

# 将标签归一化
# label_list = label_list / 500

print(f"shape of csi_tensor: {csi_tensor.shape}")
print(f"shape of label_list: {len(label_list)}")


label_list = np.array(label_list)
label_list = torch.tensor(label_list,dtype=torch.long)
# 对CSI数据进行归一化处理
csi_mean = csi_tensor.mean(dim=0)
csi_std = csi_tensor.std(dim=0)
csi = (csi_tensor - csi_mean) / (csi_std + 1e-8)

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 模型第一层的输入个数 
num_features = csi.shape[1]


# 将数据分为训练集和数据集
X_train, X_test, y_train, y_test = train_test_split(
    csi, label_list, test_size=0.2, random_state=42, shuffle=True
)

# y 要变成 [N,1]
# y_train = y_train.unsqueeze(1)
# y_test = y_test.unsqueeze(1)

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


# 2. 实例化模型
model = SimpleDNN(num_features).to(device)

# 3. 定义损失函数和优化器
# criterion = nn.MSELoss()
# 对于分类问题，不需要手动one-hot 编码label
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

num_epochs = 200
print(f"In training:...\n")
for epoch in range(num_epochs):

    model.train()

    epoch_loss = 0.0

    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:

        # 放到 GPU
        batch_x = batch_x.to(device).float()
       
        batch_y = batch_y.to(device).long()
        

        # 前向传播
        outputs = model(batch_x)
    
        # loss
        loss = criterion(outputs, batch_y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算accuracy
        # dim 表示想要消失的维度。dim = 1 表示列维度消失，意思是寻找每一行中最大元素的索引值
        pred = torch.argmax(outputs, dim=1)
        if epoch % 10 == 0:
            # print(f"size of batch_x: {batch_x.size()}\n")
            # print(f"size of batch_y: {batch_y.size()}\n")
            # print(f"size outputs: {outputs.size()}\n")

            print(f"predicted value :{pred[0:9]}\n")
            print(f"label: {batch_y[0:9]}\n")

        correct += (pred == batch_y).sum().item()
        total += batch_y.size(0)

        epoch_loss += loss.item()

    # 平均 loss
    avg_loss = epoch_loss / len(train_loader)

    # accuracy
    accuracy = correct / total

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy:.4f}"
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


# 评估模式
model.eval() 

all_preds = []
all_labels = []

correct = 0

with torch.no_grad():

    for batch_x, batch_y in test_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        predictions = model(batch_x)

        pred = torch.argmax(outputs, dim=1)

        correct += (pred == batch_y).sum().item()
        total += batch_y.size(0)

accuracy = correct / total

print(f"Accuracy: {accuracy:.4f}")

# 拼接
# predictions = torch.cat(all_preds, dim=0)
# y_test = torch.cat(all_labels, dim=0)

# MSE
# val_loss = criterion(predictions, y_test)

# print(f"\n[验证结果] 测试集平均 MSE Loss: {val_loss.item():.6f}")


# print(type(data['CSI_Dataset']['distance_1']))




