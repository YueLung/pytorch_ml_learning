# 資料集

# 匯入資料集 base class
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset


# 繼承 base class 創造資料集
class MyDataset(Dataset):
    # 給予資料集大小，並隨機創造資料
    def __init__(self, size: int):
        self.x = torch.rand(size)
        self.y = 2 * self.x**2 + 3 * self.x + 17

    # 定義總資料數
    def __len__(self):
        return len(self.x)

    # 定義取出單一資料的方法
    def __getitem__(self, index):
        return self.x[index], self.y[index]


# 創造資料集
my_dataset = MyDataset(10)
# 取得總資料數
print(len(my_dataset))
# 取出單一資料
print(my_dataset[0])

# ==================================

# 匯入資料集抽樣工具
from torch.utils.data import DataLoader


# 定義格式化的方法
def collate_fn(batch):
    x_list = []
    y_list = []

    for x, y in batch:
        x_list.append([x])
        y_list.append([y])
    # 最終回傳的維度為 [(batch_size, 1), (batch_size, 1)]
    # 最終回傳的維度為 [(batch_size, n_features), (batch_size, label_dim)]
    return [torch.tensor(x_list), torch.tensor(y_list)]


# 創造 DataLoader 實例
batch_size = 3
my_data_loader = DataLoader(
    my_dataset,  # 對資料集 my_dataset 進行抽樣
    batch_size=batch_size,  # 設定每次抽樣的數量
    shuffle=True,  # 設定隨機抽樣
    collate_fn=collate_fn,  # 指定格式化的方法
)

# 透過 my_data_loader 對資料集 my_dataset 進行抽樣
for x, y in my_data_loader:
    print(x.size(), y.size())

# ============================

# 建立模型

# 匯入神經網路模型
import torch.nn as nn

# 匯入啟動函數
import torch.nn.functional as F


# 模型需要繼承自 nn.Module
class MyModel(nn.Module):
    # 定義模型結構, 輸入層維度, 隱藏層維度, 輸出層維度
    def __init__(self, in_dim, hid_dim, out_dim):
        # 繼承 nn.Module 所有屬性
        super(MyModel, self).__init__()

        # 創造線性層 self.layer1
        self.layer1 = nn.Linear(
            # 設定線性層輸入維度
            in_features=in_dim,
            # 設定線性層輸出維度
            out_features=hid_dim,
        )
        # 創造線性層 self.layer2
        self.layer2 = nn.Linear(
            # 設定線性層輸入維度
            in_features=hid_dim,
            # 設定線性層輸出維度
            out_features=out_dim,
        )

    # [Important] 定義運算流程
    def forward(self, batch_x):
        # Why use ReLU?
        # 使用線性層 self.layer1 輸入 batch_x 計算得到 h
        # batch_x's shape: (batch_size, in_dim)
        h = self.layer1(batch_x)  # h = Wx + b, h's shape: (batch_size, hid_dim)
        # 使用 ReLU 啟動函數輸入 h 得到 a
        a = F.relu(h)  # a = ReLU(h), a's shape: (batch_size, hid_dim)
        # 使用線性層 self.layer2 輸入 a 計算得到 y
        y = self.layer2(a)  # y = Wa + b, y's shape: (batch_size, out_dim)

        # 輸出 y
        return y


# 創造 MyModel 模型實例
my_model = MyModel(
    # 設定輸入層維度
    in_dim=1,
    # 設定隱藏層維度
    hid_dim=10,
    # 設定輸出層維度
    out_dim=1,
)

# 透過 my_data_loader 對資料集 my_dataset 進行抽樣
for batch_x, batch_y in my_data_loader:
    print("batch_x shape:", batch_x.shape)
    print("batch_y shape:", batch_y.shape)
    pred_y = my_model(batch_x)  # this part calls my_model.forward(batch_x)
    print(
        "pred_y shape:", pred_y.shape
    )  # should be the same as batch_y in most of the cases, sometimes this should be postprocessed
    # to match the shape of batch_y, HERE is the simplest case that the shape of pred_y is the same as batch_y
    break

# ===========================================

# 目標函數

# 創造均方誤差計算工具
criterion = nn.MSELoss()
# 透過 my_data_loader 對資料集 my_dataset 進行抽樣
for batch_x, batch_y in my_data_loader:
    # 自動呼叫 forward 計算 batch_x 得到 pred_y
    pred_y = my_model(batch_x)

    # 計算 pred_y 與 batch_y 的均方誤差
    loss = criterion(pred_y, batch_y)
    print(loss)

    # 使用向後傳播計算梯度
    loss.backward()

# ======================================

# 最佳化

# 匯入計算梯度下降演算法的工具
from torch.optim import SGD

# 創造計算隨機梯度下降的工具
optimizer = SGD(
    # 設定計算梯度下降的目標
    my_model.parameters(),
    # 設定學習率
    lr=0.0001,
)

# 透過 my_data_loader 對資料集 my_dataset 進行抽樣
for batch_x, batch_y in my_data_loader:
    # 自動呼叫 forward 計算 batch_x 得到 pred_y
    pred_y = my_model(batch_x)

    # 計算 pred_y 與 batch_y 的均方誤差
    loss = criterion(pred_y, batch_y)
    # 使用向後傳播計算梯度
    loss.backward()

    # 使用梯度下降更新模型參數
    optimizer.step()

    # 清空計算過後的梯度值
    optimizer.zero_grad()

# =============================

# 驗證

# 如果有可用 GPU 時採用 GPU cuda:0
if torch.cuda.is_available():
    device = torch.device("cuda:0")
# 若無 GPU 可用則使用 CPU
else:
    device = torch.device("cpu")

# 創造訓練資料集
train_dataset = MyDataset(1000)
# 創造測試資料集
test_dataset = MyDataset(500)

# 設定超參數

# 設定每次抽樣的數量
batch_size = 50
# 設定資料集總訓練次數
n_epoch = 5
# 設定隱藏層維度
hid_dim = 100

# 創造 DataLoader 實例
train_data_loader = DataLoader(
    # 對資料集 train_dataset 進行抽樣
    train_dataset,
    # 設定每次抽樣的數量
    batch_size=batch_size,
    # 設定隨機抽樣
    shuffle=True,
    # 指定格式化的方法
    collate_fn=collate_fn,
)
# 創造 DataLoader 實例
test_data_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)

# 創造 MyModel 模型實例
model = MyModel(
    # 設定輸入層維度
    in_dim=1,
    # 設定隱藏層維度
    hid_dim=hid_dim,
    # 設定輸出層維度
    out_dim=1,
)
# 將模型搬移至 GPU
model = model.to(device)

# 創造均方誤差計算工具
criterion = nn.MSELoss()

# 創造計算隨機梯度下降的工具
optimizer = SGD(model.parameters(), lr=0.0001)

# 總共訓練 n_epoch 次
for epoch in range(n_epoch):
    for batch_x, batch_y in train_data_loader:
        # 將訓練資料搬移至 GPU
        batch_x = batch_x.to(device)
        # 將訓練資料標記搬移至 GPU
        batch_y = batch_y.to(device)

        # 自動呼叫 forward 計算 batch_x 得到 pred_y
        pred_y = model(batch_x)  # shape: (batch_size, 1)
        # 計算 pred_y (預測標記）與 batch_y （真實標記）的均方誤差
        loss = criterion(pred_y, batch_y)

        # 使用向後傳播計算梯度
        loss.backward()
        # 使用梯度下降更新模型參數
        optimizer.step()
        # 清空計算過後的梯度值
        optimizer.zero_grad()

    # 此區塊不會計算梯度
    with torch.no_grad():
        # 統計訓練資料誤差
        total_loss = 0
        for batch_x, batch_y in train_data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred_y = model(batch_x)
            loss = criterion(pred_y, batch_y)

            total_loss += float(loss) / len(train_data_loader)

        print("Epoch {}, training loss: {}".format(epoch, total_loss))

        # 統計測試資料誤差
        total_loss = 0
        for batch_x, batch_y in test_data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred_y = model(batch_x)
            loss = criterion(pred_y, batch_y)

            total_loss += float(loss) / len(test_data_loader)

        print("Epoch {}, testing loss: {}".format(epoch, total_loss))


# ================================

with torch.no_grad():
    for batch_x, batch_y in train_data_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred_y = model(batch_x)

        batch_x = batch_x.to("cpu")
        batch_y = batch_y.to("cpu")
        pred_y = pred_y.to("cpu")
        # 畫出訓練資料答案分佈
        plt.scatter(batch_x, batch_y, color="red")
        # 畫出訓練資料預測分佈
        plt.scatter(batch_x, pred_y, color="blue")

    plt.title("Training data performance")
    plt.show()

    for batch_x, batch_y in test_data_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred_y = model(batch_x)

        batch_x = batch_x.to("cpu")
        batch_y = batch_y.to("cpu")
        pred_y = pred_y.to("cpu")
        # 畫出測試資料答案分佈
        plt.scatter(batch_x, batch_y, color="red")
        # 畫出測試資料預測分佈
        plt.scatter(batch_x, pred_y, color="blue")

    plt.title("Testing data performance")
    plt.show()

# 儲存 & 載入模型

# 儲存模型參數
# torch.save(model.state_dict(), './data/model.ckpt')
# 載入模型參數
# model.load_state_dict(torch.load('./data/model.ckpt'))
