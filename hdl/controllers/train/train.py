# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/controllers/train/train.py
# 说明：训练流程与 Trainer 实现
# 模块功能：分子图回归模型（模型由外部环境提供）的脚本式训练入口：用 PyG DataLoader 按 8:2 切分训练/测试数据，逐批做前向、算均方根误差（RMSE）、反向传播，固定跑 2000 轮（epoch）后用 seaborn 画损失曲线
# 说明：本文件不是可导入模块，model、data 等名字都依赖外部命名空间，也没有参数解析、日志与检查点（checkpoint）保存逻辑
from torch_geometric.data import DataLoader
# 导入并全局屏蔽告警，避免训练脚本被 warning 刷屏
import warnings
warnings.filterwarnings("ignore")

# 损失与优化器都在模块顶层直接构建，没有函数封装也没有参数解析：loss_fn 为均方误差（MSE），反向传播时用的是它的平方根
# Root mean squared error
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)  

# Adam 优化器作用于 model 的全部参数，学习率固定 0.0007，未设置权重衰减（weight decay）
# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# data 为外部提供的图数据序列：前 80% 当训练集、后 20% 当测试集，每批 64 张图，两个加载器（DataLoader）都打乱顺序，没有单独验证集
# Wrap data in a data loader
data_size = len(data)
NUM_GRAPHS_PER_BATCH = 64
loader = DataLoader(data[:int(data_size * 0.8)], 
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(data[int(data_size * 0.8):], 
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

def train(data):
    """单轮（epoch）训练：形参 data 未被使用，真正迭代的是外层闭包里的全局 loader；每个批次前向得到 (预测, 节点嵌入)，以 sqrt(MSE) 作为损失反向更新，返回最后一个批次的 (loss, embedding)"""
    # Enumerate over the data
    for batch in loader:
      # 把当前批次的特征、边索引与标签搬到训练设备（就地修改 batch）
      # Use GPU
      batch.to(device)  
      # 梯度清零，否则 PyTorch 默认把上一批梯度累加进来
      # Reset gradients
      optimizer.zero_grad() 
      # 前向输入为节点特征（转成 float32）、COO 边索引和批次图归属向量，同时返回预测值与图嵌入 embedding
      # Passing the node features and the connection info
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
      # 损失取 MSE 的平方根，即均方根误差（RMSE），再对它做一次反向传播求梯度
      # Calculating the loss and gradients
      loss = torch.sqrt(loss_fn(pred, batch.y))       
      loss.backward()  
      # 用刚算好的梯度执行一次参数更新
      # Update using the gradients
      optimizer.step()   
    # 只返回最后一个批次的损失与嵌入，不做跨批平均；外层循环只取用 loss，嵌入 h 被丢弃
    return loss, embedding

# 主训练循环：轮数（epoch）硬编码 2000，每轮调用一次 train() 并把该轮损失张量追加进 losses；只做训练，没有在 test_loader 上评估指标，也不保存检查点（checkpoint）
print("Starting training...")
losses = []
for epoch in range(2000):
    loss, h = train(data)
    losses.append(loss)
    # 每 100 轮（epoch）只打印一次当前损失，没有验证集评估、早停或检查点（checkpoint）保存
    if epoch % 100 == 0:
      print(f"Epoch {epoch} | Train Loss {loss}")
# Visualize learning (training loss)
# 逐条把损失张量搬到 CPU 并脱离计算图转成 float，再和轮次序号一起交给 seaborn 画折线；此处 plt 被重新绑定为 lineplot 返回的坐标轴对象
import seaborn as sns
losses_float = [float(loss.cpu().detach().numpy()) for loss in losses] 
loss_indices = [i for i,l in enumerate(losses_float)] 
# 这一行裸写的 plt 只是把 Axes 对象显示出来（交互环境下生效），未调用 show/savefig
plt = sns.lineplot(loss_indices, losses_float)
plt
# 下一行是原稿粘贴进来的未加引号英文说明文字，HEAD 版本就存在该语法错误，此处按任务约定原样保留、不做修复
As result we get something like this: