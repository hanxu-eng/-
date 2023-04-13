import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.optimize

device = torch.device('cpu')

#初始化LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

#LSTM前向传递
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

#初始化检测实例，输入目标的一些特征
class Detection:
    def __init__(self, state, score, features):
        self.state = state
        self.score = score
        self.features = features

class Track:
    #初始化轨迹实例
    def __init__(self, detection):
        self.states = [detection.state]
        self.lost = 0
        self.score = detection.score
        self.features = detection.features

    def update(self, detection):
        self.states.append(detection.state)
        self.lost = 0
        self.score += detection.score
        self.features = detection.features

    @property
    def state(self):
        return self.states[-1]

class MHT:
    def __init__(self, max_lost=3, threshold=0.5, min_score=0.0,lstm_model=None):
        self.max_lost = max_lost
        self.threshold = threshold
        self.min_score = min_score
        self.lstm_model = lstm
        self.tracks = []
        self.track_id = 0

    def update(self, detections):
        N = len(detections)
        M = len(self.tracks)

        if N == 0:
            for track in self.tracks:
                track.lost += 1
                if track.lost > self.max_lost:
                    self.tracks.remove(track)
            return

        if M == 0:
            for i in range(N):
                self.tracks.append(Track(detections[i]))
                self.track_id += 1
            return

        cost = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                diff = self.tracks[i].state - detections[j].state
                cost[i][j] = np.sum(diff ** 2)

        row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost)

        unassigned_tracks = []
        for i in range(M):
            if i not in row_idx:
                unassigned_tracks.append(i)

        unassigned_detections = []
        for j in range(N):
            if j not in col_idx:
                unassigned_detections.append(j)

        assignments = []
        for i, j in zip(row_idx, col_idx):
            if cost[i][j] > self.threshold:
                unassigned_tracks.append(i)
                unassigned_detections.append(j)
            else:
                assignments.append((i, j))

        for i, j in assignments:
            self.tracks[i].update(detections[j])
            self.tracks[i].features = self.lstm_forward(self.tracks[i].state)

        for i in unassigned_tracks:
            self.tracks[i].lost += 1
            if self.tracks[i].lost > self.max_lost:
                self.tracks.remove(self.tracks[i])
            for j in unassigned_detections:
                track = Track(detections[j])
                track.features = self.lstm_forward(track.state)
                self.tracks.append(track)
                self.track_id += 1

    def associate_features(self, features, cost_threshold=0.5):
        N = len(self.tracks)
        M = features.shape[0]
        if N == 0 or M == 0:
            return []

        cost = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                cost[i][j] = np.sum((self.tracks[i].features - features[j]) ** 2)

        row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost)

        assignments = []
        for i, j in zip(row_idx, col_idx):
            if cost[i][j] < cost_threshold:
                assignments.append((i, j))

        return assignments

    def run_mht(self, detections, features):
        self.update(detections)
        assignments = self.associate_features(features)
        tracks = []
        for i, j in assignments:
            track = self.tracks[i]
            detection = detections[j]
            features = detection.features.unsqueeze(0)
            with torch.no_grad():
            state = self.lstm_model(features)
            track.update(detection.bbox, state)
            tracks.append(track)

        for i, track in enumerate(self.tracks):
            if i not in [assignment[0] for assignment in assignments]:
                track.increment_age()
                if track.age > self.max_age:
                    self.tracks.remove(track)

        return tracks


features = lstm_model(inputs)
tracks = mht.run_mht(detections, features)

# 定义超参数
input_size = 4
hidden_size = 32
num_layers = 2
output_size = 1
learning_rate = 0.001
num_epochs = 10

# 加载数据
train_data = np.random.randn(100, 10, input_size)
train_labels = np.random.randn(100, output_size)

# 初始化LSTM模型和优化器
model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(train_data.shape[0]):
        x = torch.Tensor(train_data[i]).unsqueeze(0).to(device)
        y = torch.Tensor(train_labels[i]).unsqueeze(0).to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = nn.MSELoss()(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss/train_data.shape[0]))

# 使用MHT进行航迹关联
mht = MHT()
detections = [Detection(np.array([1, 2, 3, 4]), 0.8, np.random.randn(16)) for _ in range(10)]
features = np.random.randn(10, 16)
tracks = mht.run_mht(detections, features)
print(tracks)
