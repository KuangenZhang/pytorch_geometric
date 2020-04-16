import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, global_mean_pool

from pointnet2_classification import MLP
from tqdm import tqdm

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')
pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
train_dataset = ModelNet(path, '10', True, transform, pre_transform)
test_dataset = ModelNet(path, '10', False, transform, pre_transform)
train_loader = DataLoader(
    train_dataset, batch_size=28, shuffle=True, num_workers=6)
test_loader = DataLoader(
    test_dataset, batch_size=28, shuffle=False, num_workers=6)


class Generator(torch.nn.Module):
    def __init__(self, feature_num = 1024, k=20, aggr='max'):
        super().__init__()
        self.conv1 = DynamicEdgeConv(MLP([2 * 3,   64 ]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64,  64 ]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64,  128]), k, aggr)
        self.conv4 = DynamicEdgeConv(MLP([2 * 128, 256]), k, aggr)

        self.lin1 = MLP([64 + 64 + 128 + 256, feature_num])

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)

        out = self.lin1(torch.cat([x1, x2, x3, x4], dim=1))
        out1 = global_max_pool(out, batch)
        out2 = global_mean_pool(out, batch)
        return torch.cat([out1, out2], dim=1)


class Classifier(torch.nn.Module):
    def __init__(self, feature_num = 1024, out_channels=10, aggr='max'):
        super().__init__()
        self.mlp = Seq(
            MLP([2 * feature_num, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, feature):
        return F.log_softmax(self.mlp(feature), dim=1)


class Net(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 3,   64 ]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64,  64 ]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64,  128]), k, aggr)
        self.conv4 = DynamicEdgeConv(MLP([2 * 128, 256]), k, aggr)

        self.lin1 = MLP([64 + 64 + 128 + 256, 1024])

        self.c1 = Seq(
            MLP([2 * 1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

        self.c2 = Seq(
            MLP([2 * 1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, data):
        feature = self.feature(data)
        prob_1 = self.classify_1(feature)
        prob_2 = self.classify_2(feature)
        return (prob_1 + prob_2) / 2

    def feature(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.conv4(x3, batch)

        out = self.lin1(torch.cat([x1, x2, x3, x4], dim=1))
        out1 = global_max_pool(out, batch)
        out2 = global_mean_pool(out, batch)
        return torch.cat([out1, out2], dim=1)

    def classify_1(self, feature):
        return F.log_softmax(self.c1(feature), dim=1)

    def classify_2(self, feature):
        return F.log_softmax(self.c2(feature), dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = Generator().to(device)
C1 = Classifier(out_channels=train_dataset.num_classes).to(device)
C2 = Classifier(out_channels=train_dataset.num_classes).to(device)

optimizer_G = torch.optim.SGD(G.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
optimizer_C1 = torch.optim.SGD(C1.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
optimizer_C2 = torch.optim.SGD(C2.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.5)
scheduler_C1 = torch.optim.lr_scheduler.StepLR(optimizer_C1, step_size=20, gamma=0.5)
scheduler_C2 = torch.optim.lr_scheduler.StepLR(optimizer_C2, step_size=20, gamma=0.5)


def train():
    G.train()
    C1.train()
    C2.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        feature = G(data)
        prob_1 = C1(feature)
        prob_2 = C2(feature)

        # loss 1: train G, C1, and C2 to minimize error
        optimizer_G.zero_grad()
        optimizer_C1.zero_grad()
        optimizer_C2.zero_grad()

        loss_error = F.nll_loss(prob_1, data.y) + F.nll_loss(prob_2, data.y)
        loss_error.backward(retain_graph=True)
        total_loss += loss_error.item() * data.num_graphs

        optimizer_C2.step()
        optimizer_C1.step()
        optimizer_G.step()

        # loss 2: train C1 and C2 to maximize discrepancy
        optimizer_C1.zero_grad()
        optimizer_C2.zero_grad()

        loss_dis = torch.mean(torch.abs(prob_1 - prob_2))

        loss_C = loss_error - 4 * loss_dis
        loss_C.backward(retain_graph=True)

        # loss 3: train G to minimize error
        optimizer_G.zero_grad()

        loss_error.backward()

        optimizer_G.step()

    return total_loss / len(train_dataset)


def test(loader):
    G.eval()
    C1.eval()
    C2.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            prob = (C1(G(data)) + C2(G(data)))/2
            pred = prob.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


best_test_acc = 0.0
pbar = tqdm(total=100, initial=0, position=0, leave=True)

for epoch in range(1, 101):
    loss = train()
    test_acc = test(test_loader)
    if best_test_acc < test_acc:
        print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
            epoch, loss, test_acc))
        best_test_acc = test_acc

    scheduler_G.step()
    scheduler_C1.step()
    scheduler_C2.step()

    pbar.update(1)
