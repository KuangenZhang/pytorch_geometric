import time

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
# from torch.utils.data import DataLoader

from tqdm import tqdm
from data import ModelNet40

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(train_dataset, test_dataset, model, epochs, batch_size, lr,
        lr_decay_factor, lr_decay_step_size, weight_decay):

    model = model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # train_loader = DataLoader(ModelNet40(partition='train', num_points=1024), num_workers=8,
    #                           batch_size=batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(ModelNet40(partition='test', num_points=1024), num_workers=8,
    #                          batch_size=batch_size, shuffle=True, drop_last=False)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    best_test_acc = 0.0
    pbar = tqdm(total=epochs, initial=0, position=0, leave=True)

    for epoch in range(1, epochs + 1):

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        train(model, optimizer, train_loader, device)
        test_acc = test(model, test_loader, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        if best_test_acc < test_acc:
            best_test_acc = test_acc

        print('Epoch: {:03d}, Test: {:.4f}, Best test: {:.4f}, Duration: {:.2f}'.format(
            epoch, test_acc, best_test_acc, t_end - t_start))

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

        pbar.update(1)

def train(model, optimizer, train_loader, device):
    model.train()

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

    # for data, label in train_loader:
    #     data, label = data.to(device), label.to(device).squeeze()
    #     optimizer.zero_grad()
    #     out = model(data, data.size()[0])
    #     loss = F.nll_loss(out, label)
    #     loss.backward()
    #     optimizer.step()

def test(model, test_loader, device):
    model.eval()

    correct = 0
    for data in test_loader:
        data = data.to(device)
        pred = model(data.pos, data.batch).max(1)[1]
        correct += pred.eq(data.y).sum().item()

    # for data, label in test_loader:
    #     data, label = data.to(device), label.to(device).squeeze()
    #     pred = model(data, data.size()[0]).max(1)[1]
    #     correct += pred.eq(label).sum().item()

    test_acc = correct / len(test_loader.dataset)

    return test_acc
