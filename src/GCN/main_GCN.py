import torch
import torch.nn as nn
import torch.nn.functional as F
from pyg_dataToGraph import DataToGraph
from torch_geometric.loader import DataLoader
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from GCN_model import GCN
import os, random
import matplotlib.pyplot as plt
#from curve import plot_metrics

# TODO 评价指标, f1-macro, f1-micro
# 添加了评估指标acc
def evaluate_metrics(model, device, data_loader, metric_type):
    model.eval()
    epoch_test_metrics = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for iter, batched_graph in enumerate(data_loader):
            batched_graph = batched_graph.to(device)
            batch_pred = model.forward(batched_graph.x, batched_graph.edge_index,
                                       batched_graph.batch)  # (batch x num_classes)
            y_true.append(batched_graph.y.view(-1, 1).detach().cpu())
            y_pred.append(torch.argmax(batch_pred.detach(), dim=1).view(-1, 1).cpu())
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        if metric_type == 'f1-macro':
            epoch_test_metrics = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        elif metric_type == 'f1-micro':
            epoch_test_metrics = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        else:
            epoch_test_metrics = accuracy_score(y_true=y_true, y_pred=y_pred) # acc

    return epoch_test_metrics



if __name__ == '__main__':
    # python dgl_data_to_graph.py --dataset 'NPS_64sensors_13type' --batch_size 128 --device 0 --epochs 10 --lr 0.001
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='TFF', choices=['NPS_64sensors_13type', 'TFF'],
                        help="Please give a value for dataset name")
    parser.add_argument('--data_dir', default='../data/', help="dataset path")
    parser.add_argument('--batch_size', default=128, help="batch_size for dataset")
    parser.add_argument('--device', default=0, help="cuda id")
    parser.add_argument('--epochs', default=300, help="training epochs")
    parser.add_argument('--lr', default=0.005, help="learning rate")
    parser.add_argument('--hidden_channels', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # TODO 加载数据集
    dataset = DataToGraph(
        raw_data_path=args.data_dir,
        dataset_name=args.dataset + '.mat')  # 格式: [(graph,label),...,(graph,label)]

    input_dim = dataset[0].x.size(1)
    num_classes = dataset.num_classes

    print("data 0 ", dataset[0])

    # TODO 获取 train, test, val 的划分索引
    split_idx = dataset.get_idx_split()  # TODO .get_idx_split() 里必须已经切分好 train、val、test

    # TODO 转成 mini-batch 格式，便于训练
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)

    # TODO 一个简单的 GNN 测试模型
    model = GCN(in_feats=input_dim, hidden_channels= args.hidden_channels,out_feats=num_classes,num_layers=3)
    model.to(device)
    print(model)
    model.reset_parameters()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    # train_f1_macro_curve = []
    # train_f1_micro_curve = []
    valid_f1_macro_curve = []
    valid_f1_micro_curve = []
    test_f1_macro_curve = []
    test_f1_micro_curve = []
    valid_acc_curve = []
    test_acc_curve = []

    # Training loop with added print and progress
    num_batch = len(train_loader)

    model.train()
    for epoch in range(args.epochs):
        # model.train()  # Set model to training mode
        for iter, batched_graph in tqdm(enumerate(train_loader), desc='Epoch: [{}]'.format(epoch),
                                        total=num_batch, ncols=60):
            batched_graph = batched_graph.to(device)

            logits = model.forward(batched_graph.x, batched_graph.edge_index, batched_graph.batch)
            loss = criterion(logits, batched_graph.y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Compute metrics after each epoch
        valid_f1_macro = evaluate_metrics(model, device, valid_loader, 'f1-macro')
        valid_f1_micro = evaluate_metrics(model, device, valid_loader, 'f1-micro')
        test_f1_macro = evaluate_metrics(model, device, test_loader, 'f1-macro')
        test_f1_micro = evaluate_metrics(model, device, test_loader, 'f1-micro')
        valid_acc = evaluate_metrics(model, device, valid_loader, 'acc')
        test_acc = evaluate_metrics(model, device, test_loader, 'acc')
        print('Valid Acc: {:.2f}, Test Acc: {:.2f}'.format(valid_acc * 100, test_acc * 100))

        # Print the current metrics
        print(f'Epoch [{epoch + 1}/{args.epochs}]')
        print(f'  Train F1 Macro: -  Train F1 Micro: -')  # This can be added if you re-enable training metrics
        print(f'  Valid F1 Macro: {valid_f1_macro:.4f}, F1 Micro: {valid_f1_micro:.4f}')
        print(f'  Test F1 Macro: {test_f1_macro:.4f}, F1 Micro: {test_f1_micro:.4f}')

        valid_f1_macro_curve.append(valid_f1_macro)
        valid_f1_micro_curve.append(valid_f1_micro)
        test_f1_macro_curve.append(test_f1_macro)
        test_f1_micro_curve.append(test_f1_micro)
        valid_acc_curve.append(valid_acc)
        test_acc_curve.append(test_acc)


    # Best model selection
    best_val_epoch = np.argmax(np.array(valid_acc_curve))
    print('Finished training!')
    print(
        'Best f1-macro validation score: {:.2f}, Test score: {:.2f}'.format(valid_f1_macro_curve[best_val_epoch] * 100,
                                                                            test_f1_macro_curve[best_val_epoch] * 100))
    print(
        'Best f1-micro validation score: {:.2f}, Test score: {:.2f}'.format(valid_f1_micro_curve[best_val_epoch] * 100,
                                                                            test_f1_micro_curve[best_val_epoch] * 100))
    print('Best acc validation score: {:.2f}, Test score: {:.2f}'.format(valid_acc_curve[best_val_epoch] * 100,
                                                                              test_acc_curve[best_val_epoch] * 100))

    torch.save(model.state_dict(), 'GCN_model.pth')