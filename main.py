import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import numpy as np
import random
import time
import os
import statistics
from tqdm import tqdm
from ogb.nodeproppred import Evaluator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import BiGTexDGL
from load_data import load_arxiv
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def train_model(model, graph, features, labels, train_idx, valid_idx, test_idx, epochs=10, batch_size=6):
    evaluator = Evaluator(name='ogbn-arxiv')
    device = next(model.parameters()).device
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
    
    train_acc = []
    val_acc = []
    test_acc = []
    
    train_batch_acc = []
    train_batch_loss = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 创建训练数据的采样器
        train_sampler = dgl.dataloading.NeighborSampler([5, 2])
        train_dataloader = dgl.dataloading.DataLoader(
            graph, train_idx, train_sampler,
            batch_size=batch_size,
            shuffle=True,
        )
        
        y_pred_train = []
        y_true_train = []
        
        batch_acc = 0.0
        batch_count = 0
        
        train_iter = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs} (Train), acc {batch_acc}')
        for input_nodes, output_nodes, blocks in train_iter:
            input_features = features[input_nodes].to(device)
            output_labels = labels[input_nodes].to(device)
            
            subgragh = dgl.node_subgraph(graph, input_nodes).to(device)
            optimizer.zero_grad()
            
            logits, _ = model(subgragh, input_features, input_nodes)
            
            loss = criterion(logits, output_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            y_pred_train.append(preds.cpu())
            y_true_train.append(output_labels.cpu())
            
            batch_correct = (preds == output_labels).sum().item()
            batch_acc = batch_correct / len(output_labels)
            
            train_iter.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{batch_acc:.4f}'
            })
            
            train_batch_acc.append(batch_acc)
            train_batch_loss.append(loss.item())
        
        y_pred_train = torch.cat(y_pred_train, dim=0)
        y_true_train = torch.cat(y_true_train, dim=0)
        train_result = evaluator.eval({
            'y_true': y_true_train.unsqueeze(1),
            'y_pred': y_pred_train.unsqueeze(1)
        })
        
        # 验证阶段
        model.eval()
        
        valid_sampler = dgl.dataloading.NeighborSampler([5, 2])
        valid_dataloader = dgl.dataloading.DataLoader(
            graph, valid_idx, valid_sampler,
            batch_size=batch_size,
        )
        
        y_pred_val = []
        y_true_val = []
        
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in tqdm(valid_dataloader, desc=f'Epoch {epoch+1}/{epochs} (Valid)'):
                input_features = features[input_nodes].to(device)
                output_labels = labels[input_nodes].to(device)
                subgragh = dgl.node_subgraph(graph, input_nodes).to(device)
                logits, _ = model(subgragh, input_features, input_nodes)
                
                preds = logits.argmax(dim=1)
                y_pred_val.append(preds.cpu())
                y_true_val.append(output_labels.cpu())
        
        y_pred_val = torch.cat(y_pred_val, dim=0)
        y_true_val = torch.cat(y_true_val, dim=0)
        val_result = evaluator.eval({
            'y_true': y_true_val.unsqueeze(1),
            'y_pred': y_pred_val.unsqueeze(1)
        })
        
        test_sampler = dgl.dataloading.NeighborSampler([5, 2])
        test_dataloader = dgl.dataloading.DataLoader(
            graph, test_idx, test_sampler,
            batch_size=batch_size,
        )
        
        y_pred_test = []
        y_true_test = []
        
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in tqdm(test_dataloader, desc=f'Epoch {epoch+1}/{epochs} (Test)'):
                input_features = features[input_nodes].to(device)
                output_labels = labels[input_nodes].to(device)
                subgraph = dgl.node_subgraph(graph, input_nodes).to(device)
                
                logits, _ = model(subgraph, input_features, input_nodes)
                
                preds = logits.argmax(dim=1)
                y_pred_test.append(preds.cpu())
                y_true_test.append(output_labels.cpu())
        
        y_pred_test = torch.cat(y_pred_test, dim=0)
        y_true_test = torch.cat(y_true_test, dim=0)
        test_result = evaluator.eval({
            'y_true': y_true_test.unsqueeze(1),
            'y_pred': y_pred_test.unsqueeze(1)
        })
        
        scheduler.step(val_result['acc'])
        
        train_acc.append(train_result['acc'])
        val_acc.append(val_result['acc'])
        test_acc.append(test_result['acc'])
        
        print(f"Epoch {epoch+1}: Train Loss: {total_loss/len(train_dataloader):.4f}, "
              f"Train Acc: {train_result['acc']:.4f}, "
              f"Val Acc: {val_result['acc']:.4f}, "
              f"Test Acc: {test_result['acc']:.4f}")
        
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_batch_acc, label="Train")
    plt.xlabel('Batch')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy for each batch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_batch_loss, label="Loss")
    plt.xlabel('Batch')
    plt.ylabel('Training Loss')
    plt.title('raining Loss for each batch')
    plt.legend()

    plt.tight_layout()
    plt.savefig("train.png")
    
    print(train_acc)
    print(val_acc)
    print(test_acc)
    max_index = val_acc.index(max(val_acc))
    return train_acc[max_index], max(val_acc), test_acc[max_index]

def main():
    # 设置参数
    batch_size = 50
    epochs = 8
    num_layers = 2
    embedding_dim = 768
    num_iterate = 1
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading ogbn-arxiv dataset...")
    graph, features, labels, train_idx, valid_idx, test_idx, texts = load_arxiv()
    
    # 将数据移至设备
    graph = graph.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    valid_idx = valid_idx.to(device)
    test_idx = test_idx.to(device)
    
    # 记录多次运行的结果
    train_accs = []
    val_accs = []
    test_accs = []
    
    start_time = time.time()
    
    for i in range(num_iterate):
        print(f"\n--- Run {i+1}/{num_iterate} ---")
        
        set_seed(42 + i)
        
        model = BiGTexDGL(
            feature_dim=features.shape[1],
            text_embedding_dim=768,
            embedding_dim=embedding_dim,
            num_classes=40,
            texts=texts,
            num_gcn_layers=num_layers
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        train_acc, val_acc, test_acc = train_model(
            model, graph, features, labels, 
            train_idx, valid_idx, test_idx,
            epochs=epochs, batch_size=batch_size
        )
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        
        print(f"Run {i+1} results: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    print("\n--- Final Results ---")
    print(f"Mean Train Acc: {statistics.mean(train_accs):.4f} ± {statistics.stdev(train_accs):.4f}")
    print(f"Mean Val Acc: {statistics.mean(val_accs):.4f} ± {statistics.stdev(val_accs):.4f}")
    print(f"Mean Test Acc: {statistics.mean(test_accs):.4f} ± {statistics.stdev(test_accs):.4f}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()