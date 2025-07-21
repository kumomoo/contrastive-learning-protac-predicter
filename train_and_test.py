import logging
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

def valids(model, test_loader, device):
    with torch.no_grad():
        criterion = nn.BCELoss()
        model.eval()
        y_true = []
        y_pred = []
        y_score = []
        loss_list = []
        iteration = 0
        
        for data_sample in test_loader:
            # 将标签移动到 device，并转换为浮点型、调整为 (batch_size, 1) 的形状
            y = data_sample["label"].to(device).float().view(-1, 1)
            
            outputs = model(
                data_sample["target_seq"],          # 目标蛋白序列
                data_sample["e3_seq"],              # E3 蛋白序列
                data_sample["smiles"].to(device),   # SMILES 数值序列
                data_sample["graph"].to(device)     # 图数据
            )
            
            loss_val = criterion(outputs, y)
            loss_list.append(loss_val.item())
            
            y_score.extend(outputs.cpu().view(-1).tolist())
            preds = (outputs >= 0.5).long()
            y_pred.extend(preds.cpu().view(-1).tolist())
            y_true.extend(y.cpu().view(-1).tolist())
            
            iteration += 1
        
        model.train()
    
    avg_loss = sum(loss_list) / iteration if iteration > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else 0.0
    return avg_loss, acc, auroc


def train(model, lr=0.001, epoch=30, train_loader=None, valid_loader=None,
          device=None, writer=None, LOSS_NAME=None, batch_size=None):
    model = model.to(device)
    
    # 冻结 target_encoder 和 e3_encoder 的参数
    for param in model.target_encoder.parameters():
        param.requires_grad = False
    for param in model.e3_encoder.parameters():
        param.requires_grad = False

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 添加学习率衰减策略，学习率衰减为原来的 0.1 倍
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.1)
    
    criterion = nn.BCELoss()
    
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    
    _ = valids(model, valid_loader, device)
    
    running_loss = 0.0
    total_num = 0
    
    best_val_acc = 0.0
    
    for epo in range(epoch):
        model.train()
        # if epo == 35:
        #     for p in model.target_encoder.parameters():
        #         p.requires_grad = True
        #     for p in model.e3_encoder.parameters():
        #         p.requires_grad = True
        #     print(">>> Unfroze target_encoder & e3_encoder at epoch", epo)
        
        train_y_true = []
        train_y_pred = []
        train_y_score = []
        
        for data_sample in train_loader:
            outputs = model(
                data_sample["target_seq"],           # 目标蛋白序列
                data_sample["e3_seq"],               # E3 蛋白序列
                data_sample["smiles"].to(device),    # SMILES 数值序列
                data_sample["graph"].to(device)      # 图数据
            )
            
            # 将标签转换为 float 类型，并调整为 (batch_size, 1) 的形状
            y = data_sample["label"].to(device).float().view(-1, 1)
            loss = criterion(outputs, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            running_loss += loss.item()
            total_num += 1
            
            train_y_score.extend(outputs.detach().cpu().view(-1).tolist())
            preds = (outputs.detach() >= 0.5).long()
            train_y_pred.extend(preds.cpu().view(-1).tolist())
            train_y_true.extend(y.cpu().view(-1).tolist())
        
        train_loss = running_loss / (total_num if total_num else 1)
        train_loss_list.append(train_loss)
        train_acc = accuracy_score(train_y_true, train_y_pred)
        train_acc_list.append(train_acc)
        
        if writer is not None:
            writer.add_scalar(LOSS_NAME + "_train", train_loss, epo)
            writer.add_scalar(LOSS_NAME + "_train_acc", train_acc, epo)
        
        logging.info("Train epoch %d, loss: %.4f, acc: %.4f" % (epo, train_loss, train_acc))
        print("Train epoch %d, loss: %.4f, acc: %.4f" % (epo, train_loss, train_acc))
        
        val_loss, val_acc, val_auroc = valids(model, valid_loader, device)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        
        if writer is not None:
            writer.add_scalar(LOSS_NAME + "_test_loss", val_loss, epo)
            writer.add_scalar(LOSS_NAME + "_test_acc", val_acc, epo)
            writer.add_scalar(LOSS_NAME + "_test_auroc", val_auroc, epo)
        
        logging.info(f"Valid epoch {epo} loss: {val_loss:.4f}, acc: {val_acc:.4f}, auroc: {val_auroc:.4f}")
        print(f"Valid epoch {epo} loss: {val_loss:.4f}, acc: {val_acc:.4f}, auroc: {val_auroc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"model/{LOSS_NAME}_best.pt")
            logging.info(f"New best model at epoch {epo} with acc: {val_acc:.4f}")
        
        running_loss = 0.0
        total_num = 0
        
        # 更新学习率
        scheduler.step()
        if writer is not None:
            writer.add_scalar(LOSS_NAME + "_lr", scheduler.get_last_lr()[0], epo)
    
    plt.figure()
    plt.plot(range(epoch), train_loss_list, label="Train Loss", color='blue')
    plt.plot(range(epoch), val_loss_list, label="Validation Loss", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"model/{LOSS_NAME}_loss_curve.png")
    plt.show()
    
    plt.figure()
    plt.plot(range(epoch), train_acc_list, label="Train Accuracy", color='blue')
    plt.plot(range(epoch), val_acc_list, label="Validation Accuracy", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(f"model/{LOSS_NAME}_acc_curve.png")
    plt.show()
    
    return model





