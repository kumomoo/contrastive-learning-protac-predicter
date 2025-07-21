# ===================================================
# IMPORTING PACKAGES
# ===================================================

import numpy as np
import random
import torch
import torch_geometric.transforms as T
import time
import os
import matplotlib.pyplot as plt


from arguments import arg_parse 
from datetime import datetime
from qm9 import QM9
from transform import Complete
from pathlib import Path
from os.path import basename
from torch_geometric.data import DataLoader


# ===================================================
# BASIC FUNCTIONS
# ===================================================

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, target]
        return data


if __name__ == '__main__':
  
    seed_everything()
    max_len = 38
    char_indices = Complete().get_char_indices()
    nchars = len(char_indices)

    args = arg_parse()

    target = args.target
    epochs = args.epochs
    sup = args.sup
    lr_decay = args.lr_decay
    lr = args.lr
    weight_decay = args.weight_decay
    bidirectional = args.bidirectional
    num_layers = args.num_layers
    temperature = args.temperature
    batch_size = args.batch_size
    sup_size = args.sup_size
    dataset_path = args.dataset_path
    output = args.output
    if not Path(output).exists():
        Path(output).mkdir(parents=True, exist_ok=True)
    load_weights = args.load_weights
    if not sup:
        aug_ratio = args.percent
        node_drop = args.node_drop
        subgraph = args.subgraph
        edge_perturbation = args.edge_perturbation
        attribute_masking = args.attribute_masking
        enumeration = args.enumeration
        smiles_noise = args.smiles_noise
        masking_pos = args.masking_pos
        xyz_pertub = args.xyz_pertub
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('Using {} for training ...'.format(device))


    from ProtacEncoder import *
    dataset = QM9(
        dataset_path,
        transform=T.Compose(
            [
            MyTransform(),
            Complete(
                True, node_drop, subgraph, edge_perturbation, attribute_masking,
                masking_pos, enumeration, smiles_noise, aug_ratio, xyz_pertub, max_len
            ),
            T.Distance(norm=False)
            ]
        )
    ).shuffle()

    # Normalize targets to mean = 0 and std = 1.
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean[:, target].item(), std[:, target].item()

    # # Split datasets.
    # test_dataset = dataset[:10000]
    # val_dataset = dataset[10000:20000]
    
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_dataset = dataset 


    model = Net(
        dataset.num_features, 64, vocab_size=nchars, max_len=max_len,
        padding_idx=char_indices[' '], embedding_dim=32,
        num_layers=num_layers, bidirectional=bidirectional
    ).to(device)
    # model = Net(
    #     dataset.num_features, 64, vocab_size=nchars, max_len=max_len,
    #     padding_idx=char_indices[' ']
    # ).to(device)

    if load_weights:

        weights = torch.load('{}/model_UNSUP.pth'.format(load_weights))
        model.load_state_dict(weights)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if lr_decay:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=10,min_lr=0.00001
        )
    log_file = "{}/history.log".format(output)

    val_loss = []
    train_loss = []
    best_val_error = None
    best_train_error = None
    best_loss = None
    for epoch in range(1, epochs):
        if lr_decay:
            lr = scheduler.optimizer.param_groups[0]['lr']
        init = time.time()
        loss = train(model, train_loader, optimizer, batch_size, temperature, device, output)
        final = time.time() - init  
        
        train_loss.append(loss)

        if best_loss is None or loss < best_loss:
            torch.save(model.state_dict(), '{}/best_model.pth'.format(output)) 
            print(f"Best unsupervised model saved at epoch {epoch} with loss {loss}")
            best_loss = loss

        msg = ('Epoch: {:03d}, LR: {:7f}, Loss: {}, time: {}'.format(epoch, lr, loss, time.strftime("%H:%M:%S", time.gmtime(final))))
        print(msg)

        if lr_decay:
            scheduler.step(val_error)
        
        with open(log_file, 'a') as f:
            f.write(msg+ '\n')
      
    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('{}/loss_per_epochs.png'.format(output))

