import pandas as pd
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torchinfo import summary
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from src.pytorchtools import EarlyStopping
from src.models import CustomResnet18, CustomResnet18_inprocess, Net, Net_inprocess
from CONSTANTS import BATCH_SIZE, EPOCHS, learning_rate, momentum, dataset, method, train_datapath, val_datapath, test_datapath, seed_size, del_size, al_steps, lambd_in, lambd_al
from src.datasets import MyDataset
from src.data_loaders import get_data_loaders
from src.metrics import acc_and_f1
from src.selection_fn import select_random_from_unlabeled, select_entropy_uncertainity_from_unlabeled, select_entropy_uncertainity_and_fairness_from_unlabeled_gt_celeba, select_entropy_uncertainity_and_fairness_from_unlabeled_gt_compas


def train(network, optimizer, n_epochs, train_data_loader, val_data_loader, device):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    early_stopping = EarlyStopping(patience=3, verbose=True, path='models/'+dataset+'_'+method+'_checkpoint.pt')

    for epoch in range(1, n_epochs + 1):
        network.train()
        if method not in ('mo-ed', 'mo-al'):
            for batch_idx, (data, target) in enumerate(train_data_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = network(data, 0)
                loss = F.nll_loss(output.log(), target)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
        else:
            if dataset == 'celebA':
                for batch_idx, (data, sensitive1, sensitive2, target) in enumerate(train_data_loader):
                    data, sensitive1, sensitive2, target = data.to(device), sensitive1.to(device), sensitive2.to(device), target.to(device)
                    optimizer.zero_grad()
                    output1, output2 = network(data, sensitive1, sensitive2, 0)
                    loss1 = F.nll_loss(output1.log(), target)
                    loss2 = F.nll_loss(output2.log(), 1 - target)
                    loss = lambd_in * loss1 + (1 - lambd_in) * loss2
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())
            elif dataset == 'compas':
                for batch_idx, (data, sensitive1, target) in enumerate(train_data_loader):
                    data, sensitive1, target = data.to(device), sensitive1.to(device), target.to(device)
                    optimizer.zero_grad()
                    output1, output2 = network(data, sensitive1, 0)
                    loss1 = F.nll_loss(output1.log(), target)
                    loss2 = F.nll_loss(output2.log(), 1 - target)
                    loss = lambd_in * loss1 + (1 - lambd_in) * loss2
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

        network.eval() # prep model for evaluation
        if method not in ('mo-ed', 'mo-al'):
            for data, target in val_data_loader:
                data, target = data.to(device), target.to(device)
                output = network(data, 0)
                loss = F.nll_loss(output.log(), target)
                valid_losses.append(loss.item())
        else:
            if dataset == 'celebA':
                for data, sensitive1, sensitive2, target in val_data_loader:
                    data, sensitive1, sensitive2, target = data.to(device), sensitive1.to(device), sensitive2.to(device), target.to(device)
                    output1, output2 = network(data, sensitive1, sensitive2, 0)
                    loss1 = F.nll_loss(output1.log(), target)
                    loss2 = F.nll_loss(output2.log(), 1 - target)
                    loss = lambd_in * loss1 + (1 - lambd_in) * loss2
                    valid_losses.append(loss.item())
            elif dataset == 'compas':
                for data, sensitive1, target in val_data_loader:
                    data, sensitive1, target = data.to(device), sensitive1.to(device), target.to(device)
                    output1, output2 = network(data, sensitive1, 0)
                    loss1 = F.nll_loss(output1.log(), target)
                    loss2 = F.nll_loss(output2.log(), 1 - target)
                    loss = lambd_in * loss1 + (1 - lambd_in) * loss2
                    valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)
        train_losses = []
        valid_losses = []
        early_stopping(valid_loss, network)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    network.load_state_dict(torch.load('models/'+dataset+'_'+method+'_checkpoint.pt'))

    return network

def select_entropy_uncertainity_and_fairness_from_unlabeled_gt(u_train, l_train, del_size, model, device):
    model.eval()
    TRANSFORM_IMG = transforms.Compose([
           transforms.Resize(224),
           transforms.ToTensor()])

    u_group_0 = u_train[u_train['Male'] == 0]
    u_group_1 = u_train[u_train['Male'] == 1]
    l_group_0 = l_train[l_train['Male'] == 0]
    l_group_1 = l_train[l_train['Male'] == 1]
    dataset_train = MyDataset(u_group_0, TRANSFORM_IMG)
    train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
    aa = None
    bb = None
    if method != "mo-al":
        for inp, tar in train_data_loader:
            inp = inp.to(device)
            otpt = model(inp)
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy() 
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)
    else:
        for inp, sensitive1, sensitive2, tar in train_data_loader:
            inp = inp.to(device)
            sensitive1 = sensitive1.to(device)
            sensitive2 = sensitive2.to(device)
            otpt, _ = model(inp, sensitive1, sensitive2)
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)

    u_group_0['pred_proba'] = aa
    u_group_0['pred_y'] = bb
    #for index, row in u_group_0.iterrows():
    #    if row['pred_y'] == 0:
    #        u_group_0.loc[index, 'pred_proba'] = 1 - u_group_0.loc[index, 'pred_proba']
    u_group_0 = u_group_0.sort_values('pred_proba')
    u_group_0 = u_group_0.drop(['pred_proba', 'pred_y'], axis=1)
    u_group_0 = u_group_0[:del_size]

    dataset_train = MyDataset(u_group_1, TRANSFORM_IMG)
    train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
    aa = None
    bb = None

    if method != "mo-al":
        for inp, tar in train_data_loader:
            inp = inp.to(device)
            otpt = model(inp)
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)
    else:
        for inp, sensitive1, sensitive2, tar in train_data_loader:
            inp = inp.to(device)
            sensitive1 = sensitive1.to(device)
            sensitive2 = sensitive2.to(device)
            otpt, _ = model(inp, sensitive1, sensitive2)
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)

    u_group_1['pred_proba'] = aa
    u_group_1['pred_y'] = bb
    #for index, row in u_group_1.iterrows():
    #    if row['pred_y'] == 0:
    #        u_group_1.loc[index, 'pred_proba'] = 1 - u_group_1.loc[index, 'pred_proba']
    u_group_1 = u_group_1.sort_values('pred_proba')
    u_group_1 = u_group_1.drop(['pred_proba', 'pred_y'], axis=1)
    u_group_1 = u_group_1[:del_size]

    batches_male = []
    ul_batches_male = []
    for _ in range(30):
        t0 = u_group_0.sample(n=int(del_size/4), replace=False)
        try:
            t1 = u_group_1.sample(n=int(del_size/4), replace=False)
        except:
            t1 = u_group_1
            print('empty')
        batches_male.append(pd.concat([t0, t1], ignore_index=True))
        z0 = u_train.drop(t0.index.append(t1.index))
        ul_batches_male.append(z0)

    l0_male = l_group_0.sample(n=int(del_size/4), replace=True).reset_index(drop=True)
    l1_male = l_group_1.sample(n=int(del_size/4), replace=True).reset_index(drop=True)
    dataset_train = MyDataset(l0_male, TRANSFORM_IMG)
    train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
    aa = None
    bb = None

    if method != "mo-al":
        for inp, tar in train_data_loader:
            inp = inp.to(device)
            otpt = model(inp)
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)
    else:
        for inp, sensitive1, sensitive2, tar in train_data_loader:
            inp = inp.to(device)
            sensitive1 = sensitive1.to(device)
            sensitive2 = sensitive2.to(device)
            otpt, _ = model(inp, sensitive1, sensitive2)
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)
    l0_male['pred_proba'] = aa
    l0_male['pred_y'] = bb
    for index, row in l0_male.iterrows():
        if row['pred_y'] == 0:
            l0_male.loc[index, 'pred_proba'] = 1 - l0_male.loc[index, 'pred_proba']

    dataset_train = MyDataset(l1_male, TRANSFORM_IMG)
    train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
    aa = None
    bb = None

    if method != "mo-al":
        for inp, tar in train_data_loader:
            inp = inp.to(device)
            otpt = model(inp)
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)
    else:
        for inp, sensitive1, sensitive2, tar in train_data_loader:
            inp = inp.to(device)
            sensitive1 = sensitive1.to(device)
            sensitive2 = sensitive2.to(device)
            otpt, _ = model(inp, sensitive1, sensitive2)
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)
    l1_male['pred_proba'] = aa
    l1_male['pred_y'] = bb
    for index, row in l1_male.iterrows():
        if row['pred_y'] == 0:
            l1_male.loc[index, 'pred_proba'] = 1 - l1_male.loc[index, 'pred_proba']

    # Eyeglasses
    u_group_0 = u_train[u_train['Eyeglasses'] == 0]
    u_group_1 = u_train[u_train['Eyeglasses'] == 1]
    l_group_0 = l_train[l_train['Eyeglasses'] == 0]
    l_group_1 = l_train[l_train['Eyeglasses'] == 1]
    dataset_train = MyDataset(u_group_0, TRANSFORM_IMG)
    train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
    aa = None
    bb = None

    if method != "mo-al":
        for inp, tar in train_data_loader:
            inp = inp.to(device)
            otpt = model(inp)
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)
    else:
        for inp, sensitive1, sensitive2, tar in train_data_loader:
            inp = inp.to(device)
            sensitive1 = sensitive1.to(device)
            sensitive2 = sensitive2.to(device)
            otpt, _ = model(inp, sensitive1, sensitive2)
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)
    u_group_0['pred_proba'] = aa
    u_group_0['pred_y'] = bb
    #for index, row in u_group_0.iterrows():
    #    if row['pred_y'] == 0:
    #        u_group_0.loc[index, 'pred_proba'] = 1 - u_group_0.loc[index, 'pred_proba']
    u_group_0 = u_group_0.sort_values('pred_proba')
    u_group_0 = u_group_0.drop(['pred_proba', 'pred_y'], axis=1)
    u_group_0 = u_group_0[:del_size]

    dataset_train = MyDataset(u_group_1, TRANSFORM_IMG)
    train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
    aa = None
    bb = None

    if method != "mo-al":
        for inp, tar in train_data_loader:
            inp = inp.to(device)
            otpt = model(inp)
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)
    else:
        for inp, sensitive1, sensitive2, tar in train_data_loader:
            inp = inp.to(device)
            sensitive1 = sensitive1.to(device)
            sensitive2 = sensitive2.to(device)
            otpt, _ = model(inp, sensitive1, sensitive2)
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)
    u_group_1['pred_proba'] = aa
    u_group_1['pred_y'] = bb
    #for index, row in u_group_1.iterrows():
    #    if row['pred_y'] == 0:
    #        u_group_1.loc[index, 'pred_proba'] = 1 - u_group_1.loc[index, 'pred_proba']
    u_group_1 = u_group_1.sort_values('pred_proba')
    u_group_1 = u_group_1.drop(['pred_proba', 'pred_y'], axis=1)
    u_group_1 = u_group_1[:del_size]

    batches_eye = []
    ul_batches_eye = []
    for _ in range(30):
        t0 = u_group_0.sample(n=int(del_size/4), replace=False)
        try:
            t1 = u_group_1.sample(n=int(del_size/4), replace=False)
        except:
            t1 = u_group_1
            print('empty')
        batches_eye.append(pd.concat([t0, t1], ignore_index=True))
        z0 = u_train.drop(t0.index.append(t1.index))
        ul_batches_eye.append(z0)

    l0_eye = l_group_0.sample(n=int(del_size/4), replace=True).reset_index(drop=True)
    l1_eye = l_group_1.sample(n=int(del_size/4), replace=True).reset_index(drop=True)
    dataset_train = MyDataset(l0_eye, TRANSFORM_IMG)
    train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
    aa = None
    bb = None

    if method != "mo-al":
        for inp, tar in train_data_loader:
            inp = inp.to(device)
            otpt = model(inp)
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)
    else:
        for inp, sensitive1, sensitive2, tar in train_data_loader:
            inp = inp.to(device)
            sensitive1 = sensitive1.to(device)
            sensitive2 = sensitive2.to(device)
            otpt, _ = model(inp, sensitive1, sensitive2)
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)
    l0_eye['pred_proba'] = aa
    l0_eye['pred_y'] = bb
    for index, row in l0_eye.iterrows():
        if row['pred_y'] == 0:
            l0_eye.loc[index, 'pred_proba'] = 1 - l0_eye.loc[index, 'pred_proba']

    dataset_train = MyDataset(l1_eye, TRANSFORM_IMG)
    train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
    aa = None
    bb = None

    if method != "mo-al":
        for inp, tar in train_data_loader:
            inp = inp.to(device)
            otpt = model(inp)
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)
    else:
        for inp, sensitive1, sensitive2, tar in train_data_loader:
            inp = inp.to(device)
            sensitive1 = sensitive1.to(device)
            sensitive2 = sensitive2.to(device)
            otpt, _ = model(inp, sensitive1, sensitive2)
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            if aa is None:
                aa = logit
                bb = pred_y
            else:
                aa = np.append(aa, logit)
                bb = np.append(bb, pred_y)
    l1_eye['pred_proba'] = aa
    l1_eye['pred_y'] = bb
    for index, row in l1_eye.iterrows():
        if row['pred_y'] == 0:
            l1_eye.loc[index, 'pred_proba'] = 1 - l1_eye.loc[index, 'pred_proba']


    batches = [pd.concat([ii,jj]).drop_duplicates().reset_index(drop=True) for ii,jj in zip(batches_male, batches_eye)]
    l0 = pd.concat([l0_male,l0_eye]).drop_duplicates().reset_index(drop=True)
    l1 = pd.concat([l1_male,l1_eye]).drop_duplicates().reset_index(drop=True)

    scores_uncer = []
    scores_fair = []
    for batch in batches:
        dataset_train = MyDataset(batch, TRANSFORM_IMG)
        train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
        aa = None
        bb = None
        if method != "mo-al":
            for inp, tar in train_data_loader:
                inp = inp.to(device)
                otpt = model(inp)
                logit = torch.max(otpt, 1)[0].data.squeeze()
                logit = logit.cpu().detach().numpy()
                pred_y = torch.max(otpt, 1)[1].data.squeeze()
                pred_y = pred_y.cpu().detach().numpy()
                if aa is None:
                    aa = logit
                    bb = pred_y
                else:
                    aa = np.append(aa, logit)
                    bb = np.append(bb, pred_y)
        else:
            for inp, sensitive1, sensitive2, tar in train_data_loader:
                inp = inp.to(device)
                sensitive1 = sensitive1.to(device)
                sensitive2 = sensitive2.to(device)
                otpt, _ = model(inp, sensitive1, sensitive2)
                pred_y = torch.max(otpt, 1)[1].data.squeeze()
                pred_y = pred_y.cpu().detach().numpy()
                logit = torch.max(otpt, 1)[0].data.squeeze()
                logit = logit.cpu().detach().numpy()
                if aa is None:
                    aa = logit
                    bb = pred_y
                else:
                    aa = np.append(aa, logit)
                    bb = np.append(bb, pred_y)
        batch['pred_proba'] = aa
        batch['pred_y'] = bb
        for index, row in batch.iterrows():
            if row['pred_y'] == 0:
                batch.loc[index, 'pred_proba'] = 1 - batch.loc[index, 'pred_proba']
        gg1 = l0.sample(n=8*del_size, replace=True).reset_index(drop=True)
        try:
            g2_m = batch[batch['Male'] == 1]
            g2_e = batch[batch['Eyeglasses'] == 1]
            gg2_m = g2_m.sample(n=8*del_size, replace=True).reset_index(drop=True)
            gg2_e = g2_e.sample(n=8*del_size, replace=True).reset_index(drop=True)
            summation1_m = np.sum(np.multiply((gg1['Young'] == gg2_m['Young']), (gg1['pred_proba'] - gg2_m['pred_proba'])**2))
            summation1_m /= (8*del_size)
            summation1_e = np.sum(np.multiply((gg1['Young'] == gg2_e['Young']), (gg1['pred_proba'] - gg2_e['pred_proba'])**2))
            summation1_e /= (8*del_size)
        except:
            summation1 = 0
            print('Alert!')

        gg1 = l1.sample(n=8*del_size, replace=True).reset_index(drop=True)
        g2_m = batch[batch['Male'] == 0]
        g2_e = batch[batch['Eyeglasses'] == 0]
        gg2_m = g2_m.sample(n=8*del_size, replace=True).reset_index(drop=True)
        gg2_e = g2_e.sample(n=8*del_size, replace=True).reset_index(drop=True)
        summation2_m = np.sum(np.multiply((gg1['Young'] == gg2_m['Young']), (gg1['pred_proba'] - gg2_m['pred_proba'])**2))
        summation2_e = np.sum(np.multiply((gg1['Young'] == gg2_e['Young']), (gg1['pred_proba'] - gg2_e['pred_proba'])**2))
        summation2_m /= (8*del_size)
        summation2_e /= (8*del_size)

        summation = 0.5 * (summation1_m + summation2_m) + 0.5 * (summation1_e + summation2_e)
        #print('Summation1:', summation1)
        #print('Summation2:', summation2)
        #print(summation)
        scores_fair.append(summation)
        #print(aa.mean())
        scores_uncer.append(aa.mean())
        #scores.append(10*summation+aa.mean())
    print(scores_fair)
    print(scores_uncer)
    scores_fair = (scores_fair - min(scores_fair)) / (max(scores_fair) - min(scores_fair))
    scores_uncer = (scores_uncer - min(scores_uncer)) / (max(scores_uncer) - min(scores_uncer))
    print(scores_fair)
    print(scores_uncer)
    scores = [lambd_al * x[0] + (1-lambd_al) * x[1] for x in zip(scores_fair, scores_uncer)]
    final = batches[np.argmin(scores)]
    ul_batch_male = ul_batches_male[np.argmin(scores)]
    ul_batch_eye = ul_batches_eye[np.argmin(scores)]
    ul_batch = pd.concat([ul_batch_male, ul_batch_eye]).drop_duplicates().reset_index(drop=True)
        #scores.append(aa.sum())
    return final, ul_batch    

def get_al_evaluation(df_train, val_data_loader, test_data_loader, df_train_unlabelled, device):
    TRANSFORM_IMG = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()])

    for i in range(al_steps+1):
        print("|S|_{} = {}, |U|_{} = {}".format(i, len(df_train), i, len(df_train_unlabelled)))
        if dataset == 'celebA':
            dataset_train = MyDataset(df_train, TRANSFORM_IMG)
            train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=1)
            if method != 'mo-al':
                network = CustomResnet18()
            else:
                network = CustomResnet18_inprocess()
        elif dataset == 'compas':
            dataset_train = MyDataset(df_train)
            train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=1)
            if method != 'mo-al':
                network = Net()
            else:
                network = Net_inprocess()

        #network = CustomResnet18()
        network.to(device)
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
        model = train(network, optimizer, EPOCHS, train_data_loader, val_data_loader, device)
        if i != (al_steps):
            if method == "rs":
                del_train, df_train_unlabelled = select_random_from_unlabeled(df_train_unlabelled)
            elif method == "us":
                del_train, df_train_unlabelled = select_entropy_uncertainity_from_unlabeled(df_train_unlabelled, model, device)
            elif method == "al" or method == "fp-o" or method == 'mo-al':
                if dataset == 'celebA':
                    del_train, df_train_unlabelled = select_entropy_uncertainity_and_fairness_from_unlabeled_gt_celeba(df_train_unlabelled, df_train, del_size, model, device)
                elif dataset == 'compas':
                    del_train, df_train_unlabelled = select_entropy_uncertainity_and_fairness_from_unlabeled_gt_compas(df_train_unlabelled, df_train, del_size, model, device)
                    
            df_train = pd.concat([df_train, del_train], ignore_index=True)
        model.eval()
        acc_and_f1(model, test_data_loader, device)
    return model

def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    torch.cuda.set_device(1)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)

    if dataset == 'celebA':
        if method not in ('mo-ed', 'mo-al'):
            network = CustomResnet18()
        else:
            network = CustomResnet18_inprocess()
    elif dataset == 'compas':
        if method not in ('mo-ed', 'mo-al'):
            network = Net()
        else:
            network = Net_inprocess()
    network.to(device)
    print(network)
    #print(summary(network, input_size=(BATCH_SIZE, 3, 224, 224)))
    #print(summary(network, input_size=(BATCH_SIZE, 15)))

    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    if method in ('ed', 'rs-b', 'mo-ed'):
        train_data_loader, val_data_loader, test_data_loader = get_data_loaders()
        model = train(network, optimizer, EPOCHS, train_data_loader, val_data_loader, device)
    else:
        df_train, val_data_loader, test_data_loader, df_train_unlabelled = get_data_loaders()
        model = get_al_evaluation(df_train, val_data_loader, test_data_loader, df_train_unlabelled, device) 

    model.eval()

    acc_and_f1(model, test_data_loader, device)
    #print("Final Test Accuracy: " + str(test_acc))
    #print("Final Test F1: " + str(test_f1))
    #get_al_evaluation(df_train, val_data_loader, test_data_loader, df_train_unlabelled, del_size, al_steps, "random")
    #get_al_evaluation(df_train, val_data_loader, test_data_loader, df_train_unlabelled, del_size, al_steps, "uncertainty")
    #get_al_evaluation(df_train, val_data_loader, test_data_loader, df_train_unlabelled, del_size, al_steps, "uncertainty_with_fairness2")
    #get_al_evaluation(df_train, val_data_loader, test_data_loader, df_train_unlabelled, del_size, al_steps, "uncertainty_with_fairness_gt")


if __name__ == '__main__':
    main()

