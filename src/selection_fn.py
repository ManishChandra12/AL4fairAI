from CONSTANTS import del_size
from torchvision import transforms
import torch
from src.datasets import MyDataset
from CONSTANTS import BATCH_SIZE, lambd_al, method
import pandas as pd
import numpy as np

def select_random_from_unlabeled(u_train):
    del_train = u_train.sample(n=del_size, random_state=42)
    unlab_train = u_train.drop(del_train.index)
    del_train = del_train.reset_index(drop=True)
    unlab_train = unlab_train.reset_index(drop=True)
    return del_train, unlab_train

def select_entropy_uncertainity_from_unlabeled(u_train, model, device):
    model.eval()
    TRANSFORM_IMG = transforms.Compose([
           transforms.Resize(224),
           transforms.ToTensor()])
    dataset_utrain = MyDataset(u_train, TRANSFORM_IMG)
    utrain_data_loader = torch.utils.data.DataLoader(dataset_utrain, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
    aa = None
    for inp, tar in utrain_data_loader:
        inp = inp.to(device)
        otpt = model(inp)
        pred_y = torch.max(otpt, 1)[0].data.squeeze()
        pred_y = pred_y.cpu().detach().numpy()
        if aa is None:
            aa = pred_y
        else:
            aa = np.append(aa, pred_y)
    u_train['pred_proba'] = aa
    u_train = u_train.sort_values('pred_proba').reset_index(drop=True)
    print(u_train['pred_proba'].head())
    u_train = u_train.drop('pred_proba', axis=1)
    del_train = u_train[:del_size]
    unlab_train = u_train[del_size:]
    del_train = del_train.reset_index(drop=True)
    unlab_train = unlab_train.reset_index(drop=True)
    return del_train, unlab_train

def select_entropy_uncertainity_and_fairness_from_unlabeled_gt_celeba(u_train, l_train, del_size, model, device):
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

def select_entropy_uncertainity_and_fairness_from_unlabeled_gt_compas(u_train, l_train, del_size, model, device):
    model.eval()

    u_group_0 = u_train[u_train['African-American'] == 0]
    u_group_1 = u_train[u_train['African-American'] == 1]
    l_group_0 = l_train[l_train['African-American'] == 0]
    l_group_1 = l_train[l_train['African-American'] == 1]
    dataset_train = MyDataset(u_group_0)
    train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
    aa = None
    if method != "mo-al":
        for inp, tar in train_data_loader:
            inp = inp.to(device)
            otpt = model(inp)
            pred_y = torch.max(otpt, 1)[0].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            if aa is None:
                aa = pred_y
            else:
                aa = np.append(aa, pred_y)
    else:
        for inp, sensitive1, tar in train_data_loader:
            inp = inp.to(device)
            sensitive1 = sensitive1.to(device)
            otpt, _ = model(inp, sensitive1)
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            if aa is None:
                aa = logit
            else:
                aa = np.append(aa, logit)
    u_group_0['pred_proba'] = aa
    u_group_0 = u_group_0.sort_values('pred_proba')
    u_group_0 = u_group_0.drop('pred_proba', axis=1)
    u_group_0 = u_group_0[:del_size]

    dataset_train = MyDataset(u_group_1)
    train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
    aa = None
    if method != "mo-al":
        for inp, tar in train_data_loader:
            inp = inp.to(device)
            otpt = model(inp)
            pred_y = torch.max(otpt, 1)[0].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            if aa is None:
                aa = pred_y
            else:
                aa = np.append(aa, pred_y)
    else:
        for inp, sensitive1, tar in train_data_loader:
            inp = inp.to(device)
            sensitive1 = sensitive1.to(device)
            otpt, _ = model(inp, sensitive1)
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            logit = torch.max(otpt, 1)[0].data.squeeze()
            logit = logit.cpu().detach().numpy()
            if aa is None:
                aa = logit
            else:
                aa = np.append(aa, logit)
    u_group_1['pred_proba'] = aa
    u_group_1 = u_group_1.sort_values('pred_proba')
    u_group_1 = u_group_1.drop('pred_proba', axis=1)
    u_group_1 = u_group_1[:del_size]

    batches = []
    ul_batches = []
    for _ in range(30):
        t0 = u_group_0.sample(n=int(del_size/2), replace=False)
        try:
            t1 = u_group_1.sample(n=int(del_size/2), replace=False)
        except:
            t1 = u_group_1
            print('empty')
        batches.append(pd.concat([t0, t1], ignore_index=True))
        z0 = u_train.drop(t0.index.append(t1.index))
        ul_batches.append(z0)

    l0 = l_group_0.sample(n=int(del_size/2), replace=True).reset_index(drop=True)
    l1 = l_group_1.sample(n=int(del_size/2), replace=True).reset_index(drop=True)
    dataset_train = MyDataset(l0)
    train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
    aa = None
    bb = None
    if method != "mo-al":
        for inp, tar in train_data_loader:
            inp = inp.to(device)
            otpt = model(inp)
            pred_prob = torch.max(otpt, 1)[0].data.squeeze()
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            pred_prob = pred_prob.cpu().detach().numpy()
            if aa is None:
                aa = pred_prob
                bb = pred_y
            else:
                aa = np.append(aa, pred_prob)
                bb = np.append(bb, pred_y)
    else:
        for inp, sensitive1, tar in train_data_loader:
            inp = inp.to(device)
            sensitive1 = sensitive1.to(device)
            otpt, _ = model(inp, sensitive1)
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
    l0['pred_proba'] = aa
    l0['pred_y'] = bb
    for index, row in l0.iterrows():
        if row['pred_y'] == 0:
            l0.loc[index, 'pred_proba'] = 1 - l0.loc[index, 'pred_proba']

    dataset_train = MyDataset(l1)
    train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
    aa = None
    bb = None
    if method != "mo-al":
        for inp, tar in train_data_loader:
            inp = inp.to(device)
            otpt = model(inp)
            pred_prob = torch.max(otpt, 1)[0].data.squeeze()
            pred_y = torch.max(otpt, 1)[1].data.squeeze()
            pred_y = pred_y.cpu().detach().numpy()
            pred_prob = pred_prob.cpu().detach().numpy()
            if aa is None:
                aa = pred_prob
                bb = pred_y
            else:
                aa = np.append(aa, pred_prob)
                bb = np.append(bb, pred_y)
    else:
        for inp, sensitive1, tar in train_data_loader:
            inp = inp.to(device)
            sensitive1 = sensitive1.to(device)
            otpt, _ = model(inp, sensitive1)
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
    l1['pred_proba'] = aa
    l1['pred_y'] = bb
    for index, row in l1.iterrows():
        if row['pred_y'] == 0:
            l1.loc[index, 'pred_proba'] = 1 - l1.loc[index, 'pred_proba']

    scores_uncer = []
    scores_fair = []
    for batch in batches:
        dataset_train = MyDataset(batch)
        train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
        aa = None
        bb = None
        if method != "mo-al":
            for inp, tar in train_data_loader:
                inp = inp.to(device)
                otpt = model(inp)
                pred_prob = torch.max(otpt, 1)[0].data.squeeze()
                pred_y = torch.max(otpt, 1)[1].data.squeeze()
                pred_y = pred_y.cpu().detach().numpy()
                pred_prob = pred_prob.cpu().detach().numpy()
                if aa is None:
                    aa = pred_prob
                    bb = pred_y
                else:
                    aa = np.append(aa, pred_prob)
                    bb = np.append(bb, pred_y)
        else:
            for inp, sensitive1, tar in train_data_loader:
                inp = inp.to(device)
                sensitive1 = sensitive1.to(device)
                otpt, _ = model(inp, sensitive1)
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
            g2 = batch[batch['African-American'] == 1]
            gg2 = g2.sample(n=8*del_size, replace=True).reset_index(drop=True)
            summation1 = np.sum(np.multiply((gg1['is_violent_recid'] == gg2['is_violent_recid']), (gg1['pred_proba'] - gg2['pred_proba'])**2))
            summation1 /= (8*del_size)
        except:
            summation1 = 0
            print('Alert!')

        gg1 = l1.sample(n=8*del_size, replace=True).reset_index(drop=True)
        g2 = batch[batch['African-American'] == 0]
        gg2 = g2.sample(n=8*del_size, replace=True).reset_index(drop=True)
        summation2 = np.sum(np.multiply((gg1['is_violent_recid'] == gg2['is_violent_recid']), (gg1['pred_proba'] - gg2['pred_proba'])**2))
        summation2 /= (8*del_size)

        summation = summation1 + summation2
        #print('Summation1:', summation1)
        #print('Summation2:', summation2)
        #print(summation)
        scores_fair.append(summation)
        #print(aa.mean())
        scores_uncer.append(aa.mean())
        #scores.append(10*summation+aa.mean())
    scores_fair = (scores_fair - min(scores_fair)) / (max(scores_fair) - min(scores_fair))
    scores_uncer = (scores_uncer - min(scores_uncer)) / (max(scores_uncer) - min(scores_uncer))
    print(scores_fair)
    print(scores_uncer)
    scores = [lambd_al * x[0] + (1-lambd_al) * x[1] for x in zip(scores_fair, scores_uncer)]
    final = batches[np.argmin(scores)]
    ul_batch = ul_batches[np.argmin(scores)]
    final = final.drop(['pred_proba', 'pred_y'], axis=1)
        #scores.append(aa.sum())
    return final, ul_batch 
