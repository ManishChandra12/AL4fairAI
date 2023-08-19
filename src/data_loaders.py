import pandas as pd
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from CONSTANTS import dataset, train_datapath, method, BATCH_SIZE, seed_size
from src.datasets import MyDataset

def get_data_loaders():
    if dataset == 'celebA':
        TRANSFORM_IMG = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()])

        if method in ('ed', 'mo-ed'):
            df_train = pd.read_csv(train_datapath)
            df_val = pd.read_csv(val_datapath)
            df_test = pd.read_csv(test_datapath)
        elif method == 'rs-b':
            df_train_all = pd.read_csv(train_datapath)
            group_0 = df_train_all[(df_train_all['Eyeglasses'] == 1) & (df_train_all['Male'] == 1)]
            group_1 = df_train_all[(df_train_all['Eyeglasses'] == 1) & (df_train_all['Male'] == 0)]
            group_2 = df_train_all[(df_train_all['Eyeglasses'] == 0) & (df_train_all['Male'] == 1)]
            group_3 = df_train_all[(df_train_all['Eyeglasses'] == 0) & (df_train_all['Male'] == 0)]
            mn = min(len(group_0), len(group_1), len(group_2), len(group_3))
            group_0_s = group_0.sample(n=mn, random_state=42)
            group_1_s = group_1.sample(n=mn, random_state=42)
            group_2_s = group_2.sample(n=mn, random_state=42)
            group_3_s = group_3.sample(n=mn, random_state=42)
            df_train = pd.concat([group_0_s, group_1_s, group_2_s, group_3_s], ignore_index=True)
            df_val = pd.read_csv(val_datapath)
            df_test = pd.read_csv(test_datapath)
        elif method in ('rs', 'us', 'al', 'fp-o', 'mo-al'):
            df_train_all = pd.read_csv(train_datapath)
            df_val = pd.read_csv(val_datapath)
            df_test = pd.read_csv(test_datapath)
            df_train_male = df_train_all.groupby('Male', group_keys=False).apply(lambda x: x.sample(n=int(len(df_train_all) * seed_size/4), random_state=42))
            df_train_eye = df_train_all.groupby('Eyeglasses', group_keys=False).apply(lambda x: x.sample(n=int(len(df_train_all) * seed_size/4), random_state=42))
            df_train = pd.concat([df_train_male, df_train_eye], ignore_index=True)
            df_train_unlabelled = df_train_all.drop(df_train_male.index.union(df_train_eye.index))
            df_train = df_train.reset_index(drop=True)
            df_train_unlabelled = df_train_unlabelled.reset_index(drop=True)
        dataset_train = MyDataset(df_train, TRANSFORM_IMG)
        dataset_val = MyDataset(df_val, TRANSFORM_IMG)
        dataset_test = MyDataset(df_test, TRANSFORM_IMG)
    elif dataset == 'compas':
        if method in ('ed', 'mo-ed'):
            df_all = pd.read_csv(train_datapath)
            for cl in df_all.columns:
                if cl in ('age', 'priors_count'):
                    df_all.loc[:, cl] = (df_all.loc[:, cl] - df_all.loc[:, cl].min()) / (df_all.loc[:, cl].max() - df_all.loc[:, cl].min())
            x, df_test = train_test_split(df_all, test_size=0.2, train_size=0.8, stratify=df_all['African-American'], random_state=42)
            df_test.to_csv('test_compas.csv', index=False)
            df_train, df_val = train_test_split(x, test_size=1/8, train_size =(1-(1/8)), stratify=x['African-American'], random_state=42)
        elif method =='rs-b':
            df_all = pd.read_csv(train_datapath)
            for cl in df_all.columns:
                if cl in ('age', 'priors_count'):
                    df_all.loc[:, cl] = (df_all.loc[:, cl] - df_all.loc[:, cl].min()) / (df_all.loc[:, cl].max() - df_all.loc[:, cl].min())
            x, df_test = train_test_split(df_all, test_size=0.2, train_size=0.8, stratify=df_all['African-American'], random_state=42)
            df_test.to_csv('test_compas.csv', index=False)
            df_train_all, df_val = train_test_split(x, test_size=1/8, train_size =(1-(1/8)), stratify=x['African-American'], random_state=42)
            #df_train_all = pd.read_csv('/scratch/manish/img_align_celeba/train.csv')
            group_0 = df_train_all[df_train_all['African-American'] == 0]
            group_1 = df_train_all[df_train_all['African-American'] == 1]
            mn = min(len(group_0), len(group_1))
            group_0_s = group_0.sample(n=mn, random_state=42)
            group_1_s = group_1.sample(n=mn, random_state=42)
            df_train = pd.concat([group_1_s, group_0_s], ignore_index=True)
        elif method in ('rs', 'us', 'fp-o', 'al', 'mo-al'):
            df_all = pd.read_csv(train_datapath)
            for cl in df_all.columns:
                if cl in ('age', 'priors_count'):
                    df_all.loc[:, cl] = (df_all.loc[:, cl] - df_all.loc[:, cl].min()) / (df_all.loc[:, cl].max() - df_all.loc[:, cl].min())
            x, df_test = train_test_split(df_all, test_size=0.2, train_size=0.8, stratify=df_all['African-American'], random_state=42)
            df_test.to_csv('test_compas.csv', index=False)
            df_train_all, df_val = train_test_split(x, test_size=1/8, train_size =(1-(1/8)), stratify=x['African-American'], random_state=42)

            df_train = df_train_all.groupby('African-American', group_keys=False).apply(lambda x: x.sample(n=int(len(df_train_all) * seed_size/2), random_state=42))
            df_train_unlabelled = df_train_all.drop(df_train.index)
            df_train = df_train.reset_index(drop=True)
            df_train_unlabelled = df_train_unlabelled.reset_index(drop=True)

        dataset_train = MyDataset(df_train)
        dataset_val = MyDataset(df_val)
        dataset_test = MyDataset(df_test)
    print("Length of train set: ", len(dataset_train))
    print("Length of val set: ", len(dataset_val))
    print("Length of test set: ", len(dataset_test))
    train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=1)
    val_data_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
    test_data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False,  num_workers=1)
    if method in ('ed', 'rs-b', 'mo-ed'):
        return train_data_loader, val_data_loader, test_data_loader
    else:
       return df_train, val_data_loader, test_data_loader, df_train_unlabelled

