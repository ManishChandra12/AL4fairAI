import torch
from CONSTANTS import dataset, method

if dataset == 'celebA':
    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe, transform=None):
            self.dataframe = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, index):
            row = self.dataframe.iloc[index]
            if dataset == 'celebA':
                if method not in ('mo-ed', 'mo-al'):
                    return (
                        #self.transform(torchvision.transforms.functional.to_tensor(Image.open(row["file"]))),
                        self.transform(Image.open('data/celebA/' + row["file"])),
                        #row["Attractive"],
                        row["Young"]
                    )
                else:
                    return (
                        #self.transform(torchvision.transforms.functional.to_tensor(Image.open(row["file"]))),
                        self.transform(Image.open('data/celebA/' + row["file"])),
                        row['Male'],
                        row["Eyeglasses"],
                        row["Young"]
                    )
if dataset == 'compas':
    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe, transform=None):
            self.dataframe = dataframe
            self.tr = self.dataframe['is_violent_recid']
            self.race = self.dataframe['African-American']
            self.dataframe = self.dataframe.drop(['African-American', 'Caucasian', 'Female', 'Male', 'is_violent_recid'], axis=1)
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, index):
            row = self.dataframe.iloc[index]
            tr_row = self.tr.iloc[index]
            race_row = self.race.iloc[index]
            if method not in ("mo-ed", "mo-al"):
                return (
                    row.values.astype(float),
                    tr_row
                )
            else:
                return (
                    row.values.astype(float),
                    race_row,
                    tr_row
                )
