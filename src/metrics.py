import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
from CONSTANTS import dataset, method

def acc_and_f1(model, generator, device):
    if dataset == 'celebA':
        aa = None
        #bb = None
        yy_true = None
        if method not in ('mo-ed', 'mo-al'):
            for inp, tar in generator:
                inp = inp.to(device)
                otpt = model(inp)
                pred_y = torch.max(otpt, 1)[1].data.squeeze()
                pred_y = pred_y.cpu().detach().numpy()
                logit = torch.max(otpt, 1)[0].data.squeeze()
                logit = logit.cpu().detach().numpy()
                if aa is None:
                    aa = pred_y
                else:
                    aa = np.append(aa, pred_y)
                #if bb is None:
                #    bb = logit
                #else:
                #    bb = np.append(bb, logit)
                if yy_true is None:
                    yy_true = tar
                else:
                    yy_true = np.append(yy_true, tar)
        else:
            for inp, sensitive1, sensitive2, tar in generator:
                inp = inp.to(device)
                sensitive1 = sensitive1.to(device)
                sensitive2 = sensitive2.to(device)
                otpt, _ = model(inp, sensitive1, sensitive2)
                pred_y = torch.max(otpt, 1)[1].data.squeeze()
                pred_y = pred_y.cpu().detach().numpy()
                logit = torch.max(otpt, 1)[0].data.squeeze()
                logit = logit.cpu().detach().numpy()
                if aa is None:
                    aa = pred_y
                else:
                    aa = np.append(aa, pred_y)
                #if bb is None:
                #    bb = logit
                #else:
                #    bb = np.append(bb, logit)
                if yy_true is None:
                    yy_true = tar
                else:
                    yy_true = np.append(yy_true, tar)

        #accuracy = Accuracy()
        #f1 = F1Score(num_classes=2)
        #print(aa.shape, bb.shape, yy_true.shape)
        np.savetxt(dataset+method+"pred.txt", aa, newline="\n")
        #np.savetxt("logit.txt", bb, newline="\n")
        np.savetxt(dataset+method+"actual.txt", yy_true, newline="\n")

        test = pd.read_csv(test_datapath)
        p = pd.read_csv(dataset+method+"pred.txt", names=['prediction'], header=None)
        test['prediction'] = p['prediction']
        FN = ((test['Young'] == 1) & (test['prediction'] == 0)).sum()
        TP = ((test['Young'] == 1) & (test['prediction'] == 1)).sum()
        FNR_all = FN / (FN + TP)
        print("FNR for test set:", FNR_all)

        eye_male = test[(test['Male'] == 1) & (test['Eyeglasses'] == 1)]
        FN = ((eye_male['Young'] == 1) & (eye_male['prediction'] == 0)).sum()
        TP = ((eye_male['Young'] == 1) & (eye_male['prediction'] == 1)).sum()
        FNR_eye_male = FN / (FN + TP)
        print("FNR for male eyeglasses group:", FNR_eye_male)

        noeye_nomale = test[(test['Male'] == 0) & (test['Eyeglasses'] == 0)]
        FN = ((noeye_nomale['Young'] == 1) & (noeye_nomale['prediction'] == 0)).sum()
        TP = ((noeye_nomale['Young'] == 1) & (noeye_nomale['prediction'] == 1)).sum()
        FNR_noeye_nomale = FN / (FN + TP)
        print("FNR for no male no eyeglasses group:", FNR_noeye_nomale)

        noeye_male = test[(test['Male'] == 1) & (test['Eyeglasses'] == 0)]
        FN = ((noeye_male['Young'] == 1) & (noeye_male['prediction'] == 0)).sum()
        TP = ((noeye_male['Young'] == 1) & (noeye_male['prediction'] == 1)).sum()
        FNR_noeye_male = FN / (FN + TP)
        print("FNR for male no eyeglasses group:", FNR_noeye_male)

        eye_nomale = test[(test['Male'] == 0) & (test['Eyeglasses'] == 1)]
        FN = ((eye_nomale['Young'] == 1) & (eye_nomale['prediction'] == 0)).sum()
        TP = ((eye_nomale['Young'] == 1) & (eye_nomale['prediction'] == 1)).sum()
        FNR_eye_nomale = FN / (FN + TP)
        print("FNR for no male eyeglasses group:", FNR_eye_nomale)

        print("FNED:", abs(FNR_all - FNR_eye_male) + abs(FNR_all - FNR_noeye_nomale) + abs(FNR_all - FNR_noeye_male) + abs(FNR_all - FNR_eye_nomale))

        #return accuracy(torch.tensor(aa), torch.tensor(yy_true)), f1(torch.tensor(aa), torch.tensor(yy_true))
        print(classification_report(yy_true, aa, digits=4))
        #return accuracy_score(yy_true, aa), f1_score(yy_true, aa)
    elif dataset == 'compas':
        aa = None
        yy_true = None
        if method not in ('mo-ed', 'mo-al'):
            for inp, tar in generator:
                inp = inp.to(device)
                otpt = model(inp)
                pred_y = torch.max(otpt, 1)[1].data.squeeze()
                pred_y = pred_y.cpu().detach().numpy()
                logit = torch.max(otpt, 1)[0].data.squeeze()
                logit = logit.cpu().detach().numpy()
                if aa is None:
                    aa = pred_y
                else:
                    aa = np.append(aa, pred_y)
                if yy_true is None:
                    yy_true = tar
                else:
                    yy_true = np.append(yy_true, tar)
        else:
            for inp, sensitive, tar in generator:
                inp = inp.to(device)
                sens = sensitive.to(device)
                otpt, _ = model(inp, sens)
                pred_y = torch.max(otpt, 1)[1].data.squeeze()
                pred_y = pred_y.cpu().detach().numpy()
                logit = torch.max(otpt, 1)[0].data.squeeze()
                logit = logit.cpu().detach().numpy()
                if aa is None:
                    aa = pred_y
                else:
                    aa = np.append(aa, pred_y)
                if yy_true is None:
                    yy_true = tar
                else:
                    yy_true = np.append(yy_true, tar) 

        #accuracy = Accuracy()
        #f1 = F1Score(num_classes=2)
        #print(aa.shape, bb.shape, yy_true.shape)
        np.savetxt(dataset+method+"pred.txt", aa, newline="\n")
        #np.savetxt("logit_young.txt", bb, newline="\n")
        np.savetxt(dataset+method+"actual.txt", yy_true, newline="\n")

        test = pd.read_csv('test_compas.csv')
        p = pd.read_csv(dataset+method+"pred.txt", names=['prediction'], header=None)
        test['prediction'] = p['prediction']
        FN = ((test['is_violent_recid'] == 1) & (test['prediction'] == 0)).sum()
        TP = ((test['is_violent_recid'] == 1) & (test['prediction'] == 1)).sum()
        FNR_all = FN / (FN + TP)
        print("FNR for test set:", FNR_all)

        aam = test[test['African-American'] == 1]
        FN = ((aam['is_violent_recid'] == 1) & (aam['prediction'] == 0)).sum()
        TP = ((aam['is_violent_recid'] == 1) & (aam['prediction'] == 1)).sum()
        FNR_aam = FN / (FN + TP)
        print("FNR for African American group:", FNR_aam)

        cauc = test[test['Caucasian'] == 1]
        FN = ((cauc['is_violent_recid'] == 1) & (cauc['prediction'] == 0)).sum()
        TP = ((cauc['is_violent_recid'] == 1) & (cauc['prediction'] == 1)).sum()
        FNR_cauc = FN / (FN + TP)
        print("FNR for Caucasian group:", FNR_cauc)

        print("FNED:", abs(FNR_all - FNR_aam) + abs(FNR_all - FNR_cauc))
        #return accuracy(torch.tensor(aa), torch.tensor(yy_true)), f1(torch.tensor(aa), torch.tensor(yy_true))
        #return accuracy_score(yy_true, aa), f1_score(yy_true, aa)
        print(classification_report(yy_true, aa, digits=4))
