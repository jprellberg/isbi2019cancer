import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, accuracy_score
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import get_dataset, TF_VALID_NOROT
from model import get_model
from utils import IncrementalAverage, to_device, set_seeds, unique_string, count_parameters


def evaluate(model, valid_loader, class_weights, device):
    model.eval()

    all_labels = []
    all_preds = []
    loss_avg = IncrementalAverage()
    for img, label in tqdm(valid_loader, leave=False):
        img, label = to_device(device, img, label)
        with torch.no_grad():
            pred = model(img).view(-1)
            loss = lossfn(pred, label.to(pred.dtype), class_weights)
            all_labels.append(label.cpu())
            all_preds.append(pred.cpu())
            loss_avg.update(loss.item())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_preds_binary = all_preds > 0

    cm = confusion_matrix(all_labels, all_preds_binary)
    auc = roc_auc_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds_binary, average='weighted')
    return loss_avg.value, cm, auc, prec, rec, f1


def train(model, opt, train_loader, class_weights, device):
    model.train()
    loss_avg = IncrementalAverage()
    for img, label in tqdm(train_loader, leave=False):
        img, label = to_device(device, img, label)
        pred = model(img)
        pred = pred.view(-1)
        loss = lossfn(pred, label.to(pred.dtype), class_weights)
        loss_avg.update(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss_avg.value


def lossfn(prediction, target, class_weights):
    pos_weight = (class_weights[0] / class_weights[1]).expand(len(target))
    return F.binary_cross_entropy_with_logits(prediction, target, pos_weight=pos_weight)


def schedule(epoch):
    if epoch < 2:
        ub = 1
    elif epoch < 4:
        ub = 0.1
    else:
        ub = 0.01
    return ub


def train_validate(args):
    model = get_model().to(args.device)
    print("Model parameters:", count_parameters(model))

    trainset, validset, validset_subjects, class_weights = get_dataset(args.dataroot, tf_valid=TF_VALID_NOROT)
    class_weights = class_weights.to(args.device)
    print(f"Trainset length: {len(trainset)}")
    print(f"Validset length: {len(validset)}")
    print(f"class_weights = {class_weights}")

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=6, shuffle=True, drop_last=True)
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=6, shuffle=False)

    opt = torch.optim.Adam([
        {'params': model.paramgroup01(), 'lr': 1e-6},
        {'params': model.paramgroup234(), 'lr': 1e-4},
        {'params': model.parameters_classifier(), 'lr': 1e-2},
    ])
    scheduler = LambdaLR(opt, lr_lambda=[lambda e: schedule(e),
                                         lambda e: schedule(e),
                                         lambda e: schedule(e)])

    summarywriter = SummaryWriter(args.out)
    recorded_data = defaultdict(list)

    def logged_eval(e):
        valid_loss, cm, auc, prec, rec, f1 = evaluate(model, valid_loader, class_weights, args.device)

        # Derive some accuracy metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / cm.sum()
        acc_hem = tn / (tn + fp)
        acc_all = tp / (tp + fn)

        print(f"epoch={e} f1={f1:.4f}")

        summarywriter.add_scalar('loss/train', train_loss, e)
        summarywriter.add_scalar('loss/valid', valid_loss, e)
        summarywriter.add_scalar('cm/tn', tn, e)
        summarywriter.add_scalar('cm/fp', fp, e)
        summarywriter.add_scalar('cm/fn', fn, e)
        summarywriter.add_scalar('cm/tp', tp, e)
        summarywriter.add_scalar('metrics/precision', prec, e)
        summarywriter.add_scalar('metrics/recall', rec, e)
        summarywriter.add_scalar('metrics/f1', f1, e)
        summarywriter.add_scalar('metrics/auc', auc, e)
        summarywriter.add_scalar('acc/acc', acc, e)
        summarywriter.add_scalar('acc/hem', acc_hem, e)
        summarywriter.add_scalar('acc/all', acc_all, e)

        recorded_data['loss_train'].append(train_loss)
        recorded_data['loss_valid'].append(valid_loss)
        recorded_data['tn'].append(tn)
        recorded_data['tn'].append(tn)
        recorded_data['fp'].append(fp)
        recorded_data['fn'].append(fn)
        recorded_data['tp'].append(tp)
        recorded_data['precision'].append(prec)
        recorded_data['recall'].append(rec)
        recorded_data['f1'].append(f1)
        recorded_data['auc'].append(auc)
        recorded_data['acc'].append(acc)
        recorded_data['acc_hem'].append(acc_hem)
        recorded_data['acc_all'].append(acc_all)
        np.savez(f'{args.out}/results', **recorded_data)

    model = torch.nn.DataParallel(model)
    train_loss = np.nan
    logged_eval(0)
    for e in trange(args.epochs, desc='Epoch'):
        scheduler.step(e)
        train_loss = train(model, opt, train_loader, class_weights, args.device)
        logged_eval(e + 1)

    torch.save(model.state_dict(), f'{args.out}/model.pt')
    summarywriter.close()

    subj_acc = evaluate_subj_acc(model, validset, validset_subjects, args.device)
    np.savez(f'{args.out}/subj_acc', **subj_acc)


def evaluate_subj_acc(model, dataset, subjects, device):
    model.eval()

    subj_pred = defaultdict(list)
    subj_label = defaultdict(list)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

    for (img, cls), subj in tqdm(zip(dataloader, subjects), total=len(subjects), leave=False):
        img, cls = to_device(device, img, cls)
        bs, nrot, c, h, w = img.size()
        with torch.no_grad():
            cls_hat = model(img.view(-1, c, h, w))
            cls_hat = cls_hat.view(bs, nrot).mean(1)
            subj_label[subj].append(cls.cpu())
            subj_pred[subj].append(cls_hat.cpu())

    for k in subj_label:
        subj_label[k] = torch.cat(subj_label[k]).numpy()
        subj_pred[k] = torch.cat(subj_pred[k]).numpy() > 0

    subj_acc = {}
    for k in subj_label:
        subj_acc[k] = accuracy_score(subj_label[k], subj_pred[k])

    return subj_acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='data', help='path to dataset')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default='results', help='output folder')
    args = parser.parse_args()
    args.out = os.path.join(args.out, unique_string())
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)

    os.makedirs(args.out, exist_ok=True)
    set_seeds(args.seed)
    torch.backends.cudnn.benchmark = True

    train_validate(args)
