import argparse
import os
import zipfile
from os.path import join

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

from model import get_model
from dataset import TF_VALID_ROT


class OrderedImages(Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.root = root
        self.transform = transform

    def __len__(self):
        return 2586

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, f'{index + 1}.bmp'))
        return self.transform(img)


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('modelroot', default='results/20190313T101236Z.LGJL', help='path to model')
parser.add_argument('dataroot', default='data/phase3', help='path to dataset')
args = parser.parse_args()

dataset = OrderedImages(args.dataroot, TF_VALID_ROT)

print(f"Loading model")
model = get_model().to('cuda:0')
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(join(args.modelroot, 'model.pt')))
model.eval()

dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=6)

print("Classifying")
all_labels = []
for x in tqdm(dataloader, total=len(dataset) // args.batch_size):
    x = x.to('cuda:0')
    bs, nrot, c, h, w = x.size()
    with torch.no_grad():
        y = model(x.view(-1, c, h, w))
        y = y.view(bs, nrot).mean(1)
    labels = y > 0
    all_labels.append(labels)

all_labels = torch.cat(all_labels)
print("Positive:", all_labels.sum().item())
print("Negative:", len(all_labels) - all_labels.sum().item())

csv_path = join(args.modelroot, 'submission.csv')
zip_path = join(args.modelroot, 'submission.zip')
np.savetxt(csv_path, all_labels.cpu().numpy(), '%d')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(csv_path, 'isbi_valid.predict')
