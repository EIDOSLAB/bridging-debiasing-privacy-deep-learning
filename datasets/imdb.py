import torch
import gdown
import tarfile
import os
import PIL
import pandas as pd
import utils

from torchvision import transforms

def bin_age(age):
    age_bins = [19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 100]
    for i, bin in enumerate(age_bins):
        if age <= bin:
            return i

class IMDB(torch.utils.data.Dataset):
    def __init__(self, root, train=True, split='EB1', target='gender'):
        if split not in ['EB1', 'EB2', 'test']:
            print('Unkown split', split)
            exit(1)
        
        if train and split == 'test':
            print('Invalid config with train=True and split=test')
            exit(1)

        if target not in ['gender', 'age']:
            print('Invalid target', target)
            exit(1)
        
        self.split = split
        self.train = train

        path = root
        utils.ensure_dir(path)

        if not os.path.isdir(os.path.join(path, 'imdb_crop')):
            self.download_dataset(path)
        path = os.path.join(path, 'imdb_crop')

        self.path = path
        self.df = pd.read_csv(os.path.join(path, f'{split}.csv'))
        self.target = target

        print('binning ages')
        self.df.age = self.df.age.apply(bin_age)

        print(f'Loaded {len(self.df)} images')

    def download_dataset(self, path):
        url = "https://drive.google.com/uc?id=16sp_9QWt_OwhaGptW_GgcWCnqxBGCNbO"
        output = os.path.join(path, 'imdb.tar.gz')
        print(f'Downloading IMDB dataset from {url}')
        gdown.download(url, output, quiet=False)

        print('Extracting dataset..')
        tar = tarfile.open(os.path.join(output), 'r')
        tar.extractall(path=path)
        tar.close()
        os.remove(output)

    def __getitem__(self, index):
        entry = self.df.iloc[index]

        age, gender = int(entry.age), int(entry.gender)
        image = PIL.Image.open(os.path.join(self.path, entry.image)).convert('RGB')

        if self.target == 'gender':
            return image, gender, age
        elif self.target == 'age':
            return image, age, gender

        print('unkown target', self.target)
        return None

    def __len__(self):
        return len(self.df)