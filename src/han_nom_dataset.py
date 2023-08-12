from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset
from torch import LongTensor, tensor, int64, from_numpy
from PIL import Image
import os
import numpy as np

IMG_CACHE = {}
class HanNomDataset(Dataset):
    def __init__(
        self,
        mode = 'train',
        root= 'demo_ds/train',
    ):
        super(HanNomDataset, self).__init__()
        self.root = root
        self.mode = mode
        self.dir = os.path.join(self.root, mode)
        self.ds_dummy: ImageFolder = ImageFolder(root= self.dir)
        self.imgs_lbs = self.ds_dummy.imgs
        self.y = LongTensor(self.ds_dummy.targets)
        self.classes = self.ds_dummy.classes


    def load_img(self, path: str)-> Image:
        if path in IMG_CACHE:
            x = IMG_CACHE[path]
        else:
            x = Image.open(path)
            IMG_CACHE[path] = x
        x = x.resize((32, 32))
        shape = 3, x.size[0], x.size[1]
        x = np.array(x, np.float32, copy=False)
        x = from_numpy(x/255)
        x = x.transpose(0, 1).contiguous().view(shape)

        return x

    def __getitem__(self, idx):
        img_path, target = self.imgs_lbs[idx]
        img = self.load_img(img_path)
        return img, tensor(target, dtype=int64)

    def __len__(self):
        return len(self.imgs_lbs)

class ClassificationDataset(Dataset):
    def __init__(
        self,
        mode: str,
        root_dir: str,
        transform = None,
    ):
        super(ClassificationDataset, self).__init__()
        self.root = root_dir
        self.mode = mode
        self.dir = os.path.join(self.root, mode)
        self.dataset: ImageFolder = ImageFolder(root= self.dir, transform= transform)

    def __len__(self):
        return len(self.dataset.imgs)
    
    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

        

if __name__ == '__main__':
    DS_PATH = 'demo_ds'
    train_path = os.path.join(DS_PATH, 'train')
    ds = HanNomDataset(root= train_path,)

    print(f'Length of training: {len(ds)}')
    print(f'Classes in training: {len(ds.classes)}')

    x, y = next(iter(ds))
    print(f'Type x: {type(x)}')
    print(f'Type y: {type(y)}')

    print(f'Size X: {x.size()}')
    print(f'Y: {y}')

    print(f"X: {x}")


    # valset = ImageFolder(root=f'{DS_PATH}/val', transform=transform)
    # testset = ImageFolder(root=f'{DS_PATH}/test', transform=transform)
