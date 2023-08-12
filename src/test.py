from pickle import load
from PIL import Image
from numpy import array, float32
from torch import from_numpy

def load_plk(filename: str):
    with open(filename, 'rb') as f:
        data = load(f)
    return data


def load_img(path):
    x = Image.open(path)
    x = x.resize((28, 28))

    shape = 1, x.size[0], x.size[1]
    x = array(x, float32, copy=False)
    x = 1.0 - from_numpy(x)
    x = x.transpose(0, 1).contiguous().view(shape)
    return x


if __name__ == '__main__':
    # Test loader
    PATH = 'tr_dataloader.plk'
    data = load_plk(PATH)

    x, y = next(iter(data))
    print(f'Type x: {type(x)}')
    print(f'Type y: {type(y)}')

    print(f'Size X: {x.size()}')
    print(f'Size Y: {y.size()}')

    print(f"X[0]: {x[0]}")
    print(f"Y[0]: {y[0]}")

    # Test dataset
    PATH = 'val_dataset.plk'
    data = load_plk(PATH)

    x, y = next(iter(data))
    print(f'Type x: {type(x)}')
    print(f'Type y: {type(y)}')

    print(f'Size X: {x.size()}')
    print(f'Y: {y}')

    print(f"X: {x}")
    print(f"Y[0]: {y}")
    # Test sample


    # PATH = 'dataset\\data\\Alphabet_of_the_Magi\\character01\\0709_01.png'
    # print(array(Image.open(PATH)))
    # print(load_img(PATH))