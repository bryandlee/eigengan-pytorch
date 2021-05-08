import os
from PIL import Image

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def infinite_loader(data_loader):
    while True:
        for batch in data_loader:
            yield batch


class Dataset:
    def __init__(self, root, transform=None):
        self.root = root
        self.imgs = []
        for dir, _, fnames in sorted(os.walk(root, followlinks=True)):
            for fname in sorted(fnames):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    self.imgs.append(os.path.join(dir, fname))

        self.transform = transform if transform is not None else lambda x: x

    def __getitem__(self, index):
        with open(self.imgs[index], 'rb') as f:
            img = Image.open(f)
            return self.transform(img.convert('RGB'))

    def __len__(self):
        return len(self.imgs)
