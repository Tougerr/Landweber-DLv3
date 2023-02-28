import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import torch
import numpy as np
import deeplearning as smp
from PIL import Image
from tqdm import tqdm


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 384)
    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


class Dataset(BaseDataset):

    #
    # CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
    #            'tree', 'signsymbol', 'fence', 'car',
    #            'pedestrian', 'bicyclist', 'unlabelled']

    CLASSES = ['_background_','c0','c10', 'c5', 'c1']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        # print(self.class_values)

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        # print(self.images_fps[i], self.masks_fps[i])
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_path = '.' + self.masks_fps[i].split('.')[1] + '.png'
        # print(mask_path)
        mask = cv2.imread(mask_path, 0)
        # print(mask)
        # print(mask.shape, type(mask))

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        # masks = [(mask == 255)]
        # print(masks)
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, self.masks_fps[i].split('/')[-1].split('.')[0]

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":

    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = [ 'c10', 'c5', 'c1']
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'
    DATA_DIR = './data/8c/'

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.Fscore(threshold=0.5), ]

    x_test_dir = os.path.join(DATA_DIR, 'image')
    y_test_dir = os.path.join(DATA_DIR, 'mask')

    best_model = torch.load('models/best_model_cDV3new.pth')

    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    test_dataloader = DataLoader(test_dataset)

    # test dataset without transformations for image visualization
    test_dataset_vis = Dataset(
        x_test_dir, y_test_dir,
        classes=CLASSES,
    )

    num = 1

    for image, mask, image_id in tqdm(test_dataset):
        out_filename = 'test_results/8c/' + image_id + '.png'
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        # print(pr_mask.shape)
        # pr_mask_c10 = pr_mask[:, 4, :].squeeze().cpu().numpy().round()
        pr_mask_c10 = pr_mask[:, 0, :].squeeze ().cpu ().numpy ().round ()
        pr_mask_c5 = pr_mask[:, 1, :].squeeze ().cpu ().numpy ().round ()
        pr_mask_b = pr_mask[:, 2, :].squeeze().cpu().numpy().round()
        # c10_mask_255 = pr_mask_c10 * 255
        c5_mask_127 = pr_mask_c10 * 255
        c_mask_127 = pr_mask_c5 * 127
        # b_mask_255 = pr_mask_b * 0
        # pr_mask_c1 = cv2.add ( cv2.resize ( c10_mask_255, (369, 369) ), cv2.resize ( c5_mask_127, (369, 369) ))
        pr_mask_c2 = cv2.add(cv2.resize ( c_mask_127, (369, 369) ) ,cv2.resize ( c5_mask_127, (369, 369) ))
        # pr_mask_c = cv2.add(pr_mask_c2,pr_mask_c1)
        result = Image.fromarray(pr_mask_c2.astype(np.uint8))
        result.save(out_filename)
