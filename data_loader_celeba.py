"""Loader and utilities to work with the CelebA dataset
TODO exchange imageA and imageP and learn from the new pair, in this case we can
TODO help to prevent learning things that are dependant on the images themselves
"""
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from os import listdir, walk, path
import random
from random import shuffle
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(self, image_path, metadata_path, transform=lambda x:x,
                 validation=(100, 25), val_samples=None):
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.transform = transform
        self.validation = validation
        self.val_samples = val_samples

        print("Preprocessing dataset...")
        self.preprocess_dataset(validation)
        print("done")

    def compute_validation_samples(self, pictures, validation=(None, None)):
        """Produces the validation dataset

        Args:
            pictures: the list of all the pictures in the dataset
            validation: tuple(#input, #cond) or tuple([input], [cond])

        Returns: the restricted set of pictures and the validation dataset
        object

        """
        shuffle(pictures)
        if type(validation[0]) == int and type(validation[1]) == int:
            num_validation = validation[0] + validation[1]
            train_pictures = pictures[:-num_validation]
            val_pictures = pictures[-num_validation:]
            val_input = val_pictures[:validation[0]]
            val_cond = val_pictures[validation[0]:]
        elif type(validation[0]) == list and type(validation[1]) == list:
            val_input = validation[0]
            val_cond = validation[1]
            train_pictures = set(pictures) - set(val_input) - set(val_cond)
            train_pictures = list(train_pictures)
        else:
            raise TypeError("validation must be an iterable of length=2 "
                            "where both elements are ints or lists")
        return train_pictures, val_input, val_cond

    def get_validation_dataset(self):
        """Returns the loader for the validation images

        Returns: loader for validation images
        """
        return ValidationDataset(self.val_input,
                                 self.val_cond,
                                 self.transform,
                                 self.image_path)

    def preprocess_dataset(self, validation):
        """Reads the dataset and creates the following variables

        self.image_ids: list of all ids
        self.pictures: list of the names of all pictures
        self.image_id2name: dict id -> list of picture names
        self.image_name2id: dict name of picture -> id
        self.ids_multiple: list of all ids that have more than one picture
        self.pictures_multiple: list of pictures whose id has several pictures

        Returns: None

        """
        self.pictures, self.val_input,self.val_cond = \
            self.compute_validation_samples(listdir(self.image_path), validation)

        pictures_set = set(self.pictures)
        shuffle(self.pictures)
        self.image_id2name, self.image_name2id = {}, {}
        self.image_ids = set()

        # read id metadata
        with open(self.metadata_path, 'r') as f:
            for line in f:
                file_name, id = line.strip().split(' ')
                id = int(id)
                self.image_ids.add(id)
                if file_name in pictures_set:
                    self.image_name2id[file_name] = id
                    if id not in self.image_id2name:
                        self.image_id2name[id] = []
                    self.image_id2name[id].append(file_name)

        self.ids_multiple = \
            list(filter(lambda x: len(self.image_id2name[x]) > 1, self.image_ids))
        ids_multiple_set = set(self.ids_multiple)
        self.pictures_multiple = \
            list(filter(lambda x: self.image_name2id[x] in ids_multiple_set,
                        self.pictures))

    def __len__(self):
        return len(self.pictures_multiple)

    def __getitem__(self, item):
        # choose an image by name
        imageA_name = self.pictures_multiple[item]
        imageA = Image.open(path.join(self.image_path, imageA_name))
        idA = self.image_name2id[imageA_name]

        pictureP = random.choice(list(
            set(self.image_id2name[idA]) - {imageA_name}
        ))
        # choose a random negative image, ensure that it's id is not idA
        while True:
            #FIXME doing choice on a list can be time consuming, better dict?
            pictureN = random.choice(self.pictures)
            if not self.image_name2id[pictureN] == idA: break
        imageP = Image.open(path.join(self.image_path, pictureP))
        imageN = Image.open(path.join(self.image_path, pictureN))

        # transform = transforms.Compose([
        #     transforms.CenterCrop(170),
        #     transforms.Resize(128, interpolation=Image.ANTIALIAS),
        #     transforms.RandomHorizontalFlip()])
        # transform(imageA).show()
        # transform(imageP).show()
        # transform(imageN).show()

        return self.transform(imageA), \
               self.transform(imageP), \
               self.transform(imageN)


class ValidationDataset(Dataset):
    """Produces validation samples that are not part of the training set
    """
    def __init__(self, input_samples, conditioning_samples, transform,
                 image_path):
        self.input_samples = input_samples
        self.conditioning_samples = conditioning_samples
        new_transform = [t for t in transform.transforms if type(t)
                         is not transforms.RandomHorizontalFlip]
        self.transform = transforms.Compose(new_transform)
        self.image_path = image_path

    def __len__(self):
        return len(self.input_samples) * len(self.conditioning_samples)

    def __getitem__(self, item):
        imageA = Image.open(path.join(self.image_path,
                                      self.input_samples[item // len(
                                          self.conditioning_samples)]))
        imageB = Image.open(path.join(self.image_path,
                                      self.conditioning_samples[item % len(
                                          self.conditioning_samples)]))
        return self.transform(imageA), self.transform(imageB)


def get_loader(image_dir, metadata_path, crop_size, image_size,
               batch_size=4, mode='train', validation=None, val_samples=None):
    """Returns the reader for the CelebA dataset

    Args:
        image_dir:
        metadata_path:
        crop_size:
        image_size:
        batch_size:
        mode:
        validation:
        val_samples:

    Returns:

    """
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size, interpolation=Image.ANTIALIAS))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = CelebADataset(image_dir,
                            metadata_path,
                            transform=transform,
                            validation=validation,
                            val_samples=val_samples)

    shuffle = True if mode == 'train' else False

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
