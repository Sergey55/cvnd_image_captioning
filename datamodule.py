import os
import unittest

from dataset import CoCoDataset
from torchvision.transforms import ToTensor

import torch.utils.data as data
import pytorch_lightning as pl

class CocoDataModule(pl.LightningDataModule):
    def __init__(self, 
                transform,
                coco_folder='./coco/',
                batch_size=32, 
                vocab_threshold=None, 
                vocab_file='./vocab.pkl', 
                start_word='<start>', 
                end_word='<end>',
                unk_word='<unk>',
                vocab_from_file=True,
                num_workers=0):
        """Constructor for DataModle

        Args:
            transform:          Default image transformation.
            batch_size:         Batch size.
            vocab_threshold:    Minimum word count threshold.
            start_word:         Special word denoting sentence start.
            end_word:           Special word denoting sentence end.
            unk_word:           Special word denoting unknown words.
            vocab_from_file:    If False, create vocab from scratch & override any existing vocab_file.
                                If True, load vocab from from existing vocab_file, if it exists.
            num_workers:        Number of subprocesses to use for data loading"""
        super().__init__()

        self.transform = transform
        self.coco_folder = coco_folder
        self.batch_size = batch_size
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.vocab_from_file = vocab_from_file
        self.num_workers = num_workers

    def train_dataloader(self):

        img_folder = os.path.join(self.coco_folder, 'images/train2014/')
        annotations_file = os.path.join(self.coco_folder, 'annotations/captions_train2014.json')

        assert os.path.exists(img_folder), "Images folder doesn't exist"
        assert os.path.exists(annotations_file), "Annotations file doesn't exist"

        dataset = CoCoDataset(transform=self.transform,
                              mode='train',
                              batch_size=self.batch_size,
                              vocab_threshold=self.vocab_threshold,
                              vocab_file=self.vocab_file,
                              start_word=self.start_word,
                              end_word=self.end_word,
                              unk_word=self.unk_word,
                              annotations_file=annotations_file,
                              vocab_from_file=self.vocab_from_file,
                              img_folder=img_folder)

        print(f'Found {len(dataset)} training records.')

        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_train_indices()
        
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        
        # data loader for COCO dataset.
        data_loader = data.DataLoader(dataset=dataset, 
                                    num_workers=self.num_workers,
                                    batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                    batch_size=dataset.batch_size,
                                    drop_last=False))

        return data_loader

    def test_dataloader(self):
        assert os.path.exists(self.vocab_file), "Must first generate vocabulary from training data."

        img_folder = os.path.join(self.coco_folder, 'images/test2014/')
        annotations_file = os.path.join(self.coco_folder, 'annotations/image_info_test2014.json')

        assert os.path.exists(img_folder), "Images folder doesn't exist"
        assert os.path.exists(annotations_file), "Annotations file doesn't exist"

        dataset = CoCoDataset(transform=self.transform,
                              mode='test',
                              batch_size=1,
                              vocab_threshold=self.vocab_threshold,
                              vocab_file=self.vocab_file,
                              start_word=self.start_word,
                              end_word=self.end_word,
                              unk_word=self.unk_word,
                              annotations_file=annotations_file,
                              vocab_from_file=self.vocab_from_file,
                              img_folder=img_folder)

        data_loader = data.DataLoader(dataset=dataset,
                                    batch_size=dataset.batch_size,
                                    shuffle=True,
                                    num_workers=self.num_workers)

        return data_loader

class Tests(unittest.TestCase):
    def test_get_train_dataloader(self):
        dm = CocoDataModule(ToTensor(), batch_size=32, vocab_from_file=True)

        self.assertNotEqual(None, dm.train_dataloader())

    def test_get_test_dataloader(self):
        dm = CocoDataModule(ToTensor(), batch_size=32, vocab_from_file=True)

        self.assertNotEqual(None, dm.test_dataloader())

if __name__ == '__main__':
    unittest.main()