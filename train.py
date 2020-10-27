from torchvision import transforms as transforms

from datamodule import CocoDataModule
from model import CaptioningModel

from pytorch_lightning import Trainer
from argparse import ArgumentParser

from pytorch_lightning.callbacks import ModelCheckpoint

def main(args):

    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    dm = CocoDataModule(transformations, batch_size=32, num_workers=8)

    vocab_size = len(dm.train_dataloader().dataset.vocab)

    model = CaptioningModel(256, 256, vocab_size, num_layers=2)

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    main(parser.parse_args())

