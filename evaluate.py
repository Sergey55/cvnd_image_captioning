import torch
import pytorch_lightning as pl

from PIL import Image
from torchvision import transforms as transforms

from argparse import ArgumentParser
from datamodule import CocoDataModule
from model import CaptioningModel
from vocabulary import Vocabulary

from utils import result_to_text

def main(args):

    # Create vocabulary. Assumes that the `vocab.pkl` file already exists.
    vocabulary = Vocabulary(vocab_from_file=True)

    # Create transformations
    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])    

    # Create the model instance.
    model = CaptioningModel(
        embed_size=256,
        hidden_size=256,
        vocab_size = len(vocabulary),
        num_layers=2,
        use_pretrained_encoder=False
    )

    # Load state dict for encoder and decoder
    model.encoder.load_state_dict(torch.load('./models/encoder.pkl')['encoder_state_dict'])
    model.decoder.load_state_dict(torch.load('./models/decoder.pkl')['decoder_state_dics'])

    # Set model to eval state
    model.eval()

    image = Image.open(args.image_path)
    tensor = transformations(image).unsqueeze(0)

    result = model.sample(tensor)
    text = result_to_text(result, vocabulary)

    print(text)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--image_path', required=True, help='Path to the image for generation of a caption.')

    main(parser.parse_args())
