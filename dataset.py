import os
import json
import nltk
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from vocabulary import Vocabulary
from pycocotools.coco import COCO

from PIL import Image

class CoCoDataset(Dataset):
    
    def __init__(self, 
        transform, 
        mode, 
        batch_size, 
        vocab_threshold, 
        vocab_file, 
        start_word, 
        end_word, 
        unk_word, 
        annotations_file, 
        vocab_from_file, 
        img_folder):
        """
        Args:
            transform:          Instance of transforms.Compose containing set of transformation applied to images.
            mode:               (train|test) mode of dataset.
            batch_size:         Size of batch in case of train mode.
            vocab_threshold:    Min word occurrence.
            vocab_file:         Location of vocabulary file
            start_word:         Sequence of characters to use as a start word.
            end_word:           Sequence of characters to use as an end word.
            unk_word:           Sequence of characters to use as an unknown word (a word that vocabulary doesn't contain).
            annotations_file:   Path to file with annotations.
            vocab_from_file:    (True|False) Load vocabulary from file.
            img_folder:         Path to folder containing image files.
        """
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        if self.mode == 'train':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print('Obtaining caption lengths...')
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item['file_name'] for item in test_info['images']]
        
    def __getitem__(self, index):
        """Returns and image and caption if in training mode overwise returns image
        
        Args:
            index:      Index of an image
        """
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']

            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.transform(image)

            tokens = nltk.tokenize.word_tokenize(str(caption).lower())

            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            return image, caption

        else:
            path = self.paths[index]

            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')

            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            return orig_image, image

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        
        return all_indices

    def __len__(self):
        """Get length of dataset"""

        if self.mode == 'train':
            return len(self.ids)
        else:
            return len(self.paths)