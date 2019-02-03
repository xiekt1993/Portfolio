# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""

from pycocotools.coco import COCO

import pickle
from PIL import Image

import torch
import torch.nn.utils.rnn as rnn_utils


def collate_fn(batch):
    batch.sort(key=lambda image_caption: len(image_caption[1]), reverse=True)
    images, captions = zip(*batch)
    
    images = torch.stack(images)
    lengths = torch.tensor([each_caption.size()[0] for each_caption in captions])
    captions = rnn_utils.pad_sequence(captions, batch_first=True, padding_value=0)

    return images, captions, lengths


class MSCOCO(torch.utils.data.Dataset):
    '''
        There is a total of #images * 5 items in the dict imagepaths_and_captions
        Each key maps to a image_path + caption pair
    '''
    def __init__(self, vocab_size, imagepaths_and_captions, transform):
        
        self.imagepaths_captions = pickle.load(open(imagepaths_and_captions, 'rb'))
        self.caption_ids = list(self.imagepaths_captions.keys())

        self.transform = transform
        self.vocab_size = vocab_size

    def __getitem__(self, index):  
        caption_id = self.caption_ids[index]
        
        imagepath_and_caption = self.imagepaths_captions[caption_id]
        image_path = imagepath_and_caption['image_path']
        caption = imagepath_and_caption['caption']
        caption[caption>=self.vocab_size] = 0

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, caption
    
    def __len__(self):
        return len(self.caption_ids)

def collate_fn_val(batch):
    batch.sort(key=lambda image_caption: len(image_caption[1]), reverse=True)
    images, captions_calc_bleu, image_ids = zip(*batch)
    images = torch.stack(images)

    captions_calc_loss = []
    lengths = []
    for i in range(len(captions_calc_bleu)):
        captions_calc_bleu[i].sort(key=lambda image_caption: len(image_caption), reverse=True)
        lengths.append(torch.tensor([each_caption.size()[0] for each_caption in captions_calc_bleu[i]]))
        captions_calc_loss.append(rnn_utils.pad_sequence(captions_calc_bleu[i], batch_first=True, padding_value=0))

    return images, captions_calc_bleu, captions_calc_loss, lengths, image_ids


class MSCOCO_VAL(torch.utils.data.Dataset):
    '''
        Define image_dict = {}
        key = image_path, value = a list of captions
        e.g image_path = "val2017/dog.jpg", value = [[21,11,111,33,66], [111,22,233,11,66], [88,22,111,11,66]]
    '''
    def __init__(self, vocab_size, val_imagepaths_and_captions, transform):
        self.image_list = pickle.load(open(val_imagepaths_and_captions, 'rb'))

        self.transform = transform
        self.vocab_size = vocab_size

    def __getitem__(self, index):  
        image_path = self.image_list[index][0]
        captions = self.image_list[index][1]
        image_id = self.image_list[index][2]


        for c in captions:
            c[c>=self.vocab_size] = 0
            
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, captions, image_id
    
    def __len__(self):
        return len(self.image_list)


class MSCOCO_TEST(torch.utils.data.Dataset):

    def __init__(self, test_json, transform, test_dir):
        coco = COCO(test_json)
        self.image_list = list(coco.imgs.values())
        self.test_dir = test_dir

        self.transform = transform

    def __getitem__(self, index):  
        image_path = self.test_dir + self.image_list[index]['file_name']
        image_id = self.image_list[index]['id']
            
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, image_id
    
    def __len__(self):
        return len(self.image_list)
