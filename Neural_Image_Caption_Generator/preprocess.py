# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""

import os
import nltk
import pickle
import itertools
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from pycocotools.coco import COCO

vocab_size = 17000

#caption_file, image_dir = ['../data/annotations/captions_train2017.json', '../data/train2017/']
caption_file, image_dir = ['../data/annotations/captions_val2017.json', '../data/val2017/']

GloVe_embeddings_file = '../pre_trained/glove.840B.300d.txt'

COCO_lookup_table = '../preprocessed_data/COCO_lookup_table'
GloVe_lookup_table = '../preprocessed_data/GloVe_lookup_table'
GloVe_embeddings_matrix = '../preprocessed_data/GloVe_embeddings'

processed_embeddings = '../preprocessed_data/embeddings'
processed_word2idx = '../preprocessed_data/word2idx'
processed_idx2word = '../preprocessed_data/idx2word'

if 'train' in caption_file:
    postfix = '.train'
elif 'val' in caption_file:
    postfix = '.val'
imagepaths_captions = '../preprocessed_data/imagepaths_captions'
imagepaths_captions = imagepaths_captions + postfix

# Preprocess data.
def preprocess(caption_file, vocab_size):

    coco = COCO(caption_file)
    captions = coco.anns
    
    all_captions = []
    for key in captions.keys():
        caption = nltk.word_tokenize(captions[key]['caption'])
        caption = [each.lower() for each in caption]
        all_captions.append(caption)
    
    all_words = list(itertools.chain.from_iterable(all_captions))
    
    unique_word, word_counts = np.unique(all_words, return_counts=True)
    
    freq_idx = np.argsort(word_counts)[::-1]
    
    word_counts = word_counts[freq_idx]
    unique_word = unique_word[freq_idx]
    
    word2count_COCO = {token: count for token, count in zip(unique_word, word_counts)}
    word2idx_COCO = {token: idx for idx, token in enumerate(unique_word)}
    idx2word_COCO = np.copy(unique_word)
    
    pickle.dump({'word2count_COCO': word2count_COCO,
                 'word2idx_COCO': word2idx_COCO,
                 'idx2word_COCO': idx2word_COCO}, open(COCO_lookup_table, 'wb'))
    
        
    #preprocess using GloVe pretrained word embeddings.
    with open(GloVe_embeddings_file, 'r', encoding='utf-8') as f:
        embeddings = f.readlines()
    
    all_embeddings = []
    all_words = []
    for emb in embeddings:
        emb = emb.strip()
        emb = emb.split(' ')
        word = emb[0]
        embedding = np.asarray(emb[1:], dtype=np.float32)
        if embedding.shape[0] != 300 or word in all_words: # performance bottleneck. May use dictionary to speed up.
            continue
        all_embeddings.append(embedding)
        all_words.append(word)
        if len(all_words)>=250000:
            break

    word2idx_GloVe = {word: idx for idx, word in enumerate(all_words[:250000])}
    idx2word_GloVe = all_words[:250000].copy()
    embeddings_GloVe = np.stack(all_embeddings[:250000])

    # introduce unknown word key, start word key and end word key.
    word2idx = {'UNK': 0, 'STK': 1, 'EDK': 2}
    idx2word = ['UNK', 'STK', 'EDK']
    no_new_keys = 3
    embeddings = np.zeros((vocab_size+no_new_keys, embeddings_GloVe.shape[1]))
    embeddings[:no_new_keys, :] = np.random.random((no_new_keys, embeddings_GloVe.shape[1])) * 0.01
    
    count = 0
    unk_count = 0
    for word in idx2word_COCO:
        idx_GloVe = word2idx_GloVe.get(word)
        if idx_GloVe is not None:
            embeddings[count+no_new_keys, :] = embeddings_GloVe[idx_GloVe, :]
            word2idx[word] = count+no_new_keys
            idx2word.append(word)
            count += 1
        else:
            unk_count += 1
        if count == vocab_size:
            break

    # save files
    pickle.dump({'word2idx_GloVe': word2idx_GloVe,
                 'idx2word_GloVe': idx2word_GloVe}, open(GloVe_lookup_table, 'wb'))
    pickle.dump(embeddings_GloVe, open(GloVe_embeddings_matrix, 'wb'))
    
    pickle.dump(word2idx, open(processed_word2idx, 'wb'))
    pickle.dump(idx2word, open(processed_idx2word, 'wb'))
    pickle.dump(embeddings, open(processed_embeddings, 'wb'))


# load files and generate new embeddings given a new vocab_size.
def generate_new_embeddings(caption_file, vocab_size):
    temp = pickle.load(open(COCO_lookup_table, 'rb'))
    idx2word_COCO = temp['idx2word_COCO']

    
    temp = pickle.load(open(GloVe_lookup_table, 'rb'))
    word2idx_GloVe = temp['word2idx_GloVe']
    embeddings_GloVe = pickle.load(open(GloVe_embeddings_matrix, 'rb'))
    
    word2idx = {'UNK': 0, 'STK': 1, 'EDK': 2}
    idx2word = ['UNK', 'STK', 'EDK']
    no_new_keys = 3
    embeddings = np.zeros((vocab_size+no_new_keys, embeddings_GloVe.shape[1]))
    embeddings[:no_new_keys, :] = np.random.random((no_new_keys, embeddings_GloVe.shape[1])) * 0.01

    count = 0
    unk_count = 0
    for word in idx2word_COCO:
        idx_GloVe = word2idx_GloVe.get(word)
        if idx_GloVe is not None:
            embeddings[count+no_new_keys, :] = embeddings_GloVe[idx_GloVe, :]
            word2idx[word] = count+no_new_keys
            idx2word.append(word)
            count += 1
        else:
            unk_count += 1
        if count == vocab_size:
            break
    
    pickle.dump(word2idx, open(processed_word2idx, 'wb'))
    pickle.dump(idx2word, open(processed_idx2word, 'wb'))
    pickle.dump(embeddings, open(processed_embeddings, 'wb'))


# tokenize captions and add image paths to dict.
def process_captions(caption_file):
    coco = COCO(caption_file)
    captions = coco.anns

    word2idx = pickle.load(open(processed_word2idx, 'rb'))

    for key in captions.keys():
        image_id = captions[key]['image_id']
        image_path = image_dir + coco.loadImgs(image_id)[0]['file_name']
        captions[key]['image_path'] = image_path

        caption = nltk.word_tokenize(captions[key]['caption'])
        caption = [each.lower() for each in caption]
        caption.insert(0, 'STK')
        caption.append('EDK')
        captions[key]['caption'] = torch.LongTensor([word2idx.get(word, word2idx['UNK']) for word in caption])

    pickle.dump(captions, open(imagepaths_captions, 'wb'))
    


transform_val = transforms.Compose([
    transforms.Resize((259, 259)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor()
])


'''
    This function generate the following list based on the imagepaths_and_captions file.
    The generated list can be more conveniently used when evaluating BLEU score for the validation set
    Each list element is a tuple like this: ("val2017/dog.jpg", [[21,11,111,33,66], [111,22,233,11,66], [88,22,111,11,66]])
'''

def generate_imagepaths_captions_for_eval(imagepaths_and_captions, imagepaths_and_captions_val):
    imagepaths_captions = pickle.load(open(imagepaths_and_captions, 'rb'))
    image_list = [] 
    imagepath2idx = {}
    idx = 0
    for caption_id in imagepaths_captions:
        # Get the caption and image path
        imagepath_and_caption = imagepaths_captions[caption_id]
        image_path = imagepath_and_caption['image_path']
        caption = imagepath_and_caption['caption']
        image_id = imagepath_and_caption['image_id']

        if image_path in imagepath2idx:
            image_idx = imagepath2idx[image_path]
            t = image_list[image_idx] 
            t[1].append(caption)
        else:
            image_list.append((image_path, [caption], image_id))
            imagepath2idx[image_path] = idx
            idx = idx + 1

    pickle.dump(image_list, open(imagepaths_and_captions_val, 'wb'))

def cal_mean_std(image_dir, transform=transform_val):
    image_paths = os.listdir(image_dir)
    
    summation = torch.tensor([0.0, 0.0, 0.0])
    for count, image_name in enumerate(image_paths, 1):
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        
        summation += image.sum(dim=(1, 2))
        
        if count % 1000 == 0:
            print(count)
    
    mean = summation / (224*224) / count
    mean = mean.reshape(3, 1, 1)

    accumulator = torch.tensor([0.0, 0.0, 0.0])
    for count, image_name in enumerate(image_paths, 1):
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        
        accumulator += ((image - mean)**2).sum(dim=(1, 2))
        
        if count % 1000 == 0:
            print(count)

    std = (accumulator / (224*224) / count)**0.5

    return mean, std

generate_imagepaths_captions_for_eval('../preprocessed_data/imagepaths_captions.val', 
                                      '../preprocessed_data/imagepaths_captions.newval')
