# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""

import os
import sys
import json
import time
import pickle
import subprocess
import numpy as np

from skimage import io
from PIL import Image

import torch
from torchvision import transforms

from dataloader import MSCOCO_TEST

def save_model(model_name, model_dir, epoch, batch_step_count, time_used_global, optimizer, encoder, decoder):
   state = {
            'epoch': epoch,
            'batch_step_count': batch_step_count,
            'time_used_global': time_used_global,
            'optimizer': optimizer.state_dict(),
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict()
            }
   torch.save(state, open(model_dir + model_name + '_' + str(epoch)+'.pth', 'wb'))


def load_model(model_dir, model_list):
    lastest_model_idx = np.argmax([int(each_model.split('_')[1][:-4]) for each_model in model_list])
    lastest_model = model_dir + model_list[lastest_model_idx]
    lastest_state = torch.load(open(lastest_model, 'rb'))
    return lastest_state

def load_model_by_filename(model_dir, model_name):
    model_name = model_dir + model_name
    model_state = torch.load(open(model_name, 'rb'))
    return model_state

# TODO: sample images from valset and save it on tensorboard. Not finished yet.
def save_images_and_captions(image, generated_captions, writer):
    im = image.cpu().numpy().transpose(0, 2, 3, 1)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    im = np.array((im * std + mean) * 255, dtype=np.uint8)
    io.imshow(im[0])
    
    pass

# TODO: Not finished yet.
def generate_caption(encoder, decoder, transform_val):
    encoder.eval()
    decoder.eval()
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open('../test_images/test1.jpg')
    image = transform_val(image)
    image = image.expand([1, -1, -1, -1])
    image = image.cuda()
    image_embeddings = encoder(image)
    with torch.no_grad():                       
#        caption = decoder.beam_search_generator(image_embeddings)
        caption = decoder.greedy_generator(image_embeddings)
        print(' '.join(caption[0]))
        caption, probs = decoder.beam_search_generator_v2(image_embeddings)
        print(' '.join(caption[0]))
        caption = decoder.beam_search_generator(image_embeddings)
        print(' '.join(caption[0]))
    return caption
    

def calc_and_save_metrics(resulting_captions_file, true_captions_file, writer, epoch):
    p = subprocess.Popen(['python2.7', 'calc_metrics.py', resulting_captions_file, true_captions_file], 
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = p.stdout.read()
    metrics = [each.split(': ') for each in str(stdout).split('\\n')[:-1][-7:]]
    print(metrics)
    for each_metric in metrics:
        writer.add_scalar('epoch/'+each_metric[0], float(each_metric[1]), epoch)


def calc_metrics_test(model_name, set_used, batch_size, num_workers):
    init_params_file = '../model_params/' + model_name
    model_dir = '../saved_model/' + model_name + '/'
    
    if set_used == 'test':
        test_json = '../data/annotations/image_info_test2014.json'
        test_dir = '../data/test2014/'
    elif set_used == 'val':
        test_json = '../data/annotations/captions_val2014.json'
        test_dir = '../data/val2014/'

    resulting_captions_dir = '../data/results/' + model_name + '/'

    BATCH_SIZE = batch_size
    NUM_WORKERS = num_workers

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # copy from github
    ])
    
    print('Loading parms...')
    params = pickle.load(open(init_params_file, 'rb'))

    encoder = params['encoder']
    decoder = params['decoder']
    encoder.cuda()
    decoder.cuda()
    
    print('Loading model...')
    model_list = os.listdir(model_dir)
    state = load_model(model_dir, model_list)
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])
    epoch = state['epoch']
    
    print('Loading testing set...')
    testset = MSCOCO_TEST(test_json, transform=transform_val, test_dir=test_dir)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=BATCH_SIZE,
                                            shuffle=False, drop_last=False, num_workers=NUM_WORKERS)

    print('Start generating captions!')
    start_time = time.time()
    resulting_captions = []
    counts = 0
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        for images, image_ids in testloader:
            
            images = images.cuda()
            image_embeddings = encoder(images)
            generated_captions_calc_bleu, probs = decoder.beam_search_generator_v2(image_embeddings)

            for idx in range(images.size(0)):

                # prepare json file for calculating bleus
                generated_caption = (' '.join(generated_captions_calc_bleu[idx][1:-1]))
                if generated_caption[-1] == '.' and generated_caption[-2] == ' ':
                    generated_caption = generated_caption[:-2] + generated_caption[-1]
                elif generated_caption[-1] != '.':
                    generated_caption = generated_caption + '.'
                resulting_captions.append({'image_id': image_ids[idx].item(), 'caption': generated_caption})

                counts += 1
                if counts % 1000 == 0:
                    print('[%d] finished, %.2f min used.'%(counts, (time.time()-start_time)/60))

    resulting_captions_file = resulting_captions_dir + 'captions_'+set_used+ '2014_' + model_name + '-' + str(epoch) +'_results.json'
    with open(resulting_captions_file, 'w') as f:
        json.dump(resulting_captions, f)

#model_name = sys.argv[1]
#model_name = 'ADAM-13000-DROPOUT-RESNET152-NOV21'
#calc_metrics_test(model_name=model_name, set_used='test', batch_size=128, num_workers=0)
#torch.cuda.empty_cache()
#calc_metrics_test(model_name=model_name, set_used='val', batch_size=128, num_workers=0)

