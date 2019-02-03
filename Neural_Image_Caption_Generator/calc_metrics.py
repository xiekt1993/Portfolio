# -*- coding: utf-8 -*-
"""
Created in Nov 2018

"""

import sys
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')


# set up file names and pathes# set u 
resulting_captions_file = sys.argv[1]
true_captions_file = sys.argv[2]


# create coco object and cocoRes object# creat 
coco = COCO(true_captions_file)
cocoRes = coco.loadRes(resulting_captions_file)


# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)


# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
cocoEval.evaluate()


for metric, score in cocoEval.eval.items():
    print '%s: %.3f'%(metric, score)
