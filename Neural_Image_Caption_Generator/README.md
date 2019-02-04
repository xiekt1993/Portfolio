# Show and tell - Neural Image Caption Generator
This project is to reimplement the show and tell paper (Vinyals et al., 2015). This application helps visually-impaired people by transforming visual signals into proper language, which involves tasks of both image classification as well as natural language processing. Using 500 GPU hours for training, the trainning process managed to improve the performance and the convergence by replacing the pre-trained model to ResNet 152, using Adam optimizer and several other experiments. As a result, the model yields better performance compared to the original paper on the MSCOCO 2014 testing set.

## Implementation
- Dataset: MSCOCO dataset (2017), divided into a training set containing 118k images and a validation set containing 5k images.
- Data Augmentation: Resize, Ramdon Changes and Normalization.
- The image captioning model(CNN+RNN with dropout): detailed model can found at model.py. The encoder is essentially a pretrained ResNet 152 model. Before outputting the image embedding vector, two FC layers and a dropout layer is applied to add non-linear factors and to reduce overfitting. The decoder is essentially a one layer LSTM model, with the initial hidden state being the image embedding produced by the encoder, the initial input of captions being the index of the start token.
- Beam Search: With beam search, instead of always selecting the word with highest probability at each time step, we maintain a priority queue that keeps track of BEAM_SIZE sequences with highest probabilities and at each time step we will generate BEAM_SIZE words with highest probabilities for each sequence that we are tracking, then we only keep the top BEAM_SIZE new sequences for the next time step.

## Results
- Training procedure of the best model
<p align="center">
  <img src="https://github.com/xiekt1993/Portfolio/blob/master/Neural_Image_Caption_Generator/examples2.png" width="750"/>
</p>
- Loss curves
<p align="center">
  <img src="https://github.com/xiekt1993/Portfolio/blob/master/Neural_Image_Caption_Generator/examples3.png" width="750"/>
</p>
- A selection of generated captions, randomly sampled from the testing set
<p align="center">
  <img src="https://github.com/xiekt1993/Portfolio/blob/master/Neural_Image_Caption_Generator/examples.png" width="750"/>
</p>

## Conclusion
It is not hard to implement it in terms of coding the end-to-end architecture. However, it is difficult to train the model since it suffers from overfitting quite a lot.
There are also some ways to improve our results. 1) The first is to try scheduled sampling when training; 2) the second is to use attention mechanism; 3) the third is to train more models using the best architecture.

## Reference
- Vinyals, Oriol, et al. "Show and tell: A neural image caption generator." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
- Chen, Xinlei, et al. "Microsoft COCO captions: Data collection and evaluation server." arXiv preprint arXiv:1504.00325(2015).
- Xu, Kelvin, et al. "Show, attend and tell: Neural image caption generation with visual attention." International conference on machine learning. 2015.
