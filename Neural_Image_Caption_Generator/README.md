# Show and tell
This repo is to reimplement the show and tell paper.

# Check list
* Tune hyperparameters.
	* Data augmentation
	* Optimizer, learning rate, weight decay
	* Clip gradients
	* Which ResNet?
	* Which RNN: LSTM? GRU?
	* Word embeddings:
		* Embedding size?
		* Vocabulary size?
	* RNN parameters?
		* Hidden size?
		* Number of layers?
		* Dropout?
	* Beam search size?
	* Google open source parameter settings of the NIC, in tensorflow:
		* [Github link](https://github.com/tensorflow/models/blob/master/research/im2txt/im2txt/configuration.py)
* Dropout + ResNet152 + RandomCrop + TrainLastBottleneck + LSTM/GRU
* Ensemble