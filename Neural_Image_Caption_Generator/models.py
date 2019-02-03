# -*- coding: utf-8 -*-
"""
Created in Oct 2018

"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.utils.rnn as rnn_utils
from copy import deepcopy

def resnet_loader(pre_trained_dir, model_name='resnet152', pretrained=True):
    if model_name == 'resnet101':
        model = torchvision.models.resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3])
    elif model_name == 'resnet152':
        model = torchvision.models.resnet.ResNet(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3])

    if pretrained:
        state_dict = torch.load(open(pre_trained_dir + model_name + '.pth', 'rb'))
        model.load_state_dict(state_dict)

    return model


class CNN(nn.Module):
    
    def __init__(self, no_word_embeddings, pre_train_dir, freeze, dropout_prob, model_name):
        super(CNN, self).__init__()

        pretrained_cnn = resnet_loader(pre_train_dir, model_name, pretrained=True)

        self.resnet = nn.Sequential(*list(pretrained_cnn.children())[:-1])
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad_(requires_grad=False)

        if dropout_prob:
            self.fc_output = nn.Sequential(
                    nn.Linear(pretrained_cnn.fc.in_features, pretrained_cnn.fc.in_features),
                    nn.ReLU(),
                    nn.BatchNorm1d(pretrained_cnn.fc.in_features),
                    nn.Dropout(p=dropout_prob),
                    
                    nn.Linear(pretrained_cnn.fc.in_features, no_word_embeddings)
                    )
        else:
            self.fc_output = nn.Linear(pretrained_cnn.fc.in_features, no_word_embeddings)


    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc_output(x)

        return x


class RNN(nn.Module):

    def __init__(self, vocab_size, no_word_embeddings, hidden_size, num_layers, pre_trained_file, freeze, dropout_prob):
        super(RNN, self).__init__()
        
        self.id2word = np.array(pickle.load(open('../preprocessed_data/idx2word', 'rb')))
        self.hidden_size = hidden_size
        
        if pre_trained_file is not None:
            pretrained_word_embeddings = torch.from_numpy(pickle.load(open(pre_trained_file, 'rb')).astype(np.float32)).cuda()
            self.word_embeddings = nn.Embedding.from_pretrained(pretrained_word_embeddings, freeze)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, no_word_embeddings)

        self.lstm = nn.LSTM(
            input_size=no_word_embeddings,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        if dropout_prob:
            self.fc_output = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(p=dropout_prob),
                    
                    nn.Linear(hidden_size, vocab_size)
                    )
        else:
            self.fc_output = nn.Linear(hidden_size, vocab_size)
        

        self.softmax = nn.LogSoftmax()
        self.softmax_v2 = nn.Softmax(dim=1)

    def decode_idx2word(self, idx_seq):
        return self.id2word[idx_seq]
    
    def forward(self, image_embeddings, captions, lengths):

        word_embeddings = self.word_embeddings(captions)
        
        _, (h_0, c_0) = self.lstm(image_embeddings.view(image_embeddings.size(0), 1, -1))
        
        inputs = rnn_utils.pack_padded_sequence(word_embeddings, lengths, batch_first=True)
        h_t, (h_n, c_n) = self.lstm(inputs, (h_0, c_0))
        
        output = self.fc_output(h_t[0])
        return output

    def greedy_generator(self, image_embeddings, max_caption_length=20, STKidx=1, EDKidx=2):
        batch_size = image_embeddings.size(0)
        _, (h_0, c_0) = self.lstm(image_embeddings.view(batch_size, 1, -1)) # h_0.shape: 1 x batch_size x hidden_size
        output_captions_one_batch = []
        for image_idx in range(batch_size):
            # Get h0 and c0 of some image in the batch
            h_image_idx = h_0[:, [image_idx], :]
            c_image_idx = c_0[:, [image_idx], :]

            # Get Xi = X0 (i.e word embeddings of the start word)
            word_embeddings_image_idx = self.word_embeddings(torch.tensor(STKidx).cuda()).view(1, 1, -1)
            
            output_caption_one_image = []            
            for seq_idx in range(max_caption_length):
                # Given last hidden state, cell state and Xi, obtain the next hidden state and cell state
                _, (h_image_idx, c_image_idx) = self.lstm(word_embeddings_image_idx, (h_image_idx, c_image_idx))
                # Output a vector of len = vocab_size 
                output_seq_idx = self.fc_output(h_image_idx.view(-1, self.hidden_size))
                # Take the word index with biggest value
                predicted_word_idx = torch.argmax(output_seq_idx).item()
                if predicted_word_idx == EDKidx:
                    break
                else:
                    output_caption_one_image.append(predicted_word_idx)
                    # Update next Xi to be the word embeddings of the predicted word
                    word_embeddings_image_idx = self.word_embeddings(torch.tensor(predicted_word_idx).cuda()).view(1, 1, -1)
            
            # Convert word indices to actual words 
            output_caption_one_image = list(self.decode_idx2word(output_caption_one_image))
            output_captions_one_batch.append(output_caption_one_image)

        return output_captions_one_batch


    def beam_search_generator(self, image_embeddings, max_caption_length=20, STKidx=1, EDKidx=2):
        # Initially, there is only an empty sequence
        beam_width = 3 

        batch_size = image_embeddings.size(0)
        _, (h_0, c_0) = self.lstm(image_embeddings.view(batch_size, 1, -1)) # h_0.shape: 1 x batch_size x hidden_size
        sequences = [[[STKidx], 0.0, h_0, c_0]]
        output_captions_one_batch = []
        #pdb.set_trace()
        for image_idx in range(batch_size):
            # Get h0 and c0 of some image in the batch
            h_image_idx = h_0[:, [image_idx], :]
            c_image_idx = c_0[:, [image_idx], :]

            output_caption_one_image = []
            for seq_idx in range(max_caption_length):
                print('seq_idx = ', seq_idx);
                all_candidates = []
                for k in range(len(sequences)):
                    seq, score, h_curr, c_curr = sequences[k]
                    last_word_in_sequence = seq[len(seq) - 1]
                    strseq = self.decode_idx2word(seq)
                    print('the seq: ', strseq)

                    # This sequence is finished, no need to generate more words for it(but we still want it in the candidate list)
                    if last_word_in_sequence == EDKidx:
                        all_candidates.append(sequences[k])
                        continue

                    word_embeddings_image_idx = self.word_embeddings(torch.tensor(last_word_in_sequence).cuda()).view(1, 1, -1)
                    # Given last hidden state, cell state and Xi, obtain the next hidden state and cell state
                    _, (h_image_idx, c_image_idx) = self.lstm(word_embeddings_image_idx, (h_curr, c_curr))
                    # Output a vector of len = vocab_size 
                    output_seq_idx = self.fc_output(h_image_idx)
                    # Take top k words's indices with biggest values
                    output_seq_idx = output_seq_idx.resize(17003)
                    softmax_output = self.softmax(output_seq_idx)
                    #softmax_output = output_seq_idx

                    topk_results = softmax_output.topk(beam_width, largest=True)
                    topk_indices = topk_results[1]
                    #if strseq[len(strseq) - 1] == 'and':
                    #    print(softmax_output)
                    print('top k words:');
                    for j in range(topk_indices.size()[0]):
                        index = topk_indices[j]
                        candidate = [seq + [index], score + softmax_output[index], h_image_idx, c_image_idx] 
                        all_candidates.append(candidate)
                        print('k = :', j+1, ' ', self.decode_idx2word([index]));

                ordered = sorted(all_candidates, reverse = True, key=lambda tup:tup[1])
                sequences = ordered[:beam_width]

            # TODO: Remove any captions that don't have END token
            for p in range(len(sequences)):
                caption = sequences[p][0]
                caption = list(self.decode_idx2word(caption))
                print(caption)

            output_caption_one_image = sequences[0][0]
            # Convert word indices to actual words 
            print(output_caption_one_image)
            output_caption_one_image = list(self.decode_idx2word(output_caption_one_image))
            print(output_caption_one_image)
            output_captions_one_batch.append(output_caption_one_image)

        return output_captions_one_batch

    def beam_search_generator_v2(self, image_embeddings, beam_width=3, alpha=0, show_all=False,
                                 max_caption_length=20, STKidx=1, EDKidx=2):

        batch_size = image_embeddings.size(0)
        _, (h_0, c_0) = self.lstm(image_embeddings.view(batch_size, 1, -1)) # h_0.shape: 1 x batch_size x hidden_size
        output_captions_one_batch = []
        output_scores_one_batch = []

        for image_idx in range(batch_size):
            h_image_idx = h_0[:, [image_idx], :]
            c_image_idx = c_0[:, [image_idx], :]

            word_embeddings_image_idx = self.word_embeddings(torch.tensor(STKidx).cuda()).view(1, 1, -1)
            _, (h_image_idx, c_image_idx) = self.lstm(word_embeddings_image_idx, (h_image_idx, c_image_idx))

            output_seq_idx = self.fc_output(h_image_idx.view(-1, self.hidden_size)).view(1, -1)
            softmax_output = self.softmax_v2(output_seq_idx)
            top_k_prob, top_k_idx = softmax_output.topk(beam_width)

            top_k_scores = top_k_prob.log().view(-1, 1)
            top_k_seq = [[STKidx, each] for each in top_k_idx.view(-1).tolist()]

            word_embeddings_image_idx = self.word_embeddings(torch.tensor(top_k_idx).cuda()).view(beam_width, 1, -1)


            h_image_idx = h_image_idx[:, [0]*beam_width, :]
            c_image_idx = c_image_idx[:, [0]*beam_width, :]


            for i in range(max_caption_length):
    
                _, (h_image_idx, c_image_idx) = self.lstm(word_embeddings_image_idx, (h_image_idx, c_image_idx))
                output_seq_idx = self.fc_output(h_image_idx.view(beam_width, self.hidden_size)).view(beam_width, -1)
                
                softmax_output = self.softmax_v2(output_seq_idx)
                next_top_k_prob, next_top_k_idx = softmax_output.topk(beam_width)
                next_top_k_score = next_top_k_prob.log()
                
                length_flag = [1.0] * beam_width
                for idx in range(beam_width):
                    if top_k_seq[idx][-1] == EDKidx:
                        next_top_k_score[idx][0] = 0.0
                        next_top_k_idx[idx][0] = EDKidx
                        length_flag[idx] = 0.0
                
                if alpha == 0:
                    scores = (top_k_scores + next_top_k_score).view(-1)
                else:
                    # normalize via seq length, not very useful yet, maybe has bugs.
                    length_factor = (torch.FloatTensor([len(each_seq)-1 for each_seq in top_k_seq]) + torch.FloatTensor(length_flag)).view(-1, 1)
                    length_factor = length_factor.cuda()
                    scores =  ((1.0 / length_factor**alpha) * (top_k_scores + next_top_k_score)).view(-1)

                top_k_scores, top_k_idx = scores.topk(beam_width)
                top_k_scores = top_k_scores.view(-1, 1)
                
                prev_top_k_seq, next_top_k_candidate = np.unravel_index(top_k_idx, next_top_k_idx.size())
                next_top_k_idx = next_top_k_idx[prev_top_k_seq, next_top_k_candidate]
                
                temp_seq = []
                for idx in range(beam_width):
                    temp_seq.append(deepcopy(top_k_seq[prev_top_k_seq[idx]]))
                top_k_seq = deepcopy(temp_seq)
                finish_counts = 0
                for idx in range(beam_width):
                    if top_k_seq[idx][-1] == EDKidx:
                        finish_counts += 1
                        continue
                    else:
                        top_k_seq[idx].append(next_top_k_idx[idx].item())

                if finish_counts == beam_width:
                    break

                h_image_idx = h_image_idx[:, prev_top_k_seq, :]
                c_image_idx = c_image_idx[:, prev_top_k_seq, :]
                word_embeddings_image_idx = self.word_embeddings(next_top_k_idx).view(beam_width, 1, -1)


            if show_all:
                output_captions_one_batch.append([list(self.decode_idx2word(top_k_seq[i])) for i in range(beam_width)])
#                output_captions_one_batch.append(top_k_seq)
                output_scores_one_batch.append(top_k_scores.exp().view(-1).tolist())
            else:
                output_captions_one_batch.append(list(self.decode_idx2word(top_k_seq[0])))
#                output_captions_one_batch.append(top_k_seq[0])
                output_scores_one_batch.append(top_k_scores[0].exp().view(-1).tolist())

        return output_captions_one_batch, output_scores_one_batch
