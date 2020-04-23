from Batch import batch2TrainData
from LoadFile import loadPrepareData, trimRareWords
from Algorithm import EncoderRNN, LuongAttnDecoderRNN
from torch import optim
from Train import trainIters
import torch.nn as nn
import random
import os
import torch
import argparse

parser = argparse.ArgumentParser(description='Train Data')
parser.add_argument("-l", "--loadtrain", action="store_true",
                    help="Load checkpoint train iterate")
parser.add_argument("-c", "--checkpoint", type=int,
                    help="Input checkpoint number")
parser.add_argument("-s", "--save", type=int,
                    help="save train iterate checkpoint every that number")

args = vars(parser.parse_args())

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Load/Assemble voc and pairs
corpus = "data"
datafile = os.path.join(corpus, "formatted_movie_lines.txt")
save_dir = os.path.join("model", "save")
voc, pairs = loadPrepareData(corpus, datafile)

MIN_COUNT = 3  # Minimum word count threshold for trimming

# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)

# Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs)
                                for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


# Configure models
model_name = 'cb_model'
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
checkpoint_iter = args['checkpoint'] if args['checkpoint'] != None else 4000

loadFilename = None
if args['loadtrain']:
    loadFilename = os.path.join(save_dir, model_name, corpus,
                                '{}-{}_{}'.format(encoder_n_layers,
                                                  decoder_n_layers, hidden_size),
                                '{}_checkpoint.tar'.format(checkpoint_iter))

# Load model if a loadFilename is provided
if loadFilename:
    checkpoint = torch.load(loadFilename, map_location=device)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']
else:
    checkpoint = None

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)

# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(
    attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
else:
    encoder = encoder.to(device)
    decoder = decoder.to(device)
print('Models built and ready to go!')

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = args['save'] if args['save'] != None else 4000

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(
    decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have cuda, configure cuda to call
if USE_CUDA:
    for states in [encoder_optimizer.state.values(), decoder_optimizer.state.values()]:
        for state in states:
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus, loadFilename, teacher_forcing_ratio, hidden_size, checkpoint)
