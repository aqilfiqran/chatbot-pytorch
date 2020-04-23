from Algorithm import GreedySearchDecoder, EncoderRNN, LuongAttnDecoderRNN
from LoadFile import loadPrepareData
from Evaluate import evaluateInput
import argparse
import os
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description='Train Data')
parser.add_argument("-c", "--checkpoint", type=int,
                    help="Input checkpoint number")

args = vars(parser.parse_args())

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# load data and model
save_dir = os.path.join("model", "save")
corpus = "data"
datafile = os.path.join(corpus, "formatted_movie_lines.txt")
model_name = 'cb_model'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
attn_model = 'dot'
checkpoint_iter = args['checkpoint'] if args['checkpoint'] != None else 4000

# call function loadPrepareData
voc, pairs = loadPrepareData(corpus, datafile)

loadFilename = os.path.join(save_dir, model_name, corpus,
                            '{}-{}_{}'.format(encoder_n_layers,
                                              decoder_n_layers, hidden_size),
                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
checkpoint = torch.load(loadFilename, map_location=device)
voc.__dict__ = checkpoint['voc_dict']

# load embedding
embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(checkpoint['embedding'])

# load encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(
    attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
encoder.load_state_dict(checkpoint['en'])
decoder.load_state_dict(checkpoint['de'])

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)
