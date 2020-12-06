import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import flask
from flask import request
import os
from torchtext import vocab
from models import Baseline, CNN_LSTM_2_conv

# Initialize the app
app = flask.Flask(__name__)

# class Baseline(nn.Module):

#     def __init__(self, embedding_dim, vocab):
#         super(Baseline, self).__init__()

#         self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
#         self.fc = nn.Linear(embedding_dim, 1)

#     def forward(self, x, lengths=None):
#         embedded = self.embedding(x)

#         average = embedded.mean(0)
#         output = self.fc(average)
#         output = nn.functional.sigmoid(output)

#         return output.squeeze()

try:
    vocab._default_unk_index
except AttributeError:
    def _default_unk_index():
        return 0
    vocab._default_unk_index = _default_unk_index

model_path = "./models_v3/"

def read_vocab(path):
    import pickle
    pkl_file = open(path, 'rb')
    vocab = pickle.load(pkl_file)
    pkl_file.close()
    return vocab
# toxic_vocab = read_vocab(model_path + 'toxic_vocab.pkl')
# severe_toxic_vocab = read_vocab(model_path + 'severe_toxic_vocab.pkl')
# obscene_vocab = read_vocab(model_path + 'obscene_vocab.pkl')
# insult_vocab = read_vocab(model_path + 'insult_vocab.pkl')
# threat_vocab = read_vocab(model_path + 'threat_vocab.pkl')
# identity_hate_vocab = read_vocab(model_path + 'identity_hate_vocab.pkl')

toxic_vocab = read_vocab(model_path + 'toxic_vocab (1).pkl')
print(toxic_vocab)
# severe_toxic_vocab = read_vocab(model_path + 'severe_toxic_vocab (1).pkl')
# obscene_vocab = read_vocab(model_path + 'obscene_vocab (1).pkl')
# insult_vocab = read_vocab(model_path + 'insult_vocab (1).pkl')
# threat_vocab = read_vocab(model_path + 'threat_vocab (1).pkl')
# identity_hate_vocab = read_vocab(model_path + 'identity_hate_vocab (1).pkl')

toxic_model = CNN_LSTM_2_conv(100, toxic_vocab, 100)
# severe_toxic_model = CNN_LSTM_2_conv(100, severe_toxic_vocab, 100)
# obscene_model = CNN_LSTM_2_conv(100, obscene_vocab, 100)
# insult_model = CNN_LSTM_2_conv(100, insult_vocab, 100)
# threat_model = CNN_LSTM_2_conv(100, threat_vocab, 100)
# identity_hate_model = CNN_LSTM_2_conv(100, identity_hate_vocab, 100)

toxic_model.load_state_dict(torch.load(model_path + 'toxic_sd_model_CNN_LSTM_v3.pt'))
# severe_toxic_model.load_state_dict(torch.load(model_path + 'severe_toxic_sd_model_CNN_LSTM_v3.pt'))
# obscene_model.load_state_dict(torch.load(model_path + 'obscene_model_sd_CNN_LSTM_v3.pt'))
# insult_model.load_state_dict(torch.load(model_path + 'insult_model_sd_CNN_LSTM_v3.pt'))
# threat_model.load_state_dict(torch.load(model_path + 'threat_model_sd_CNN_LSTM_v3.pt'))
# identity_hate_model.load_state_dict(torch.load(model_path + 'identity_hate_model_sd_CNN_LSTM_v3.pt'))

toxic_model.eval()
# severe_toxic_model.eval()
# obscene_model.eval()
# insult_model.eval()
# threat_model.eval()
# identity_hate_model.eval()

def tokenizer(text):
    spacy_en = spacy.load("en")
    return [tok.text for tok in spacy_en(text)]

def predict(sentence):
    tokens = tokenizer(sentence)
    token_ints = [vocab.stoi[tok] for tok in tokens]
    token_tensor = torch.LongTensor(token_ints).view(-1,1)
    lengths = torch.Tensor([len(token_ints)])

    toxic_output            = toxic_model(token_tensor, lengths)
    # severe_toxic_output     = severe_toxic_model(token_tensor, lengths)
    # obscene_output          = obscene_model(token_tensor, lengths)
    # insult_output           = insult_model(token_tensor, lengths)
    # threat_output           = threat_model(token_tensor, lengths)
    # identity_hate_output    = identity_hate_model(token_tensor, lengths)
    return toxic_output
    # return [toxic_output, severe_toxic_output, obscene_output, insult_output, threat_output, identity_hate_output]

@app.route('/')
def index():
    print("args", request.args)
    # Contains a dictionary containing the parsed contents of the query string
    if(request.args):
        # Passes contents of query string to the prediction function contained in model.py
        preds = predict(request.args['text_in'])
        print("preds", preds)
        prediction = preds
        # Indexes the returned dictionary for the sentiment probability
        # prediction = []
        # for pred in preds:
        #     if(prediction > 0.5):
        #         prediction += "Subjective"    
        #     else:
        #         prediction += "Objective"
        return flask.render_template('index.html', text_in=request.args['text_in'], prediction=prediction, header_comment="Comment:", header_prediction="Bad or Good:")
    # If the parsed query string does not contain anything then return index page
    else:
        return flask.render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
