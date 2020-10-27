import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import pytorch_lightning as pl

class CaptioningModel(pl.LightningModule):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CaptioningModel, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Encoder
        self.encoder = EncoderCNN(self.embed_size)

        # Decoder
        self.decoder = DecoderRNN(self.embed_size, self.hidden_size, self.vocab_size, num_layers=self.num_layers)

    def forward(self, images, captions):
        batch_size = images.size(0)

        # Encoder
        features = self.encoder(images).unsqueeze(1)

        # Decoder
        out = self.decoder(features, captions)

        return out
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        return optimizer

    def training_step(self, batch, batch_idx):
        images, captions = batch

        y_hat = self(images, captions)

        loss = F.cross_entropy(y_hat.view(-1, self.vocab_size), captions.view(-1))

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        images, captions = batch
        y_hat = self(images)
        loss = F.cross_entropy(y_hat, captions)

        return {'test_loss': loss}

    def sample(self, images, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        features = self.encoder(images).unsqueeze(1)

        return self.decoder.sample(features)


class EncoderCNN(nn.Module):
    
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module): 
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        batch_size = features.size(0)

        embedded_words = self.word_embeddings(captions[:, :-1])

        lstm_input = torch.cat((features.view((batch_size, 1, -1)), embedded_words), dim=1)
        
        lstm_out, _ = self.lstm(lstm_input)

        out = self.fc(lstm_out)
        
        return out
        

    def sample(self, inputs, states=None, max_len=20):
        caption = []
        
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
                                
            out = self.fc(lstm_out)
            
            word_idx = out.max(2)[1]
            
            caption.append(word_idx.item())
            
            inputs = self.word_embeddings(word_idx)
                                
        return caption