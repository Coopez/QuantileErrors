import torch
import torch.nn as nn
import random
## Define simple LSTM with attention

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
 # set device to device avail 
    def forward(self, x, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        out, _ = self.lstm(x)
        
        # Compute attention weights
        attn_weights = self.attention(out)
        attn_weights = self.softmax(attn_weights.squeeze(-1)).unsqueeze(-1)
        
        # Apply attention weights
        context_vector = torch.sum(attn_weights * out, dim=1)
        
        out = self.tanh(context_vector)
        out = self.fc(out)
        
        return out
    

# make a normal LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    
        out, _ = self.lstm(x)
        
        out = self.fc(out[:, -1, :])
        
        return out


# make a encoder decoder LSTM
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        return hn, cn

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_size, dim_output_size = 1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = dim_output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.output_size)
        
    def forward(self, x, hn, cn, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        out, (hn, cn) = self.lstm(x, (hn, cn))
        out = self.fc(out[:, -1, :])
        return out.unsqueeze(1), hn, cn

class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon_size, output_size = 1,training="normal",teacher_forcing_ratio=0.5):
        super(EncoderDecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon_size = horizon_size
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(input_size=1,hidden_size=hidden_size, num_layers=num_layers,latent_size = hidden_size)
        self.output_size = output_size
        self.training = training
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
    def forward(self, x,y, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        batch_size = x.size(0)
        
        # Encode the input sequence
        hn, cn = self.encoder(x, device)
        
        # Initialize the decoder input with zeros
        decoder_input = torch.zeros(batch_size, 1 , self.output_size).to(device) # 1 timesstep
        
        # Initialize the output tensor
        outputs = torch.zeros(batch_size,self.horizon_size,1).to(device)
        
        # Decode the sequence
        if self.training == "normal":
            for t in range(self.horizon_size):
                decoder_output, hn, cn = self.decoder(decoder_input, hn, cn, device)
                outputs[:, t, :] = decoder_output.squeeze(1)
                decoder_input = decoder_output
        elif self.training == "teacher_forcing":
            # use teacher forcing
            if random.random() < self.teacher_forcing_ratio:
                for t in range(self.horizon_size):
                    decoder_output, hn, cn = self.decoder(decoder_input, hn, cn, device)
                    outputs[:, t, :] = decoder_output.squeeze(1)
                    decoder_input = y[:, t, :]
            else:
                for t in range(self.horizon_size):
                    decoder_output, hn, cn = self.decoder(decoder_input, hn, cn, device)
                    outputs[:, t, :] = decoder_output.squeeze(1)
                    decoder_input = decoder_output
        else:
            raise ValueError("training must be either 'normal' or 'teacher_forcing")
        return outputs.squeeze(-1)
