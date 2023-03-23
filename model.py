import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var

from transformers import BertModel, GPT2Model

class TransformerVAE(nn.Module):
    def __init__(self, bert_model, gpt2_model, latent_dim):
        super(VAE, self).__init__()
        self.bert_model = bert_model
        self.gpt2_model = gpt2_model
        self.latent_dim = latent_dim
        self.hidden_dim = 768  # output dimension of BERT and GPT-2
        
        # Encoder layers
        self.encoder_fc1 = nn.Linear(self.hidden_dim, 512)
        self.encoder_fc2 = nn.Linear(512, 256)
        self.encoder_fc31 = nn.Linear(256, self.latent_dim)  # mean
        self.encoder_fc32 = nn.Linear(256, self.latent_dim)  # log variance
        
        # Decoder layers
        self.decoder_fc1 = nn.Linear(self.latent_dim, 256)
        self.decoder_fc2 = nn.Linear(256, 512)
        self.decoder_fc3 = nn.Linear(512, self.hidden_dim)
        
    def encode(self, x):
        # Encode the input sentence using BERT
        with torch.no_grad():
            bert_output = self.bert_model(x)[0]  # output of last layer
            
        # Flatten and encode the BERT output
        h = bert_output.flatten(start_dim=1)
        h = F.relu(self.encoder_fc1(h))
        h = F.relu(self.encoder_fc2(h))
        mean = self.encoder_fc31(h)
        logvar = self.encoder_fc32(h)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        # Reparameterize the latent variable
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mean
        return z
    
    def decode(self, z):
        # Decode the latent variable using GPT-2
        gpt2_output = self.gpt2_model(z)[0]  # output of last layer
        h = gpt2_output[:, -1, :]  # take the last hidden state as the output
        h = F.relu(self.decoder_fc1(h))
        h = F.relu(self.decoder_fc2(h))
        recon_x = torch.sigmoid(self.decoder_fc3(h))
        return recon_x
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decode(z)
        return recon_x, mean, logvar

class TextAutoencoder(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', gpt_model_name='gpt2'):
        super()
        
        # Encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Decoder
        self.gpt = GPT2Model.from_pretrained(gpt_model_name)
        self.decoder = nn.Linear(gpt_config.hidden_size, gpt_config.vocab_size)
        
    def forward(self, input_ids, attention_mask, target_ids=None):
        # Encode input sequence with BERT
        hidden_states, _ = self.bert(input_ids, attention_mask)
        
        if target_ids is not None:
            # Use teacher forcing
            output = self.gpt(inputs_embeds=hidden_states, past_key_values=None, use_cache=True, input_ids=target_ids)
        else:
            # Generate output sequence from scratch
            output = self.gpt(inputs_embeds=hidden_states, past_key_values=None, use_cache=True)
            
        logits = self.decoder(output.last_hidden_state)
        
        return logits
    
    def generate(self, input_ids, attention_mask, max_length):
        # Generate output sequence from scratch
        hidden_states, _ = self.bert(input_ids, attention_mask)
        input_shape = hidden_states.size()[:-1]
        device = hidden_states.device
        
        past_key_values = None
        outputs = input_ids
        
        for i in range(max_length):
            # Use the decoder to generate the next token
            output = self.gpt(inputs_embeds=hidden_states, past_key_values=past_key_values, use_cache=True, input_ids=outputs)
            logits = self.decoder(output.last_hidden_state[:, -1, :])
            next_token = logits.argmax(dim=-1).view(*input_shape).to(device)
            outputs = torch.cat((outputs, next_token), dim=-1)
            
            # Update the past key values for the next time step
            past_key_values = output.past_key_values
            
        return outputs

class VAEDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, embedding, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.embedding = embedding
        self.word_dropout_rate = word_dropout
        # self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.decoder_hidden_factor = num_layers
        
        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM
            self.latent2c = nn.Sequential(
                nn.Linear(latent_size, latent_size),
                nn.ReLU(),
		        nn.Linear(latent_size, hidden_size * self.decoder_hidden_factor),
            )
        else:
            raise ValueError()

        self.latent2hidden = nn.Sequential(
                nn.Linear(latent_size, latent_size),
                nn.ReLU(),
		        nn.Linear(latent_size, hidden_size * self.decoder_hidden_factor),
            )

        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers,
                               batch_first=True)
        
        self.outputs2vocab = nn.Sequential(
		nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, vocab_size),
            )

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, use_teacher_forcing, params):
        hidden = None
        if use_teacher_forcing:
            logits, b, s = self.teacher_forward(*params)
        else:       
            input_sequence, z, is_first_word = params
            b, s = input_sequence.shape[0], 1
            if is_first_word:
                hidden = self.latent2hidden(z)

                if self.num_layers > 1:
                    # unflatten hidden state
                    if self.rnn_type == 'lstm':
                        hidden = hidden.view(self.decoder_hidden_factor, b, self.hidden_size), self.latent2c(z).view(self.decoder_hidden_factor, b, self.hidden_size)
                    else:
                        hidden = hidden.view(self.decoder_hidden_factor, b, self.hidden_size)
                else:
                    if self.rnn_type == 'lstm':
                        hidden = (hidden.unsqueeze(0), self.latent2c(z).unsqueeze(0))
                    else:
                        hidden = hidden.unsqueeze(0)


                # hidden = hidden.view(-1, b, hidden.shape[1])  # D∗num_layers,N,H out
                # print(6, hidden.shape)
            else:
                hidden = z

            # print(1, input_sequence.shape, z.shape)
            # print(2, hidden.shape)
            if not is_first_word and self.word_dropout_rate > 0:
                # randomly replace decoder input with <unk>
                prob = torch.rand(input_sequence.size())
                if torch.cuda.is_available():
                    prob=prob.cuda()
                # prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
                # decoder_input_sequence = input_sequence.clone()
                # print(input_sequence)
                input_sequence[prob < self.word_dropout_rate] = self.unk_idx
                # print(input_sequence)
                # input_embedding = self.embedding(input_sequence)
            input_sequence = input_sequence.unsqueeze(1)
            # print(3, input_sequence.shape)
            input_embedding = self.embedding(input_sequence)

            # print(4, input_embedding.shape)
            # print(5, hidden.shape)
            output, hidden = self.decoder_rnn(input_embedding, hidden)

            # print(11, hidden.shape)

            logits = self.outputs2vocab(output)
            # print("input_sequence.size():", input_sequence.size())
            # print(input_sequence.shape[0], 1)
            # 0/0

        # project outputs to vocab
        logp = self.log_softmax(logits).view(b, s, self.embedding.num_embeddings)
        # print(7, logp.shape)
        return logp, hidden

    def teacher_forward(self, input_sequence, batch_size, sorted_idx, mean, logv, z, reversed_idx, sorted_lengths):
        # DECODER
        # if False:

        #     generations, z, padded_outputs = self.inference(z=z)
        
        # else:
        hidden = self.latent2hidden(z)
    
        if self.num_layers > 1:
            # unflatten hidden state
            if self.rnn_type == 'lstm':
                hidden = hidden.view(self.decoder_hidden_factor, batch_size, self.hidden_size), self.latent2c(z).view(self.decoder_hidden_factor, batch_size, self.hidden_size)
            else:
                hidden = hidden.view(self.decoder_hidden_factor, batch_size, self.hidden_size)
        else:
            if self.rnn_type == 'lstm':
                hidden = (hidden.unsqueeze(0), self.latent2c(z).unsqueeze(0))
            else:
                hidden = hidden.unsqueeze(0)
        # decoder input
        if self.word_dropout_rate > 0:
            input_sequence = input_sequence[sorted_idx]
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        # input_embedding = self.embedding_dropout(self.embedding(input_sequence))
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]

        ###########
            
        padded_outputs = padded_outputs.contiguous()
        # print(padded_outputs.shape)
        # print(padded_outputs)
        # 0/0
        # _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        return self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), b, s

    def generate_seq(self, batch_size, batch_length, z):
        # print(batch_size, batch_length, z.shape)
        # exit()
        use_teacher_forcing = False
        input_sequence = to_var(torch.Tensor(batch_size).fill_(self.eos_idx).long())
        params = input_sequence, z, True
        hidden = None
        output = []
        for di in range(max(batch_length)):
            logp, hidden = self(use_teacher_forcing, params)
            # decoder_output, decoder_hidden = decoder(use_teacher_forcing, params)
            topv, topi = logp.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            # print(8, target_tensor[:, di].shape, target_tensor[:, di])
            # print(8, target_tensor[:, di].shape)
            # print(9, target_tensor.shape)

            output.append(decoder_input)
            # input(":")

            # logp = logp.squeeze(1)
            
            # local_loss = NLL(logp, target_tensor[:, di])
            # print("criterion(logp, target_tensor[di])", local_loss.shape)
            # print(local_loss)
            # 1/0
            # nll_loss += NLL(logp, target_tensor[:, di])
            
            # for seq_index, decoder_next_input in enumerate(decoder_input):
            #     if decoder_next_input.item() == eos_idx:
            #         ended_sequence_indices_set.add(seq_index)

            # print(params[0], decoder_input)
            params = decoder_input, hidden, False
        return output
    
    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)
        
        if self.bidirectional or self.num_layers > 1:  # newly added bidirectional
            # unflatten hidden state
            if self.rnn_type == 'lstm':
                hidden = hidden.view(self.decoder_hidden_factor, batch_size, self.hidden_size), self.latent2c(z).view(self.decoder_hidden_factor, batch_size, self.hidden_size)
            else:
                hidden = hidden.view(self.decoder_hidden_factor, batch_size, self.hidden_size)
                
                # hidden = hidden.unsqueeze(0)  # newly added insync with the original code
        else:
            if self.rnn_type == 'lstm':
                hidden = (hidden.unsqueeze(0), self.latent2c(z).unsqueeze(0))
            else:
                hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
        # all idx of batch which are still generating
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long()

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()
        # print(batch_size, self.max_sequence_length)
        padded_outputs = self.tensor(batch_size, self.max_sequence_length, self.hidden_size).fill_(self.pad_idx)
        # print(padded_outputs.shape)

        t = 0
        while t < self.max_sequence_length and len(running_seqs) > 0:

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())  # newly changed from eos to sos
            # no dropout in inference time
            # elif self.word_dropout_rate > 0:
            #     # randomly replace decoder input with <unk>
            #     prob = torch.rand(input_sequence.size())
            #     if torch.cuda.is_available():
            #         prob=prob.cuda()
            #     # prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            #     # decoder_input_sequence = input_sequence.clone()
            #     # print(input_sequence)
            #     input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            #     # print(input_sequence)
            #     # input_embedding = self.embedding(input_sequence)
            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)
            # print(logits.shape, input_sequence.shape)
            # save next input
            output = output.squeeze(1)
            # print(output.shape)
            generations = self._save_sample(generations, input_sequence, sequence_running, t)
            padded_outputs = self._save_sample(padded_outputs, output, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx)
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                # print(input_sequence.shape)
                if self.rnn_type == 'lstm':
                    h, c = hidden
                    hidden = h[:, running_seqs], c[:, running_seqs]
                else:
                    hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z, padded_outputs

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.reshape(-1)

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
    

class VAEEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.decoder_hidden_factor = num_layers
        
        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM
            self.latent2c = nn.Sequential(
                nn.Linear(latent_size, latent_size),
                nn.ReLU(),
		        nn.Linear(latent_size, hidden_size * self.decoder_hidden_factor),
            )
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)


        self.hidden2mean_logv = nn.Sequential(
                nn.Linear(hidden_size * self.hidden_factor, hidden_size * self.hidden_factor),
                nn.ReLU(),
		        nn.Linear(hidden_size * self.hidden_factor, 2* latent_size)
            )
       

    def forward(self, input_sequence, length):
        
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding_dropout(self.embedding(input_sequence))

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            if self.rnn_type == 'lstm':
                hidden = hidden[0].view(batch_size, self.hidden_size*self.hidden_factor)
            else:
                hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            if self.rnn_type == 'lstm':
                hidden = hidden[0].squeeze()
            else:
                hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean, logv = torch.chunk(self.hidden2mean_logv(hidden), 2, dim=-1)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean
        _, reversed_idx = torch.sort(sorted_idx)

        return batch_size, sorted_idx, mean, logv, z, reversed_idx, sorted_lengths

