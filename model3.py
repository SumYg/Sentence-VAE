import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var

from transformers import AutoModel, AutoTokenizer

# def get_tokenizer(pretrained_name="princeton-nlp/sup-simcse-bert-base-uncased"):
#     tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
#     return tokenizer

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
        
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        # self.bert.eval()
        # embedding_size = self.bert.config.hidden_size
        # self.bert.requires_grad = False
        self.word_dropout_rate = word_dropout
        # self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.decoder_hidden_factor = num_layers
        

        # self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
        #                        batch_first=True)
        self.encoder = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

        # disable the gradient of encoder
        # for param in self.encoder.parameters():
            # param.requires_grad = False

        self.hidden2mean_logv = nn.Sequential(
                # nn.Linear(hidden_size * self.hidden_factor, hidden_size * self.hidden_factor),
                # nn.ReLU(),
		        nn.Linear(hidden_size * self.hidden_factor, 2* latent_size)
            )
       

    def forward(self, input_sequence, attention_mask):
        
        batch_size = input_sequence.size(0)

        # ENCODER
        hidden = self.encoder(input_ids=input_sequence, attention_mask=attention_mask, output_hidden_states=False, return_dict=True).pooler_output
        # print(hidden.shape)
        # if self.bidirectional or self.num_layers > 1:
        #     # flatten hidden state
        #     if self.rnn_type == 'lstm':
        #         hidden = hidden[0].view(batch_size, self.hidden_size*self.hidden_factor)
        #     else:
        #         hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        # else:
        #     if self.rnn_type == 'lstm':
        #         hidden = hidden[0].squeeze()
        #     else:
        #         hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean, logv = torch.chunk(self.hidden2mean_logv(hidden), 2, dim=-1)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean
        # print(z.shape)
        return mean, logv, z



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
        self.vocab_size = vocab_size

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
            input_embedding = self.embedding(input_sequence.to('cuda:0')).to('cuda:1')

            # print(4, input_embedding.shape)
            # print(5, hidden.shape)
            output, hidden = self.decoder_rnn(input_embedding, hidden)

            # print(11, hidden.shape)

            logits = self.outputs2vocab(output)
            # print("input_sequence.size():", input_sequence.size())
            # print(input_sequence.shape[0], 1)
            # 0/0

        # project outputs to vocab
        logp = self.log_softmax(logits).view(b, s, self.vocab_size)
        # print(7, logp.shape)
        return logp, hidden

    def teacher_forward(self, input_sequence, batch_size, sorted_idx, mean, logv, z, reversed_idx, sorted_lengths):
        # DECODER
        # if False:

        #     generations, z, padded_outputs = self.inference(z=z)
        
        # else:
        input_sequence = input_sequence[sorted_idx]
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
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence.to(self.embedding.weight.device)).to('cuda:1')
        else:
            input_embedding = self.embedding(input_sequence).to('cuda:1')
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

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

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
    
