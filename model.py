import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var


class SentenceVAE(nn.Module):
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
                nn.Linear(latent_size, hidden_size * self.decoder_hidden_factor),
                nn.ReLU(),
            )
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers,
                               batch_first=True)


        self.hidden2mean = nn.Sequential(
                nn.Linear(hidden_size * self.hidden_factor, latent_size),
                nn.ReLU(),
            )
        self.hidden2logv = nn.Sequential(
                nn.Linear(hidden_size * self.hidden_factor, latent_size),
                nn.ReLU(),
            )
        
        self.latent2hidden = nn.Sequential(
                nn.Linear(latent_size, hidden_size * self.decoder_hidden_factor),
                nn.ReLU(),
            )
        
        self.outputs2vocab = nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_size, vocab_size),
            )

    def forward(self, input_sequence, length):
        # print(input_sequence[-1])
        # print(input_sequence.shape, length)
        batch_size, sorted_idx, mean, logv, z, reversed_idx, input_embedding, sorted_lengths = self.get_representation(input_sequence, length)

        # DECODER
        if True:

            generations, z, padded_outputs = self.inference(z=z)
        # print(padded_outputs)
        # print(padded_outputs.shape, generations.shape)
        
        # padded_outputs = generations
        # padded_outputs = padded_outputs.contiguous()
        # print(padded_outputs.size())
        
        # if not teacher forcing
                # padded_outputs = padded_outputs[reversed_idx]
                # b,s, _ = padded_outputs.size()
                # # print(padded_outputs.view(-1, padded_outputs.size(2)))
                # # project outputs to vocab
                # logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
                # # print(logp.shape)
                # logp = logp.view(b, s, self.embedding.num_embeddings)
                # # print(logp.shape)
                # return logp, mean, logv, z
        
        else:
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
                input_embedding = self.embedding(decoder_input_sequence)
            input_embedding = self.embedding_dropout(input_embedding)
            packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

            # decoder forward pass
            outputs, _ = self.decoder_rnn(packed_input, hidden)

            # process outputs
            padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
            
        padded_outputs = padded_outputs.contiguous()
        # print(padded_outputs.shape)
        # print(padded_outputs)
        # 0/0
        # _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp, mean, logv, z
        
        
        
        
        
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
        # if self.word_dropout_rate > 0:
        #     # randomly replace decoder input with <unk>
        #     prob = torch.rand(input_sequence.size())
        #     if torch.cuda.is_available():
        #         prob=prob.cuda()
        #     prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
        #     decoder_input_sequence = input_sequence.clone()
        #     decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
        #     input_embedding = self.embedding(decoder_input_sequence)
        
        input_embedding = self.embedding_dropout(self.embedding(self.tensor([self.eos_idx for _ in range(batch_size)]).long()))
        # print("input_embedding", input_embedding.shape, len(sorted_lengths.data.tolist()), sorted_lengths.data.sum(), sorted_lengths.data.tolist())
        # packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True, enforce_sorted=True)
        # print("packed_input")
        # print(packed_input)
        # print(packed_input.data.shape, packed_input.batch_sizes.sum())
        # print(hidden[0].shape)
        # decoder forward pass
        print(input_embedding.unsqueeze(1).shape)
        prev_embeddings = input_embedding.unsqueeze(1)
        print(prev_embeddings.shape)
        
        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long()  # all idx of batch
        # all idx of batch which are still generating
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long()
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()
        
        # generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()
        
        outputs = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx)
        for i in range(self.max_sequence_length):
            
            output, hidden = self.decoder_rnn(prev_embeddings, hidden)
            print(output.shape)
            
            
            # save next input
            outputs = self._save_sample(outputs, input_sequence, sequence_running, t)
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
                
                if self.rnn_type == 'lstm':
                    h, c = hidden
                    hidden = h[:, running_seqs], c[:, running_seqs]
                else:
                    hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()
                
            # exit()
        
        
        
        # print("Outputs")
        # print(outputs)
        # print(outputs.data.shape[0])
        # print( outputs.data.shape[0] == outputs.batch_sizes.sum())
        # print(outputs.data.shape, outputs.batch_sizes.sum())
        # process outputs
        # padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        # padded_outputs = padded_outputs.contiguous()
        padded_outputs = padded_outputs[reversed_idx]
        print(padded_outputs.size())
        print(padded_outputs.view(-1, padded_outputs.size(2)).shape)
        
        generations, z = self.inference(z=z)
        
        padded_outputs = generations
        # padded_outputs = padded_outputs.contiguous()
        print(padded_outputs.size())
        exit()
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp, mean, logv, z

    def get_representation(self, input_sequence, length):
        
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)

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
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean
        _, reversed_idx = torch.sort(sorted_idx)

        return batch_size, sorted_idx, mean, logv, z, reversed_idx, input_embedding, sorted_lengths

    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

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

        # if self.bidirectional or self.num_layers > 1:
        #     # unflatten hidden state
        #     if self.rnn_type == 'lstm':
        #         hidden = (hidden.view(self.hidden_factor, batch_size, self.hidden_size), self.latent2c(z).view(self.hidden_factor, batch_size, self.hidden_size))
        #     else:
        #         hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        # else:
        #     if self.rnn_type == 'lstm':
        #         hidden = (hidden.unsqueeze(0), self.latent2c(z).unsqueeze(0))
        #     else:
        #         hidden = hidden.unsqueeze(0)

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
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.eos_idx).long())
            elif self.word_dropout_rate > 0:
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
