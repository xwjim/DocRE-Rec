import torch
import torch.autograd as autograd
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch.nn.utils import rnn

class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, num_units, nlayers, bidir, dropout):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)

        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.init_c = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

        self.dropout = LockedDropout(dropout)
        self.nlayers = nlayers

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i,hidden_init = None,ceil_init = None):
        if hidden_init is not None:
            return hidden_init,ceil_init
        else:
            return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None,hidden_init = None,ceil_init = None):
        lengths = input_lengths.clone()
        lens, indices = torch.sort(lengths, 0, True)
        input = input[indices]
        _, _indices = torch.sort(indices, 0)

        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        #if input_lengths is not None:
        #    lens = input_lengths.data.cpu().numpy()
        lens[lens==0] = 1

        for i in range(self.nlayers):

            hidden, c = self.get_init(bsz,i,hidden_init,ceil_init)

            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens.cpu(), batch_first=True)

            self.rnns[i].flatten_parameters()
            output, (hidden, c) = self.rnns[i](output, (hidden, c))


            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)

            outputs.append(output)
            outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            outputs.append(c.permute(1, 0, 2).contiguous().view(bsz, -1))

        for i, output in enumerate(outputs):
            outputs[i] = output[_indices]
        return outputs[0],outputs[1],outputs[2]
