import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Distribution(nn.Module):
    def __init__(self, nslot, hidden_size, dropout):
        super(Distribution, self).__init__()

        self.query = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.key = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.beta = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.hidden_size = hidden_size

    def init_p(self, bsz, nslot):
        return None

    @staticmethod
    def process_softmax(beta, prev_p):
        if prev_p is None:
            return torch.zeros_like(beta), torch.ones_like(beta), torch.zeros_like(beta)

        beta_normalized = beta - beta.max(dim=-1)[0][:, None]
        x = torch.exp(beta_normalized)

        prev_cp = torch.cumsum(prev_p, dim=1)
        mask = prev_cp[:, 1:]
        mask = mask.masked_fill(mask < 1e-5, 0.)
        mask = F.pad(mask, (0, 1), value=1)

        x_masked = x * mask

        p = F.normalize(x_masked, p=1)
        cp = torch.cumsum(p, dim=1)
        rcp = torch.cumsum(p.flip([1]), dim=1).flip([1])
        return cp, rcp, p

    def forward(self, in_val, prev_out_M, prev_p):
        query = self.query(in_val)
        key = self.key(prev_out_M)
        beta = self.beta(query[:, None, :] + key).squeeze(dim=2)
        beta = beta / math.sqrt(self.hidden_size)
        cp, rcp, p = self.process_softmax(beta, prev_p)
        return cp, rcp, p


class Cell(nn.Module):
    def __init__(self, hidden_size, dropout, activation=None):
        super(Cell, self).__init__()
        self.hidden_size = hidden_size
        self.cell_hidden_size = 4 * hidden_size

        self.input_t = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, self.cell_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.cell_hidden_size, hidden_size * 4),
        )

        self.gates = nn.Sequential(
            nn.Sigmoid(),
        )

        assert activation is not None
        self.activation = activation

        self.drop = nn.Dropout(dropout)

    def forward(self, vi, hi):
        input = torch.cat([vi, hi], dim=-1)

        g_input, cell = self.input_t(input).split(
            (self.hidden_size * 3, self.hidden_size),
            dim=-1
        )

        gates = self.gates(g_input)
        vg, hg, cg = gates.chunk(3, dim=1)
        output = self.activation(vg * vi + hg * hi + cg * cell)
        return output


class OrderedMemoryRecurrent(nn.Module):
    def __init__(self, input_size, slot_size, nslot,
                 dropout=0.2, dropoutm=0.2):
        super(OrderedMemoryRecurrent, self).__init__()

        self.activation = nn.LayerNorm(slot_size)
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, slot_size),
            self.activation
        )

        self.distribution = Distribution(nslot, slot_size, dropoutm)

        self.cell = Cell(slot_size, dropout, activation=self.activation)

        self.nslot = nslot
        self.slot_size = slot_size
        self.input_size = input_size

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        zeros = weight.new(bsz, self.nslot, self.slot_size).zero_()
        p = self.distribution.init_p(bsz, self.nslot)
        return (zeros, zeros, p)

    def omr_step(self, in_val, prev_M, prev_out_M, prev_p):
        batch_size, nslot, slot_size = prev_M.size()
        _batch_size, slot_size = in_val.size()

        assert self.slot_size == slot_size
        assert self.nslot == nslot
        assert batch_size == _batch_size

        cp, rcp, p = self.distribution(in_val, prev_out_M, prev_p)

        curr_M = prev_M * (1 - rcp)[:, :, None] + prev_out_M * rcp[:, :, None]

        M_list = []
        h = in_val
        for i in range(nslot):
            if i == nslot - 1 or cp[:, i + 1].max() > 0:
                h = self.cell(h, curr_M[:, i, :])
                h = in_val * (1 - cp)[:, i, None] + h * cp[:, i, None]
            M_list.append(h)
        out_M = torch.stack(M_list, dim=1)

        output = out_M[:, -1]
        return output, curr_M, out_M, p

    def forward(self, X, hidden, mask=None):
        prev_M, prev_memory_output, prev_p = hidden
        output_list = []
        p_list = []
        X_projected = self.input_projection(X)
        if mask is not None:
            padded = ~mask
        for t in range(X_projected.size(0)):
            output, prev_M, prev_memory_output, prev_p = self.omr_step(
                X_projected[t], prev_M, prev_memory_output, prev_p)
            if mask is not None:
                padded_1 = padded[t, :, None]
                padded_2 = padded[t, :, None, None]
                output = output.masked_fill(padded_1, 0.)
                prev_p = prev_p.masked_fill(padded_1, 0.)
                prev_M = prev_M.masked_fill(padded_2, 0.)
                prev_memory_output = prev_memory_output.masked_fill(padded_2, 0.)
            output_list.append(output)
            p_list.append(prev_p)

        output = torch.stack(output_list)
        probs = torch.stack(p_list)

        return (output,
                probs,
                (prev_M, prev_memory_output, prev_p))


class OrderedMemory(nn.Module):
    def __init__(self, config):
        super(OrderedMemory, self).__init__()

        input_size = config["hidden_size"]
        slot_size = config["hidden_size"]
        nslot = config["memory_slots"]
        if not self.training:
            nslot = 2 * nslot
        dropoutm = config["memory_dropout"]
        dropout = config["dropout"]
        bidirection = False #config["bidirection"]
        self.output_last = False #config["output_last"]

        self.OM_forward = OrderedMemoryRecurrent(input_size, slot_size, nslot,
                                                 dropout=dropout, dropoutm=dropoutm)
        if bidirection:
            self.OM_backward = OrderedMemoryRecurrent(input_size, slot_size, nslot,
                                                      dropout=dropout, dropoutm=dropoutm)

        self.bidirection = bidirection

    def init_hidden(self, bsz):
        return self.OM_forward.init_hidden(bsz)

    def forward(self, X, input_mask, **kwargs):

        X = X.permute(1, 0, 2).contiguous()
        mask = input_mask.permute(1, 0).contiguous()
        mask = mask.bool()

        output_last = False #self.output_last

        bsz = X.size(1)
        lengths = mask.sum(0)
        init_hidden = self.init_hidden(bsz)

        output_list = []
        prob_list = []

        om_output_forward, prob_forward, _ = self.OM_forward(X, init_hidden, mask)
        if output_last:
            output_list.append(om_output_forward[-1])
        else:
            output_list.append(om_output_forward[lengths - 1, torch.arange(bsz).long()])
        prob_list.append(prob_forward)

        if self.bidirection:
            om_output_backward, prob_backward, _ = self.OM_backward(X.flip([0]), init_hidden, mask.flip([0]))
            output_list.append(om_output_backward[-1])
            prob_list.append(prob_backward.flip([0]))

        output = torch.cat(output_list, dim=-1)
        self.probs = prob_list[0]

        return {"global_state": output, "sequence": om_output_forward.permute(1, 0, 2).contiguous(),
                "input_mask": input_mask}
    