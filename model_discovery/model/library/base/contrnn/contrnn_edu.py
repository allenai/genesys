import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.layers import Linear
from models.utils import gelu
import math


class FOCN(nn.Module):
    def __init__(self, config):
        super(FOCN, self).__init__()

        self.config = config
        self.hidden_size = config["hidden_size"]
        self.cell_hidden_size = config["cell_hidden_size"]
        self.window_size = config["window_size"]
        self.stop_threshold = config["stop_threshold"]
        # self.switch_threshold = config["switch_threshold"]
        self.entropy_gamma = config["entropy_gamma"]
        self.structure_gamma = 0.01  # config["structure_gamma"]
        self.speed_gamma = config["speed_gamma"]
        self.in_dropout = config["in_dropout"]
        self.hidden_dropout = config["hidden_dropout"]
        self.recurrent_momentum = config["recurrent_momentum"]
        self.small_d = config["small_d"]

        self.START = nn.Parameter(T.randn(self.hidden_size))
        self.END = nn.Parameter(T.randn(self.hidden_size))

        if self.recurrent_momentum:
            self.past_transition_features = nn.Parameter(T.randn(self.small_d))
            self.past_non_transition_features = nn.Parameter(T.randn(self.small_d))
            self.conv_layer = Linear(self.window_size * self.hidden_size + self.small_d, self.hidden_size)
        else:
            self.conv_layer = Linear(self.window_size * self.hidden_size, self.hidden_size)

        self.scorer = Linear(self.hidden_size, 1)

        self.initial_transform_layer = Linear(self.hidden_size, self.hidden_size)
        self.wcell1 = Linear(2 * self.hidden_size, self.cell_hidden_size)
        self.wcell2 = Linear(self.cell_hidden_size, 4 * self.hidden_size)
        self.LN = nn.LayerNorm(self.hidden_size)

        self.eps = 1e-8

    # %%
    def sum_normalize(self, logits, dim=-1):
        return logits / T.sum(logits + self.eps, keepdim=True, dim=dim)

    # %%
    def augment_sequence(self, sequence, input_mask):
        N, S, D = sequence.size()
        assert input_mask.size() == (N, S, 1)

        """
        AUGMENT SEQUENCE WITH START AND END TOKENS
        """
        # ADD START TOKEN
        START = self.START.view(1, 1, D).repeat(N, 1, 1)
        sequence = T.cat([START, sequence], dim=1)
        assert sequence.size() == (N, S + 1, D)
        input_mask = T.cat([T.ones(N, 1, 1).float().to(input_mask.device), input_mask], dim=1)
        assert input_mask.size() == (N, S + 1, 1)

        # ADD END TOKEN
        input_mask_no_end = T.cat([input_mask.clone(), T.zeros(N, 1, 1).float().to(input_mask.device)], dim=1)
        input_mask_yes_end = T.cat([T.ones(N, 1, 1).float().to(input_mask.device), input_mask.clone()], dim=1)
        END_mask = input_mask_yes_end - input_mask_no_end
        assert END_mask.size() == (N, S + 2, 1)

        END = self.END.view(1, 1, D).repeat(N, S + 2, 1)
        sequence = T.cat([sequence, T.zeros(N, 1, D).float().to(sequence.device)], dim=1)
        sequence = END_mask * END + (1 - END_mask) * sequence

        input_mask = input_mask_yes_end
        input_mask_no_start = T.cat([T.zeros(N, 1, 1).float().to(input_mask.device),
                                     input_mask[:, 1:, :]], dim=1)

        return sequence, input_mask, END_mask, input_mask_no_start, input_mask_no_end

    # %%
    def compute_neighbor_probs(self, active_probs, input_mask):
        N, S, _ = input_mask.size()
        assert input_mask.size() == (N, S, 1)
        input_mask = input_mask.permute(0, 2, 1).contiguous()
        assert input_mask.size() == (N, 1, S)

        assert active_probs.size() == (N, S, 1)
        active_probs = active_probs.permute(0, 2, 1).contiguous()
        assert active_probs.size() == (N, 1, S)

        input_mask_flipped = T.flip(input_mask.clone(), dims=[2])
        active_probs_flipped = T.flip(active_probs.clone(), dims=[2])

        input_mask = T.stack([input_mask_flipped, input_mask], dim=1)
        active_probs = T.stack([active_probs_flipped, active_probs], dim=1)

        assert input_mask.size() == (N, 2, 1, S)
        assert active_probs.size() == (N, 2, 1, S)

        active_probs_matrix = active_probs.repeat(1, 1, S, 1) * input_mask
        assert active_probs_matrix.size() == (N, 2, S, S)
        right_probs_matrix = T.triu(active_probs_matrix, diagonal=1)  # mask self and left

        right_probs_matrix_cumsum = T.cumsum(right_probs_matrix, dim=-1)
        assert right_probs_matrix_cumsum.size() == (N, 2, S, S)
        remainders = 1.0 - right_probs_matrix_cumsum

        remainders_from_left = T.cat([T.ones(N, 2, S, 1).float().to(remainders.device), remainders[:, :, :, 0:-1]],
                                     dim=-1)
        assert remainders_from_left.size() == (N, 2, S, S)

        remainders_from_left = T.max(T.zeros(N, 2, S, 1).float().to(remainders.device), remainders_from_left)
        assert remainders_from_left.size() == (N, 2, S, S)

        right_neighbor_probs = T.where(right_probs_matrix_cumsum > 1.0,
                                       remainders_from_left,
                                       right_probs_matrix)

        right_neighbor_probs = right_neighbor_probs * input_mask

        left_neighbor_probs = right_neighbor_probs[:, 0, :, :]
        left_neighbor_probs = T.flip(left_neighbor_probs, dims=[1, 2])
        right_neighbor_probs = right_neighbor_probs[:, 1, :, :]

        return left_neighbor_probs, right_neighbor_probs

    # %%
    def make_window(self, sequence, left_child_probs, right_child_probs):

        N, S, D = sequence.size()

        left_children_list = []
        right_children_list = []
        left_children_k = sequence.clone()
        right_children_k = sequence.clone()

        for k in range(self.window_size // 2):
            left_children_k = T.matmul(left_child_probs, left_children_k)
            left_children_list = [left_children_k.clone()] + left_children_list

            right_children_k = T.matmul(right_child_probs, right_children_k)
            right_children_list = right_children_list + [right_children_k.clone()]

        windowed_sequence = left_children_list + [sequence] + right_children_list
        windowed_sequence = T.stack(windowed_sequence, dim=-2)

        assert windowed_sequence.size() == (N, S, self.window_size, D)

        return windowed_sequence

    # %%
    def initial_transform(self, sequence):
        sequence = self.LN(self.initial_transform_layer(sequence))
        return sequence

    # %%
    def score_fn(self, windowed_sequence, transition_feats):
        N, S, W, D = windowed_sequence.size()
        windowed_sequence = windowed_sequence.view(N, S, W * D)

        if self.recurrent_momentum:
            windowed_sequence = T.cat([windowed_sequence, transition_feats], dim=-1)

        scores = self.scorer(gelu(self.conv_layer(windowed_sequence)))

        transition_scores = scores[:, :, 0].unsqueeze(-1)
        # reduce_probs = T.sigmoid(scores[:,:,1].unsqueeze(-1))
        no_op_scores = T.zeros_like(transition_scores).float().to(transition_scores.device)
        scores = T.cat([transition_scores, no_op_scores], dim=-1)
        scores = scores / self.temperature
        max_score = T.max(scores)
        exp_scores = T.exp(scores - max_score)

        return exp_scores

    # %%
    def composer(self, child1, child2):
        N, S, D = child1.size()

        concated = T.cat([child1, child2], dim=-1)
        assert concated.size() == (N, S, 2 * D)

        # concated = F.dropout(concated, p=self.hidden_dropout, training=self.training)
        if self.config["hidden_activation"].lower() == "relu":
            intermediate = F.relu(self.wcell1(concated))
        else:
            intermediate = gelu(self.wcell1(concated))
        intermediate = F.dropout(intermediate, p=self.hidden_dropout, training=self.training)
        contents = self.wcell2(intermediate)

        contents = contents.view(N, S, 4, D)
        gates = T.sigmoid(contents[:, :, 0:3, :])
        parent = contents[:, :, 3, :]
        f1 = gates[..., 0, :]
        f2 = gates[..., 1, :]
        i = gates[..., 2, :]

        transition = self.LN(f1 * child1 + f2 * child2 + i * parent)

        return transition

    # %%
    def compute_entropy_penalty(self, active_probs, last_token_mask):
        N, S = active_probs.size()
        active_prob_dist = self.sum_normalize(active_probs, dim=-1)
        nll_loss = - T.log(T.sum(last_token_mask * active_prob_dist, dim=1) + self.eps)
        nll_loss = nll_loss.view(N)
        return nll_loss

    # %%
    def compute_speed_penalty(self, steps, input_mask):
        steps = T.max(steps, dim=1)[0]
        speed_penalty = steps.squeeze(-1) / (T.sum(input_mask.squeeze(-1), dim=1) - 2.0)
        return speed_penalty

    # %%
    def encoder_block(self, sequence, input_mask):

        sequence, input_mask, END_mask, \
        input_mask_no_start, input_mask_no_end = self.augment_sequence(sequence, input_mask)

        N, S, D = sequence.size()

        """
        Initial Preparations
        """
        active_probs = T.ones(N, S, 1).float().to(sequence.device) * input_mask
        steps = T.zeros(N, S, 1).float().to(sequence.device)
        zeros_sequence = T.zeros(N, 1, 1).float().to(sequence.device)
        last_token_mask = T.cat([END_mask[:, 1:, :], zeros_sequence], dim=1)
        START_END_LAST_PAD_mask = input_mask_no_start * input_mask_no_end * (1.0 - last_token_mask)
        self.START_END_LAST_PAD_mask = START_END_LAST_PAD_mask
        halt_ones = T.ones(N).float().to(sequence.device)
        halt_zeros = T.zeros(N).float().to(sequence.device)
        improperly_terminated_mask = halt_ones.clone()
        update_mask = T.ones(N).float().to(sequence.device)
        left_transition_probs = T.zeros(N, S, 1).float().to(sequence.device)

        """
        Initial Transform
        """
        sequence = self.initial_transform(sequence)
        sequence = sequence * input_mask
        """
        Start Recursion
        """
        t = 0
        while t < (S - 2):
            original_active_probs = active_probs.clone()
            original_sequence = sequence.clone()
            residual_sequence = sequence.clone()
            original_steps = steps.clone()

            left_neighbor_probs, right_neighbor_probs \
                = self.compute_neighbor_probs(active_probs=active_probs.clone(),
                                              input_mask=input_mask.clone())

            windowed_sequence = self.make_window(sequence=sequence,
                                                 left_child_probs=left_neighbor_probs,
                                                 right_child_probs=right_neighbor_probs)

            if self.recurrent_momentum:
                transition_feats = left_transition_probs * self.past_transition_features.view(1, 1, -1) \
                                   + (1 - left_transition_probs) * self.past_non_transition_features.view(1, 1, -1)
            else:
                transition_feats = None

            exp_scores = self.score_fn(windowed_sequence, transition_feats)
            exp_transition_scores = exp_scores[:, :, 0].unsqueeze(-1)
            exp_no_op_scores = exp_scores[:, :, 1].unsqueeze(-1)

            exp_transition_scores = exp_transition_scores * START_END_LAST_PAD_mask

            if self.config["no_modulation"] is True:
                exp_scores = T.cat([exp_transition_scores,
                                    exp_no_op_scores], dim=-1)
            else:
                exp_left_transition_scores = T.matmul(left_neighbor_probs, exp_transition_scores)
                exp_right_transition_scores = T.matmul(right_neighbor_probs, exp_transition_scores)

                exp_scores = T.cat([exp_transition_scores,
                                    exp_no_op_scores,
                                    exp_left_transition_scores,
                                    exp_right_transition_scores], dim=-1)

            normalized_scores = self.sum_normalize(exp_scores, dim=-1)
            transition_probs = normalized_scores[:, :, 0].unsqueeze(-1)
            transition_probs = transition_probs * START_END_LAST_PAD_mask

            left_transition_probs = T.matmul(left_neighbor_probs, transition_probs)
            left_transition_probs = left_transition_probs * input_mask_no_start * input_mask_no_end
            left_sequence = windowed_sequence[:, :, self.window_size // 2 - 1, 0:self.hidden_size]

            transition_sequence = self.composer(child1=left_sequence, child2=sequence)
            transition_sequence = transition_sequence * input_mask

            tp = left_transition_probs
            sequence = tp * transition_sequence + (1 - tp) * residual_sequence
            sequence = sequence * input_mask
            steps = steps + active_probs

            bounded_probs = transition_probs
            active_probs = active_probs * (1.0 - bounded_probs) * input_mask

            active_probs = T.where(update_mask.view(N, 1, 1).expand(N, S, 1) == 1.0,
                                   active_probs,
                                   original_active_probs)

            steps = T.where(update_mask.view(N, 1, 1).expand(N, S, 1) == 1.0,
                            steps,
                            original_steps)

            sequence = T.where(update_mask.view(N, 1, 1).expand(N, S, D) == 1.0,
                               sequence,
                               original_sequence)

            t += 1
            discrete_active_status = T.where(active_probs > self.stop_threshold,
                                             T.ones_like(active_probs).to(active_probs.device),
                                             T.zeros_like(active_probs).to(active_probs.device))

            halt_condition_component = T.sum(discrete_active_status.squeeze(-1), dim=1) - 2.0
            update_mask = T.where((halt_condition_component <= 1) | (T.sum(input_mask.squeeze(-1), dim=-1) - 2.0 < t),
                                  halt_zeros,
                                  halt_ones)

            proper_termination_condition = T.sum(discrete_active_status * last_token_mask, dim=1).squeeze(-1)
            improperly_terminated_mask_ = T.where((halt_condition_component == 1) & (proper_termination_condition == 1),
                                                  halt_zeros,
                                                  halt_ones)

            improperly_terminated_mask = improperly_terminated_mask * improperly_terminated_mask_

            if self.config["early_stopping"]:
                if T.sum(update_mask) == 0.0:
                    break

        steps = steps * START_END_LAST_PAD_mask
        sequence = sequence * (1 - END_mask)
        active_probs = active_probs * (1 - END_mask)
        sequence = sequence[:, 1:-1, :]  # remove START and END
        active_probs = active_probs[:, 1:-1, :]  # remove START and END

        last_token_mask = END_mask[:, 2:, :]
        global_state = T.sum(sequence * last_token_mask, dim=1)

        assert active_probs.size(1) == sequence.size(1)

        entropy_penalty = self.compute_entropy_penalty(active_probs.squeeze(-1),
                                                       last_token_mask.squeeze(-1))

        speed_penalty = self.compute_speed_penalty(steps, input_mask)

        entropy_penalty = entropy_penalty * improperly_terminated_mask
        penalty = self.entropy_gamma * entropy_penalty + self.speed_gamma * speed_penalty

        return sequence, global_state, penalty

    # %%
    def forward(self, sequence, input_mask, **kwargs):

        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        else:
            self.temperature = 1.0

        self.temperature = 1.0 if self.temperature is None else self.temperature

        input_mask = input_mask.unsqueeze(-1)
        sequence = sequence * input_mask

        sequence, global_state, penalty = self.encoder_block(sequence, input_mask)
        sequence = sequence * input_mask
        return {"sequence": sequence, "penalty": penalty, "global_state": global_state}