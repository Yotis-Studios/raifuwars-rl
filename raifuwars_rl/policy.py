"""A policy that SCORES actions instead of classifying over them.

    net = ActionScorer()
    logits = net(state_vec, action_mat)     # (D_STATE,), (N, D_ACTION) -> (N,)
    probs  = softmax(logits)                # over exactly the N offered

TWO TOWERS AND A DOT PRODUCT. The state is embedded once, each candidate action is embedded, and
the score is their interaction. Nothing about the architecture knows how many actions there are,
which is the requirement: the legal set runs from 2 to ~670 between decisions and varies with board
size, dice roll and hand.

WHY NOT A FIXED HEAD. A classifier over "all possible actions" would need an output for every tile
on every map (27x27 = 729 for movement alone, times action types), would have to mask nearly all of
them on every step, and would learn nothing transferable between a 17x21 board and a 27x27 one.
Scoring makes an illegal action unrepresentable rather than merely penalised, and makes the same
weights work on any board.

THE INTERACTION TERM MATTERS. An early version scored actions from action features alone -- the
state embedding was concatenated but the model could ignore it -- and a policy that cannot condition
on the board is a policy that plays the same move everywhere. The elementwise product forces the
score to depend on both.

SMALL ON PURPOSE. 33 + 26 inputs and a couple of hundred hidden units. The bottleneck here is the
simulator at ~1.6 agent steps per second, so a bigger network buys nothing but a slower step -- and
a small one is auditable, which matters when a run costs a day.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import D_ACTION, D_STATE


class ActionScorer(nn.Module):
    def __init__(self, d_state=D_STATE, d_action=D_ACTION, hidden=128, embed=64):
        super().__init__()
        self.state_tower = nn.Sequential(
            nn.Linear(d_state, hidden), nn.ReLU(),
            nn.Linear(hidden, embed), nn.ReLU(),
        )
        self.action_tower = nn.Sequential(
            nn.Linear(d_action, hidden), nn.ReLU(),
            nn.Linear(hidden, embed), nn.ReLU(),
        )
        # Scores the INTERACTION, not the concatenation. See the module docstring.
        self.head = nn.Sequential(
            nn.Linear(embed * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # A value head for later -- PPO needs V(s) and it costs nothing to carry now. Reading it
        # is optional; behaviour cloning ignores it.
        self.value = nn.Sequential(
            nn.Linear(embed, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, actions):
        """state: (D_STATE,) or (B, D_STATE); actions: (N, D_ACTION) or (B, N, D_ACTION)."""
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)
            actions = actions.unsqueeze(0)

        s = self.state_tower(state)                       # (B, E)
        a = self.action_tower(actions)                    # (B, N, E)
        n = a.shape[1]
        s_rep = s.unsqueeze(1).expand(-1, n, -1)          # (B, N, E)
        pair = torch.cat([s_rep, a, s_rep * a], dim=-1)   # (B, N, 3E)
        logits = self.head(pair).squeeze(-1)              # (B, N)
        return logits.squeeze(0) if single else logits

    def value_of(self, state):
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)
        v = self.value(self.state_tower(state)).squeeze(-1)
        return v.squeeze(0) if single else v


def masked_log_probs(logits, lengths):
    """Log-softmax over each row's true length, padding excluded.

    Batching decisions with different numbers of offered actions means padding, and padding that
    is merely zeroed still enters the softmax and steals probability mass. -inf before the softmax
    is the only version that is arithmetically the same as scoring each row alone.
    """
    mask = torch.arange(logits.shape[1], device=logits.device)[None, :] < lengths[:, None]
    logits = logits.masked_fill(~mask, float("-inf"))
    return F.log_softmax(logits, dim=1)
