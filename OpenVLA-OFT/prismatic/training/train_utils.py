"""Token-mask helpers required by the OpenVLA Hugging Face model."""

import torch

from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def get_current_action_mask(token_ids: torch.Tensor) -> torch.Tensor:
    positions = torch.cumsum((token_ids != IGNORE_INDEX).to(torch.int64), dim=1)
    action_tokens = token_ids > ACTION_TOKEN_BEGIN_IDX
    return action_tokens & (positions >= 1) & (positions <= ACTION_DIM)


def get_next_actions_mask(token_ids: torch.Tensor) -> torch.Tensor:
    positions = torch.cumsum((token_ids != IGNORE_INDEX).to(torch.int64), dim=1)
    action_tokens = token_ids > ACTION_TOKEN_BEGIN_IDX
    return action_tokens & (positions > ACTION_DIM)
