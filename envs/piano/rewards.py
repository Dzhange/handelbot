import torch
from .music import constants as consts

_DEFAULT_VALUE_AT_MARGIN = 0.1
_FINGER_CLOSE_ENOUGH_TO_KEY = 0.01 

def _sigmoids(x, value_at_1, sigmoid: str):
    if sigmoid in ('cosine', 'linear', 'quadratic'):
        if not 0 <= value_at_1 < 1:
            raise ValueError(f'`value_at_1` must be in [0,1), got {value_at_1}.')
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(f'`value_at_1` must be in (0,1), got {value_at_1}.')

    if sigmoid == 'gaussian':
        scale = torch.sqrt(-2 * torch.log(torch.tensor(value_at_1)))
        return torch.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == 'hyperbolic':
        scale = torch.acosh(torch.tensor(1.0) / value_at_1)
        return 1 / torch.cosh(x * scale)

    elif sigmoid == 'long_tail':
        scale = torch.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == 'reciprocal':
        scale = 1 / value_at_1 - 1
        return 1 / (torch.abs(x) * scale + 1)

    elif sigmoid == 'cosine':
        scale = torch.arccos(2 * value_at_1 - 1) / torch.pi
        scaled_x = x * scale
        cos_pi_scaled_x = torch.cos(torch.pi * scaled_x)
        return torch.where(torch.abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, torch.tensor(0.0))

    elif sigmoid == 'linear':
        scale = 1 - value_at_1
        scaled_x = x * scale
        return torch.where(torch.abs(scaled_x) < 1, 1 - scaled_x, torch.tensor(0.0))

    elif sigmoid == 'quadratic':
        scale = torch.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return torch.where(torch.abs(scaled_x) < 1, 1 - scaled_x ** 2, torch.tensor(0.0))

    elif sigmoid == 'tanh_squared':
        scale = torch.atanh(torch.sqrt(1 - value_at_1))
        return 1 - torch.tanh(x * scale) ** 2

    else:
        raise ValueError(f'Unknown sigmoid type {sigmoid!r}.')


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian',
              value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
    lower, upper = bounds
    if lower > upper:
        raise ValueError('Lower bound must be <= upper bound.')
    if margin < 0:
        raise ValueError('`margin` must be non-negative.')

    in_bounds = (x >= lower) & (x <= upper)
    if margin == 0:
        value = torch.where(in_bounds, 1, 0)
    else:
        d = torch.where(x < lower, lower - x, x - upper) / margin
        value = torch.where(in_bounds, 1, _sigmoids(d, value_at_margin, sigmoid))

    return value


class BaseReward:
    def __call__(self, info: dict) -> torch.Tensor:
        return 0.0

class KeyPressReward(BaseReward):
    name: str = 'key_press'

    def __init__(self, coef, key_on, n_fingers=5):
        self.coef = coef
        self.key_on = key_on
        self.n_fingers = n_fingers

    def __call__(self, info: dict) -> torch.Tensor:
        on = torch.nonzero(info['goal_current'][:-1]).flatten() # remove sustain, keys to press are same for parallel env
        rew = 0.0

        pressed = info['piano_activation']        
        if len(on) > 0:
            rew = self.key_on * pressed[:, on].mean(axis=1)
        # If there are any false positives, the remaining 0.5 reward is lost.
        off = torch.nonzero(1 - info['goal_current'][:-1]).flatten()
        rew += (1 - self.key_on) * (1 - pressed[:, off].any(axis=1).float())
        return rew * self.coef 

class FingeringReward(BaseReward):
    name: str = 'finger'

    def __init__(self, coef, n_fingers=5):
        self.coef = coef
        self.n_fingers = n_fingers

    def __call__(self, info: dict) -> torch.Tensor:
        def _distance_finger_to_key(
            hand_keys, hand
        ):
            distances = []
            for key, fingering in hand_keys:
                idx = fingering
                if self.n_fingers == 4:
                    idx = fingering - 1
                fingertip_pos = hand[idx]
                key_pos = info['piano_poses'][:, key]
                diff = key_pos - fingertip_pos
                distances.append(torch.linalg.norm(diff, dim=-1))
            return distances

        distances = []
        if 'left_hand' in info:
            distances += _distance_finger_to_key(info['lh_keys_current'], info['left_hand'])
        if 'right_hand' in info:
            distances += _distance_finger_to_key(info['rh_keys_current'], info['right_hand'])

        # Case where there are no keys to press at this timestep.
        if not distances:
            return 0.0 

        rews = tolerance(
            torch.vstack(distances),
            bounds=(0, _FINGER_CLOSE_ENOUGH_TO_KEY),
            margin=(_FINGER_CLOSE_ENOUGH_TO_KEY * 10),
            sigmoid="gaussian",
        )
        return rews.mean(axis=0) * self.coef

class CompositeReward:
    def __init__(self, components):
        self.components = components

    def __call__(self, info: dict) -> torch.Tensor:
        reward_dict = dict()
        total = torch.zeros(info["piano_activation"].shape[0], device=info["piano_activation"].device)
        for component in self.components:
            reward = component(info)
            if not isinstance(reward, float):
                reward_dict[component.name] = reward
            total += reward
        return total, reward_dict

