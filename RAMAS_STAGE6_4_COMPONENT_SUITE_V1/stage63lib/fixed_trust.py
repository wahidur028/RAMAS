"""Fixed positive influence for the matched memory ablation.

This does not implement a learning rule. The adaptive rule remains unchanged in
the vendor source solely for provenance; it is never called by this experiment.
"""
import math


REGIMES = ('bear', 'bull', 'mix')
FIXED_BETA = 0.05


def validate_fixed_trust_config(config):
    value = config.get('fixed_beta')
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value != FIXED_BETA):
        raise ValueError('This frozen experiment requires fixed_beta exactly 0.05')
    if config.get('agent_trust_mode') != 'FIXED_POSITIVE':
        raise ValueError('agent_trust_mode must be FIXED_POSITIVE')
    if config.get('common_legacy_trust_rule') is not False:
        raise ValueError('Adaptive legacy trust must be disabled in both arms')
    if config.get('arms') != ['memory', 'no_memory']:
        raise ValueError('The matched arms must be memory and no_memory in frozen order')
    if config.get('continue_across_years') is not True:
        raise ValueError('State must continue across years')


class FixedTrust:
    """A constant, reconstructible trust API compatible with result reporting."""
    def __init__(self, config):
        validate_fixed_trust_config(config)
        self.events = []

    @property
    def beta(self):
        # Return a snapshot. Callers cannot mutate the actual fixed value.
        return {regime: FIXED_BETA for regime in REGIMES}

    def value(self, regime):
        if regime not in REGIMES:
            raise ValueError('Unknown regime for fixed trust')
        return FIXED_BETA
