"""Separate final development protocol; leaves first staged study untouched."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_staged_recovery as runner
from vguard.native_soft_recovery import soft_recover

FROZEN = Path('docs/evidence/2026-11-09-staged/final/protocol.json')
original_optimizer = runner.staged_recover


def dispatch(model, original, target_embeddings, contents, theta, psd_max, attr, pp, **kwargs):
    if kwargs['objective'] != 'soft':
        return original_optimizer(model, original, target_embeddings, contents, theta, psd_max, attr, pp, **kwargs)
    protocol = runner.read(FROZEN)
    with np.load(protocol['native_target_bank']) as bank:
        prototypes = bank['embeddings']
    return soft_recover(model, original, target_embeddings, theta, psd_max, attr, pp, prototypes,
        strong_steps=kwargs['strong_steps'], conceal_steps=kwargs['conceal_steps'],
        learning_rate=kwargs['learning_rate'], searches=protocol['soft_searches'],
        epsilon=protocol['soft_epsilon'], callback=kwargs['callback'])


if __name__ == '__main__':
    runner.PROTOCOL = FROZEN
    runner.staged_recover = dispatch
    runner.main()
