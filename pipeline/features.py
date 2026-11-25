"""Stage 2 · Interpretable Feature Engineering (`pipeline/features.py`).

Synthesized by sovLoop (Model: qwen3.5:9b)
Active Operations: synth:log1p:is_self_pair, synth:log1p:agent_id
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_FEATURE_OPS: tuple[str, ...] = ('synth:log1p:is_self_pair', 'synth:log1p:agent_id')
PRIORITY_SIGNAL_COLUMNS: list[str] = ['lab_id', 'dist_min_kp', 'frame', 'dist_nb', 'dist_bc_rm_l', 'tg_speed_bc_rm_m']


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit, human-readable engineered features one transformation per block."""
    out = df.copy()
    if not ACTIVE_FEATURE_OPS:
        return out

    fallback_cols = [c for c in PRIORITY_SIGNAL_COLUMNS if c in out.columns]
    if len(fallback_cols) < 2:
        for col in out.columns:
            if col not in fallback_cols and not str(col).endswith(("_isna", "_target_enc")):
                fallback_cols.append(str(col))
            if len(fallback_cols) >= 4:
                break

    if 'is_self_pair' in out.columns:
        out['is_self_pair_log1p'] = np.sign(out['is_self_pair']) * np.log1p(np.abs(out['is_self_pair']))
    if 'agent_id' in out.columns:
        out['agent_id_log1p'] = np.sign(out['agent_id']) * np.log1p(np.abs(out['agent_id']))

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def run_features_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Alias preserving modular orchestrator compatibility."""
    return feature_engineering(df)
