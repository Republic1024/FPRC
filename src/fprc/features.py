"""Label-free, same-date ranks and optional shared P/Q purification."""
import numpy as np
import pandas as pd


def rank_pct(values, dates):
    values = np.asarray(values, dtype=float)
    return pd.Series(values).groupby(dates, sort=False).rank(
        pct=True, method="average"
    ).to_numpy(dtype=float)


def residualize(y, x, dates, min_samples):
    out = np.full(len(y), np.nan)
    _, starts, counts = np.unique(dates, return_index=True, return_counts=True)
    for start, count in zip(starts, counts):
        stop = start + count
        xx, yy = x[start:stop], y[start:stop]
        good = np.isfinite(yy) & np.isfinite(xx).all(axis=1)
        if good.sum() < max(min_samples, x.shape[1] + 2):
            continue
        design = np.column_stack([np.ones(good.sum()), xx[good]])
        beta = np.linalg.lstsq(design, yy[good], rcond=None)[0]
        ids = start + np.flatnonzero(good)
        out[ids] = yy[good] - design @ beta
    return out


def build_features(raw, dates, specs, parents, closure_features, apex, q_features,
                   min_samples):
    used = list(dict.fromkeys(
        list(parents) + list(closure_features)
        + [x for spec in specs for x in spec.local_features]
        + ([apex] if apex else []) + list(q_features)
    ))
    ranks = {name: rank_pct(raw[name], dates) for name in used}
    parent_x = np.column_stack([ranks[name] for name in parents])
    apex_rank, p_local = None, []
    if apex is not None:
        apex_rank = rank_pct(residualize(
            ranks[apex], parent_x, dates, min_samples
        ), dates)
        # Frozen FPRC-PQ Q construction conditions on raw-AB-purified local P's.
        for spec in specs:
            p_local.append(rank_pct(residualize(
                raw[apex].to_numpy(float), raw[[spec.a, spec.b]].to_numpy(float),
                dates, min_samples
            ), dates))
    q_core = np.column_stack([parent_x, *p_local]) if p_local else parent_x
    q_columns = [rank_pct(residualize(
        ranks[q], q_core, dates, min_samples
    ), dates) for q in q_features]

    local, required, local_names = [], [], []
    for spec in specs:
        cols = [ranks[spec.a], ranks[spec.b]]
        names = [f"rank:{spec.a}", f"rank:{spec.b}"]
        gate = np.isfinite(np.column_stack(cols)).all(axis=1)
        if apex_rank is not None:
            cols.append(apex_rank)
            names.append(f"shared_P_perp:{apex}")
            gate &= np.isfinite(apex_rank)
        cols.extend(q_columns)
        names.extend(f"Q_perp:{q}" for q in q_features)
        # Optional Q / local-feature NaNs are native XGBoost missing values.
        extras = [x for x in spec.local_features if x not in (spec.a, spec.b)]
        cols.extend(ranks[name] for name in extras)
        names.extend(f"rank:{name}" for name in extras)
        local.append(np.column_stack(cols).astype(np.float32))
        required.append(gate)
        local_names.append(names)
    closure = np.column_stack([ranks[name] for name in closure_features]).astype(np.float32)
    closure_gate = np.isfinite(parent_x).all(axis=1)
    return ranks, local, required, closure, closure_gate, local_names
