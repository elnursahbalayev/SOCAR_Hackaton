"""Our OWN permeability and skin per well, from our own build-up analysis (no external PTA results).

Chicken-and-egg: PTA needs the VFM oil rate q before each shut-in, and the VFM needs J, which comes
from PTA's k and skin. Solved by iterating:
    step 0: VFM with J FITTED from the well tests (needs no PTA)  -> q before each shut-in
    step 1: PTA (Horner) with that q                               -> k, skin for wells with a shut-in
    step 2: Darcy J from those k, skin -> VFM again -> new q -> PTA again ... until k stops changing.
Near test dates q hardly depends on J, so this settles in 1-2 rounds.

Usage (project root):  python src/own_well_properties.py
Writes data/processed/own_pta_k_skin.csv and rebuilds test_features / daily_features parquet files.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')

import pandas as pd

import features as F
import load
import pta
import vfm


def physics_daily_oil(test_table, daily_table, options):
    """Daily oil from the physics model (oil in the final model is pure physics)."""
    trends = vfm.fit_well_trends(test_table, **options)
    flowing = daily_table[daily_table.dh_pressure_psia.notna()]
    oil = vfm.physics_predict(flowing, trends).oil_rate_stbd.reindex(daily_table.index).fillna(0.0)
    return daily_table[['well_name', 'date']].assign(oil=oil * daily_table.on_stream_frac)


def build_k_skin(hf, daily_oil, pvt, completion):
    """Horner k and skin for every shut-in, averaged per well (q = oil rate on the day before shut-in)."""
    rows = []
    for _, event in pta.find_shut_ins(hf).iterrows():
        r = daily_oil[daily_oil.well_name == event.well_name]
        q = r[r.date == event.t_shut.normalize() - pd.Timedelta(days=1)].oil.iloc[0]
        cum = completion.loc[event.well_name, 'cum_oil_at_2025_01_01_stb'] + r[r.date < event.t_shut.normalize()].oil.sum()
        res = pta.analyze_event(hf, event, q, cum, pvt.loc[event.well_name])
        rows.append({'well_name': event.well_name, 'k_md': res['k_md'], 'skin': res['skin']})
    return pd.DataFrame(rows).groupby('well_name').mean()


def estimate(max_rounds=5, tolerance_pct=0.1, verbose=True):
    hf = load.load_high_freq_clean()
    pvt_raw, completion_raw = load.load_pvt(), load.load_completion()
    pvt, completion = pvt_raw.set_index('well_name'), completion_raw.set_index('well_name')
    base_tests = F.build_test_table(load.load_well_tests_clean(), hf)
    base_daily = F.build_daily_table(hf)

    k_skin = None
    for round_no in range(max_rounds):
        wells = F.build_well_table(pvt_raw, completion_raw, k_skin)
        tests = F.add_physics_features(base_tests, wells)
        daily = F.add_physics_features(base_daily, wells)
        options = {**vfm.FINAL_PHYSICS_OPTIONS, 'j_source': 'fitted' if k_skin is None else 'auto'}
        new = build_k_skin(hf, physics_daily_oil(tests, daily, options), pvt, completion)
        if verbose:
            print(f'round {round_no}:', {w: (round(r.k_md, 2), round(r.skin, 3)) for w, r in new.iterrows()}, flush=True)
        if k_skin is not None and ((new.k_md / k_skin.k_md - 1).abs().max() * 100 < tolerance_pct):
            k_skin = new
            break
        k_skin = new

    k_skin.to_csv(load.PROCESSED / 'own_pta_k_skin.csv')
    wells = F.build_well_table(pvt_raw, completion_raw, k_skin)
    F.add_physics_features(base_tests, wells).to_parquet(load.PROCESSED / 'test_features.parquet', index=False)
    F.add_physics_features(base_daily, wells).to_parquet(load.PROCESSED / 'daily_features.parquet', index=False)
    return k_skin, wells


if __name__ == '__main__':
    k_skin, wells = estimate()
    print(wells[['well_name', 'J_from_pta', 'k_md', 'skin', 'J_darcy']].round(3).to_string(index=False))
