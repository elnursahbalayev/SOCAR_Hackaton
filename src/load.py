"""Load the hackathon data files.

All raw files live in data/data/ and are never modified.
The big per-minute file (257 MB CSV) is converted once to parquet in data/processed/,
because parquet loads ~10x faster and keeps the column types (dates stay dates).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "data"
PROCESSED = ROOT / "data" / "processed"


def load_high_freq():
    """Per-minute sensor data for all wells.

    Columns (psia = absolute pressure, degF = temperature, mscfd = thousand scf/day):
      whp_psia               wellhead pressure, upstream of the choke
      whp_downstream_psia    pressure just after the choke (flowline side)
      dh_pressure_psia       downhole gauge pressure ~ flowing bottomhole pressure Pwf
      dh_temperature_degf    downhole temperature
      annulus_pressure_psia  casing pressure = pressure of the injected lift gas
      gas_lift_rate_mscfd    lift gas injected into the annulus
    """
    parquet = PROCESSED / "high_freq.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)

    df = pd.read_csv(RAW / "high_freq_data.csv", parse_dates=["timestamp"])
    df = df.sort_values(["well_name", "timestamp"]).reset_index(drop=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet, index=False)
    return df


def load_high_freq_clean():
    """Cleaned per-minute data (made by src/clean.py, see notebooks/02_cleaning.ipynb).
    Extra columns: choke_opening_pct, shut_in (True when choke = 0 %)."""
    parquet = PROCESSED / "high_freq_clean.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)

    import clean  # local import: clean.py lives next to this file
    df, _ = clean.clean_high_freq(load_high_freq(), load_choke_events())
    df.to_parquet(parquet, index=False)
    return df


def load_well_tests():
    """Well tests = our LABELS (measured oil / water / gas rates), plus sensor averages during the test."""
    return pd.read_csv(RAW / "well_tests.csv", parse_dates=["test_start", "test_end"])


def load_well_tests_clean():
    """Well tests with sign flips and gas-unit errors fixed (src/clean.py, notebooks/03_label_cleaning.ipynb)."""
    import clean
    tests, _ = clean.clean_well_tests(load_well_tests())
    return tests


def load_choke_events():
    """Choke opening (%) changes. The value holds until the next event for that well (step function)."""
    df = pd.read_csv(RAW / "choke_events.csv", parse_dates=["timestamp"])
    return df.sort_values(["well_name", "timestamp"]).reset_index(drop=True)


def load_pvt():
    """Per-well rock and fluid properties used in PTA formulas (h, phi, ct, rw, Bo, mu ...)."""
    return pd.read_csv(RAW / "reservoir_pvt.csv")


def load_completion():
    """Per-well completion: gauge / gas-lift valve depths, tubing size, choke size, cum oil at 2025-01-01."""
    return pd.read_csv(RAW / "well_completion.csv", parse_dates=["first_production_date"])


def load_own_k_skin():
    """OUR OWN build-up results per well (k_md, skin), made by src/own_well_properties.py.
    Returns None if not computed yet."""
    path = PROCESSED / "own_pta_k_skin.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col='well_name')
