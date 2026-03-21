import pandas as pd
import sys
sys.path.insert(0, '.')
from evolution.gp import FeatureLibrary

prices = pd.read_parquet('data/cache/prices_master.parquet').loc[:'2018-07-02']
lib = FeatureLibrary(enable_oil=True)

for name, spec in lib.feature_specs.items():
    try:
        lib._compute_feature(prices.iloc[:-1], None, spec)
    except Exception as e:
        print(f'BROKEN: {name} -> {e}')