import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

TRAIN_PATH = '/Users/kris/Desktop/COMP5434/Project/comp-5434-20251-project-task-1/train.csv'
RESULTS_CSV = '/Users/kris/Desktop/COMP5434/Project/comp-5434-20251-project-task-1/ablation_features_results.csv'

RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_FEATURES = ['magnitude', 'depth', 'cdi', 'mmi', 'sig']


def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    X = df[BASE_FEATURES].copy()

    if config.get('sig_abs', False):
        X['sig_abs'] = np.abs(X['sig'])

    if config.get('interactions', False):
        X['magnitude_depth_ratio'] = X['magnitude'] / (X['depth'] + 1)
        X['magnitude_depth_product'] = X['magnitude'] * X['depth']
        X['cdi_mmi_diff'] = np.abs(X['cdi'] - X['mmi'])
        X['cdi_mmi_sum'] = X['cdi'] + X['mmi']
        X['cdi_mmi_product'] = X['cdi'] * X['mmi']
        X['magnitude_cdi'] = X['magnitude'] * X['cdi']
        X['magnitude_mmi'] = X['magnitude'] * X['mmi']
        X['sig_magnitude'] = X['sig'] * X['magnitude']

    if config.get('binning', False):
        X['depth_shallow'] = (X['depth'] < 50).astype(int)
        X['depth_medium'] = ((X['depth'] >= 50) & (X['depth'] < 200)).astype(int)
        X['depth_deep'] = (X['depth'] >= 200).astype(int)
        X['magnitude_low'] = (X['magnitude'] < 7.0).astype(int)
        X['magnitude_high'] = (X['magnitude'] >= 7.5).astype(int)

    return X


def run_variant(df: pd.DataFrame, name: str, cfg: dict) -> dict:
    X = build_features(df, cfg)
    y = df['label']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    return {
        'setting': name,
        'num_features': X.shape[1],
        'f1_macro': f1
    }


def main():
    df = pd.read_csv(TRAIN_PATH)

    variants = [
        ('baseline_only', {'sig_abs': False, 'interactions': False, 'binning': False}),
        ('+sig_abs', {'sig_abs': True, 'interactions': False, 'binning': False}),
        ('+interactions', {'sig_abs': False, 'interactions': True, 'binning': False}),
        ('+binning', {'sig_abs': False, 'interactions': False, 'binning': True}),
        ('+sig_abs+interactions', {'sig_abs': True, 'interactions': True, 'binning': False}),
        ('+sig_abs+binning', {'sig_abs': True, 'interactions': False, 'binning': True}),
        ('+interactions+binning', {'sig_abs': False, 'interactions': True, 'binning': True}),
        ('all_features', {'sig_abs': True, 'interactions': True, 'binning': True}),
    ]

    results = []
    for name, cfg in variants:
        res = run_variant(df, name, cfg)
        print(f"{name}: F1_macro={res['f1_macro']:.4f} (features={res['num_features']})")
        results.append(res)

    pd.DataFrame(results).sort_values('f1_macro', ascending=False).to_csv(RESULTS_CSV, index=False)
    print(f"Saved results -> {RESULTS_CSV}")


if __name__ == '__main__':
    main()


