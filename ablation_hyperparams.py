import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

TRAIN_PATH = '/Users/kris/Desktop/COMP5434/Project/comp-5434-20251-project-task-1/train.csv'
RESULTS_CSV = '/Users/kris/Desktop/COMP5434/Project/comp-5434-20251-project-task-1/ablation_hyperparams_results.csv'

RANDOM_STATE = 42
TEST_SIZE = 0.2
FEATURES = ['magnitude', 'depth', 'cdi', 'mmi', 'sig']


def eval_with_params(df: pd.DataFrame, params: dict, label: str):
    X = df[FEATURES]
    y = df['label']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    base = {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'class_weight': 'balanced'
    }
    base.update(params)

    model = RandomForestClassifier(**base)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    return {'setting': label, 'params': base, 'f1_macro': f1}


def main():
    df = pd.read_csv(TRAIN_PATH)
    results = []

    # Baseline
    results.append(eval_with_params(df, {}, 'baseline'))

    # Single-parameter ablations (change one, keep others at baseline)
    for n in [50, 200, 300]:
        results.append(eval_with_params(df, {'n_estimators': n}, f'n_estimators={n}'))

    for d in [10, 20, None]:
        results.append(eval_with_params(df, {'max_depth': d}, f'max_depth={d}'))

    for mss in [2, 5, 10]:
        results.append(eval_with_params(df, {'min_samples_split': mss}, f'min_samples_split={mss}'))

    for msl in [1, 2, 4]:
        results.append(eval_with_params(df, {'min_samples_leaf': msl}, f'min_samples_leaf={msl}'))

    for mf in ['sqrt', 'log2']:
        results.append(eval_with_params(df, {'max_features': mf}, f'max_features={mf}'))

    df_out = pd.DataFrame(results)
    # flatten params to string for CSV readability
    df_out['params'] = df_out['params'].apply(lambda d: str(d))
    df_out.sort_values('f1_macro', ascending=False).to_csv(RESULTS_CSV, index=False)
    print(df_out.sort_values('f1_macro', ascending=False).to_string(index=False))
    print(f"Saved results -> {RESULTS_CSV}")


if __name__ == '__main__':
    main()


