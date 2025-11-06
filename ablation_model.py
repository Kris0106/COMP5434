import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

TRAIN_PATH = '/Users/kris/Desktop/COMP5434/Project/comp-5434-20251-project-task-1/train.csv'
RESULTS_CSV = '/Users/kris/Desktop/COMP5434/Project/comp-5434-20251-project-task-1/ablation_model_results.csv'

RANDOM_STATE = 42
TEST_SIZE = 0.2
FEATURES = ['magnitude', 'depth', 'cdi', 'mmi', 'sig']


def evaluate_model(df: pd.DataFrame, model_name: str):
    X = df[FEATURES]
    y = df['label']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    if model_name == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
    elif model_name == 'XGBoost':
        if not HAS_XGB:
            return None, None
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            objective='multi:softmax',
            eval_metric='mlogloss'
        )
    else:
        raise ValueError('Unknown model')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return model_name, f1_score(y_val, y_pred, average='macro')


def main():
    df = pd.read_csv(TRAIN_PATH)
    results = []

    for name in ['RandomForest', 'XGBoost']:
        model_name, f1 = evaluate_model(df, name)
        if f1 is None:
            print(f"{name}: skipped (package not available)")
            continue
        print(f"{name}: F1_macro={f1:.4f}")
        results.append({'model': model_name, 'f1_macro': f1})

    if results:
        pd.DataFrame(results).sort_values('f1_macro', ascending=False).to_csv(RESULTS_CSV, index=False)
        print(f"Saved results -> {RESULTS_CSV}")
    else:
        print('No results to save.')


if __name__ == '__main__':
    main()


