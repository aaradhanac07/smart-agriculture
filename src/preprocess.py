import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def preprocess_dataset(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Returns:
        X_train_scaled, X_test_scaled, y_train_enc, y_test_enc,
        scaler, label_encoder, feature_columns
    """
    # Clean column names (safe)
    df.columns = [c.strip().lower() for c in df.columns]

    # Expected columns
    # N, P, K, temperature, humidity, ph, rainfall, label
    # Dataset may have slightly different casing
    target_col = "label"
    if target_col not in df.columns:
        raise ValueError("Dataset must contain a 'label' column for crop name.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_columns = list(X.columns)

    # Handle missing values (simple)
    X = X.fillna(X.median(numeric_only=True))

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le, feature_columns
