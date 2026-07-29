"""
Retrain ML and DL models — fixed version
"""
import pandas as pd
import numpy as np
import joblib, os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report

DATA_PATH = r'C:\Users\HT\Desktop\AI_Water_Energy_Nexus_Dataset_2016_2026.csv'
BASE = os.path.dirname(os.path.abspath(__file__))

# The 37 features the frontend sends (in order)
FEATURE_COLS = [
    'year','month','quarter','country','region','climate_zone',
    'water_stress_level','organization','collaboration_type','sector',
    'ai_technique','water_application','energy_application','nexus_focus',
    'deployment_scale','status','sdg_alignment','funding_usd','investment_roi',
    'population_served','citation_count','impact_score','innovation_index',
    'model_performance_metric','model_performance_value','co2_reduction_tons',
    'water_savings_liters','energy_savings_kwh','renewable_energy_share_pct',
    'open_access','venue_h_index','patent_class','patent_family_size',
    'policy_type','policy_level','policy_stringency_score','language'
]

print("[1/6] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

target_col = 'entry_type'
print(f"  Target: {target_col}")
print(f"  Classes:\n{df[target_col].value_counts()}")

# Keep only the 37 feature columns + target
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"  WARNING: Missing columns: {missing}")
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

X_raw = df[FEATURE_COLS].copy()
y_raw = df[target_col]

print(f"\n[2/6] Encoding {len(FEATURE_COLS)} features...")

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(y_raw)
print(f"  Classes: {list(le_target.classes_)}")

# Encode categoricals
cat_cols = X_raw.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    X_raw[col] = le.fit_transform(X_raw[col].astype(str).fillna('Unknown'))

# Fill remaining NaNs with median
X_raw = X_raw.fillna(X_raw.median(numeric_only=True))
X_raw = X_raw.fillna(0)

X = X_raw.values.astype(float)
print(f"  Feature matrix: {X.shape}, NaN count: {np.isnan(X).sum()}")

# Scale
print("\n[3/6] Scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train ML
print("\n[4/6] Training ML model (GradientBoosting, max_depth=6, 300 trees)...")
ml_model = GradientBoostingClassifier(
    n_estimators=300, max_depth=6, min_samples_split=5,
    min_samples_leaf=3, learning_rate=0.1, subsample=0.8,
    max_features='sqrt', random_state=42
)
ml_model.fit(X_train, y_train)
y_pred_ml = ml_model.predict(X_test)
print(classification_report(y_test, y_pred_ml, target_names=le_target.classes_))

# Feature importance
imp = ml_model.feature_importances_
pairs = sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1])
print("Top 15 feature importances:")
for name, v in pairs[:15]:
    print(f"  {name:35s} => {v:.4f}")
non_zero = sum(1 for _, v in pairs if v > 0.001)
print(f"  Features with >0.1% importance: {non_zero}/{len(pairs)}")

# Train DL
print("\n[5/6] Training DL model...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

n_classes = len(le_target.classes_)
y_train_cat = to_categorical(y_train, n_classes)
y_test_cat = to_categorical(y_test, n_classes)
X_train_dl = X_train.reshape(-1, 1, X_train.shape[1])
X_test_dl = X_test.reshape(-1, 1, X_test.shape[1])

dl_model = Sequential([
    LSTM(128, input_shape=(1, X_train.shape[1]), return_sequences=True),
    Dropout(0.3), BatchNormalization(),
    LSTM(64, return_sequences=False),
    Dropout(0.3), BatchNormalization(),
    Dense(64, activation='relu'), Dropout(0.2),
    Dense(n_classes, activation='softmax')
])
dl_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train_dl, y_train_cat, validation_data=(X_test_dl, y_test_cat),
             epochs=50, batch_size=64,
             callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=1)
dl_loss, dl_acc = dl_model.evaluate(X_test_dl, y_test_cat, verbose=0)
print(f"\nDL Test Accuracy: {dl_acc:.4f}")

# Save
print("\n[6/6] Saving...")
joblib.dump(ml_model, os.path.join(BASE, 'best_ml_model.pkl'))
joblib.dump(scaler, os.path.join(BASE, 'scaler.pkl'))
joblib.dump(le_target, os.path.join(BASE, 'le_target.pkl'))
dl_model.save(os.path.join(BASE, 'best_dl_model.h5'))

# Verify
print("\n--- Verification ---")
for i in range(5):
    s = X_test[i:i+1]
    ml_p = ml_model.predict_proba(s)[0]
    dl_p = dl_model.predict(s.reshape(1,1,-1), verbose=0)[0]
    print(f"  #{i+1}: ML={le_target.classes_[np.argmax(ml_p)]} {max(ml_p)*100:.1f}% | DL={le_target.classes_[np.argmax(dl_p)]} {max(dl_p)*100:.1f}%")

print("\n✅ Done! Models saved.")
