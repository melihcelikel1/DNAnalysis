import gzip
import pandas as pd
import io
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# VCF Verisini İşleme Fonksiyonu
def process_vcf(file_path):
    with gzip.open(file_path, mode='rt') as f:
        lines = [line for line in f if not line.startswith('##')]

    vcf_df = pd.read_csv(
        io.StringIO(''.join(lines)),
        sep='\t',
        low_memory=False
    )

    vcf_df['Clinical_Significance'] = vcf_df['INFO'].apply(
        lambda info: "Benign" if "CLNSIG=Benign" in info else
                     "Pathogenic" if "CLNSIG=Pathogenic" in info else
                     "Uncertain"
    )

    vcf_df = vcf_df[vcf_df['Clinical_Significance'] != "Uncertain"]

    features = pd.DataFrame({
        'CHROM': vcf_df['#CHROM'],
        'POS': vcf_df['POS'],
        'ALT_LEN': vcf_df['ALT'].apply(len),
        'QUAL': vcf_df['QUAL']
    })

    # Veriyi sayısallaştır ve eksik değerleri doldur
    features = features.apply(pd.to_numeric, errors='coerce')
    features = features.fillna(0)

    return features, vcf_df['Clinical_Significance']

# Model Eğitimi Fonksiyonu
def train_model(file_paths, model_output_path):
    all_features = []
    all_labels = []

    for file_path in file_paths:
        features, labels = process_vcf(file_path)
        all_features.append(features)
        all_labels.append(labels)

    # Tüm verileri birleştir
    all_features = pd.concat(all_features, ignore_index=True)
    all_labels = pd.concat(all_labels, ignore_index=True)

    # Eğitim ve test veri setlerini ayır
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, all_labels, test_size=0.2, random_state=42
    )

    # Etiketleri sayısallaştır
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Farklı modellerin eğitilmesi ve değerlendirilmesi
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    best_model = None
    best_score = 0

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average="binary")
        print(f"{model_name} F1-Score: {score}")

        if score > best_score:
            best_score = score
            best_model = model

    print(f"Best model: {best_model}")

    # Hiperparametre optimizasyonu (GridSearch ile örnek)
    if isinstance(best_model, RandomForestClassifier):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        grid_search = GridSearchCV(best_model, param_grid, cv=3, scoring='f1', verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Optimized Model Parameters: {grid_search.best_params_}")

    # En iyi modelin değerlendirilmesi
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Confusion matrix görselleştirme
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

    # Modeli kaydet
    joblib.dump(best_model, model_output_path)
    print(f"Model saved to {model_output_path}")

# Ana Program
if __name__ == "__main__":
    file_paths = ["clinvar.vcf.gz", "clinvar_20241230.vcf.gz"]  # Eğitim veri setleri
    model_output_path = "optimized_model.joblib"  # Modelin kaydedileceği dosya
    train_model(file_paths, model_output_path)
