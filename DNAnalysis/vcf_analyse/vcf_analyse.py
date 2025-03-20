from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QWidget, QProgressBar, QLineEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sys
import gzip
import pandas as pd
import io
import joblib
import matplotlib.pyplot as plt
import os
from pathlib import Path

class WorkerThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list, pd.DataFrame)  # İki ayrı parametre
    error = pyqtSignal(str)

    def __init__(self, file_path, model_path):
        super().__init__()
        self.file_path = file_path
        self.model_path = model_path

    def run(self):
        try:
            # VCF dosyasını oku
            if self.file_path.endswith('.gz'):
                with gzip.open(self.file_path, 'rt') as f:
                    lines = [line for line in f if not line.startswith('##')]
            elif self.file_path.endswith('.vcf'):
                with open(self.file_path, 'r') as f:
                    lines = [line for line in f if not line.startswith('##')]
            else:
                raise ValueError("Unsupported file format. Please upload a .vcf or .vcf.gz file.")

            vcf_df = pd.read_csv(
                io.StringIO(''.join(lines)),
                sep='\t',
                low_memory=False
            )

            # Varyant türlerini belirle
            def determine_variant_type(row):
                ref = row['REF']
                alt = row['ALT']
                if len(ref) == 1 and len(alt) == 1:
                    return 'SNV'
                elif len(alt) > len(ref):
                    if ref in alt:
                        return 'Insertion'
                    return 'Duplication'
                elif len(alt) < len(ref):
                    if alt in ref:
                        return 'Deletion'
                    return 'Complex'
                elif len(ref) > 1 and len(alt) > 1:
                    if len(ref) == len(alt):
                        if ref == alt[::-1]:
                            return 'Inversion'
                        return 'Substitution'
                    return 'MNV'
                return 'Microsatellite_Expansion' if set(ref) == set(alt) else 'Complex'

            vcf_df['Variant_Type'] = vcf_df.apply(determine_variant_type, axis=1)

            # Model özelliklerini oluştur
            features = pd.DataFrame({
                'CHROM': vcf_df['#CHROM'],
                'POS': vcf_df['POS'],
                'ALT_LEN': vcf_df['ALT'].apply(len),
                'QUAL': vcf_df['QUAL']
            })
            features = features.apply(pd.to_numeric, errors='coerce').fillna(0)

            # Modeli yükle
            model = joblib.load(self.model_path)

            # Tahmin yap ve sonuçları topla
            results = []
            if 'ID' in vcf_df.columns and 'INFO' in vcf_df.columns:
                total_rows = len(vcf_df)
                for index, row in vcf_df.iterrows():
                    variant_id = row['ID'] if row['ID'] != '.' else f"{row['#CHROM']}:{row['POS']}"
                    prediction = model.predict(features.iloc[[index]])[0]
                    results.append((variant_id, prediction, row['Variant_Type']))
                    self.progress.emit(int((index + 1) / total_rows * 100))

            self.finished.emit(results, vcf_df)  # İki ayrı değer gönder
        except Exception as e:
            self.error.emit(str(e))

class DNAAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DNA Variant Analyzer")
        self.setGeometry(100, 100, 800, 600)
        self.main_widget = QWidget()
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)

        self.initUI()

    def initUI(self):
        self.upload_button = QPushButton("Upload DNA File")
        self.upload_button.clicked.connect(self.upload_file)
        self.layout.addWidget(self.upload_button)

        self.file_label = QLabel("No file selected")
        self.layout.addWidget(self.file_label)

        self.analyze_button = QPushButton("Analyze Variants")
        self.analyze_button.clicked.connect(self.analyze_variants)
        self.layout.addWidget(self.analyze_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search Variant...")
        self.search_box.textChanged.connect(self.filter_table)
        self.layout.addWidget(self.search_box)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)  # Variant, Impact, Variant Type
        self.results_table.setHorizontalHeaderLabels(["Variant", "Impact", "Variant Type"])
        self.layout.addWidget(self.results_table)

        self.variant_distribution_button = QPushButton("Show Variant Type Distribution")
        self.variant_distribution_button.clicked.connect(self.show_variant_distribution)
        self.layout.addWidget(self.variant_distribution_button)

        self.impact_distribution_button = QPushButton("Show Impact Distribution")
        self.impact_distribution_button.clicked.connect(self.show_impact_distribution)
        self.layout.addWidget(self.impact_distribution_button)

        self.worker = None
        self.results = []
        self.vcf_df = None

    def upload_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a DNA File (VCF or VCF.GZ)", "", "VCF Files (*.vcf *.vcf.gz);;All Files (*)", options=options)

        if file_path:
            self.file_label.setText(f"Selected File: {file_path}")
            self.file_path = file_path
        else:
            self.file_label.setText("No file selected")

    def analyze_variants(self):
        if hasattr(self, 'file_path'):
            self.progress_bar.setValue(0)
            if self.worker:
                self.worker.terminate()

            base_dir = Path(__file__).resolve().parent.parent
            model_path = base_dir / "models" / "trained_model.joblib"

            self.worker = WorkerThread(self.file_path, model_path)
            self.worker.progress.connect(self.update_progress)
            self.worker.finished.connect(self.display_results)
            self.worker.error.connect(self.display_error)
            self.worker.finished.connect(self.worker_finished)
            self.worker.start()
        else:
            self.file_label.setText("Please upload a file first.")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def display_results(self, results, vcf_df):
        self.results = results
        self.vcf_df = vcf_df
        self.results_table.setRowCount(0)
        for index, (variant_id, prediction, variant_type) in enumerate(self.results):
            self.results_table.insertRow(index)
            self.results_table.setItem(index, 0, QTableWidgetItem(variant_id))
            self.results_table.setItem(index, 1, QTableWidgetItem(prediction))
            self.results_table.setItem(index, 2, QTableWidgetItem(variant_type))

    def filter_table(self, text):
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, 0)  # First column (Variant ID)
            self.results_table.setRowHidden(row, text.lower() not in item.text().lower())

    def display_error(self, message):
        print(f"Error: {message}")

    def worker_finished(self):
        self.worker = None

    def show_variant_distribution(self):
        if self.vcf_df is not None:
            distribution = self.vcf_df['Variant_Type'].value_counts()
            plt.figure(figsize=(8, 6))
            distribution.plot(kind='bar', color='skyblue')
            plt.title('Distribution of Variant Types')
            plt.xlabel('Variant Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("No data to display.")

    def show_impact_distribution(self):
        if self.results:
            predictions = [result[1] for result in self.results]
            benign_count = predictions.count('Benign')
            pathogenic_count = predictions.count('Pathogenic')
            plt.figure(figsize=(6, 6))
            plt.pie([benign_count, pathogenic_count], labels=['Benign', 'Pathogenic'],
                    autopct='%1.1f%%', colors=['green', 'red'])
            plt.title('Impact Distribution')
            plt.show()
        else:
            print("No results to display.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DNAAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
