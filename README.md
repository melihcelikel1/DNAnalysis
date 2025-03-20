# DNAnalysis

DNAnalysis is a Python-based project developed for analyzing DNA variants. It processes VCF files to identify various genetic variants.

## 🚀 Features
- DNA variant analysis from VCF files
- Variant impact prediction using trained machine learning models
- User-friendly PyQt interface

## ⚙️ Installation

### 1. Install the required libraries:
```bash
pip install -r requirements.txt
```

## 📥 Pre-trained Models (External Download)

Due to GitHub's storage limits, please download the pre-trained models from these external links and place them in the `models/` folder:

- [Download Models (Google Drive)](https://drive.google.com/drive/folders/1NWHGTvO8Uc2U252CwyE3KLTMU3om600d?usp=sharing)

### 2. Analyze using the trained model:
```bash
python vcf_analyse/vcf_analyse.py
```


## 📂 Project Structure
```
DNAnalysis/
├── model_training/          # Model training scripts
├── models/                  # Trained models
├── vcf_analyse/             # Analysis scripts and interface
├── .gitignore               # Excludes unnecessary files
├── requirements.txt         # Required Python libraries
└── README.md                # This file
```

## 📌 Preparing Datasets
You can download the required datasets (`.vcf.gz`) from the following source:

- [ClinVar Database](https://www.ncbi.nlm.nih.gov/clinvar/)



## 🤝 Contributing
Contributions are welcome! Feel free to submit pull requests.

## 📄 License
This project is licensed under the MIT License.
