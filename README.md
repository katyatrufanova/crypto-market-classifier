# Crypto Market Classifier

Welcome to the Crypto Market Classifier project! This repository explores cryptocurrency price trends using Multi-View Time Series Classification (MVTSC), providing deeper insights by analyzing different data view combinations.

This project is the result of a collaborative effort between [Alberto G. Valerio](https://github.com/albertovalerio) and myself. We both contributed equally to all aspects of its development.

## 🚀 Project Overview

In the rapidly evolving world of cryptocurrencies, understanding market trends is crucial. This project tackles the challenge of the volatile cryptocurrency market by employing a **Bagging Classifier** model in a multi-view analysis context. Our goal? To create a model that enhances the accuracy and robustness of time series data classification for various cryptocurrencies.

🔍 **Key Focus**: Improving predictive accuracy in cryptocurrency market analysis and advancing the field of financial data mining and cryptocurrency research.

## ⚠️ Important Note on Data Availability

**Please be aware**: The dataset used in this project is proprietary and cannot be publicly shared. The data download link has been removed from the repository (previously located in `notebooks/utils/global_config.py` under the `REMOTE_DATA` variable).

This repository is primarily for descriptive and presentation purposes. While the code and documentation are available for review, **users will not be able to run the code or reproduce the results without access to the proprietary dataset**.

## 🧠 Model Implementation

We're leveraging the power of [sklearn.ensemble.BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) for our model implementation.

## 🛠️ Requirements

- **Python**: Version 3.8 or higher (tested on 3.11.4)
- **System Requirements**: Check out [requirements.txt](/requirements.txt) for detailed dependencies

Our experiments were conducted using the following setup:
- CPU: 1 x Apple M2 Max - 12 Core
- GPU: 1 x Apple M2 Max - 38 Core
- RAM: 64 GB

## 📊 Dataset Overview

While we can't provide the actual data, here's what our dataset looked like:

- **Sources**: BINANCE, HUOBI, and OKX exchanges
- **Cryptocurrencies**: ADA, AVAX, BNB, BTC, DOGE, DOT, ETH, NEO, and SOL
- **Data Types**: 
  - Candles (open, high, low, close, volume, etc.)
  - Order Book Snapshots (bid/ask prices and sizes)

## 📓 Notebooks

Explore our analysis process through these Jupyter notebooks:

1. [Base Classifier](/notebooks/base-classifier.ipynb): Single-view and multi-view experiments with default parameters for each crypto.
2. [Hyperparameter Tuning](/notebooks/hyperparam-tuning.ipynb): Optimization of hyperparameters for both single-view and multi-view experiments.
3. [Results Analysis](/notebooks/results-analysis.ipynb): In-depth analysis of our experimental results.

## 📈 Key Results

Please refer to the [project documentation](docs/project_report.pdf) for detailed performance metrics and insights derived from the analysis.

### Average metrics by view

![accuracy_by_view](/graphs/accuracybyview.png)

### Average metrics by crypto

![accuracy_by_crypto](/graphs/accuracybycrypto.png)

### Trade-off between metrics and execution times

![accuracy_by_time](/graphs/accuracybytime.png)

## Authors

This project was 

## Acknowledgments

- **[@Scikit-learn](https://scikit-learn.org/)**

## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See `LICENSE.txt` for more information.