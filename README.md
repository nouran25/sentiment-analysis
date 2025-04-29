# Sentiment Analysis on Movie Reviews 🎬

This project uses an LSTM-based neural network to classify movie reviews as Positive or Negative.

## Features
- Preprocessing using word-to-index conversion
- LSTM network for binary sentiment classification
- Interactive Jupyter notebook with a UI (text box and colored output)
- Displays prediction confidence and sentiment label

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Jupyter notebook:
```bash
jupyter notebook notebook/Sentiment_Analysis_LSTM.ipynb
```

## Folder Overview

### `model/`
Contains saved PyTorch model files.

- `trained_model.pt`: The trained LSTM model for sentiment classification.

### `scripts/`
Contains Python scripts for preprocessing, model definition, and predictions.

- `preprocess.py`: Preprocessing functions for text conversion (word-to-index).
- `model_def.py`: Definition of the LSTM model architecture.
- `predict.py`: Prediction logic for sentiment analysis.

### `notebook/`
Contains Jupyter notebook for interactive analysis and UI.

- `Sentiment_Analysis_LSTM.ipynb`: Interactive notebook with a UI for entering movie reviews and getting sentiment predictions.

## Dependencies
The project requires the following Python libraries:

- `torch` (for PyTorch)
- `numpy`
- `ipywidgets`
- `matplotlib`
- `pandas`
- `scikit-learn`

You can install them by running:

```bash
pip install -r requirements.txt
```
## Usage
1. Launch the Jupyter notebook using the command above.
2. Enter a movie review in the provided text box.
3. The sentiment (Positive/Negative) will be displayed.

## License
This project is for educational and research purposes only.

