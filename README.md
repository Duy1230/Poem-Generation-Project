# Vietnamese Poem Generation using GPT-2

This project implements a fine-tuned GPT-2 model for generating Vietnamese poems. The model is trained on a dataset of Vietnamese poems scraped from thivien.net.

## Features

- Web scraping of Vietnamese poems from thivien.net
- Data preprocessing and cleaning
- Fine-tuning GPT-2 model for Vietnamese poem generation
- Interactive poem generation interface

## Prerequisites

- Python 3.x
- CUDA-compatible GPU (recommended for training)
- Chrome WebDriver (for web scraping)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd vietnamese-poem-generation
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run streamlit interface:

```bash
streamlit run interface.py
```

## Usage

### 1. Data Collection (poem_mining.ipynb)

The script scrapes Vietnamese poems from thivien.net:

- Collects poem content, titles, and sources
- Processes and cleans the data
- Saves the dataset to `poem_dataset.csv`

To run the scraping:

```bash
jupyter notebook poem_mining.ipynb
```

### 2. Model Training (training.ipynb)

The notebook includes:

- Data preprocessing
- Model fine-tuning using GPT-2
- Poem generation testing

To train the model:

```bash
jupyter notebook training.ipynb
```

Training parameters:

- Base model: `danghuy1999/gpt2-viwiki`
- Learning rate: 2e-5
- Epochs: 10
- Max sequence length: 100
- FP16 training enabled

## Project Dependencies

```
streamlit
torch
transformers
pandas
selenium
tqdm
huggingface_hub
datasets
```

## Acknowledgments

- Base model: [danghuy1999/gpt2-viwiki](https://huggingface.co/danghuy1999/gpt2-viwiki)
- Poem source: [thivien.net](https://www.thivien.net/)
