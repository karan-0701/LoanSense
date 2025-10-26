# LoanSense Project

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## Run Training Pipeline
From the project root, run:
* Train both DL and RL models (default):

```bash
python src/main.py
```

* Train only DL model:

```bash
python src/main.py --model dl
```

* Train only RL agent:

```bash
python src/main.py --model rl
```

## Project Structure
* `data/raw/` – place raw CSVs here
* `data/processed/` – processed data saved here
* `models/` – trained models saved here
* `src/` – source code and pipelines

### Read the project report for detailed analysis
