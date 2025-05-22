# LLM Finetuning: Comparing PEFT Techniques on Gemma-3-1b-it

## 1. Project Overview and Objectives

The primary goal of this project is to finetune the `google/gemma-3-1b-it` large language model using two distinct Parameter-Efficient Fine-Tuning (PEFT) techniques: QLoRA and GaLore (or a modified LoRA approach if GaLore is unavailable). The project aims to compare their effectiveness based on performance metrics (BLEU-4, ROUGE-L), resource consumption (peak GPU memory, training time), and to discuss potential generalization behaviors.

The base model for finetuning is `google/gemma-3-1b-it`, chosen for its strong instruction-following capabilities and manageable size for experimentation.

## 2. Setup Instructions

### Environment
*   Python 3.8+
*   We recommend using a virtual environment (e.g., `venv` or `conda`) to manage dependencies.

### Dependencies
Install the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
Ensure you have CUDA installed and configured correctly if you plan to use GPUs for training.

**Note on `requirements.txt`**:
This file should include the following packages (specific versions should be tested for compatibility):
```
torch
datasets
transformers
peft
trl
evaluate
bitsandbytes
accelerate # Often a dependency for transformers/peft
scipy # Often a dependency for evaluate or other libraries
# Add any other specific versions or libraries you used
```

## 3. Step-by-Step Guide to Run Scripts

The main script `finetune_gemma_peft.py` handles data preprocessing, model training, and evaluation.

### a. Data Sampling and Preprocessing
This step samples data from Alpaca, Tulu v2 SFT, and Ultrachat 200k datasets, formats it, and saves training and test sets as JSONL files.

**Command:**
```bash
python finetune_gemma_peft.py prep <your_data_output_directory>
```
*   Replace `<your_data_output_directory>` with the path where `train.jsonl` and `test.jsonl` will be saved (e.g., `./data`).

### b. Finetuning Models

#### i. Finetuning with QLoRA
This command trains the `gemma-3-1b-it` model using QLoRA.

**Command:**
```bash
python finetune_gemma_peft.py qlora <your_data_output_directory> <your_qlora_model_output_directory>
```
*   `<your_data_output_directory>`: Path to the directory containing `train.jsonl` and `test.jsonl` (from the `prep` step).
*   `<your_qlora_model_output_directory>`: Path where the finetuned QLoRA model and checkpoints will be saved (e.g., `./models/qlora_gemma`).

#### ii. Finetuning with GaLore (or Modified LoRA)
This command trains the `gemma-3-1b-it` model using GaLore. If GaLore is not available in your `peft` installation, the script will automatically fall back to a modified high-rank LoRA configuration as an alternative.

**Command:**
```bash
python finetune_gemma_peft.py galore <your_data_output_directory> <your_galore_model_output_directory>
```
*   `<your_data_output_directory>`: Path to the directory containing `train.jsonl` and `test.jsonl`.
*   `<your_galore_model_output_directory>`: Path where the finetuned GaLore (or modified LoRA) model and checkpoints will be saved (e.g., `./models/galore_gemma`).

### c. Evaluation
This step evaluates a finetuned model (or the base model) on the test set using BLEU-4 and ROUGE-L metrics.

**Command:**
```bash
python finetune_gemma_peft.py eval <path_to_your_model_or_model_directory> <path_to_your_test_jsonl>
```
*   `<path_to_your_model_or_model_directory>`: Path to the directory containing the saved model (e.g., `./models/qlora_gemma` or `google/gemma-3-1b-it` for base model evaluation).
*   `<path_to_your_test_jsonl>`: Path to the `test.jsonl` file created during the `prep` step (e.g., `./data/test.jsonl`).

**Example Evaluation Flow:**
1.  Evaluate base model:
    ```bash
    python finetune_gemma_peft.py eval google/gemma-3-1b-it ./data/test.jsonl
    ```
2.  Evaluate finetuned QLoRA model:
    ```bash
    python finetune_gemma_peft.py eval ./models/qlora_gemma ./data/test.jsonl
    ```
3.  Evaluate finetuned GaLore model:
    ```bash
    python finetune_gemma_peft.py eval ./models/galore_gemma ./data/test.jsonl
    ```-