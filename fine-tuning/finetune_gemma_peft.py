import argparse
import os
import random
import time
from pathlib import Path
import json

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer

# Check for GaLoreConfig availability
has_galore = False
try:
    from peft import GaLoreConfig
    has_galore = True
except (ImportError, AttributeError):
    # GaLoreConfig is not available in this PEFT version
    pass

import evaluate

TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{response}"
)

DATASET_SPECS = [
    ("tatsu-lab/alpaca", "alpaca"),
    ("allenai/tulu-v2-sft-mixture", "tulu"),
    ("HuggingFaceH4/ultrachat_200k", "ultrachat"),
]

TRAIN_SAMPLES = 5000
TEST_SAMPLES = 2000
SEED = 42
MODEL_NAME = "google/gemma-3-1b-it"


# --------------------------- misc helpers ---------------------------

def _write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


def _set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# --------------------------- dataset mappers ---------------------------

def _map_alpaca(ex):
    return {
        "text": TEMPLATE.format(
            instruction=ex["instruction"],
            input=ex.get("input", ""),
            response=ex["output"],
        )
    }


def _chat_to_strings(msgs):
    user, assistant = [], []
    for m in msgs:
        if m["role"] == "user":
            user.append(m["content"])
        elif m["role"] == "assistant":
            assistant.append(m["content"])
    return "\n".join(user), "\n".join(assistant)


def _map_tulu(ex):
    instr, resp = _chat_to_strings(ex["messages"])
    return {"text": TEMPLATE.format(instruction=instr, input="", response=resp)}


def _map_ultrachat(ex):
    instr, resp = _chat_to_strings(ex["messages"])
    return {"text": TEMPLATE.format(instruction=instr, input="", response=resp)}


# --------------------------- data prep ---------------------------

def prep_datasets(out_dir: str):
    """Sample TRAIN_SAMPLES/TEST_SAMPLES from each dataset and save JSONL."""
    _set_seed(SEED)
    train, test = [], []
    for hf_name, tag in DATASET_SPECS:
        print(f"Loading {hf_name}…")
        # Handle different split names for different datasets
        if "ultrachat" in hf_name:
            split_name = "train_sft"  # Use train_sft for ultrachat
        else:
            split_name = "train"
        
        ds = load_dataset(hf_name, split=split_name)
        idx = list(range(len(ds)))
        random.shuffle(idx)
        train_idx = idx[:TRAIN_SAMPLES]
        test_idx = idx[TRAIN_SAMPLES : TRAIN_SAMPLES + TEST_SAMPLES]
        mapper = globals()[f"_map_{tag}"]
        train.extend([mapper(ds[i]) for i in train_idx])
        test.extend([mapper(ds[i]) for i in test_idx])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    _write_jsonl(Path(out_dir) / "train.jsonl", train)
    _write_jsonl(Path(out_dir) / "test.jsonl", test)


# --------------------------- training ---------------------------

def train_qlora(data_dir: str, out_dir: str, bs: int = 4, epochs: int = 1):
    tok = _load_tokenizer()
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", quantization_config=bnb_cfg)
    model = prepare_model_for_kbit_training(model)
    l_cfg = LoraConfig(r=64, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, l_cfg)

    ds_train = load_dataset("json", data_files=str(Path(data_dir) / "train.jsonl"), split="train")
    ds_eval = load_dataset("json", data_files=str(Path(data_dir) / "test.jsonl"), split="train").select(range(1000))

    args = TrainingArguments(
        out_dir,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        peft_config=l_cfg,
        processing_class=tok
    )
    start = time.time()
    trainer.train()
    print(f"QLoRA training time: {(time.time()-start)/3600:.2f}h")
    trainer.save_model(out_dir)


def train_galore(data_dir: str, out_dir: str, bs: int = 4, epochs: int = 1):
    if not has_galore:
        import peft
        print(f"GaLore training is not available in your installation despite PEFT version {peft.__version__}.")
        return
        
    tok = _load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    g_cfg = GaLoreConfig(rank=64, proj_type="gated", beta1=0.85, beta2=0.95)
    model = get_peft_model(model, g_cfg)

    ds_train = load_dataset("json", data_files=str(Path(data_dir) / "train.jsonl"), split="train")
    ds_eval = load_dataset("json", data_files=str(Path(data_dir) / "test.jsonl"), split="train").select(range(1000))

    args = TrainingArguments(
        out_dir,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        peft_config=g_cfg,
        processing_class=tok
    )
    start = time.time()
    trainer.train()
    print(f"GaLore training time: {(time.time()-start)/3600:.2f}h")
    trainer.save_model(out_dir)


def train_modified_galore(data_dir: str, out_dir: str, bs: int = 4, epochs: int = 1):
    """
    A function that mimics some aspects of GaLore using LoRA configuration
    since GaLoreConfig is not available in this PEFT installation.
    """
    print("Using LoRA as an alternative to GaLore since GaLoreConfig is not available in your installation.")
    print("This will use a high-rank LoRA configuration to achieve similar effects.")
    
    tok = _load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    
    # Using LoRA with higher rank and alpha as an alternative
    l_cfg = LoraConfig(
        r=128,                      # Higher rank than standard LoRA
        lora_alpha=32,              # Higher alpha for stronger adaptation
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, l_cfg)

    ds_train = load_dataset("json", data_files=str(Path(data_dir) / "train.jsonl"), split="train")
    ds_eval = load_dataset("json", data_files=str(Path(data_dir) / "test.jsonl"), split="train").select(range(1000))

    args = TrainingArguments(
        out_dir,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        peft_config=l_cfg,
        processing_class=tok
    )
    start = time.time()
    trainer.train()
    print(f"Modified LoRA training time: {(time.time()-start)/3600:.2f}h")
    trainer.save_model(out_dir)


# --------------------------- evaluation ---------------------------

def _generate(model, tok, prompts, max_new_tokens=128):
    inp = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens)
    return tok.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)


def evaluate_model(model_path: str, test_jsonl: str, batch: int = 8, max_examples: int = 100):
    tok = _load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    
    print(f"Loading dataset from {test_jsonl}")
    ds = load_dataset("json", data_files=test_jsonl, split="train")
    print(f"Dataset loaded, size: {len(ds)}")
    
    # Limit the number of examples for evaluation
    test_size = min(max_examples, len(ds))
    print(f"Using {test_size} examples for evaluation")
    
    preds, refs = [], []
    for i in range(0, test_size, batch):
        batch_end = min(i + batch, test_size)
        batch_prompts = []
        batch_responses = []
        
        for idx in range(i, batch_end):
            item = ds[idx]
            text = None
            
            if isinstance(item, dict) and "text" in item:
                text = item["text"]
            elif isinstance(item, str):
                text = item
                
            if text and "### Response:" in text:
                response_idx = text.find("### Response:")
                prompt = text[:response_idx] + "### Response:"
                response = text[response_idx + len("### Response:"):].strip()
                batch_prompts.append(prompt)
                batch_responses.append(response)
        
        if batch_prompts:
            generated = _generate(model, tok, batch_prompts)
            preds.extend(generated)
            refs.extend(batch_responses)
            print(f"Evaluated {len(preds)}/{test_size} examples")
    
    if not preds:
        print("Error: No predictions generated.")
        return
        
    print("\nExample prediction:")
    print(f"Generated: {preds[0][:100]}...")
    print(f"Reference: {refs[0][:100]}...")
    
    bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])["bleu"]
    rouge_score = rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"])["rougeL"]
    print(json.dumps({"BLEU-4": bleu_score, "ROUGE-L": rouge_score}, indent=2))


# --------------------------- CLI ---------------------------

def main():
    p = argparse.ArgumentParser("Gemma PEFT pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prep")
    sp.add_argument("out_dir")

    sq = sub.add_parser("qlora")
    sq.add_argument("data_dir")
    sq.add_argument("out_dir")

    sg = sub.add_parser("galore")
    sg.add_argument("data_dir")
    sg.add_argument("out_dir")

    se = sub.add_parser("eval")
    se.add_argument("model_path")
    se.add_argument("test_jsonl")

    args = p.parse_args()
    if args.cmd == "prep":
        prep_datasets(args.out_dir)
    elif args.cmd == "qlora":
        train_qlora(args.data_dir, args.out_dir)
    elif args.cmd == "galore":
        if not has_galore:
            import peft
            print(f"GaLore training is not available in your installation despite PEFT version {peft.__version__}.")
            print("Using an alternative approach with high-rank LoRA settings instead.")
            train_modified_galore(args.data_dir, args.out_dir)
            return
        train_galore(args.data_dir, args.out_dir)
    elif args.cmd == "eval":
        evaluate_model(args.model_path, args.test_jsonl)


if __name__ == "__main__":
    main()
