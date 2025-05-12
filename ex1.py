from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, TrainingArguments, Trainer, \
    DataCollatorWithPadding
from datasets import load_dataset
import numpy as np
from evaluate import load
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max_train_samples", type=int, default=-1)
parser.add_argument("--max_eval_samples", type=int, default=-1)
parser.add_argument("--max_predict_samples", type=int, default=-1)
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=0.00005)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_predict", action="store_true")
parser.add_argument("--model_path", type=str, default="")
args = parser.parse_args()

wandb.login()

config = AutoConfig.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
ds = load_dataset("nyu-mll/glue", "mrpc")
metric = load("accuracy")
wandb.init(project="anlp_ex1", name=f"epoch_num_{args.num_train_epochs}_lr_{args.lr}_batch_size_{args.batch_size}")

args.max_train_samples = min(len(ds["train"]), args.max_train_samples) if args.max_train_samples != -1 else len(ds["train"])
args.max_eval_samples = min(len(ds["validation"]), args.max_eval_samples) if args.max_eval_samples != -1 else len(ds["validation"])
args.max_predict_samples = min(len(ds["test"]), args.max_predict_samples) if args.max_predict_samples != -1 else len(ds["test"])
train_dataset = ds["train"].shuffle().select(range(args.max_train_samples))
val_dataset = ds["validation"].shuffle().select(range(args.max_eval_samples))
test_dataset = ds["test"].shuffle().select(range(args.max_predict_samples))
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def tokenize(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=56)


def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    return result


tokenized_train = train_dataset.map(tokenize, batched=True)
tokenized_val = val_dataset.map(tokenize, batched=True)
tokenized_test = test_dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    eval_strategy=("epoch" if args.do_train else "no"),
    learning_rate=args.lr,
    logging_strategy="steps",
    logging_steps=1,
    logging_dir="./logs",
    report_to="wandb",
    run_name=f"epoch_num_{args.num_train_epochs}_lr_{args.lr}_batch_size_{args.batch_size}"
)


if args.do_train:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        processing_class=tokenizer
    )

    trainer_results = trainer.train()
    metric = trainer.evaluate(eval_dataset=tokenized_val)

    with open("res.txt", "a") as f:
        f.write(f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {metric['eval_accuracy']:.4f}\n")

if args.do_predict and args.model_path:
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    logits = trainer.predict(tokenized_test)
    preds = logits.predictions.argmax(axis=-1)
    with open("predictions.txt", "w") as file:
        for i in range(len(preds)):
            file.write(f"<{test_dataset[i]['sentence1']}>###<{test_dataset[i]['sentence2']}>###<predicted label {preds[i]}>\n")
    metric = trainer.evaluate(eval_dataset=tokenized_test)

wandb.finish()