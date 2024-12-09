import json
import os
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import Dataset
import torch

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

with open("finetune_dataset.json") as f:
    data = json.load(f)

inputs = [item['input'] for item in data]
outputs = [item['output'] for item in data]
dataset = Dataset.from_dict({"text": [inp + "\n" + out for inp, out in zip(inputs, outputs)]})

model_name = "gpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=256)  

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./fine_tuned_llama2",
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    fp16=False,  
    report_to="clearml",  
    optim="adamw_torch", 
    max_grad_norm=0.3,  
)

torch.cuda.empty_cache()

model.gradient_checkpointing_enable()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

try:
    trainer.train()
   
    model.save_pretrained("./fine_tuned_llama2")
    tokenizer.save_pretrained("./fine_tuned_llama2")
    print("Fine-tuned model saved successfully.")
except RuntimeError as e:
    print(f"An error occurred during training: {e}")
    print("Consider further reducing batch size, sequence length, or using a smaller model.")
