import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 1. 读取CSV
df = pd.read_csv("/Users/songkui/mycode/text_classifier/data/class_data_v3.csv")

# 2. 转换为 (text, label) 格式
rows = []
for col in df.columns:
    for val in df[col].dropna().astype(str).tolist():
        rows.append({"text": val, "label": col})

data = pd.DataFrame(rows)

# 3. 标签编码
label2id = {label: i for i, label in enumerate(data["label"].unique())}
id2label = {i: label for label, i in label2id.items()}
data["label_id"] = data["label"].map(label2id)

# 4. 划分数据集
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 5. 加载分词器
model_name = "prajjwal1/bert-tiny"  # 极小的模型
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label_id", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset = test_dataset.rename_column("label_id", "labels")
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 6. 定义模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# 7. 训练
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

trainer.train()

# 8. 推理
text = "R203"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
pred = outputs.logits.argmax(-1).item()
print(f"输入: {text} -> 预测类别: {id2label[pred]}")
