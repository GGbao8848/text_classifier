# 模型导出工具使用指南

本项目提供了将训练好的BERT模型导出为ONNX或TorchScript格式的完整解决方案。

## 已完成的工作

1. **安装了必要的依赖包**：
   - onnx：用于模型转换为ONNX格式
   - onnxruntime：用于运行ONNX格式的模型

2. **创建了模型导出脚本**：
   - `export_model.py`：支持导出为ONNX和TorchScript格式

3. **创建了演示脚本**：
   - `demo_onnx_inference.py`：展示如何使用导出的ONNX模型进行推理

4. **更新了依赖配置**：
   - 已将新依赖添加到`requirements.txt`文件中

## 安装依赖

确保已安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

## 导出模型

### 导出为ONNX格式

```bash
python export_model.py --format onnx
```

### 导出为TorchScript格式

```bash
python export_model.py --format torchscript
```

### 导出并测试模型

您可以在导出后立即测试模型功能：

```bash
# 导出并测试ONNX模型
python export_model.py --format onnx --test

# 导出并测试TorchScript模型
python export_model.py --format torchscript --test
```

### 自定义参数

您可以自定义以下参数：

```bash
python export_model.py --format onnx --model_path ./results/checkpoint-3000 --output_dir ./exported_models --test --test_text "自定义测试文本"
```

## 使用导出的模型

### 使用ONNX模型

```python
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# 加载分词器
.tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-3000")

# 加载ONNX模型
session = ort.InferenceSession('./exported_models/model.onnx')

# 准备输入数据
text = "要分类的文本"
inputs = tokenizer(text, return_tensors="np")
input_ids = inputs["input_ids"].astype(np.int64)
attention_mask = inputs["attention_mask"].astype(np.int64)

# 运行推理
outputs = session.run(['logits'], {
    'input_ids': input_ids,
    'attention_mask': attention_mask
})

# 获取预测结果
logits = outputs[0]
predicted_class = np.argmax(logits, axis=-1)
print(f"预测类别索引: {predicted_class[0]}")
```

### 使用TorchScript模型

```python
import torch
from transformers import AutoTokenizer

# 加载分词器
.tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-3000")

# 加载TorchScript模型
traced_model = torch.jit.load('./exported_models/model.pt')

# 准备输入数据
text = "要分类的文本"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 运行推理
with torch.no_grad():
    outputs = traced_model(input_ids, attention_mask)
    
    # 检查输出类型，如果是字典则直接获取logits
    if isinstance(outputs, dict) and 'logits' in outputs:
        logits = outputs['logits']
    else:
        logits = outputs.logits

# 获取预测结果
predicted_class = torch.argmax(logits, dim=-1)
print(f"预测类别索引: {predicted_class.item()}")
```

## 演示脚本

本项目包含一个完整的ONNX模型推理演示脚本：

```bash
python demo_onnx_inference.py
```

运行此脚本将展示如何加载ONNX模型并进行文本分类推理。

## 常见问题解决方案

1. **问题**：导出ONNX模型时出现`ModuleNotFoundError: No module named 'onnx'`
   **解决方案**：安装onnx包：`pip install onnx`

2. **问题**：使用ONNX模型进行推理时出现错误
   **解决方案**：安装onnxruntime：`pip install onnxruntime`

3. **问题**：导出TorchScript模型时出现`Encountering a dict at the output of the tracer`错误
   **解决方案**：脚本已自动处理此问题，通过添加`strict=False`参数

## 目录结构

```
├── data/                  # 数据文件夹
│   ├── class_data_v3.csv  # 训练数据
│   └── valid_data_v3.csv  # 验证数据
├── exported_models/       # 导出的模型保存位置
├── results/               # 训练结果
│   └── checkpoint-3000/   # 训练好的模型检查点
├── demo_onnx_inference.py # ONNX模型推理演示脚本
├── export_model.py        # 模型导出工具
├── face_train.py          # 原始训练脚本
└── requirements.txt       # 项目依赖
```

## 注意事项

1. 导出的模型将保存在`./exported_models`目录中
2. 确保您使用的PyTorch版本与ONNX兼容
3. ONNX模型可以在不依赖PyTorch的环境中运行，适合部署场景
4. TorchScript模型保留了更多PyTorch功能，适合在PyTorch生态系统中使用