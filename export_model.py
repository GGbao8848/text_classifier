import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import onnxruntime as ort
import numpy as np

"""
模型导出工具 - 将训练好的BERT模型导出为ONNX或TorchScript格式

使用方法:
1. 导出为ONNX: python export_model.py --format onnx
2. 导出为TorchScript: python export_model.py --format torchscript
3. 导出后进行测试: python export_model.py --format onnx --test
"""

def export_to_onnx(model, tokenizer, output_path):
    """将模型导出为ONNX格式"""
    # 创建一个示例输入
    inputs = tokenizer("示例文本", return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # 设置模型为评估模式
    model.eval()
    
    # 导出为ONNX
    torch.onnx.export(
        model,
        (
            input_ids,
            attention_mask
        ),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        }
    )
    
    print(f"模型已成功导出为ONNX格式: {output_path}")

def export_to_torchscript(model, tokenizer, output_path):
    """将模型导出为TorchScript格式"""
    # 创建一个示例输入
    inputs = tokenizer("示例文本", return_tensors="pt")
    
    # 设置模型为评估模式
    model.eval()
    
    # 跟踪模型，设置strict=False以允许字典输出
    traced_model = torch.jit.trace(model, (
        inputs["input_ids"],
        inputs["attention_mask"]
    ), strict=False)
    
    # 保存TorchScript模型
    torch.jit.save(traced_model, output_path)
    
    print(f"模型已成功导出为TorchScript格式: {output_path}")

def load_trained_model(model_path, num_labels=None):
    """加载训练好的模型"""
    # 从checkpoint加载模型
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def test_onnx_model(onnx_path, tokenizer, text="测试文本"):
    """测试ONNX模型的推理功能"""
    # 创建ONNX运行时会话
    session = ort.InferenceSession(onnx_path)
    
    # 使用分词器处理输入文本
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # 运行推理
    outputs = session.run(['logits'], {'input_ids': input_ids, 'attention_mask': attention_mask})
    logits = outputs[0]
    predicted_class = np.argmax(logits, axis=-1)
    
    print(f"\nONNX模型测试:")
    print(f"输入文本: {text}")
    print(f"预测类别索引: {predicted_class[0]}")
    return predicted_class[0]

def test_torchscript_model(torchscript_path, tokenizer, text="测试文本"):
    """测试TorchScript模型的推理功能"""
    # 加载TorchScript模型
    traced_model = torch.jit.load(torchscript_path)
    
    # 使用分词器处理输入文本
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
    
    predicted_class = torch.argmax(logits, dim=-1)
    
    print(f"\nTorchScript模型测试:")
    print(f"输入文本: {text}")
    print(f"预测类别索引: {predicted_class.item()}")
    return predicted_class.item()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="模型导出工具")
    parser.add_argument("--format", type=str, choices=["onnx", "torchscript"], 
                        default="onnx", help="导出的模型格式")
    parser.add_argument("--model_path", type=str, 
                        default="./results/checkpoint-3000", 
                        help="训练好的模型路径")
    parser.add_argument("--output_dir", type=str, 
                        default="./exported_models", 
                        help="导出模型的保存目录")
    parser.add_argument("--test", action="store_true", 
                        help="导出后测试模型")
    parser.add_argument("--test_text", type=str, 
                        default="测试文本", 
                        help="测试模型使用的文本")
    
    args = parser.parse_args()
    
    try:
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 加载模型
        print(f"正在加载模型: {args.model_path}")
        model, tokenizer = load_trained_model(args.model_path)
        
        # 导出模型
        if args.format == "onnx":
            output_path = os.path.join(args.output_dir, "model.onnx")
            export_to_onnx(model, tokenizer, output_path)
        elif args.format == "torchscript":
            output_path = os.path.join(args.output_dir, "model.pt")
            export_to_torchscript(model, tokenizer, output_path)
        
        # 测试模型
        if args.test:
            if args.format == "onnx":
                test_onnx_model(output_path, tokenizer, args.test_text)
            else:
                test_torchscript_model(output_path, tokenizer, args.test_text)
        
        print("\n使用指南:")
        if args.format == "onnx":
            print("1. ONNX模型可以使用ONNX Runtime或其他支持ONNX的推理框架加载")
            print("2. 示例代码:")
            print("""
import onnxruntime as ort
import numpy as np

# 加载ONNX模型
session = ort.InferenceSession('exported_models/model.onnx')

# 准备输入数据
input_ids = np.array([[...]]).astype(np.int64)  # 根据实际分词后的input_ids填写
attention_mask = np.array([[...]]).astype(np.int64)  # 根据实际分词后的attention_mask填写

# 运行推理
outputs = session.run(['logits'], {'input_ids': input_ids, 'attention_mask': attention_mask})
logits = outputs[0]
predicted_class = np.argmax(logits, axis=-1)
""")
        else:
            print("1. TorchScript模型可以使用PyTorch加载")
            print("2. 示例代码:")
            print("""
import torch

# 加载TorchScript模型
traced_model = torch.jit.load('exported_models/model.pt')

# 准备输入数据
input_ids = torch.tensor([[...]])  # 根据实际分词后的input_ids填写
attention_mask = torch.tensor([[...]])  # 根据实际分词后的attention_mask填写

# 运行推理
with torch.no_grad():
    outputs = traced_model(input_ids, attention_mask)
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=-1)
""")
        
    except Exception as e:
        print(f"导出过程中出现错误: {str(e)}")
        print("请检查以下几点:")
        print("1. 确保模型路径正确")
        print("2. 确保已安装所有依赖: pip install -r requirements.txt")
        print("3. 确保PyTorch版本与ONNX兼容")