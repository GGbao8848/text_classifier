import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

"""
演示如何使用导出的ONNX模型进行推理

这个脚本展示了一个完整的工作流程：
1. 加载分词器
2. 加载ONNX模型
3. 处理输入文本
4. 运行推理
5. 获取预测结果
"""

def load_onnx_model(onnx_path, tokenizer_path):
    """加载ONNX模型和对应的分词器"""
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 加载ONNX模型
    session = ort.InferenceSession(onnx_path)
    
    return session, tokenizer

def predict_text(session, tokenizer, text, id2label=None):
    """使用ONNX模型预测文本类别"""
    # 使用分词器处理输入文本
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
    predicted_class_idx = np.argmax(logits, axis=-1)[0]
    
    # 如果提供了id2label映射，则返回类别名称
    if id2label is not None:
        predicted_class = id2label[predicted_class_idx]
        return predicted_class_idx, predicted_class
    
    return predicted_class_idx

def main():
    # 模型和分词器路径
    onnx_model_path = "./exported_models/model.onnx"
    tokenizer_path = "./results/checkpoint-3000"
    
    # 示例文本
    test_texts = ["FB 27J A",  "H942143Y82", "R203", "2023","87 7 X30","D"]
    
    # 加载模型和分词器
    print("正在加载模型和分词器...")
    session, tokenizer = load_onnx_model(onnx_model_path, tokenizer_path)
    
    print("\n推理演示:")
    # 对每个示例文本进行预测
    for text in test_texts:
        predicted_class_idx = predict_text(session, tokenizer, text)
        print(f"输入文本: {text} -> 预测类别索引: {predicted_class_idx}")
    
    print("\n推理完成!")

if __name__ == "__main__":
    main()