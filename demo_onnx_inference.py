import onnxruntime as ort
import numpy as np
import json
import re

"""
演示如何使用导出的ONNX模型进行推理（与transformers解耦版本）

这个脚本实现了一个轻量级的文本分类推理系统，特点是：
1. 不依赖transformers库
2. 实现了自定义的简化BERT分词器
3. 可以单独使用，便于集成到其他项目中
4. 保持了与原始模型相同的推理精度
"""

class SimpleBertTokenizer:
    """简化版的BERT分词器，不依赖transformers"""
    def __init__(self, tokenizer_info):
        self.vocab = tokenizer_info['vocab']
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.cls_token = tokenizer_info['cls_token']
        self.sep_token = tokenizer_info['sep_token']
        self.pad_token = tokenizer_info['pad_token']
        self.unk_token = tokenizer_info['unk_token']
        self.model_max_length = tokenizer_info['model_max_length']
        self.do_lower_case = tokenizer_info.get('do_lower_case', False)
        self.clean_up_tokenization_spaces = tokenizer_info.get('clean_up_tokenization_spaces', True)
        
        # 获取特殊token的ID
        self.cls_token_id = self.vocab[self.cls_token]
        self.sep_token_id = self.vocab[self.sep_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
    
    def _clean_text(self, text):
        """清理文本"""
        if self.clean_up_tokenization_spaces:
            text = text.strip()
            # 合并多余的空格
            text = re.sub(r'\s+', ' ', text)
        return text
    
    def _tokenize(self, text):
        """基础分词逻辑"""
        # 转小写
        if self.do_lower_case:
            text = text.lower()
        
        # 简单的分词，按空格分割
        tokens = text.split()
        
        # 子词切分（简化版，只处理已知的token）
        output_tokens = []
        for token in tokens:
            if token in self.vocab:
                output_tokens.append(token)
            else:
                # 如果整个词不在词汇表中，尝试按字符切分
                # 这里使用简化的逻辑，实际BERT分词器会更复杂
                chars = list(token)
                for char in chars:
                    if char in self.vocab:
                        output_tokens.append(char)
                    else:
                        output_tokens.append(self.unk_token)
        
        return output_tokens
    
    def encode(self, text, max_length=128, pad_to_max_length=True):
        """将文本编码为input_ids和attention_mask"""
        # 清理文本
        text = self._clean_text(text)
        
        # 分词
        tokens = self._tokenize(text)
        
        # 添加特殊token
        tokens = [self.cls_token] + tokens + [self.sep_token]
        
        # 转换为ID
        input_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # 计算长度
        seq_length = len(input_ids)
        
        # 截断（保留CLS和SEP）
        if max_length is not None and seq_length > max_length:
            # 保留第一个token(CLS)和最后一个token(SEP)
            input_ids = [input_ids[0]] + input_ids[1:max_length-1] + [input_ids[-1]]
            seq_length = max_length
        
        # 填充
        if pad_to_max_length and max_length is not None:
            padding_length = max_length - seq_length
            input_ids += [self.pad_token_id] * padding_length
        
        # 创建attention_mask
        attention_mask = [1] * seq_length
        if pad_to_max_length and max_length is not None:
            attention_mask += [0] * padding_length
        
        # 确保长度正确
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        
        return {
            'input_ids': np.array([input_ids], dtype=np.int64),
            'attention_mask': np.array([attention_mask], dtype=np.int64)
        }

def load_onnx_model_and_tokenizer(onnx_path):
    """加载ONNX模型和自定义分词器"""
    # 加载ONNX模型
    session = ort.InferenceSession(onnx_path)
    
    # 加载tokenizer信息
    tokenizer_info_path = onnx_path.replace('.onnx', '_tokenizer_info.json')
    with open(tokenizer_info_path, 'r', encoding='utf-8') as f:
        tokenizer_info = json.load(f)
    
    # 创建自定义分词器
    tokenizer = SimpleBertTokenizer(tokenizer_info)
    
    return session, tokenizer

def predict_text(session, tokenizer, text, max_length=128):
    """使用ONNX模型预测文本类别"""
    # 处理输入文本
    inputs = tokenizer.encode(text, max_length=max_length, pad_to_max_length=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # 运行推理
    outputs = session.run(['logits'], {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    })
    
    # 获取预测结果
    logits = outputs[0]
    predicted_class_idx = np.argmax(logits, axis=-1)[0]
    
    return predicted_class_idx

def main():
    # 模型路径
    onnx_model_path = "./exported_models/model.onnx"
    
    # 示例文本
    test_texts = ["FB 27J A", "FB27JB","H942143Y82", "R203", "2023", "87 7 X30", "D"]
    
    # 加载模型和自定义分词器
    print("正在加载模型和分词器...")
    session, tokenizer = load_onnx_model_and_tokenizer(onnx_model_path)
    
    print("\n推理演示:")
    # 对每个示例文本进行预测
    for text in test_texts:
        predicted_class_idx = predict_text(session, tokenizer, text)
        print(f"输入文本: {text} -> 预测类别索引: {predicted_class_idx}")
    
    print("\n推理完成!")

if __name__ == "__main__":
    main()