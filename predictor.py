import pandas as pd
import numpy as np
from fastai.text.all import *
import os
import re

class TextPredictor:
    def __init__(self, model_path='./model'):
        self.model_path = model_path
        self.learn = None
        self.load_model()
        
    def load_model(self):
        """加载已训练的模型"""
        model_file = os.path.join(self.model_path, 'export.pkl')
        if not os.path.exists(model_file):
            raise FileNotFoundError("模型文件不存在，请先运行text_classifier.py训练模型")
            
        self.learn = load_learner(model_file)
        
    def _preprocess_text(self, text):
        """预处理文本数据"""
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text.strip())
        # 统一格式：转换为大写
        text = text.upper()
        return text
        
    def predict(self, text, top_k=3):
        """预测文本类别"""
        if self.learn is None:
            self.load_model()
            
        # 预处理输入文本
        processed_text = self._preprocess_text(text)
            
        # 获取预测结果和概率
        pred, pred_idx, probs = self.learn.predict(processed_text)
        
        # 获取所有类别的概率
        categories = self.learn.dls.vocab[1]
        probs_dict = {category: float(prob) for category, prob in zip(categories, probs)}
        
        # 按概率排序，返回top_k个结果
        sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return sorted_probs

# 交互式预测脚本
if __name__ == '__main__':
    try:
        # 创建预测器实例
        predictor = TextPredictor()
        
        print("文本分类预测器已启动！")
        print("请输入文本进行预测，输入'exit'退出程序")
        
        while True:
            # 获取用户输入
            user_input = input("\n请输入要预测的文本: ").strip()
            
            # 检查是否退出
            if user_input.lower() == 'exit':
                print("感谢使用文本分类预测器，再见！")
                break
            
            # 执行预测
            try:
                results = predictor.predict(user_input)
                
                # 打印预测结果
                print("预测类别（按置信度排序）:")
                for category, prob in results:
                    print(f"  - {category}: {prob:.4f}")
                
            except Exception as e:
                print(f"预测出错: {str(e)}")
                
    except Exception as e:
        print(f"程序异常: {str(e)}")
        print("请确保已成功运行text_classifier.py训练模型")