import pandas as pd
import numpy as np
from fastai.text.all import *
import os
import re
import logging
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextClassifier:
    def __init__(self, model_path='./model'):
        self.model_path = model_path
        self.learn = None
        self.data_loader = None
        self.categories = None
        self.history = {}
        self.best_lr = None
        
    def load_data(self, train_csv, valid_csv):
        """加载训练和验证数据"""
        # 读取CSV文件
        train_df = pd.read_csv(train_csv)
        valid_df = pd.read_csv(valid_csv)
        
        # 获取类别名称
        categories = train_df.columns.tolist()
        
        # 保存类别信息
        self.categories = categories
        
        # 准备训练数据：将每个字段的值与其类别组合
        train_data = []
        for category in categories:
            values = train_df[category].dropna().unique()
            for value in values:
                if pd.notna(value):
                    # 文本预处理
                    processed_text = self._preprocess_text(str(value))
                    train_data.append((processed_text, category))
        
        # 准备验证数据
        valid_data = []
        for category in categories:
            values = valid_df[category].dropna().unique()
            for value in values:
                if pd.notna(value):
                    # 文本预处理
                    processed_text = self._preprocess_text(str(value))
                    valid_data.append((processed_text, category))
        
        logger.info(f"训练数据大小: {len(train_data)}，验证数据大小: {len(valid_data)}")
        
        # 创建DataFrame
        train_df_processed = pd.DataFrame(train_data, columns=['text', 'label'])
        valid_df_processed = pd.DataFrame(valid_data, columns=['text', 'label'])
        
        # 创建TextDataLoaders
        self.data_loader = TextDataLoaders.from_df(
            train_df_processed,
            valid_df=valid_df_processed,
            text_col='text',
            label_col='label',
            is_lm=False,
            seq_len=72,
            bs=64,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # 记录类别分布
        self._log_class_distribution(train_df_processed, valid_df_processed)
        
        return self.data_loader
        
    def _preprocess_text(self, text):
        """预处理文本数据"""
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text.strip())
        # 统一格式：转换为大写
        text = text.upper()
        return text
        
    def _log_class_distribution(self, train_df, valid_df):
        """记录训练和验证数据的类别分布"""
        train_counts = train_df['label'].value_counts()
        valid_counts = valid_df['label'].value_counts()
        
        logger.info("\n训练数据类别分布:")
        for category, count in train_counts.items():
            logger.info(f"  {category}: {count} ({count/len(train_df):.2%})")
            
        logger.info("\n验证数据类别分布:")
        for category, count in valid_counts.items():
            logger.info(f"  {category}: {count} ({count/len(valid_df):.2%})")
        
    def train_model(self, epochs=8, drop_mult=0.3):
        """训练模型"""
        if self.data_loader is None:
            raise ValueError("请先加载数据")
            
        # 创建学习器
        self.learn = text_classifier_learner(
            self.data_loader,
            AWD_LSTM,
            drop_mult=drop_mult,
            metrics=[accuracy, F1Score(average='macro')]
        )
        
        # 查找最佳学习率
        lr_finder = self.learn.lr_find()
        self.best_lr = lr_finder.valley
        logger.info(f"最佳学习率: {self.best_lr:.6f}")
        
        # 冻结预训练层，先训练头部
        logger.info("开始训练模型头部...")
        self.learn.fit_one_cycle(3, self.best_lr)
        
        # 解冻所有层，继续训练
        logger.info("解冻所有层，继续训练...")
        self.learn.unfreeze()
        self.learn.fit_one_cycle(epochs-3, slice(self.best_lr/10, self.best_lr))
        
        # 保存训练历史
        self.history = self.learn.recorder.values
        
        return self.learn
        
    def save_model(self):
        """保存模型"""
        if self.learn is None:
            raise ValueError("请先训练模型")
            
        # 创建模型目录（如果不存在）
        os.makedirs(self.model_path, exist_ok=True)
        
        # 保存模型
        self.learn.export(os.path.join(self.model_path, 'export.pkl'))
        
    def load_model(self):
        """加载已训练的模型"""
        model_file = os.path.join(self.model_path, 'export.pkl')
        if not os.path.exists(model_file):
            raise FileNotFoundError("模型文件不存在，请先训练并保存模型")
            
        self.learn = load_learner(model_file)
        
    def evaluate_model(self):
        """评估模型性能"""
        if self.learn is None:
            self.load_model()
            
        # 获取验证集预测结果
        logger.info("评估模型性能...")
        valid_dl = self.learn.dls.valid
        
        # 存储真实标签和预测标签
        true_labels = []
        pred_labels = []
        
        # 获取所有验证集的预测结果
        # get_preds返回的是(preds, targets)
        preds, targets = self.learn.get_preds(dl=valid_dl)
        
        # 添加到列表中
        true_labels.extend(targets.tolist())
        pred_labels.extend(preds.argmax(dim=1).tolist())
        
        # 转换为类别名称
        vocab = self.learn.dls.vocab[1]
        true_categories = [vocab[i] for i in true_labels]
        pred_categories = [vocab[i] for i in pred_labels]
        
        # 打印分类报告
        logger.info("\n模型分类报告:\n")
        report = classification_report(true_categories, pred_categories, digits=4)
        print(report)
        
        return true_categories, pred_categories
        
            
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

# 示例用法
if __name__ == '__main__':
    # 创建分类器实例
    classifier = TextClassifier()
    
    # 加载数据
    print("加载数据...")
    classifier.load_data('./data/class_data_v3.csv', './data/valid_data_v3.csv')
    
    # 训练模型
    print("训练模型...")
    classifier.train_model(epochs=40, drop_mult=0.3)
    
    # 保存模型
    print("保存模型...")
    classifier.save_model()
    
    # 评估模型
    print("\n评估模型性能...")
    classifier.evaluate_model()
    
    
    # 测试预测
    print("\n测试预测结果：")
    test_cases = ["FE 82JD", "H942143Y82", "2023-02-03", "87 7 X30", "D"]
    for test_text in test_cases:
        results = classifier.predict(test_text)
        print(f"\n输入: {test_text}")
        print("预测类别（按置信度排序）:")
        for category, prob in results:
            print(f"  - {category}: {prob:.4f}")