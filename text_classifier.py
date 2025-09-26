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
            seq_len=50,  # 减小序列长度以减少计算量
            bs=32,       # 适当减小batch size
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # 记录类别分布
        self._log_class_distribution(train_df_processed, valid_df_processed)
        
        return self.data_loader
        
    def _preprocess_text(self, text):
        """预处理文本数据"""
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 特殊格式处理
        # 标准化日期格式（尝试将各种格式的日期统一表示）
        if re.search(r'\d{4}[-/]?\d{1,2}[-/]?\d{1,2}', text):
            # 识别为日期，添加特殊标记帮助模型识别
            text = f"[DATE] {text}"
        
        # 识别IQI_tag模式（通常以F开头，后跟字母和数字）
        elif re.match(r'^F[A-Z0-9]', text) and len(text) <= 10:
            text = f"[IQI] {text}"
        
        # 识别单个字符的情况（通常是Slice类别）
        elif len(text.strip()) == 1 and text.isalnum():
            text = f"[SINGLE_CHAR] {text}"
        
        # 识别WelderNo模式（可能包含字母和数字的组合）
        elif re.match(r'^[A-Z0-9]+$', text) and len(text) > 3:
            text = f"[WELDER] {text}"
        
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
        
    def train_model(self, epochs=20, drop_mult=0.5, use_small_model=True):
        """训练模型"""
        if self.data_loader is None:
            raise ValueError("请先加载数据")
            
        # 计算类别权重，解决类别不平衡问题
        # 特别是针对WeldNo(召回率低)和WelderNo(精确度低)等类别
        def get_class_weights():
            # 安全地获取训练数据中的类别分布
            # 避免直接从Dataset构建DataFrame可能出现的问题
            label_counts = {}            
            # 遍历训练数据集，统计每个类别的样本数
            for item in self.data_loader.train_ds:
                label = item[1]
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
            
            total = sum(label_counts.values())
            
            # 计算权重（使用1/频率的倒数）
            weights = {label: total / count for label, count in label_counts.items()}
            
            # 为表现不佳的类别增加权重
            # 根据分类报告，Slice召回率很低，单个字符如"D"被错误预测为WelderNo
            weights['Slice'] = weights.get('Slice', 1.0) * 10.0  # 大幅增加Slice权重
            weights['WeldNo'] = weights.get('WeldNo', 1.0) * 1.2
            # 降低WelderNo权重以减少它对其他类别的"占用"
            if 'WelderNo' in weights:
                weights['WelderNo'] = min(weights['WelderNo'], 1.0)
            
            # 转换为张量并按照类别顺序排序
            categories = self.data_loader.vocab[1]
            weight_tensor = torch.tensor([weights.get(cat, 1.0) for cat in categories], 
                                         device=self.data_loader.device)
            return weight_tensor
        
        # 选择模型：使用更轻量级的模型架构，但增加对关键特征的提取能力
        if use_small_model:
            # 使用更轻量但更强大的模型架构
            logger.info("使用优化的轻量级模型进行训练...")
            # 自定义优化的小型模型
            class EnhancedSmallRNN(nn.Module):
                def __init__(self, vocab_sz, emb_sz, n_hid, n_out, n_layers=2, drop_p=0.3):
                    super().__init__()
                    # 嵌入层
                    self.emb = nn.Embedding(vocab_sz, emb_sz)
                    # 使用双向GRU代替RNN，提高特征提取能力
                    self.rnn = nn.GRU(emb_sz, n_hid, n_layers, 
                                     batch_first=True, 
                                     dropout=drop_p if n_layers > 1 else 0, 
                                     bidirectional=True)
                    # 注意力机制层，帮助模型关注重要特征
                    self.attention = nn.Sequential(
                        nn.Linear(n_hid * 2, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
                    # 输出层
                    self.out = nn.Linear(n_hid * 2, n_out)
                    self.drop = nn.Dropout(drop_p)
                
                def forward(self, x):
                    # 嵌入层
                    x = self.emb(x)
                    x = self.drop(x)
                    # RNN层
                    output, hidden = self.rnn(x)
                    # 注意力机制
                    attn_weights = torch.softmax(self.attention(output).squeeze(2), dim=1)
                    attn_output = torch.sum(attn_weights.unsqueeze(2) * output, dim=1)
                    # 输出层
                    x = self.out(attn_output)
                    return x
            
            # 创建自定义模型
            vocab_sz = len(self.data_loader.vocab[0])
            emb_sz = 128  # 稍微增加嵌入维度以捕获更多特征
            n_hid = 128   # 隐藏层大小
            n_out = len(self.data_loader.vocab[1])
            model = EnhancedSmallRNN(vocab_sz, emb_sz, n_hid, n_out, n_layers=2, drop_p=drop_mult)
            
            # 获取类别权重
            class_weights = get_class_weights()
            logger.info(f"使用类别权重: {dict(zip(self.data_loader.vocab[1], class_weights.tolist()))}")
            
            # 创建学习器，使用加权交叉熵损失函数
            self.learn = Learner(
                self.data_loader,
                model,
                loss_func=CrossEntropyLossFlat(weight=class_weights),
                metrics=[accuracy, F1Score(average='macro')],
                cbs=[SaveModelCallback(monitor='accuracy', fname='best_model')]
            )
        else:
            # 使用传统的AWD_LSTM模型
            self.learn = text_classifier_learner(
                self.data_loader,
                AWD_LSTM,
                drop_mult=drop_mult,
                metrics=[accuracy, F1Score(average='macro')],
                cbs=[SaveModelCallback(monitor='accuracy', fname='best_model')]
            )
            
        # 查找最佳学习率
        lr_finder = self.learn.lr_find()
        self.best_lr = lr_finder.valley
        logger.info(f"最佳学习率: {self.best_lr:.6f}")
        
        # 改进的训练策略：使用学习率调度和早停
        # 1. 先使用较低学习率训练所有层
        logger.info("开始训练模型...")
        self.learn.fit_one_cycle(epochs, self.best_lr, cbs=[
            EarlyStoppingCallback(monitor='accuracy', patience=3),  # 早停防止过拟合
            ReduceLROnPlateau(monitor='valid_loss', min_delta=0.01, patience=2)  # 自动降低学习率
        ])
        
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
    # print("加载数据...")
    # classifier.load_data('./data/class_data_v3.csv', './data/valid_data_v3.csv')
    
    # 训练模型 - 使用优化后的策略和轻量级模型
    # print("训练模型...")
    # classifier.train_model(epochs=20, drop_mult=0.5, use_small_model=True)
    
    # 保存模型
    # print("保存模型...")
    # classifier.save_model()
    
    # 评估模型
    # print("\n评估模型性能...")
    # classifier.evaluate_model()
    
    
    # 测试预测
    print("\n测试预测结果：")
    test_cases = ["FE 82JD", "H942143Y82", "2023-02-03", "87 7 X30", "D", "202302", "H841"]
    for test_text in test_cases:
        results = classifier.predict(test_text)
        print(f"\n输入: {test_text}")
        print("预测类别（按置信度排序）:")
        for category, prob in results:
            print(f"  - {category}: {prob:.4f}")