class N_config():
    def __init__(self):
        self.embedding_pretrained = None # 是否使用预训练的词向量
        self.n_vocab = 100 # 词表中单词的个数
        self.embed_size = 300 # 词向量的维度
        self.cuda = True # 是否使用gpu
        self.filter_num = 100 # 每种尺寸卷积核的个数
        self.filters = [3,4,5] # 卷积核的尺寸
        self.label_num = 2 # 标签个数
        self.dropout = 0.5 # dropout的概率
        self.batch_size = 50 #最大句子长度
        self.epoch = 200 #训练的轮数
        self.gpu   = 0   #是否使用GPU
        self.learning_rate = 0.0005 #学习率
        self.seed = 1 #随机种子
        self.l2   = 0.004 #l2正则化权重
        self.use_pretrained_embed = True #是否使用预训练
        self.k = 0 #交叉验证的k值
