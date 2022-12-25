from parameter import N_config
from data_process import MR_Dataset
from  model_buiding import TextCNN
import torch
import torch.optim as optim
import torch.nn as nn
from tool import EarlyStopping
import numpy as np
import torch.autograd as autograd
config = N_config()

def get_parameter():
    print("参数准备完毕！")
    print()
def get_data():
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
    training_set = MR_Dataset(state="train", k=config.k)
    config.n_vocab = training_set.n_vocab
    training_iter = torch.utils.data.DataLoader(dataset=training_set,
                                                batch_size=config.batch_size,
                                                shuffle=True,
                                                num_workers=2)
    if config.use_pretrained_embed:
        config.embedding_pretrained = torch.from_numpy(training_set.weight).float()
    else:
        pass
    valid_set = MR_Dataset(state="valid", k=config.k)
    valid_iter = torch.utils.data.DataLoader(dataset=valid_set,
                                            batch_size=config.batch_size,
                                            shuffle=False,
                                            num_workers=2)
    test_set = MR_Dataset(state="test", k=config.k)
    test_iter = torch.utils.data.DataLoader(dataset=test_set,
                                            batch_size=config.batch_size,
                                            shuffle=False,
                                            num_workers=2)
    print("数据准备完毕！")
    print()
    return training_iter, valid_iter, test_iter,valid_set,test_set

def get_model(pa):
    model = TextCNN(pa)
    if config.cuda and torch.cuda.is_available():
        model.cuda()
        config.embedding_pretrained.cuda()
    print('模型构建完毕！')
    print()
    return model

def get_test_result(data_iter,data_set,model,criterion):
    model.eval()
    data_loss = 0
    true_sample_num = 0
    for data, label in data_iter:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        else:
            data = torch.autograd.Variable(data).long()
        out = model(data)
        loss = criterion(out, autograd.Variable(label.long()))
        data_loss += loss.data.item()
        true_sample_num += np.sum((torch.argmax(out, 1) == label).cpu().numpy()) #(0,0.5)
    acc = true_sample_num / data_set.__len__()
    return data_loss,acc


def get_loss_and_opitm(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    print('损失函数和优化器构建完毕！')
    print()
    return criterion, optimizer
def train():
    
    acc = 0
    for i in range(0, 10):  # 10-cv
        print("现在进行第{}fold的运算".format(i+1))
        early_stopping = EarlyStopping(patience=10, verbose=True, cv_index=i)
        print("-----获取数据集--------------")
        training_iter,valid_iter,test_iter,valid_set,test_set = get_data()
        print('------构建模型---------------')
        model = get_model(config)
        if config.cuda and torch.cuda.is_available():
            model.cuda()
            config.embedding_pretrained.cuda()
        print('------构建损失函数和优化器---------')
        criterion ,optimizer = get_loss_and_opitm(model)
        count = 0
        loss_sum = 0

        print("-----现在开始训练--------")
        for epoch in range(config.epoch):
            # 开始训练
            model.train()
            for data, label in training_iter:
                if config.cuda and torch.cuda.is_available():
                    data = data.cuda()
                    label = label.cuda()
                else:
                    data = torch.autograd.Variable(data).long()
                label = torch.autograd.Variable(label).squeeze()
                out = model(data)
                l2_loss = config.l2 * torch.sum(torch.pow(list(model.parameters())[1], 2))
                loss = criterion(out, autograd.Variable(label.long())) + l2_loss
                loss_sum += loss.data.item()
                count += 1
                if count % 100 == 0:
                    print("epoch", epoch, end='  ')
                    print("The loss is: %.5f" % (loss_sum / 100))
                    loss_sum = 0
                    count = 0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # save the model in every epoch
            # 一轮训练结束，在验证集测试
            valid_loss, valid_acc = get_test_result(valid_iter, valid_set,model,criterion)
            early_stopping(valid_loss, model)
            print("The valid acc is: %.5f" % valid_acc)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        # 1 fold训练结果
        model.load_state_dict(torch.load('./checkpoints/checkpoint%d.pt' % i))
        test_loss, test_acc = get_test_result(test_iter, test_set,model,criterion)
        print("The test acc is: %.5f" % test_acc)
        acc += test_acc / 10
    # 输出10-fold的平均acc
    print("The test acc is: %.5f" % acc)

if __name__ == "__main__":
    torch.cuda.set_device(0)
    train()