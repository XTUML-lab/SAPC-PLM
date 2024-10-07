from transformers import AutoModelForMaskedLM,AutoTokenizer,AutoModelForSequenceClassification,LlamaModel,LlamaForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import BertModel, BertTokenizer, BertConfig
import math
import torch.distributed as dist
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score,precision_recall_fscore_support
from transformers import AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig,TaskType,get_peft_model,PeftModel
from tqdm import tqdm
import numpy as np
from accelerate.utils import DummyScheduler, DummyOptim
from accelerate import Accelerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import random

from peft import PeftConfig

def is_main_process():
    # 如果没有设置环境变量，假设该进程就是主进程
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    return True

# 在初始化Accelerator之前检查是否是主进程

pretrained_llama2_path = "/home/user/yzb/PLM-Patent-ChatGLM/model/llama2-model"
#A800和A100路径是一样的
A100_labels_path = "/home/user/yzb/data/fenlei-12w/label_matrix.txt"
A100_cases_path = "/home/user/yzb/data/fenlei-12w/patent_claims_text.txt"

# pretrained_lora_path = "/root/autodl-tmp/PLM-Patent-ChatGLM/output/llama2-structure-pretrained-r8"
# A100_model_path = pretrained_llama2_path
out_model_path = "/home/user/yzb/Patent_Claims_Retrieval/outputs/llama2-finetune"#存储路径
checkpoint_path = "/home/user/yzb/Patent_Claims_Retrieval/outputs/llama2-finetune/checkpoint.pth"

tokenizer = AutoTokenizer.from_pretrained(pretrained_llama2_path,use_fast = True)
tokenizer.pad_token = tokenizer.eos_token
embedder = LlamaForCausalLM.from_pretrained(pretrained_llama2_path, output_hidden_states=True, return_dict=True)

#embedder需要先融合权重
# peft_config = PeftConfig.from_pretrained(pretrained_lora_path)
# embedder = PeftModel.from_pretrained(embedder, pretrained_lora_path, is_trainable=True,config=peft_config)
#
# embedder = embedder.merge_and_unload()

epochs = 5
learning_rate = 0.00001
batch_size = 2
test_batch_size = 2
val_batch_size = 2
accumulation_steps = 64

#要改的地方包括 1、Bert_finetune的预训练模型加载位置 2、模型声明时是否要load参数 3、最后的模型保存位置  4、process和定义损失权重的三个数据文件加载的位置

class Bert_finetune(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = embedder
        self.liner = torch.nn.Linear(4096, 656)
    def forward(self, input_ids,attention_mask):
        features = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
        # print(features)
        features = features.hidden_states[-1]
        # cls_feature = torch.mean(features, dim=1)
        cls_feature = features[:,-1,:]#取最后一个单词
        features = self.liner(cls_feature)
        return features


class TrainDataset(Dataset):
    def __init__(self,pairs,tokenizer):
        super().__init__()
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.embedder = embedder

    @torch.no_grad()
    def __getitem__(self, item):#将每个样例都编码为向量，每个样例为最长500的字符


        # text = self.tokenizer.encode_plus(self.pairs[item]['text_a'],self.pairs[item]['text_b'],add_special_tokens=True,
        #                                   max_length = 500,padding = 'max_length',truncation = True,return_tensors='pt')#要有special_token

        text = self.tokenizer.encode_plus(self.pairs[item]['claim_text'],
                                          add_special_tokens=True,
                                          max_length=1000, padding='max_length', truncation=True,
                                          return_tensors='pt')  # 要有special_token

        # features = self.embedder(input_ids = text['input_ids'],attention_mask = text['attention_mask']).hidden_states[-1]#batchsize blocksize embsize
        # features = features[:,-1,:]
        # return features,self.pairs[item]['label']
        return text['input_ids'],text['attention_mask'],self.pairs[item]['label']

    def __len__(self):
        return len(self.pairs)



def process_dataset_fenlei():
    truelabels = np.loadtxt(A100_labels_path, delimiter=',')#668类
    ciaims_file = open(A100_cases_path, "r", encoding='UTF-8', errors="ignore")#591797个样本，只有独权
    claims_text = ciaims_file.readlines()
    data_list = []
    print(len(claims_text))

    for i in range(0,len(claims_text)):
        sample = {}
        sample["claim_text"] = claims_text[i]
        sample["label"] = truelabels[i]
        data_list.append(sample)
    random.seed(1024)
    random.shuffle(data_list)
    train = data_list[0:int(len(claims_text) * 0.8)]
    eval = data_list[int(len(claims_text) * 0.8):int(len(claims_text) * 0.9)]
    test = data_list[int(len(claims_text) * 0.9):int(len(claims_text))]
    #下面的是为了统计训练集和验证集中的标签分布，从而给损失函数设定类别权重
    combin = train+eval
    label = []
    for j in combin:
       label.append(j["label"])
    label_array = np.array(label)

    return train,eval,test,label_array

train_data,val_data,test_data,label_array = process_dataset_fenlei()
print(len(train_data))
print(len(val_data))
print(len(test_data))
train_dataset = TrainDataset(train_data,tokenizer)
val_dataset = TrainDataset(val_data,tokenizer)
test_dataset = TrainDataset(test_data,tokenizer)

train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size)
val_dataloader = DataLoader(dataset=val_dataset,batch_size=val_batch_size)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=test_batch_size)


# structure_state_dict_pth = torch.load(pretrained_structure_path)
#改下名字，encoder和embedder都是我自己定义的名字，不改名字，层匹配不上
# structure_state_dict = {k.replace('encoder.bert','embedder'): v for k,v in structure_state_dict_pth.items()}

model = Bert_finetune()
peft_config = LoraConfig(inference_mode = False,r=8,lora_alpha=16,lora_dropout=0.05,target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"])
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# num_training_steps = int(len(train_data)*0.9/batch_size*epochs)#数据集大小除以batchsize，乘以epochs
# num_warmup_steps = int(num_training_steps*0.1)
total_steps = int((epochs * len(train_dataloader))/(2*accumulation_steps))#2是显卡数量

optimizer = DummyOptim(model.parameters(), lr=learning_rate)
scheduler = DummyScheduler(optimizer, total_num_steps=total_steps, warmup_num_steps=1000)

accelerator = Accelerator(gradient_accumulation_steps=accumulation_steps)
model, scheduler,optimizer, train_dataloader,val_dataloader,test_dataloader = accelerator.prepare(model, scheduler,optimizer, train_dataloader,val_dataloader,test_dataloader)


weight = np.sum(label_array, axis=0)

#这里的类别权重需要设定上限，不然会造成NAN
if accelerator.is_main_process:
    print("weight1",weight)
for idx, i in enumerate(weight):  # 对于每一个二分类
    weight[idx] = (len(label_array) - float(i)) / i#负样本数量/正样本数量
# 寻找数组中不是inf的最大值
max_val = np.max(weight[np.isfinite(weight)])
# 将inf替换为找到的最大值
weight[np.isinf(weight)] = max_val
weight[weight > 1000] = 1000
weight = torch.from_numpy(weight)
weight = weight.to(accelerator.device)
if accelerator.is_main_process:
    print("weight2",weight)
loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean',pos_weight=weight)

def precision_recall_f1_at_k(y_true, y_pred, k):
    top_k_precision = []
    top_k_recall = []
    top_k_f1 = []

    for i, label in enumerate(y_true):
        # 取出前k个最高分数的预测标签
        top_k_preds = y_pred[i].argsort()[-k:].tolist()
        # print(top_k_preds)
        # print(label)
        true_labels = torch.where(label == 1)[0].tolist()
        if len(true_labels) == 0:#说明这个样本没有标签，所以跳过这个样本
            continue
        # print(type(true_labels))
        # print(type(top_k_preds))
        # print(true_labels)
        # print(top_k_preds)
        # 真实标签与预测标签的交集
        true_positives = len(set(top_k_preds) & set(true_labels))

        # 计算top-k精确率
        precision_k = true_positives / k

        # 计算top-k召回率

        recall_k = true_positives / len(true_labels)

        f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0

        top_k_precision.append(precision_k)
        top_k_recall.append(recall_k)
        top_k_f1.append(f1_k)

    # 计算所有样本的平均top-k精确率、召回率和F1分数
    avg_precision_k = np.mean(top_k_precision)
    avg_recall_k = np.mean(top_k_recall)
    avg_f1_k = np.mean(top_k_f1)

    return avg_precision_k, avg_recall_k, avg_f1_k

@torch.no_grad()
def validate_model_loss(model, val_loader):
    # 开启模型评估模式
    model.eval()
    # 初始化损失
    total_loss = 0.0
    # 不需要计算梯度
    n_samples = 0
    avg_loss = None
    with torch.no_grad():
        for batch in tqdm(val_loader,desc="验证进度"):
            input_ids,attention_mask, label = batch
            input_ids = input_ids.squeeze()
            attention_mask = attention_mask.squeeze()

            labels = torch.tensor(label).to(torch.float16)
            logits = model(input_ids,attention_mask)  # batch_size,num_classes
            loss = loss_function(logits, labels)

            gathered_losses = accelerator.gather(loss)

            # 只在主进程上进行计算
            if accelerator.is_main_process:
                total_loss += gathered_losses.sum()
                n_samples += len(gathered_losses)
                print("n_samples",n_samples)

    # 计算平均损失
    if accelerator.is_main_process:
        avg_loss = total_loss / n_samples
        print(f'Validation Loss: {avg_loss:.4f}')
    model.train()
    # 返回验证集上的平均损失
    return avg_loss if avg_loss is not None else 0

@torch.no_grad()
def model_test(k):
    precision_list = []
    recall_list = []
    f1socore_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader,desc="测试进度"):
            input_ids,attention_mask, label = batch
            input_ids = input_ids.squeeze()
            attention_mask = attention_mask.squeeze()
            # print(input_ids.shape)

            labels = torch.tensor(label).to(torch.float16)
            # print(label)
            # print(one_hot_label)
            logits = model(input_ids,attention_mask)  # batch_size,num_classes
            predicted_probs = F.sigmoid(logits)
            # print("pred", predicted_probs)
            # print(F.sigmoid(logits))
            # print(predictions.shape)
            # print(predictions)
            # print(labels)
            # accuracy = accuracy_score(labels, predictions.cpu())
            # precision = precision_score(labels, predictions.cpu())
            # recall = recall_score(labels, predictions.cpu())
            # print("pred",predictions)
            # print("label",labels)
            # precision_micro, recall_micro, fbeta_score_micro, _ = precision_recall_fscore_support(
            #     labels, predictions.cpu(), average='micro')
            #
            # precision_macro, recall_macro, fbeta_score_macro, _ = precision_recall_fscore_support(
            #     labels, predictions.cpu(), average='macro')
            #
            # precision_micro_list.append(precision_micro)
            # recall_micro_list.append(recall_micro)
            # fbeta_score_micro_list.append(fbeta_score_micro)
            #
            # precision_macro_list.append(precision_macro)
            # recall_macro_list.append(recall_macro)
            # fbeta_score_macro_list.append(fbeta_score_macro)
            #
            # print("micro",precision_micro,recall_micro,fbeta_score_micro)
            # print("macro", precision_macro, recall_macro, fbeta_score_macro)
            avg_precision_k, avg_recall_k, avg_f1_k = precision_recall_f1_at_k(labels, predicted_probs, k)
            precision_list.append(avg_precision_k)
            recall_list.append(avg_recall_k)
            f1socore_list.append(avg_f1_k)
        Top_k_precision = sum(precision_list)/len(precision_list)
        Top_k_recall = sum(recall_list)/len(recall_list)
        Top_k_f1score = sum(f1socore_list)/len(f1socore_list)

        gathered_Top_k_precision = accelerator.gather(torch.tensor(Top_k_precision).to(accelerator.device)).mean()
        gathered_Top_k_recall = accelerator.gather(torch.tensor(Top_k_recall).to(accelerator.device)).mean()
        gathered_Top_k_f1score = accelerator.gather(torch.tensor(Top_k_f1score).to(accelerator.device)).mean()

        if accelerator.is_main_process:
            print(f"Top-{k} precision: ",gathered_Top_k_precision)
            print(f"Top-{k} recall: ",gathered_Top_k_recall)
            print(f"Top-{k} F1 Score: ",gathered_Top_k_f1score)

    model.train()



patience = 2  # 设置耐心参数为3，即2个epoches模型在验证集上都没有提高则停止训练
patience_counter = 0  # 初始化耐心计数器
best_loss = float('inf')  # 初始化最佳损失为无穷大
best_model = None  # 初始化最佳模型为空

for epoch in range(epochs):
    loss_list = []
    model.train()
    for i,batch in enumerate(tqdm(train_dataloader,desc="训练进度")):
        input_ids,attention_mask,label = batch
        input_ids = input_ids.squeeze()
        attention_mask = attention_mask.squeeze()
        # print("model_input",model_input.shape)
        label = torch.tensor(label).to(torch.float16)
        # print("label",label)
        logits = model(input_ids,attention_mask)#batch_size,num_classes

        loss = loss_function(logits,label)

        loss = loss / accumulation_steps #损失缩放

        accelerator.backward(loss)

        loss_value = accelerator.gather(loss).mean()

        if accelerator.is_main_process:
            loss_list.append(loss_value)

        if (i + 1) % accumulation_steps == 0 or (i+1) == len(train_dataloader):  # 检查是否达到了累积步骤
            optimizer.step()  # 更新模型参数
            scheduler.step()
            optimizer.zero_grad()  # 重置参数梯度
            if accelerator.is_main_process:
                print("batch avg loss",sum(loss_list))#这里因为损失在梯度累积上缩放过，所以不需要再除以accumulation_steps了
                loss_list = []

    val_loss = validate_model_loss(model, val_dataloader)
    print("epoch is",epoch)
    # 在所有进程上初始化早停标记变量
    early_stopping_triggered = False

    if accelerator.is_main_process:
        # 检查是否找到了新的最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            model.save_pretrained(out_model_path)
            torch.save({
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            patience_counter = 0  # 重置耐心计数器
        else:
            patience_counter += 1  # 增加耐心计数器

        # 是否达到早停条件
        if patience_counter >= patience:
            print("epoch is ",epoch)
            print("Early stopping triggered.")
            early_stopping_triggered = True

    # 同步早停标记变量
    # if accelerator.state.distributed_type != DistributedType.NO:
    early_stopping_triggered_tensor = torch.tensor(early_stopping_triggered, dtype=torch.bool).to(
        accelerator.device)
    dist.broadcast(early_stopping_triggered_tensor, src=0)
    early_stopping_triggered = early_stopping_triggered_tensor.item()
    # early_stopping_triggered = accelerator.broadcast(early_stopping_triggered, src=0)

    # 所有进程根据早停标记做出决策
    if early_stopping_triggered:
        break

# torch.save(model.state_dict(), A100_output_domain_claims_desciption_path)
# print("保存路径",output_bert_path)
model_test(k=1)
model_test(k=5)

