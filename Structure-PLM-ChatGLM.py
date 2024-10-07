import json
import os
import sys
import time
from accelerate import Accelerator
from transformers import AutoModelForMaskedLM,AutoTokenizer,AutoModelForCausalLM,LlamaForCausalLM,LlamaConfig,BitsAndBytesConfig
import torch
from peft import LoraConfig,TaskType,get_peft_model,PeftModel
from accelerate.utils import DummyScheduler, DummyOptim
import torch.nn.functional as F
import argparse
import datetime
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import DataCollatorForSeq2Seq
import torch.distributed as dist

def is_main_process():
    # 如果没有设置环境变量，假设该进程就是主进程
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    return True

epochs = 3
#2e-4，如果不收敛，就用5e-5
learning_rate = 0.00001
batch_size = 2
accumulation_steps = 16

shuXueYuan_output_path = '/root/autodl-tmp/PLM-Patent-ChatGLM/output/llama2-structure-pretrained-r8'
shuXueYuan_checkpoint_path = '/root/autodl-tmp/PLM-Patent-ChatGLM/output/llama2-structure-pretrained-r8/checkpoint.pth'
shuXueYuan_pretrained_path = '/root/autodl-tmp/llama2'
shuXueYuan_cases_path = "/root/autodl-tmp/data/new-cpc-uspto"

# model_path = shuXueYuan_model_path
case_path = shuXueYuan_cases_path

class AlpacaDataset(Dataset):
    def __init__(self,pairs,tokenizer):
        super().__init__()
        self.pairs = pairs
        self.tokenizer = tokenizer
    def __getitem__(self, item):#将每个样例都编码为向量，每个样例为最长500的字符
        sample = self.pairs[item]
        # claim_text = self.tokenizer.encode(sample["claim"],add_special_tokens=True,max_length = 500,padding = True)#要有special_token
        # description_text = self.tokenizer.encode(sample["description"],add_special_tokens=True,max_length = 500,padding = True)
        # # #这里有字典嵌套，claim_text应该是一个包括inputid和attentionmask的字典
        # text_dict = {}
        # text_dict["claim"] = claim_text
        # text_dict["description"] = description_text
        return sample

    def __len__(self):
        return len(self.pairs)

#该函数负责把文本块token变成inputid,labels，并且进行部分掩码

def coll(batch):
    # print(len(batch))
    # print(batch)
    claims_batch_data = []
    description_batch_data = []
    cpc_text_batch_data = []
    claims_batch_labels = []
    description_batch_labels = []
    cpc_text_batch_labels = []

    for i in batch:
        claim_text = tokenizer(i["claim"], add_special_tokens=True, max_length=1000,truncation = True,)
        if len(claim_text["input_ids"]) < 1000:
            label = torch.cat((torch.tensor(claim_text["input_ids"]),torch.full((1000-len(claim_text["input_ids"]),), -100)),dim=0)
        else:
            label = torch.tensor(claim_text["input_ids"])
        label = label.tolist()
        claims_batch_labels.append(label)

        description_text = tokenizer(i["description"], add_special_tokens=True, max_length=500, truncation=True)
        if len(description_text["input_ids"]) < 500:
            label = torch.cat(
                (torch.tensor(description_text["input_ids"]), torch.full((500 - len(description_text["input_ids"]),), -100)),
                dim=0)
        else:
            label = torch.tensor(description_text["input_ids"])
        label = label.tolist()
        description_batch_labels.append(label)

        cpc_text = tokenizer(i["cpc_text"], add_special_tokens=True, max_length=500, truncation=True)
        if len(cpc_text["input_ids"]) < 500:
            label = torch.cat(
                (torch.tensor(cpc_text["input_ids"]), torch.full((500 - len(cpc_text["input_ids"]),), -100)),
                dim=0)
        else:
            label = torch.tensor(cpc_text["input_ids"])
        label = label.tolist()
        cpc_text_batch_labels.append(label)


        claim_text = tokenizer(i["claim"], add_special_tokens=True, max_length=1000,truncation = True,padding='max_length')
        # claim_text = tokenizer.encode(i["claim"], add_special_tokens=True, max_length=1000,truncation = True,
        #                               padding='max_length',)  # 要有special_token
        description_text = tokenizer(i["description"], add_special_tokens=True, max_length=500,truncation = True, padding='max_length')
        cpc_text = tokenizer(i["cpc_text"], add_special_tokens=True, max_length=500,truncation = True,padding='max_length')

        claims_batch_data.append(claim_text)
        description_batch_data.append(description_text)
        cpc_text_batch_data.append(cpc_text)

    # claim = claims_datacollecter(claims_batch_data)
    # print(claims_batch_data)
    claim = tokenizer.pad(claims_batch_data, return_tensors='pt', padding=True)
    claim['labels'] = torch.tensor(claims_batch_labels)#因为input_ids有一部分是pad的，对于这部分pad的token，应该使用-100
    description = tokenizer.pad(description_batch_data,return_tensors='pt', padding=True)
    description['labels'] = torch.tensor(description_batch_labels)
    cpc_text = tokenizer.pad(cpc_text_batch_data,return_tensors='pt', padding=True)
    cpc_text['labels'] = torch.tensor(cpc_text_batch_labels)

    # torch.set_printoptions(threshold=10000)
    # if is_main_process():
    #     print("claims",claim)
    #     print("description",description)
    #     print("cpc_text",cpc_text)

    # return claim,description,cpc_text
    return claim

layers = 1
configuration = LlamaConfig(num_hidden_layers = layers,output_hidden_states = True)#其他的不用配置，自动产生和bert-uncased类似的配置
#可能得改一下类名
encoder = LlamaForCausalLM.from_pretrained(shuXueYuan_pretrained_path, output_hidden_states=True)
#定义模型，因为解码器和编码器的损失是共同计算的，所以模型要同时包含解码器和编码器两部分，把他们放在不同的层
decoder_description = LlamaForCausalLM(configuration)
decoder_cpc_text = LlamaForCausalLM(configuration)

tokenizer = AutoTokenizer.from_pretrained(shuXueYuan_pretrained_path,use_fast = True)
tokenizer.pad_token = tokenizer.eos_token

# claims_datacollecter = DataCollatorForLanguageModeling(tokenizer,mlm = True,mlm_probability = 0.2)
# description_datacollecter = CustomDataCollatorForSeq2Seq(tokenizer = tokenizer,mask_ratio = 0.6)#这个mask是指60%的token不计入损失

class Model(torch.nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        self.encoder = encoder
        # self.decoder_description = decoder_description
        # self.decoder_cpc_text = decoder_cpc_text
    def forward(self,claims_input,claims_labels,claim_attention_mask,description_input=None,description_labels=None,description_attention_mask=None,
                cpc_text_input=None,cpc_text_labels=None,cpc_text_attention_mask=None):

        #claims应该包括 input_ids,attention_mask,token_type_ids,position_ids,labels
        #其实只有input_ids和labels就行，其他的会默认初始，位置嵌入是随机初始化的一个嵌入矩阵，注意力掩码是默认计算所有位置的注意力
        # domain_input = torch.cat([claims_input,description_input],dim=1)
        # domain_input = torch.cat([domain_input,cpc_text_input],dim=1)
        # domain_attention_mask = torch.cat([claim_attention_mask,description_attention_mask],dim=1)
        # domain_attention_mask = torch.cat([domain_attention_mask,cpc_text_attention_mask],dim=1)
        # domain_labels = torch.cat([claims_labels,description_labels],dim=1)
        # domain_labels = torch.cat([domain_labels,cpc_text_labels],dim=1)
        # out = self.encoder(input_ids = domain_input,labels = domain_labels,attention_mask = domain_attention_mask)
        # loss = out.loss

        out1 = self.encoder(input_ids = claims_input,labels = claims_labels,attention_mask = claim_attention_mask)
        # cls = torch.mean(out1.hidden_states[-1],dim=1)#这个应该是可以取到最后一层吧，这里要取一下最后一个隐藏层的平均词向量
        # cls = out1.hidden_states[-1][:,-1,:]
        # cls = torch.unsqueeze(cls,dim=1)
        # #将cls加入description的input_id，开启attention_mask，labels默认为0，位置嵌入该怎么搞？
        # # print(description_input.shape) #[32,500]
        # # print(description_labels.shape)#[32,500]
        # tmp_out1 = self.decoder_description(input_ids = description_input).hidden_states[-1]#这里是直接拿词表的嵌入好还是拿经过自注意力层后的嵌入好呢
        # new_decoder_input1 = torch.cat([cls,tmp_out1],dim=1)#[32 500 768] [32 1 768]#这里做了编码器的文本向量和解码器的文本向量的拼接
        # device = next(model.parameters()).device
        # new_labels1 = torch.cat([torch.full((batch_size, 1), -100).to(device),description_labels],dim=1)#标签也要拼接
        # new_attention_mask1 = torch.cat([torch.full((batch_size, 1), 1).to(device),description_attention_mask],dim=1)
        # out2 = self.decoder_description(inputs_embeds=new_decoder_input1, labels=new_labels1, attention_mask=new_attention_mask1)
        #
        # tmp_out2 = self.decoder_cpc_text(input_ids=cpc_text_input).hidden_states[-1]
        # new_decoder_input2 = torch.cat([cls,tmp_out2],dim=1)#[32 500 768] [32 1 768]#这里做了编码器的文本向量和解码器的文本向量的拼接
        # device = next(model.parameters()).device
        # new_labels2 = torch.cat([torch.full((batch_size, 1), -100).to(device),cpc_text_labels],dim=1)#标签也要拼接
        # new_attention_mask2 = torch.cat([torch.full((batch_size, 1), 1).to(device),cpc_text_attention_mask],dim=1)
        # out3 = self.decoder_cpc_text(inputs_embeds=new_decoder_input2, labels=new_labels2, attention_mask=new_attention_mask2)
        # if is_main_process():
        #     print("encoder loss",out1.loss)
        #     print("decoder description loss",out2.loss)
        #     print("decoder cpc_text loss",out3.loss)
        # loss = out1.loss+out2.loss+out3.loss
        loss = out1.loss
        return loss

    def embedding(self,batch_input,attention_mask=None,token_type_ids=None):
        embed = self.encoder(input_ids = batch_input,attention_mask = attention_mask,token_type_ids = token_type_ids).hidden_states[-1]#[batch_size,seq_length,embedding_size]
        return embed


def process_dataset():
    data_list = []
    for file_name in os.listdir(case_path):
        file_path = case_path + '/' + file_name
        print(file_path)
        file = open(file_path,'r',encoding='utf-8')
        data = file.read()
        json_data = json.loads(data,strict = False)
        # print(json_data)
        #如果没有claims或者description，直接跳过
        if json_data["claims"] is None:
            continue
        if json_data["description"] is None:
            continue
        if json_data["CPC"] is None:
            continue
        if json_data["CPC_TEXT"] is None:
            continue
        part2 = json_data["description"].split("SUMMARY")[0]
        if len(part2) == 0:
            part2 = json_data["description"].split("summary")[0]
        if len(part2) == 0:
            continue
        claims = json_data["claims"]
        description = part2

        a = json_data["CPC"].split(" | ")[0:-1]#因为cpc多一个" | "，所以切分后最后一个是空的，不取
        b = json_data["CPC_TEXT"].split("\n")[0:-1]
        s = ""
        for i in range(0,len(a)):
            s = s +" " + a[i]+" " + b[i]
        cpc_text = s

        #把切开的部分做组合，一个组合是一个样本
        sample = {}
        sample["claim"] = claims
        sample["description"] = description
        sample["cpc_text"] = cpc_text
        data_list.append(sample)
    # for i in data_list:
    #     print(len(i))
    random.seed(32)
    random.shuffle(data_list)
    print(len(data_list))#489272
    return data_list

pairs = process_dataset()
train_pairs = pairs[0:int(len(pairs)*0.8)]
dev_pairs = pairs[int(len(pairs)*0.8):len(pairs)]

train_dataset = AlpacaDataset(train_pairs,tokenizer)
val_dataset = AlpacaDataset(dev_pairs,tokenizer)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,collate_fn=coll,drop_last=True,num_workers=4)
val_dataloader = DataLoader(dataset=val_dataset,batch_size=batch_size,collate_fn=coll,drop_last=True,num_workers=4)
model = Model()

# model = PeftModel.from_pretrained(model, model_path, is_trainable=True)

peft_config = LoraConfig(inference_mode = False,r=8,lora_alpha=16,lora_dropout=0.05,target_modules="encoder\.model\.layers\..*self_attn\.[qkvo]_proj|encoder\.model\.layers\..*\.gate_proj|encoder\.model\.layers\..*\.down_proj|encoder\.model\.layers\..*\.up_proj")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# model = model.to(device)
# print(model.state_dict())

# 总的训练步数是 (epochs * 数据集大小)/(显卡数*梯度累积步数)
total_steps = int((epochs * len(train_dataloader))/(2*accumulation_steps))


optimizer = DummyOptim(model.parameters(), lr=learning_rate)
scheduler = DummyScheduler(optimizer, total_num_steps=total_steps, warmup_num_steps=1000)

accelerator = Accelerator(gradient_accumulation_steps=accumulation_steps)
model, scheduler,optimizer, train_dataloader,val_dataloader = accelerator.prepare(model, scheduler,optimizer, train_dataloader,val_dataloader)

@torch.no_grad()
def validate_model_loss(model, val_loader):
    # 开启模型评估模式
    model.eval()
    # 初始化损失
    total_loss = 0.0
    avg_loss = None
    # 不需要计算梯度
    n_samples = 0
    with torch.no_grad():
        # for claim, description,cpc_text in val_loader:
        for claim in val_loader:
            claim_input = claim['input_ids']
            claim_labels = claim['labels']
            claim_attention_mask = claim['attention_mask']
            loss = model(claims_input = claim_input, claims_labels = claim_labels,claim_attention_mask = claim_attention_mask)
            # description_input = description['input_ids']
            # description_labels = description['labels']
            # description_attention_mask = description['attention_mask']
            #
            # cpc_text_input = cpc_text['input_ids']
            # cpc_text_attention_mask = cpc_text['attention_mask']
            # cpc_text_labels = cpc_text['labels']
            #
            # loss = model(claims_input = claim_input, claims_labels = claim_labels,claim_attention_mask = claim_attention_mask,
            #              description_input = description_input, description_labels = description_labels,description_attention_mask = description_attention_mask,
            #              cpc_text_input = cpc_text_input,cpc_text_attention_mask = cpc_text_attention_mask,cpc_text_labels = cpc_text_labels)

            # 使用accelerator.gather()来汇总所有进程上的损失
            gathered_losses = accelerator.gather(loss)

            # 只在主进程上进行计算
            if accelerator.is_main_process:
                total_loss += gathered_losses.sum()
                n_samples += len(gathered_losses)
                print("n_samples",n_samples)
            # # 累加损失
            # total_loss += loss.item()

    # 计算平均损失
    if accelerator.is_main_process:
        avg_loss = total_loss / n_samples
        print(f'Validation Loss: {avg_loss:.4f}')
    model.train()
    # 返回验证集上的平均损失
    return avg_loss if avg_loss is not None else 0

# 设置早停参数
patience = 1  # 一轮在验证集上不提高就直接停止训练
patience_counter = 0  # 初始化耐心计数器
best_loss = float('inf')  # 初始化最佳损失为无穷大
best_model = None  # 初始化最佳模型为空

val_loss_list = []
# 训练循环
for epoch in range(epochs):
    model.train()
    loss_list = []
    for i,batch in enumerate(tqdm(train_dataloader,desc="训练进度")):
        # claim,description,cpc_text = batch
        claim = batch
        claim_input = claim['input_ids']
        claim_labels = claim['labels']
        claim_attention_mask = claim['attention_mask']

        loss = model(claims_input = claim_input,claims_labels = claim_labels,claim_attention_mask = claim_attention_mask)
        # description_input = description['input_ids']
        # description_labels = description['labels']
        # description_attention_mask = description['attention_mask']
        #
        # cpc_text_input = cpc_text['input_ids']
        # cpc_text_labels = cpc_text['labels']
        # cpc_text_attention_mask = cpc_text['attention_mask']
        # loss = model(claims_input = claim_input, claims_labels = claim_labels,claim_attention_mask = claim_attention_mask,
        #              description_input = description_input, description_labels = description_labels,description_attention_mask = description_attention_mask,
        #              cpc_text_input = cpc_text_input,cpc_text_labels = cpc_text_labels,cpc_text_attention_mask = cpc_text_attention_mask)

        loss = loss / accumulation_steps  # 损失缩放
        accelerator.backward(loss)

        loss_value = accelerator.gather(loss).mean()

        if accelerator.is_main_process:
            loss_list.append(loss_value)


        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):  # 检查是否达到了累积步骤
            optimizer.step()  # 更新模型参数
            scheduler.step()
            optimizer.zero_grad()  # 重置参数梯度
            current_lr = scheduler.get_lr()
            print("step",i/accumulation_steps," lr=",current_lr)

            if accelerator.is_main_process:
                print("batch avg loss",sum(loss_list))#这里因为损失在梯度累积上缩放过，所以不需要再除以accumulation_steps了
                loss_list = []

    val_loss = validate_model_loss(model, val_dataloader)
    val_loss_list.append(val_loss)

    print("epoch is ", epoch)
    # 在所有进程上初始化早停标记变量
    early_stopping_triggered = False

    if accelerator.is_main_process:
        # 检查是否找到了新的最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            model.save_pretrained(shuXueYuan_output_path)
            torch.save({
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, shuXueYuan_checkpoint_path)
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
if is_main_process():
    print("保存路径",shuXueYuan_output_path)
    print("层数为",layers,"训练完成。")
    print("val_loss_list",val_loss_list)