import argparse
#if __name__ == '__main__':

import os
import random
from datetime import datetime
import time

import numpy as np
import torch
from utils.loader import DataLoader
from utils.GraphMaker import GraphMaker
from utils import torch_utils, helper
from model.trainer import CrossTrainer

parser=argparse.ArgumentParser()
# dataset part
parser.add_argument('--dataset', type=str,default='group1',help='')
parser.add_argument('--clients',type=str, default='clothes,electronic,phone,sports',help='all clients')
parser.add_argument("--target",type=str,default='clothes',help="the target client")
# model part
parser.add_argument('--model',type=str,default="fr",help="The model name.")
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--feature_dim', type=int, default=128, help='Initialize network embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=128, help='GNN network hidden embedding dimension.')
parser.add_argument('--GNN', type=int, default=2, help='GNN layer.')

parser.add_argument('--dropout', type=float, default=0.3, help='GNN layer dropout rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--decay_epoch', type=int, default=10, help='Decay learning rate after this epoch.')
parser.add_argument('--leakey', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.9)
# train part
parser.add_argument('--seed',type=int,default=2040)
parser.add_argument('--id',type=str,default='00',help='Model id under which to save models.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size.')
parser.add_argument('--num_epoch', type=int, default=50, help='Number of total training epochs.')
parser.add_argument('--load', dest='load', action='store_true', default=False,  help='Load pretrained model.')
parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

def seed_everything(seed=1005):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONASHSEED']=str(seed)   # 设置python环境变量
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = parser.parse_args()
if args.cpu:
    args.cuda=False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time=time.time()
# make opt字典对象
opt = vars(args)
seed_everything(opt["seed"])

if "fr" in opt["model"]:
    # 确定一组clients
    group=opt["dataset"]
    clients_list=opt["clients"].split(",")
    #target
    target=opt["target"]
    target_client="../data/"+group+"/train/"+target+".txt"
    target_G=GraphMaker(opt,target_client)
    target_UV=target_G.UV
    target_VU=target_G.VU
    target_adj=target_G.adj
    #source
    source_GS={}
    source_UVS={}
    source_VUS={}
    source_adjs={}
    source_num=0
    for i in clients_list:
        if i != target:
            source_num+=1
            source_client="../data/"+group+"/train/"+i+".txt"
            source_G=GraphMaker(opt,source_client)
            source_GS[source_num]=source_G
            source_UV=source_G.UV
            source_UVS[source_num]=source_UV
            source_VU = source_G.VU
            source_VUS[source_num]=source_VU
            source_adj=source_G.adj
            source_adjs[source_num]=source_adj
print("graphs loaded")

model_id=opt['id']
model_save_dir=opt['save_dir']+'/'+ model_id
helper.ensure_dir(model_save_dir,verbose=True)
#save config
helper.save_config(opt,model_save_dir+'/config.json',verbose=True)
file_logger=helper.FileLogger(model_save_dir+'/'+opt['log'],
                              header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")
# print model info
helper.print_config(opt)

print("Loading data from {} with batch size {}...".format(opt['dataset'], opt['batch_size']))
train_batch = DataLoader(opt['dataset'], opt['target'],opt['clients'],opt['batch_size'], opt, evaluation = -1)
source_dev_batch = DataLoader(opt['dataset'], opt['target'],opt['clients'],opt["batch_size"], opt, evaluation = 1)
target_dev_batch = DataLoader(opt['dataset'], opt['target'],opt['clients'],opt["batch_size"], opt, evaluation = 2)

print("user_num", opt["target_user_num"])
num=0
for i in clients_list:
    if i !=target:
        num+=1
        print("source"+str(num)+"_item_num", opt["source"+str(num)+"_item_num"])
print("target_item_num", opt["target_item_num"])
#print("source train data : {}, target train data {}, source test data : {}, source test data : {}".format(len(train_batch.source_train_data),len(train_batch.target_train_data),len(train_batch.source_test_data),len(train_batch.target_test_data)))

if opt["cuda"]:
    num=0
    for i in clients_list:
        if i !=target:
            num+=1
            source_UV=source_UVS[num].cuda()
            source_UVS[num]=source_UV
            source_VU=source_VUS[num].cuda()
            source_VUS[num]=source_VU
            source_adj=source_adjs[num].cuda()
            source_adjs[num]=source_adj
    target_UV = target_UV.cuda()
    target_VU = target_VU.cuda()
    target_adj = target_adj.cuda()

# model
if not opt['load']:
    trainer = CrossTrainer(opt)
else:
    # load pretrained model
    model_file = opt['model_file']
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = CrossTrainer(opt)
    trainer.load(model_file)

dev_score_history = [0]
current_lr = opt['lr']
global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps=DataLoader.all_len(train_batch)
for key in max_steps:
    max_steps[key]*=opt['num_epoch']


#start training
for epoch in range(1,opt['num_epoch']+1):
    train_loss=0
    start_time=time.time()
    current_source = 1
    steps_in_epoch = 0
    count=len(train_batch.data[current_source])
    for i, batch in enumerate(train_batch):
        global_step+=1
        steps_in_epoch+=1
        if steps_in_epoch>count:
            current_source+=1
            count+=len(train_batch.data[current_source])
        loss=trainer.reconstruct_graph(current_source,batch,source_UVS[current_source],source_VUS[current_source],target_UV,target_VU,source_adjs[current_source],target_adj,epoch)
        train_loss+=loss
    duration = time.time() - start_time
    train_loss = train_loss / count
    print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                            opt['num_epoch'], train_loss, duration, current_lr))
    if epoch % 10:
        # pass
        continue

    # eval model
    print("Evaluating on dev set...")
    trainer.model.eval()
    trainer.evaluate_embedding(current_source,source_UVS[current_source], source_VUS[current_source], target_UV, target_VU, source_adjs[current_source], target_adj)

    '''NDCG =0.0
    HT =0.0
    valid_entity = 0.0
    for i,batch in enumerate(source_dev_batch):
        predictions=trainer.source_predict(batch)
        for pred in predictions:
            rank=(-pred).argsort().argsort()[0].item()
            valid_entity+=1
            if rank<10:
                NDCG+=1 / np.log2(rank+2)
                HT+=1
            if valid_entity % 100==0:
                print('.',end='')
    s_ndcg = NDCG / valid_entity
    s_hit = HT / valid_entity'''

    NDCG = 0.0
    HT = 0.0
    valid_entity = 0.0
    for i, batch in enumerate(target_dev_batch):
        predictions = trainer.target_predict(batch)
        for pred in predictions:
            rank = (-pred).argsort().argsort()[0].item()

            valid_entity += 1

            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
            if valid_entity % 100 == 0:
                print('.', end='')
    t_ndcg = NDCG / valid_entity
    t_hit = HT / valid_entity

    print(
        "epoch {}: train_loss = {:.6f},  target_hit = {:.4f}, target_ndcg = {:.4f}".format(
            epoch, \
            train_loss, t_hit, t_ndcg))
    dev_score = t_ndcg
    file_logger.log(
        "{}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_score, max([dev_score] + dev_score_history)))

