import torch
from model import *
from dataset import *
import torch.optim as optim
import logging
import os
from Levenshtein import distance as lvd

def train():
    # logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO,filename='STRtrainLog.log',filemode='w',format=log_format,force = True)
    # device 
    deviceName = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(deviceName)
    logging.info('train using %s'%deviceName)
    print('train using ',deviceName)
    # parameter for dataLoader
    batch_size = 8
    max_length = 40+1
    # data_loader
    dataset_STR = STRDataset(root='train_data',labelPath='train_high_crop_list3.txt',charsetPath='myDict.txt')
    train_size = int(dataset_STR.__len__() * 0.9)
    valid_size = dataset_STR.__len__() - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset_STR,[train_size,valid_size])
    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0, collate_fn = collate_fn
    )
    data_loader_valid = torch.utils.data.DataLoader(
        valid_dataset, batch_size = 1, shuffle = False, num_workers = 0, collate_fn = collate_fn
    )
    # chars dictionary
    charsDict = dataset_STR.charsDict
    # model
    model = clsScore(len(charsDict.keys()),max_length)
    model.load_state_dict(torch.load("train_models/merge_STR_model/epoch_40.pth"))
    model.to(device)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr = 0.00005)
    loss_fn = nn.CTCLoss()
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=2000)
    # set model to train mode
    model.train()
    # parameter for training and save
    step = 1
    epoches = 100
    log_step = 1000
    best_valid = 0.0
    model_save_flag = False
    epoch_datanum = math.ceil(train_dataset.__len__() / batch_size)
    model_save_root = 'train_models'
    model_save_dir = "merge2_STR_model"
    if not os.path.exists(os.path.join(os.getcwd(),model_save_root,model_save_dir)):
        os.mkdir(os.path.join(os.getcwd(),model_save_root,model_save_dir))
    model_save_epoch = 20 # save model every xxx epoch while training
    model_valid_epoch = 10 # valid model every xxx epoch while training
    # epoch for training
    for epoch in range(epoches):
        for imgs,labels in data_loader:
            # load img and label
            imgs = torch.stack(list(img.to(device) for img in imgs))
            labels = list(labels)
            labelIdx, labelLength = labels_IndexAndLength(labels = labels,charsDict=charsDict)

            # pred and calculate ctc loss
            optimizer.zero_grad()
            outputs = model(imgs) # Batch_size / max_length / class_num(chars_num)
            outputs = outputs.permute(1,0,2) #  max_length / Batch_size / class_num(chars_num) for ctc loss
            outputLength = torch.full(size=(outputs.shape[1],),fill_value=max_length,dtype=torch.long)
            loss = loss_fn(outputs,labelIdx,outputLength,labelLength)
            if loss < 0:
                logging.info(labels)
            # step to step and writing log file
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if step % log_step == 0:
                logging.info("epoch: [%d] step: %d / %d, loss is %3.5f, lr : %3.5f"%(epoch,step%epoch_datanum,epoch_datanum,loss,optimizer.param_groups[0]["lr"]))
                # print("epoch: [%d] step: %d / %d, loss is %3.5f, lr : %3.5f"%(epoch,step%epoch_datanum,epoch_datanum,loss,optimizer.param_groups[0]["lr"]))
            step += 1
        
        # model valid
        if epoch%model_valid_epoch == 0:
            logging.info("epoch %d valid!"%(epoch))
            score = valid(data_loader_valid,model,device,charsDict)
            logging.info("1_NED : " + str(score))
            if score > best_valid:
                model_save_flag = True
        # model save
        if epoch % model_save_epoch == 0 or model_save_flag:
            model_save_name = "epoch_"+str(epoch)+".pth"
            path = os.path.join(model_save_root,model_save_dir,model_save_name)
            torch.save(model.state_dict(),path)
            logging.info("epoch %d save! model save at %s, loss is %f"%(epoch,path,loss))
            model_save_flag = False
    # model save after train
    model_save_name = 'final.pth'
    path = os.path.join(model_save_root,model_save_dir,model_save_name)
    torch.save(model.state_dict(),path)
    logging.info("That's it! model save at "+str(path))

def valid(dataloader,model,device,charsDict):
    model.eval()
    preds = []
    ans = []
    for imgs,labels in dataloader:
        imgs = torch.stack(list(img.to(device) for img in imgs))
        labels = list(labels)
        outputs = model(imgs) # Batch_size / max_length / class_num(chars_num)
        predIdx = torch.argmax(outputs,dim = 2)
        key_list = list(charsDict.keys())
        sentence = ""
        last_idx = 1
        for l in range(predIdx.shape[1]): # max_length
            pIdx = predIdx[0][l].cpu().detach().numpy()
            if pIdx == 1:
                break
            elif pIdx != 0 and pIdx != last_idx :
                sentence += key_list[predIdx[0][l]]
                last_idx = pIdx
        preds.append(sentence)
        ans.append(labels[0])
    score = 0
    for i in range(len(ans)): # all data
        score += lvd(preds[i],ans[i]) / max(len(preds[i]),len(ans[i]))
    score = 1 - score / len(ans)
    return score

def labels_IndexAndLength(labels,charsDict):
    global option_max_length
    global lost_word
    labelIdx = []
    labelLength = []
    for label in labels:
        for char in label:
            try:
                labelIdx.append(charsDict[char])
            except KeyError:
                if char not in lost_word:
                    lost_word.append(char)
                    logging.info("KeyError >> %c"%char)
                labelIdx.append(charsDict['@']) # @ mean don't care
        labelIdx.append(1) # append EOS symbol
        tmp = len(label)+1 # + EOS symbol
        if tmp > option_max_length:
            option_max_length = tmp
            print('option_max_length : %d / %s'%(option_max_length,label))
            logging.info('option_max_length : %d / %s'%(option_max_length,label))
        labelLength.append(tmp) 
    return torch.LongTensor(labelIdx), torch.LongTensor(labelLength)

if __name__ == "__main__":
    option_max_length = 0
    lost_word = []
    train()
    with open('lost_word.txt','w',encoding = 'utf-8') as f:
        for char in lost_word:
            f.write(char+'\n')


