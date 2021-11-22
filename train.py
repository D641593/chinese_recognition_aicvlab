import torch
from model import *
from dataset import *
import torch.optim as optim
import logging
import os

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
    max_length = 30+1
    # data_loader
    dataset_STR = STRDataset(root='train_data',labelPath='public_crop_list.txt',charsetPath='chinese_cht_dict.txt')
    data_loader = torch.utils.data.DataLoader(
        dataset_STR, batch_size = batch_size, shuffle = True, num_workers = 0, collate_fn = collate_fn
    )
    # chars dictionary
    charsDict = dataset_STR.charsDict
    # model
    model = clsScore(len(charsDict.keys()))
    model.to(device)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    loss_fn = nn.CTCLoss()
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=2000)
    # set model to train mode
    model.train()
    # parameter for training and save
    step = 1
    epoches = 1
    log_step = 100
    epoch_datanum = math.ceil(dataset_STR.__len__() / batch_size)
    model_save_root = 'train_models'
    model_save_dir = "public_STR_model"
    if not os.path.exists(os.path.join(os.getcwd(),model_save_root,model_save_dir)):
        os.mkdir(os.path.join(os.getcwd(),model_save_root,model_save_dir))
    model_save_epoch = 50 # save model every xxx epoch while training
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
            # print(sum(outputs[0,0,:]))
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
                print("epoch: [%d] step: %d / %d, loss is %3.5f, lr : %3.5f"%(epoch,step%epoch_datanum,epoch_datanum,loss,optimizer.param_groups[0]["lr"]))
            step += 1
        
        # model eval
        # if epoch%model_eval_epoch == 0:

        # model save
        if epoch % model_save_epoch == 0:
            model_save_name = "epoch_"+str(epoch)+".pth"
            path = os.path.join(model_save_root,model_save_dir,model_save_name)
            torch.save(model.state_dict(),path)
            logging.info("epoch %d save! model save at %s, loss is %f"%(epoch,path,loss))
    # model save after train
    model_save_name = 'final.pth'
    path = os.path.join(model_save_root,model_save_dir,model_save_name)
    torch.save(model.state_dict(),path)
    logging.info("That's it! model save at "+str(path))

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
            f.write(char)


