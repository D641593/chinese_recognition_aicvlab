import torch
from model import *
from dataset import *
import torch.optim as optim
import logging
import os
from Levenshtein import distance as lvd

def eval():
    # logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO,filename='STRevalLog.log',filemode='w',format=log_format,force = True)
    # device 
    deviceName = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(deviceName)
    logging.info('eval using %s'%deviceName)
    print('eval using ',deviceName)
    # parameter for dataLoader
    batch_size = 1
    max_length = 30+1
    # data_loader
    dataset_STR = STRDataset(root='train_data',labelPath='train_crop_list.txt',charsetPath='chinese_cht_dict.txt')
    data_loader = torch.utils.data.DataLoader(
        dataset_STR, batch_size = batch_size, shuffle = False, num_workers = 0, collate_fn = collate_fn
    )
    # chars dictionary
    charsDict = dataset_STR.charsDict
    # model
    model = clsScore(len(charsDict.keys()))
    # parameter for eval
    model_save_root = 'train_models'
    model_save_dir = "first_STR_model"
    if not os.path.exists(os.path.join(os.getcwd(),model_save_root,model_save_dir)):
        raise NameError("model path not exists")
    model.load_state_dict(torch.load(os.path.join(model_save_root,model_save_dir,"epoch_250.pth")))
    model.to(device)
    # set model to eval mode
    model.eval()
    # epoch for training
    preds = []
    ans = []
    for imgs,labels in data_loader:
        # load img and label
        imgs = torch.stack(list(img.to(device) for img in imgs))
        labels = list(labels)
        # pred and calculate ctc loss
        outputs = model(imgs) # Batch_size / max_length / class_num(chars_num)
        # print(sum(outputs[0,0,:]))
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
        logging.info(preds[i]+" / "+ans[i])
        score += lvd(preds[i],ans[i]) / max(len(preds[i]),len(ans[i]))
    score = 1 - score / len(ans)
    logging.info("1_NED : " + str(score))
if __name__ == "__main__":
    eval()