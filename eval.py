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
    max_length = 40+1
    # data_loader
    dataset_STR = STRDataset(root='train_data',labelPath='train_high_crop_list3.txt',charsetPath='myDict.txt')
    data_loader = torch.utils.data.DataLoader(
        dataset_STR, batch_size = batch_size, shuffle = False, num_workers = 0, collate_fn = collate_fn
    )
    # chars dictionary
    charsDict = dataset_STR.charsDict
    # model
    model = clsScore(len(charsDict.keys()),max_length)
    # parameter for eval
    model_save_root = 'train_models'
    model_save_dir = "merge2_STR_model"
    if not os.path.exists(os.path.join(os.getcwd(),model_save_root,model_save_dir)):
        raise NameError("model path not exists")
    model.load_state_dict(torch.load(os.path.join(model_save_root,model_save_dir,"epoch_40.pth")))
    model.to(device)
    # set model to eval mode
    model.eval()
    preds = []
    ans = []
    for imgs,labels in data_loader:
        # load img and label
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
            elif pIdx == 0:
                last_idx = 0
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