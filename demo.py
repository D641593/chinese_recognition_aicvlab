import logging
import torch
from os
from dataset import *
from model import *
from PIL import Image

def demo(image_fname):
    # logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO,filename='STRdemoLog.log',filemode='a',format=log_format,force = True)
    # device 
    deviceName = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(deviceName)
    logging.info('demo using %s'%deviceName)
    print('demo using ',deviceName)
    # Dict
    charsDict = {} 
    with open(os.path.join("train_data/myDict.txt"),'r',encoding='utf-8') as f:
        txt = f.readlines()
        charsDict['blank'] = 0
        charsDict['EOS'] = 1
        idx = 2
        for char in txt:
            charsDict[char[0]] = idx # char[0] to expect '\n'
            idx += 1
    charsDict['@'] = len(charsDict)
    # model 
    max_length = 40+1
    model = clsScore(len(charsDict.keys()),max_length)
    model.load_state_dict(torch.load("train_models/merge2_STR_model/epoch_80.pth"))
    model.to(device)
    # image
    img = Image.open(image_fname)
    trans = get_eval_transforms()
    img = trans(img)
    img = torch.unsqueeze(img) # batch_size = 1
    # predict
    pred = model(img)
    predIdx = torch.argmax(outputs,dim=2)
    predconfs = torch.max(outputs,dim=2)
    key_list = list(charsDict.keys())
    sentence = ""
    confs = []
    last_idx = 1
    for l in range(predIdx.shape[1]): # max_length
        pIdx = predIdx[0][l].cpu().detach().numpy()
        if pIdx == 1:
            break
        elif pIdx != 0 and pIdx != last_idx :
            sentence += key_list[predIdx[0][l]]
            confs.append(predconfs[0][l])
            last_idx = pIdx

    logging.info("%s predict result : %s"%(image_fname,sentence))
    print("%s predict result : %s"%(image_fname,sentence))
    logging.info("predict confidence : %s"%(",".join(confs)))
    print("predict confidence : %s"%(",".join(confs))

if __name__ == "__main__":
    image_fname = "testImg.jpg"
    demo(image_fname)

