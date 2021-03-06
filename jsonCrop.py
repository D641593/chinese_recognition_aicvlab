import numpy as np
import argparse
import json
import cv2
import os
import re


def doNotCareCheck(line:str):
    line = line.replace("###","@")
    return line

def getTargetPoints(points):
    maxh = max(points[:,1])
    maxw = max(points[:,0])
    minh = min(points[:,1])
    minw = min(points[:,0])
    h = maxh - minh
    w = maxw - minw
    return np.array([[0,0],[w,0],[w,h],[0,h]],dtype=np.float32),(int(w),int(h))

def warpImg(img,points):
    points = np.array(points,dtype=np.float32)
    points = points.reshape(4,2)
    points = orderPoints(points)
    targets,shape = getTargetPoints(points)
    M = cv2.getPerspectiveTransform(points,targets)
    transImg = cv2.warpPerspective(img,M,shape,cv2.INTER_LINEAR)
    return transImg

def orderPoints(points):
    mw = points[:,0].mean()
    mh = points[:,1].mean()
    order = []
    for point in points:
      x = point[0] - mw
      y = point[1] - mh
      if x < 0 and y < 0:
        if 0 in order:
          order = [0,1,2,3]
          break
        order.append(0)
      elif x >= 0 and y < 0:
        if 1 in order:
          order = [0,1,2,3]
          break
        order.append(1)
      elif x >= 0 and y >= 0:
        if 2 in order:
          order = [0,1,2,3]
          break
        order.append(2)
      else:
        if 3 in order:
          order = [0,1,2,3]
          break
        order.append(3)
    return points[order]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", help = 'path of the folder of train_data', type = str,
                        default="train_data/")
    parser.add_argument("--json_dir", help = 'path of the folder of json file', type = str,
                        default="train_high/json/")
    parser.add_argument("--img_dir",help='path of the folder of image file',type = str,
                        default='train_high/img/')
    parser.add_argument("--output_dir", help = 'output path of crop image', type = str,
                        default='train_high_crop3/')
    parser.add_argument("--label_fileName", help = 'output path of fileName.txt', type = str,
                        default='train_high_crop_list3.txt')  

    args = parser.parse_args()
    main_dir = args.main_dir
    json_dir = args.json_dir
    img_dir = args.img_dir
    output_dir = args.output_dir
    label_fname = args.label_fileName

    assert os.path.isdir(main_dir),"main_dir not exist"
    assert os.path.isdir(os.path.join(main_dir,json_dir)),"json_dir not exist"
    assert os.path.isdir(os.path.join(main_dir,img_dir)),"img_dir not exist"
    if not os.path.isdir(os.path.join(main_dir,output_dir)):
        os.mkdir(os.path.join(main_dir,output_dir))
        print("creating folder : ",os.path.join(main_dir,output_dir))

    files = sorted(os.listdir(os.path.join(main_dir,json_dir)))
    chineselimit = re.compile(u"[\u4e00-\u9fa5]+")
    txt = []
    for jsonfname in files:
        imgfname = jsonfname[:-4] + "jpg"
        with open(os.path.join(main_dir,json_dir,jsonfname),'r',encoding='utf-8') as f:
            img = cv2.imread(os.path.join(main_dir,img_dir,imgfname))
            assert img is not None , "the path '" + os.path.join(main_dir,img_dir,imgfname)+"' can't load image successfully."
            jsonfile = json.load(f)
            shapes = jsonfile['shapes']
            num = 0
            for shape in shapes:
                group_id = shape['group_id']
                if group_id == 1: # ????????????
                    continue
                label = shape['label']
                if label == "" or label == "###":
                    label = "@"
                points = shape['points']
                label = doNotCareCheck(label)
                transImg = warpImg(img,points)
                h,w = transImg.shape[:2]
                if h > w:
                    transImg = cv2.rotate(transImg,cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                transImgfname = imgfname[:-4] + "_" + str(num) + ".jpg"
                num += 1
                cv2.imwrite(os.path.join(main_dir,output_dir,transImgfname),transImg)
                row = os.path.join(output_dir,transImgfname)+"\t"+label+"\n"
                txt.append(row)
        print(jsonfname," finish!")
  
    with open(os.path.join(main_dir,label_fname),'w',encoding = 'utf-8') as wf:
        for row in txt:
            wf.write(row)
    print("txt file save at '",os.path.join(main_dir,label_fname),"'")
        

