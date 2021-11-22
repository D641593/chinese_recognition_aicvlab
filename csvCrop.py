import numpy as np
import argparse
import csv
import cv2
import os
import re


def checkChinese(line:str):
    chineselimit = re.compile(u"[\u4e00-\u9fa5]+")
    if chineselimit.fullmatch(line) is not None:
        return line
    flag = 0
    new_line = ""
    for i in range(len(line)):
        if chineselimit.match(line[i]) is not None:
            flag = 0
            new_line += line[i]
        else:
            if flag == 0:
                new_line += "@"
                flag = 1
    return new_line

def getTargetPoints(points):
    maxh = max(points[:,1])
    maxw = max(points[:,0])
    minh = min(points[:,1])
    minw = min(points[:,0])
    h = maxh - minh
    w = maxw - minw
    return np.array([[0,0],[w,0],[w,h],[0,h]],dtype=np.float32),(w,h)

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


def warpImg(img,points):
    points = np.array(points,dtype=np.float32)
    points = points.reshape(4,2)
    points = orderPoints(points)
    targets,shape = getTargetPoints(points)
    M = cv2.getPerspectiveTransform(points,targets)
    transImg = cv2.warpPerspective(img,M,shape,cv2.INTER_LINEAR)
    return transImg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", help = 'path of the folder of train_data', type = str,
                        default="PaddleOCR/train_data/")
    parser.add_argument("--csvfileName", help = 'path of the csv file', type = str,
                        default="public/Task2_Public_String_Coordinate.csv")
    parser.add_argument("--img_dir",help='path of the folder of image file',type = str,
                        default='public/img_public/')
    parser.add_argument("--output_dir", help = 'output path of crop image', type = str,
                        default='public_crop/')
    parser.add_argument("--label_fileName", help = 'output path of fileName.txt', type = str,
                        default='public_crop_list.txt')  

    args = parser.parse_args()
    main_dir = args.main_dir
    csv_fname = args.csvfileName
    img_dir = args.img_dir
    output_dir = args.output_dir
    label_fname = args.label_fileName

    assert os.path.isdir(main_dir),"main_dir not exist"
    assert os.path.exists(os.path.join(main_dir,csv_fname)),"csvfile not exist"
    assert os.path.isdir(os.path.join(main_dir,img_dir)),"img_dir not exist"
    if not os.path.isdir(os.path.join(main_dir,output_dir)):
        os.mkdir(os.path.join(main_dir,output_dir))
        print("creating folder : ",os.path.join(main_dir,output_dir))

    chineselimit = re.compile(u"[\u4e00-\u9fa5]+")
    lastImgfname = ""
    num = 0
    noLabelOutput = 0
    txt = []
    with open(os.path.join(main_dir,csv_fname),'r',encoding='utf-8') as f:
      rows = csv.reader(f)
      for row in rows:
        imgfname = row[0] + ".jpg"
        if lastImgfname == imgfname:
          num += 1
        else:
          print(lastImgfname," finish!")
          lastImgfname = imgfname
          num = 0
          img = cv2.imread(os.path.join(main_dir,img_dir,imgfname))
          assert img is not None,"the path '" + os.path.join(main_dir,img_dir,imgfname)+"' can't load image successfully."
        points = row[1:9]
        transImg = warpImg(img,points)
        h,w = transImg.shape[:2]
        if h > w:
          transImg = cv2.rotate(transImg,cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        transImgfname = imgfname[:-4] + "_" + str(num) + ".jpg"
        cv2.imwrite(os.path.join(main_dir,output_dir,transImgfname),transImg)
        if not noLabelOutput:
          try : 
            label = row[9]
            label = checkChinese(label)
            txtrow = os.path.join(output_dir,transImgfname)+"\t"+label+"\n"
            txt.append(txtrow)
          except IndexError:
            noLabelOutput = 1

    if not noLabelOutput:
      with open(os.path.join(main_dir,label_fname),'w',encoding = 'utf-8') as wf:
          for row in txt:
              wf.write(row)
    else:
      print(csv_fname," miss some label. ",label_fname," will not output.")
        