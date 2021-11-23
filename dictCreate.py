files = [
    'train_data/train_crop_list2.txt',
    'train_data/public_crop_list2.txt',
    'train_data/train_high_crop_list2.txt',
]

chars = []
exceptList = [' ','\n','\t','\r','@']
for f in files:
    with open(f,'r',encoding='utf-8') as reader:
        txt = reader.readlines()
        for line in txt:
            label = line.split('\t')[1]
            for char in label:
                if char in exceptList:
                    break
                if char not in chars:
                    chars.append(char)
chars = sorted(chars)
with open("myDict.txt",'w',encoding='utf-8') as wf:
    for char in chars:
        wf.write(char+'\n')
        