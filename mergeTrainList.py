files = [
    'train_data/train_crop_list2.txt',
    'train_data/public_crop_list2.txt',
    'train_data/train_high_crop_list2.txt',
]

outputfile = 'train_data/merge_crop_list.txt'

with open(outputfile,'w',encoding = 'utf-8') as wf:
    for f in files:
        with open(f,'r',encoding='utf-8') as reader:
            txt = reader.readlines()
            for line in txt:
                wf.write(line)