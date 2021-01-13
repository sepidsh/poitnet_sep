from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement

import cv2 
import matplotlib.pyplot as plt
def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

       
        self.datapath = []
        with open("../fs_data/public_100/processed/seperate_room_data/test/list.txt") as fp: 
                    Lines = fp.readlines() 
                    lll=0
                    for line in Lines: 
                        line="../fs_data/public_100/processed/seperate_room_data/test/"+line[:-1]
                        k=np.load(line,allow_pickle=True)
                        if(lll>2000):
                            break
                        k3=k.tolist()['room_instances_annot']
                        for i in range(len(k3)):
                              lll=lll+1
                              if(lll>2000):
                                  break
                              #cr=[]

                              cr=(k3[i]['room_corners'])
                              for knmn in range(len(cr)):
                                      cr[knmn]=[cr[knmn]]
                              cr=[(np.array(cr,dtype=np.int32))]
                              
                              """mask=np.zeros((256,256),dtype=np.uint8)
                              mask[k3[i]['mask_large']!=False]=1
                              mask.resize((256,256))
                              mask[mask>0]=1
                              mask[mask<0]=0
                              ret,thresh=cv2.threshold(mask,127,255,0)
            
                              cr, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
           
                              print("cr contour is",cr[0].type)"""
                              """if len(cr) > 0:              
                                 cr = [c for c in cr] 
                              cr_new=np.squeeze(cr[0])
                              
                              #print(cr_new)
                              #plt.imshow(mask)
                              #plt.show()
                              cr_new=cr_new.tolist()
                              #[:][:][0]
                              #print(x)
                              x_min=10000
                              x_max=0
                              y_min=10000
                              y_max=0
                              for kn in range(len(cr_new)):
                                    if(x_min>cr_new[kn][0]):
                                        x_min=cr_new[kn][0]
                                    if(x_max<cr_new[kn][0]):
                                        x_max=cr_new[kn][0]
                                    if(y_min>cr_new[kn][1]):
                                        y_min=cr_new[kn][1]

                                    if(y_max<cr_new[kn][1]):
                                        y_max=cr_new[kn][1]
                              #print(cr_new[:,0]) 
                              #x_min=min(cr_new[:][0])
                              #x_max=max(cr_new[:][0])
                              #mask_o=np.zeros((abs(y_min-y_max),abs(x_min-x_max)),dtype=np.uint8)
                              #y_min=min(cr_new[:][1])
                              #y_max=max(cr_new[:][1])
                              #print(y_min,y_max,x_min,x_max)
                              mask_o=mask[y_min:y_max, x_min:x_max]
                              #mask_o.resize((256,256))
                              #plt.imshow(mask_o)
                              #plt.show()
                              mask_f=mask_o.copy()
                              #mask=mask_f
                              mask_f.resize((256,256))
                              mask=mask_f
                              mask[mask>0]=1
                              mask[mask<0]=0
                              ret,thresh=cv2.threshold(mask,127,255,0)
            
                              cr, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
           
                              if len(cr) > 0:              
                                 cr = [c for c in cr]

                              #mask_o=mask_o.resize((256,256))"""

                              
                              mask_out=np.zeros((256,256),dtype=np.uint8)
                              points_final=[]
                              cnt_m=np.zeros((256,256), dtype='uint8')
                              cv2.drawContours(cnt_m,cr,-1,(256,256,256),1)
                              #plt.imshow(cnt_m)
                              #plt.show()
                              points_cc=[]
                              for k in range(256):
                                  for j in range(256):
                                     if(cnt_m[k][j]!=0):
                                           points_final.append([(j-128)/128,(k-128)/128,0])
                                           points_cc.append([j,k])
                              seg_final=[]
                              cr=np.array(cr)
                              #print("cr is", cr[0][1][0][1])
                              
                              for k in range(len(points_final)):
                                  hm=0     
                                  for kmn in range(len(cr[0])):
                                      #print(cr[0][kmn][0][0])
                                      
                                      if (points_cc[k][0]==cr[0][kmn][0][0] )&(cr[0][kmn][0][1]==points_cc[k][1]):
                                          
                                          hm=5
                                      elif (abs(points_cc[k][0]-cr[0][kmn][0][0])<2)&(abs(cr[0][kmn][0][1]-points_cc[k][1])<2):
                                          if(hm<4):   
                                              hm=4
                                      elif (abs(points_cc[k][0]-cr[0][kmn][0][0])<3 )&(abs(cr[0][kmn][0][1]-points_cc[k][1])<3):
                                          
                                          if(hm<3):   
                                              hm=3
                                          #hm=0.9
                                      elif (abs(points_cc[k][0]-cr[0][kmn][0][0])<4 )&(abs(cr[0][kmn][0][1]-points_cc[k][1])<4):
                                          
                                          if(hm<2):   
                                              hm=2
                                          #hm=0.8
                                          
                                      elif (abs(points_cc[k][0]-cr[0][kmn][0][0])<5)&(abs(cr[0][kmn][0][1]-points_cc[k][1])<5):
                                          
                                          if(hm<1):   
                                              hm=1
                                          #hm=0.6
                                      #elif (abs(points_cc[k][0]-cr[0][kmn][0][0])<30 )&(abs(cr[0][kmn][0][1]-points_cc[k][1])<25):
                                          
                                       #   if(hm<0.3):   
                                        #      hm=0.3
                                          #hm=0.30
                                      #elif (abs(points_cc[k][0]-cr[0][kmn][0][0])<30 )&(abs(cr[0][kmn][0][1]-points_cc[k][1])<30):
                                          
                                       #   hm=0.5
                                      #elif (abs(points_cc[k][0]-cr[0][kmn][0][0])<35)&(abs(cr[0][kmn][0][1]-points_cc[k][1])<35):
                                          
                                       #   hm=0.4
                                      #if (points_cc[k][0]==cr[0][kmn][0][0] )&(cr[0][kmn][0][1]==points_cc[k][1]):
                                          
                                       #   hm=1
                                      #if (points_cc[k][0]==cr[0][kmn][0][0] )&(cr[0][kmn][0][1]==points_cc[k][1]):
                                          
                                       #   hm=1
                                      #if (points_cc[k][0]==cr[0][kmn][0][0] )&(cr[0][kmn][0][1]==points_cc[k][1]):
                                          
                                       #   hm=1
                                      #if (points_cc[k][0]==cr[0][kmn][0][0] )&(cr[0][kmn][0][1]==points_cc[k][1]):
                                          
                                       #   hm=1
                                      #if (points_cc[k][0]==cr[0][kmn][0][0] )&(cr[0][kmn][0][1]==points_cc[k][1]):
                                          
                                       #   hm=1
                                  #if(hm>0):
                                   #      hm=1
                                         #print(points_cc[k])
                                  hm=float(hm)
                                  seg_final.append((hm+1))
                                  #else:
                                   #   seg_final.append(0)
                              #print(len(seg_final),len(points_final))
                              seg_final=np.array(seg_final)
                              #plt.imshow(cnt_m)
                              #plt.show()
                        
        #self.datapath = []
        #for item in self.cat:
         #   for fn in self.meta[item]:
                              self.datapath.append((np.array(points_final),seg_final))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = 8
        self.num_sef_class=8#self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        #print("hhuihihiohoihhioohiohiohou") 
        #print("fn is:",fn)
        k=np.load("../fs_data/public_100/processed/seperate_room_data/test/0Kajc_nnyZ6K0cRGCQJW56.npy",allow_pickle=True)        
        cls =0# self.classes[self.datapath[index][0]]
        #print("cls is",cls)
        #cls2=0
        
        point_set = fn[0].astype(np.float32)
        #np.loadtxt(fn[1]).astype(np.float32)
        #point_set_2=k.tolist()['topview_image']
        #k3=k.tolist()['room_instances_annot']
        #corners=[]
        #import cv2 
        #import matplotlib.pyplot as plt
        #for i in range(len(k3)):
        #    cr=k3[i]['room_corners']
            #print(k3[i]['mask'])
         #   mask=np.zeros((256,256),dtype=np.uint8)
         #   mask[k3[i]['mask']!=False]=1
            #plt.imshow(mask)
            #plt.show()
           #im, contours, k112=cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #maks=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY);
          #  ret,thresh=cv2.threshold(mask,127,255,0)
            
           # cr, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #mask_new=cv2.drawContours(np.zeros((256,256),dtype=np.uint8), im, 0, (0, 255, 0), 2)
           # if len(cr) > 0:              
            #    cr = [c for c in cr] 

            #mask_out=np.zeros((256,256),dtype=np.uint8)
            #points_final=[]
            #cnt_m=np.zeros((256,256), dtype='uint8')
            #c=max(cr,key=cv2.contourArea)
            #print("cr is",cr)
            #cv2.drawContours(cnt_m,cr,-1,(256,256,256),2)
            #points_cc=[]
            #plt.imshow(cnt_m)
            #plt.show()
        seg = fn[1].astype(np.int32)
        #print("seg", np.unique(seg))

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        seg = seg[choice]
        #print("final seg is",seg)
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            #print("point set is",len(point_set))
            #print("seg is",seg.unique())
            return point_set, seg

    def __len__(self):
        return len(self.datapath)

class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


    def __len__(self):
        return len(self.fns)

if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(),seg.type())

        d = ShapeNetDataset(root = datapath, classification = True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d = ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])

