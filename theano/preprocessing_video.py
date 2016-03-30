# -*- coding: utf-8 -*-

dataset_path='/home/chuckgu/Desktop/project/preprocessing/UCF101/UCF-101-Frames'
import csv
import numpy as np
import cPickle as pkl
import glob
import os
import pandas as pd


def main():
    data=[]
    lable=[]
    index=1
    
    for index,clas in enumerate(sorted(os.listdir(dataset_path))):
        for directory in sorted(os.listdir(dataset_path+'/'+clas)):
            os.chdir(dataset_path+'/'+clas+'/'+directory)
            img=[]
            for j,ff in enumerate(sorted(glob.glob("*.jpg"))):
                print ff,index
                #img.append(img_to_array(load_img(ff)))
                img.append(dataset_path+'/'+clas+'/'+directory+'/'+ff)
                if (j+1)%16==0:
                    data.append(img)
                    lable.append(index)
                    img=[]
	#if index==9: break
    #data=np.asarray(data)
                
    n_samples = len(data)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * 0.2))
    train_x = [data[s] for s in sidx[n_train:]]
    train_y = [lable[s] for s in sidx[n_train:]]
    test_x = [data[s] for s in sidx[:n_train]]
    test_y = [lable[s] for s in sidx[:n_train]]
    
    currdir = os.getcwd()
    os.chdir('%s/' % '/home/chuckgu/Desktop/project/Alphabeta/data')
    print n_samples,max(lable)
    print 'Saving..'
    f = open('ucf_data_all.pkl', 'wb')
    pkl.dump(((train_x,train_y),(test_x,test_y)), f, -1)
    f.close()


    
    return data,lable
    


if __name__ == '__main__':
    data,label=main()
    #print len(set(y))
