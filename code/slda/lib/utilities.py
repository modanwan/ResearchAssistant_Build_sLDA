#!/usr/bin/env python2
import os
import csv
import itertools
import pandas 


def read_dir(dir_path, ends=None):
    """
    :param ends: suffix of files
    :param dir_path: directory path
    :return: all files' name and files' path in that directory
    """
    all_files = []
    if os.path.isdir(dir_path):
        file_list = os.listdir(dir_path)
        for f_name in file_list:
            f_path = dir_path + '/' + f_name
            if os.path.isdir(f_path):
                sub_files = read_dir(f_path)
                all_files = sub_files + all_files
            elif (ends is None) or (f_path.endswith(ends)):
                all_files.append((f_name, f_path))
            else:
                pass
        return all_files
    else:
        return None


def load_doc_list():
    with open("/Users/wangmodan/Desktop/For_Modan/code/slda/data/docs_for_lda.csv", "r") as f:
        return [l.strip() for l in f.readlines()]
#new
def load_surprise():
    fa_dir = '/Users/wangmodan/Desktop/For_Modan/code/slda' 
    mydata = pandas.read_csv(fa_dir + '/data/surprise.csv',index_col=0)
    mydata.index=mydata.index+'.txt'

   
   
    res={}
    for row in mydata.index: 
        res[row]=mydata.loc[row,'local_adv']
    
    #res=dict(itertools.islice(res.items(), 60)) ### only for slice 
    return res




def load_csvdic(path):
    res = {}
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            res[row[0]] = row[1]
        return res

