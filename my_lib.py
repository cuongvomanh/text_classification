"""my_lib.py: Preprocessing Lib for classification for 20news-bydate data."""

__author__      = "cuongvomanh"
__copyright__="Apr,2017"
import os
import numpy as np
from nltk.stem.lancaster import *
from nltk.stem.wordnet import *
import re
import random
# from HTMLParser import HTMLParser
# htmlParser = HTMLParser()
def read_file(file_path) :
    # f is a file object
    data_file=open(file_path)
    result=data_file.read()
    data_file.close()
    return result
def delete_special_symbol_and_lower_word(data):
    # data_list=[x.strip('"').strip('(').rstrip(')').rstrip("'s").rstrip('.').rstrip('?').rstrip('!').rstrip(':').rstrip(';').rstrip('...').strip('>').lower() \
    # for x in data_list if ( x.isalpha() or x.startswith('(') or x.endswith(')')  or x.endswith(':') or x.endswith(';') or \
    #  x.startswith('"') or x.endswith('"') or x.endswith("'s") or x.endswith('.') or x.endswith('?') or x.endswith('!')  or x.endswith('...') or  x.startswith('>'))and \
    #   x.strip('"').strip('(').rstrip(')').rstrip(':').rstrip(';').rstrip("'s").rstrip('.').rstrip('?').rstrip('!').rstrip('...').strip('>').lower().isalpha() ]
    # data_list=[xre.compile('[0-9A-Za-z]')
    # change currency sign followed by number to ' currency '
    data = re.compile(r'(\€|\¥|\£|\$)\d+([\.\,]\d+)*').sub(' currency ', data )
    
    # change link to ' urllink '
    data = re.compile(r'(((http|https):*\/\/[^\s]*)|((www)\.[^\s]*)|([^\s]*(\.com|\.co\.uk|\.net)[^\s]*))').sub('', data)

    # change email to ' emailaddr '
    data = re.compile(r'[^\s]+@[^\s]+').sub('', data)

    # change html entities to unicode characters
    # line = htmlParser.unescape(line)

    # change sequence of number to ' numb '
    data = re.compile(r'\d+[\.\,]*\d*').sub('', data)

    # lowercase and split line by characters are not in  0-9A-Za-z
    data=data.replace("'s",' ').replace("'re",' ').replace("'ve",' ').replace("'d",' ').replace("n't",' ').replace("'ll",' ')
    data_string_list=list(data)
    # print(data_string_list)
    symbol='?!.,_^@:;`~-/&()%$#!<>*+={}[]|'
    symbol=list(symbol)
    symbol+=['"',"'",'\\']
    new_data_string_list=[]
    for char in data_string_list:
        if not char in symbol:
            new_data_string_list.append(char)
        else:
            new_data_string_list.append(' ')
    data=''.join(new_data_string_list)
    data_list=data.split()

    # data_list = re.compile(r'\W+').split(data.lower())
    # data_list=[element.strip('_') for element in data_list if len(element.strip('_'))>0 ]   
    # line = ' '.join(lineArr).lower()
    # return
    # return line
    return data_list
    # return data_list
def lemmatize_stem_word_and_delete_stop_word(data_list,stop_word_list,lemmatize=False):  
    lemmatizer = WordNetLemmatizer()
    if lemmatize:
        new_data_list=[]
        for word in data_list:
            if word.endswith('ly'):
                continue
            # if len(word) <= 1: continue
            rootWord = lemmatizer.lemmatize(word, pos='n')
            if rootWord == word:
                rootWord = lemmatizer.lemmatize(word, pos='a')
                if rootWord == word:
                    rootWord = lemmatizer.lemmatize(word, pos='v')
                    if rootWord == word:
                        rootWord = lemmatizer.lemmatize(word, pos='r')
                        if rootWord == word:
                            rootWord = lemmatizer.lemmatize(word, pos='s')
                    
            if not rootWord in stop_word_list: 
                new_data_list.append(rootWord)
        data_list=new_data_list
    else:
        stemmer=LancasterStemmer()
        data_list= [stemmer.stem(lemmatizer.lemmatize(word,pos='v')) for word in data_list if not stemmer.stem(lemmatizer.lemmatize(word,pos='v')) in stop_word_list]
    
    return data_list
def lemmatize_stem_word(data_list):
    lemmatizer = WordNetLemmatizer()
    stemmer=LancasterStemmer()
    data_list= [stemmer.stem(lemmatizer.lemmatize(word,pos='v')) for word in data_list]
    return data_list

def dictionary_merge(a,b):
    result={}
    a_iter=iter(a)
    b_iter=iter(b)
    a_value=next(a_iter,None)
    b_value=next(b_iter,None)
    while a_value!=None and b_value!=None :
        if a_value==b_value:
            result[a_value]=a[a_value]+b[b_value]
            a_value=next(a_iter,None)
            b_value=next(b_iter,None)
        elif a_value<b_value:
            result[a_value]=a[a_value]
            a_value=next(a_iter,None)
        else:
            result[b_value]=b[b_value]
            b_value=next(b_iter,None)
    while a_value !=None:
        result[a_value]=a[a_value]
        a_value=next(a_iter,None)
    while b_value !=None:
        result[b_value]=b[b_value]
        b_value=next(b_iter,None)
    return result

def write_matrix_numpy_to_file(matrix,file_path):
    file= open(file_path,'w')
    file.truncate()
    count=0
    for matrix_row in matrix:
        file.write(str(matrix_row.tolist()))
        if count!= len (matrix)-1:
            file.write('\n')
        count+=1
    file.close()
def write_matrix_to_file(matrix,file_path):
    file= open(file_path,'w')
    file.truncate()
    count=0
    for matrix_row in matrix:
        file.write(str(matrix_row))
        if count!= len (matrix)-1:
            file.write('\n')
        count+=1
    file.close()

def read_input(file_path):
    file=open(file_path)
    list=[]
    count=0
    while True:
        line=file.readline().rstrip('\n')
        # if line =='' or count>200:
        if line =='':
            break
        list.append(np.array(eval(line)).reshape(-1,1))
        # list.append(eval(line).reshape(-1,1))
        # print(count)
        count+=1
    #list=np.array(list)
    print("hello")
    #print(list)
    return list
def read_lable(file_path):
    file=open(file_path)
    list=[]
    count=0
    while True:
        line=file.readline().rstrip('\n')
        # if line =='' or count>200:
        if line =='' :
            break
        list.append(np.array(eval(line)).reshape(-1,1))
        # list.append(eval(line).reshape(-1,1))
        # print(count)
        count+=1
    #list=np.array(list)
    print("hello")
    #print(list)
    return list
def tranform_from_sparse_matrix_to_dense_matrix(path,subject_number):
    source_file=open(path)
    input_list=[]
    label_list=[]


    files=source_file.read().split('\n')
    # print(files[-1])
    input_list=[]
    label_list=[]
    for file in files:
        if len(file)<2: break
        label=int(file[0])
        if label>subject_number-1: break
        b=[]
        b+=label*[0.0]
        b.append(float(label))
        b+=(subject_number-1-label)*[0.0]
        b=np.array(b,dtype='float32').reshape(-1,1)
        # print(np.shape(b))
        label_list.append(b)

        couples=file[2:].split(' ')
        # print(couples[0])
        a=[]
        for couple in couples:
            if len(couple)<2 :
                # print('eh')
                break
            # print(couple)
            pair=couple.split(':')
            # print(pair)
            index=int(pair[0])
            value=float(pair[1].strip(', '))
            # print(len(a()))
            number=index-len(a)
            # print(number)
            a+=[0.0]*number
            a.append(value)
        # a+=[0.0]*(9182-len(a))
        a+=[0.0]*(17809-len(a))
        if len(a) <2 : break
        array=np.array(a,dtype='float32').reshape(-1,1)
        input_list.append(array)
    label_input_list=list(zip(input_list,label_list))
    random.shuffle(label_input_list)
    index=int(0.8*len(label_input_list))
    train_lable_input_list=label_input_list[:index]
    test_lable_input_list=label_input_list[index:]
    source_file.close()
    return train_lable_input_list,test_lable_input_list