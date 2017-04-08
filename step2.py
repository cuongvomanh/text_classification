import os
from nltk.stem.lancaster import *
from nltk.stem.wordnet import *

dir_path = 'C:/Users/user/Documents/machine_learning_btl/preprocessing/data/step1_data'
new_dir_path='C:/Users/user/Documents/machine_learning_btl/preprocessing/data/step2_data'
stop_word_file=open('C:/Users/user/Documents/machine_learning_btl/preprocessing/merged-stop-word-list.txt')
stop_words = stop_word_file.read().split()
stop_word_file.close()
lemmatizer = WordNetLemmatizer()

#Dua cac tu ve tu goc
for subject_dir_name in os.listdir(dir_path) :
    subject_dir_path = dir_path + "/" + subject_dir_name
    new_subject_dir_path=new_dir_path+'/'+subject_dir_name
    if not os.path.exists(new_subject_dir_path):
        os.makedirs(new_subject_dir_path)
    for file_name in os.listdir(subject_dir_path) :
        file_path = subject_dir_path + "/" + file_name
        file = open(file_path)
        data=file.read()
        file.close()
        data_list= [lemmatizer.lemmatize(word,pos='v') for word in data.split() if not lemmatizer.lemmatize(word,pos='v') in stop_words]
        #Ghi vao file
        data_file=open(new_subject_dir_path+'/'+file_name, 'w')
        data_file.truncate()
        data_file.write('\n'.join(data_list))
        data_file.close()

#for file_name in os.listdir(dir) :
#    f = open(dir + '/' + file_name)
#    data=f.read()
#    data_list=data.split()
#    dictionary=dictionary+data_list
#dictionary= [lemmatizer.lemmatize(word,pos='v') for word in dictionary]
# Xoa nhung phan tu giong nhau
#temp_dictionary=[]
#for word in dictionary:
#    if not word in temp_dictionary:
#        temp_dictionary.append(word)
#dictionary=temp_dictionary
#Luu vao second_dictionary.txt
#new_file=open(new_file_dir+'/second_dictionary.txt', 'w')
#new_file.truncate()
#new_file.write('\n'.join(dictionary))
#new_file.close()
