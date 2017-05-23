__author__      = "cuongvomanh"
__copyright__="Apr,2017"

import os
import my_lib
import numpy as np
import pickle
import math

def save_dictionary_file(dictionary,file_path):
	data_file=open(file_path, 'w')
	data_file.truncate()
	data_file.write(str(len(dictionary))+'\n')
	count=0
	for key,value in dictionary.items():
		data_file.write(str(key)+'-->')
		data_file.write(str(value))
		if count!=len(dictionary)-1:
			data_file.write('\n')
		count+=1
	data_file.close()
def save_data_dict_list(subject_list,data_dict_list,file_path):
	new_data_file=open(file_path,'w')
	new_data_file.truncate()
	for index in range(len(subject_list)):
		new_data_file.write(str(subject_list[index])+'-->'+str(data_dict_list[index]))
		if index!=len(subject_list)-1:
			new_data_file.write('\n')
	new_data_file.close()
def save_data(data_dict_list,subject_list,path):
	file=open(path,'w')
	file.truncate()
	for i,data_dict,subject in zip(range(len(data_dict_list)),data_dict_list,subject_list):
		data_str_list=[str(subject)]
		for key,value in data_dict.items():
			data_str_list+=list(' '+str(key)+':'+str(value))
		data_str=''.join(data_str_list)
		file.write(data_str)
		if i != len(data_dict_list)-1:
			file.write('\n')
	file.close()
def create_dictionary(data_dir_path,stop_word_list,lemmatize=False):
	dictionary={}
	data_dict_list=[]
	subject_list=[]
	count=0
	for subject_dir_name in os.listdir(data_dir_path) :
		# if subject_dir_name=='comp.sys.ibm.pc.hardware': break
		subject_dir_path = data_dir_path + "/" + subject_dir_name
		for file_name in os.listdir(subject_dir_path) :
			file_path = subject_dir_path + "/" + file_name
			data=my_lib.delete_header(file_path)
			data_list=my_lib.delete_special_symbol_and_lower_word(data)
			data_list=my_lib.lemmatize_stem_word_and_delete_stop_word(data_list,stop_word_list,lemmatize)
			
			if len(data_list)==0:
				continue
			data_list.sort()
			data_dict={}
			for word in data_list:
				if len(data_dict.keys()) ==0 or  word != list(data_dict.keys())[-1] :
					data_dict[word]=1
				else:
					data_dict[word]+=1
			data_dict_list.append(data_dict)
			subject_list.append(count)
			dictionary=my_lib.dictionary_merge(dictionary,data_dict)
		count+=1
	print('1')
	return dictionary,data_dict_list,subject_list
def filter_dictionary(dictionary,data_dict_list,subject_list):
	dictionary_len=len(dictionary)
	print("Size of dictionary="+str(dictionary_len))
	for key in list(dictionary.keys()):
		if dictionary[key]<= 4 or dictionary[key] >= 4000:
			dictionary.pop(key)
	print('2')
	# Dem so văn bản mà mỗi tu trong dictionary xuất hiện 
	
	new_dictionary={}
	idf_list=[]
	delete_indexs=[]
	index=0
	for dictionary_word,dictionary_value in dictionary.items():
		word_count=0
		for data_dict,subject in zip(data_dict_list,subject_list):
			for data_word,data_value in data_dict.items():
				if dictionary_word==data_word:
					word_count+=1
					break
		if word_count >4 and word_count <3000:
			new_dictionary[dictionary_word]=dictionary_value
			idf_list.append(word_count)

		else:
			word_index=0
			for data_dict,subject in zip(data_dict_list,subject_list):
				if dictionary_word in data_dict.keys():

					data_dict.pop(dictionary_word)
					if len(data_dict) <=3 and word_index not in delete_indexs:
						delete_indexs.append(word_index)
				word_index+=1
		index+=1
	print('3')
	# Loai nhung file co so tu qua it
	new_data_dict_list=[]
	new_subject_list=[]
	dictionary=new_dictionary
	for i,data_dict,subject in zip(range(len(data_dict_list)),data_dict_list,subject_list):
		if i not in delete_indexs:
			new_data_dict_list.append(data_dict)
			new_subject_list.append(subject)
	data_dict_list=new_data_dict_list
	subject_list=new_subject_list

	file_number=len(data_dict_list)
	idf_list=[math.log10(float(file_number)/ element ) for element in idf_list]
	print('4')
	return dictionary,data_dict_list,subject_list,idf_list
	
def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0: 
       return v
    return v/norm
def create_data(dictionary,data_dict_list,idf_list):
	file_number=len(data_dict_list)
	max_word_count_list=[max(data_dict.values()) for data_dict in data_dict_list]
	dense_matrix=[]
	for data_dict in data_dict_list:
		number_data_dict={}
		for i,dictionary_key in zip(range(len(dictionary)),dictionary.keys()):
			for data_id,data_key,data_value in zip(range(file_number),data_dict.keys(),data_dict.values()):
				if dictionary_key==data_key:
					tf_idf=float(data_value)/float(max_word_count_list[data_id])*idf_list[i]
					number_data_dict[i]=tf_idf
		dense_matrix.append(number_data_dict)
	print('6')
	return dense_matrix

def preprocessing(path,data_dir_path,destination_dir_path,lemmatize=False):

	if lemmatize:
		stop_word_path=path+'/preprocessing/lemmatized-merged-stop-word-list.txt'
	else:
		stop_word_path=path+'/preprocessing/merged-stop-word-list.txt'

	stop_word_file=open(stop_word_path)
	stop_word_list = stop_word_file.read().split()
	stop_word_file.close()

	dictionary,data_dict_list,subject_list =create_dictionary(data_dir_path,stop_word_list,lemmatize)		
	dictionary,data_dict_list,subject_list,idf_list=filter_dictionary(dictionary,data_dict_list,subject_list)
	dictionary_len=len(dictionary)
	print("Size of dictionary after filter bad data="+str(dictionary_len)) 

	#Luu dictionary va data
	save_dictionary_file(dictionary,destination_dir_path+'/standard_dictionary_view.txt')
	save_data_dict_list(subject_list,data_dict_list,destination_dir_path+'/standard_step1_data.txt')
	print('5')

	dense_matrix=create_data(dictionary,data_dict_list,idf_list)
	save_data(dense_matrix,subject_list,destination_dir_path+'/data.txt')
	print('9')	

path='C:/Users/user/Documents/machine_learning_btl2'
data_dir_path = path+'/20_newsgroups'
destination_dir_path=path+'/preprocessing/data'


preprocessing(path,data_dir_path,destination_dir_path,lemmatize=False)