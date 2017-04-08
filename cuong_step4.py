'''
import os
dir_path = 'C:/Users/user/Documents/machine_learning_btl/preprocessing/data/step3_data'
new_dir_path='C:/Users/user/Documents/machine_learning_btl/preprocessing/data/step4_data'

dictionary_dict={}
for subject_dir_name in os.listdir(dir_path) :
    subject_dir_path = dir_path + "/" + subject_dir_name
    new_subject_dir_path=new_dir_path+'/'+subject_dir_name
    if not os.path.exists(new_subject_dir_path):
            os.makedirs(new_subject_dir_path)
    for file_name in os.listdir(subject_dir_path) :
        file_path = subject_dir_path + "/" + file_name
        file = open(file_path)
        data=file.read()
        data_dict=eval(data)
        dictionary
        file.close()
        
        #
        count_dict={}
        for word in data_list:
        	if len(count_dict.keys()) ==0 or  word != list(count_dict.keys())[-1] :
        		count_dict[word]=0
        	else:
        		count_dict[word]+=1
        
        #Ghi vao file
        data_file=open(new_subject_dir_path+'/'+file_name, 'w')
        data_file.truncate()
        data_file.write(str(count_dict))
        data_file.close()
'''

def merge(a,b):
    result=[]
    a_iter=iter(a)
    
    b_iter=iter(b)
    a_value=next(a_iter,None)
    b_value=next(b_iter,None)
    while a_value!=None and b_value!=None :
        if a_value<=b_value:
            result.append(a_value)
            a_value=next(a_iter,None)
        else:
            result.append(b_value)
            a_value=next(a_iter,None)   
    while a_value !=None:
        result.append(a_value)
        a_value=next(a_iter,None)
    while b_value !=None:
        result.append(b_value)
        b_value=next(b_iter,None)
    return result

a=[1,5,7,9,11,17,19]
b=[1,2,4,6,8,8,9,12,13,15,15,18,21]
print(merge(a,b))
