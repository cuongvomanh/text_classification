import os
import bisect
dir_path = 'C:/Users/user/Documents/machine_learning_btl/preprocessing/data/step2_data'
new_dir_path='C:/Users/user/Documents/machine_learning_btl/preprocessing/data/step3_data'

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
        data_list=data.split()
        data_list.sort()
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
dir_path = "/home/luuphuoc/data/new_data"
data = words.split_words(open("/home/luuphuoc/data/dict.txt",'r'))
counter = []

for x in data :
    counter.append(0)

print(len(data))
print(len(counter))

for dirname in os.listdir(dir) :
    for dirname1 in os.listdir(dir + '/' + dirname) :        
        for filename in os.listdir(dir + '/' + dirname + '/' + dirname1) : 
            path = dir + '/' + dirname+ '/' + dirname1 + '/' + filename

            file_in = words.split_words(open(path,'r'))
            for i in range(len(data)) :
                if data[i] in file_in :
                    counter[i] += 1

open("/home/luuphuoc/data/quantity.txt",'w').write(' '.join(str(x) for x in counter)) 
# The 'quantity.txt' file save the quantities of document which the word is in
'''