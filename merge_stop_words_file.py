import os
import heapq
import my_lib
dir='C:/Users/user/Documents/machine_learning_btl2/preprocessing/stop-word-list-dir'
file_list=[open(dir+'/'+f) for f in os.listdir(dir)]
data_list=[f.read() for f in file_list ]
for f in file_list:
    f.close()
list_data_list=[d.split() for d in data_list]
for list_data in list_data_list:
	list_data.sort()
lemmatized_data_list=[my_lib.lemmatize_stem_word(d) for d in list_data_list]
#print(data1_list)
#print(data3_list)
final_data_list=list(heapq.merge(*lemmatized_data_list))
temp_data_list=[final_data_list[0]]
for x in final_data_list:
    if (x != temp_data_list[-1]):
        temp_data_list.append( x )
final_data_list=temp_data_list
print(final_data_list)
merged_f=open('C:/Users/user/Documents/machine_learning_btl2/preprocessing/'+'lemmatized-merged-stop-word-list.txt','w')
merged_f.write('\n'.join(final_data_list))
merged_f.close()
