import os
import heapq
dir='C:/Users/user/Documents/machine_learning_btl/preprocessing/stop-word-list-dir'
file=[open(dir+'/'+f) for f in os.listdir(dir)]
data=[f.read() for f in file ]
for f in file:
    f.close()
file_data_list=[d.split() for d in data]
#print(data1_list)
#print(data3_list)
data_list=list(heapq.merge(*file_data_list))
temp_data_list=[data_list[0]]
for x in data_list:
    if (x != temp_data_list[-1]):
        temp_data_list.append( x )
data_list=temp_data_list
print(data_list)
merged_f=open('C:/Users/user/Documents/machine_learning_btl/preprocessing/'+'merged-stop-word-list.txt','w')
merged_f.write('\n'.join(data_list))
merged_f.close()
