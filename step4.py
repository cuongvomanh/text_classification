import os
import words
import numpy

dir = "/home/luuphuoc/data/new_data/train"
out = open("/home/luuphuoc/data/input.txt",'w')
vector = words.split_words(open('/home/luuphuoc/data/last_dict.txt','r'))
quan  = words.split_words_and_num(open('/home/luuphuoc/data/last_quan.txt','r'))
N = 11314 + 7532 # size of 
label = 0
for dirname in os.listdir(dir) :
        for filename in os.listdir(dir + '/' + dirname) : 
            
            path = dir + '/' + dirname+ '/' + filename
            file_in = words.split_words(open(path, 'r'))
            out.writelines(str(label) + '*')
            for i in range(len(vector)) :
                if vector[i] in file_in :
                    TF = file_in.count(vector[i])
                    IDF = numpy.log10(float(N)/float(quan[i]))
                    TFIDF = TF*IDF
                    out.writelines(str(i) + ' ' + str(TFIDF) + ' ')
                    
            out.writelines('\n')
        label += 1
        
        