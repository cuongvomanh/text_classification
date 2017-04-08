import os

def delete_by_null_line(f) :
    # f is a file object
    f.seek(0)
    temp = ''
    loop = True
    while(loop) :
        data = f.readline()
        if data == '' :
            break
        else :
            if data == '\n' :
                temp = f.read()
                break
    f.seek(0)
    f.truncate()
    f.write(temp)
    f.close()
    
    return True

def delete_by_title(f) :
    # f is a file object
    f.seek(0)
    temp = ''
    wrong_line = 0
    counter = 1
    while(True) :
        data = f.readline()
        if data == '' :
            break
        if 'writes:' in data :
            wrong_line = counter
        counter += 1
    
    f.seek(0)
    for i in range(wrong_line) :
        f.readline()
    temp = f.read()

    f.seek(0)
    f.truncate()
    f.write(temp)
    f.close()

    return True

# main
#  

dir_path = 'C:/Users/user/Documents/machine_learning_btl/20news-bydate/20news-bydate-test'

for dirname in os.listdir(dir_path) :
    for filename in os.listdir(dir_path  + '/' + dirname) : 
        file_path = dir_path + '/' + dirname+ '/' + filename
        delete_by_null_line(open(file_path, 'r+'))
        delete_by_title(open(file_path, 'r+'))


