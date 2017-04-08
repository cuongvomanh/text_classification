import os

dir_path = 'C:/Users/user/Documents/machine_learning_btl/20news-bydate/20news-bydate-test'
new_dir_path = 'C:/Users/user/Documents/machine_learning_btl/preprocessing/data/step1_data'


# Loai bo cac ky tu dac biet, lower cac tu va loai bo cac stop-word

for subject_dir_name in os.listdir(dir_path) :
    subject_dir_path = dir_path + "/" + subject_dir_name
    #Tao thu muc moi
    new_subject_dir_path=new_dir_path+'/'+subject_dir_name
    if not os.path.exists(new_subject_dir_path):
            os.makedirs(new_subject_dir_path)

    for file_name in os.listdir(subject_dir_path) :
        file_path = subject_dir_path + "/" + file_name
        file = open(file_path)
        data=file.read()
        file.close()
        data_list=[x.strip('"').rstrip("'s").rstrip('.').rstrip('?').rstrip('!').lower() for x in data.split() if(x.isalpha() or x.startswith('"') or x.endswith('"') or x.endswith("'s") or x.endswith('.') or x.endswith('?') or x.endswith('!') ) and x.strip('"').rstrip("'s").rstrip('.').rstrip('?').rstrip('!').lower().isalpha() ]
        #Ghi vao file
        data_file=open(new_subject_dir_path+'/'+file_name, 'w')
        data_file.truncate()
        data_file.write('\n'.join(data_list))
        data_file.close()
    







