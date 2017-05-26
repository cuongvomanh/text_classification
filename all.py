import os
import numpy as np
import pickle
import random

import matplotlib.pyplot as plt
import math

def normalize(v):
	norm=np.linalg.norm(v)
	if norm==0: 
	   return v
	return v/norm

def create_data(source_file_path):
	label_input_list=read_label(source_file_path)
	data=np.array(label_input_list)
	data=data.T
	for j in range(len(data)):
		sum=0
		count=0
		for i in range(len(data[j])):
			if data[j][i] !=None:
				sum+=data[j][i]
			count+=1
		aval=sum/count
		if count!=0:
			data[j]=[x if x!=None else aval for x in data[j]]
	#Nomalize
	for i in range(len(data)):
		data[i]=np.array(normalize(data[i]))# print(data)
	label=data[-1]
	input=data[:-1]
	input=input.T
	new_input=[]
	for i in range(len(input)):
		new_input.append(input[i].reshape(-1,1))
	input=new_input
	label=label.T
	label=[1.0 if element!=0.0 else 0.0 for element in label]
	new_label=[]
	for i in range(len(label)):
		if label[i]==0.0:
			new_label.append(np.array([1.0,0.0]).reshape(-1,1))
		else:
			new_label.append(np.array([0.0,1.0]).reshape(-1,1))
	label=new_label
	label_input_list=[]
	for i in range(len(input)):
		label_input=[input[i],label[i]]
		label_input_list.append(label_input)
	random.shuffle(label_input_list)
	denta=int(len(label_input_list)/5)
	final_list=[label_input_list[k:k+denta] for k in range(0,len(label_input_list),denta)]
	return final_list

print('1')
def function(x):
	return 2.0/(1+np.exp(-x.astype(float)))-1.0
def function_derivative(x):
	return (2.0*(np.exp(-x))/(1+np.exp(-x))**2)
def descartes_mul(a_array,b_array):
	return int(np.dot(a_array,b_array))
class Network:
	def __init__(self, size_layer_list):
		self.size_layer_list=size_layer_list
		if len(size_layer_list) >2:
			self.subnet_list = [np.random.randn(post_neron_list_size,pre_neron_list_size ) for pre_neron_list_size, post_neron_list_size in zip(size_layer_list[:-1], size_layer_list[1:])]# subnet= weight_list
		else:
			self.subnet_list = [np.zeros(post_neron_list_size*pre_neron_list_size,dtype='float32').reshape((post_neron_list_size,pre_neron_list_size )) for pre_neron_list_size, post_neron_list_size in zip(size_layer_list[:-1], size_layer_list[1:])]
		# self.subnet_list = [np.zeros(post_neron_list_size*pre_neron_list_size,dtype='float32').reshape((post_neron_list_size,pre_neron_list_size ))+0.0001 for pre_neron_list_size, post_neron_list_size in zip(size_layer_list[:-1], size_layer_list[1:])]
		for subnet in self.subnet_list:
			print(subnet.shape)
	def train(self,label_input_list,test_label_input_list,epoch_size,mini_batch_size,change_learning_rate=True,show_Train_Accurany=False):
		def error_function(last_neron_list,label ):
			error=0.0
			for neron,label_element in zip(last_neron_list,label):
				error+=1/2*(label_element-neron)**2
			return error
		def update_learning_rate(learning_rate_weight,derivative_weight,new_derivative_weight):
			return np.where(new_derivative_weight*derivative_weight <0,\
				np.where( learning_rate_weight>0.0001,0.5*learning_rate_weight,0.0),\
				np.where(new_derivative_weight*derivative_weight >0, np.where(learning_rate_weight <10.0,1.5*learning_rate_weight,10.0),0.01) )
		num_layers = len(self.size_layer_list)
		accurancy_list=[]
		if show_Train_Accurany:
			train_accurancy_list=[]
		index_list=[]
		if change_learning_rate:
			learning_rate_subnet_list=[np.ones(post_neron_list_size*pre_neron_list_size,dtype='float32').reshape((post_neron_list_size,pre_neron_list_size )) for pre_neron_list_size, post_neron_list_size in zip(size_layer_list[:-1:-1], size_layer_list[1::-1])]
		derivative_subnet_list=[]
		for epoch in range(epoch_size):
			mini_batchs=[label_input_list[k:k+mini_batch_size] for k in range(0,len(label_input_list),mini_batch_size)]
			for mini_batch in mini_batchs:
				mini_batch_avarage_error=0.0
				epoch_derivative_subnet_list=[np.zeros(post_neron_list_size*pre_neron_list_size,dtype='float32').reshape((post_neron_list_size,pre_neron_list_size )) for pre_neron_list_size, post_neron_list_size in zip(self.size_layer_list[-2::-1], self.size_layer_list[-1:0:-1])]
				epoch_learning_rate_negative_count=[np.zeros(post_neron_list_size*pre_neron_list_size,dtype='float32').reshape((post_neron_list_size,pre_neron_list_size )) for pre_neron_list_size, post_neron_list_size in zip(self.size_layer_list[:-1], self.size_layer_list[1:])]
				for label_input in mini_batch:
					label=label_input[1]
					input=label_input[0]
					layer_list=[]
					layer_list.append(input)
					# feedforward
					for i in range(num_layers-1):
						layer_list.append(self.subnet_list[i].dot(function(layer_list[i])) )
					# Dao ham cac neron
					derivative_layer_list=[]
					derivative_layer_list.append( (-label+ function(layer_list[-1]))*function_derivative(layer_list[-1]) )
					for i in range(num_layers-2):
						derivative_neron_list=(self.subnet_list[-(i+1)].T).dot(derivative_layer_list[i])*(function_derivative(layer_list[-(i+2)]))
						# print("Shape of derivative_neron_list ={}".format(derivative_neron_list.shape))
						# print(derivative_neron_list)
						derivative_layer_list.append(derivative_neron_list)
					# Dao ham cac trong so
					new_derivative_subnet_list=[]
					for i in range(len(self.subnet_list)):
						new_derivative_subnet_list.append( derivative_layer_list[i] .dot(function(layer_list[-(i+2)]).T) )
						# print("Shape of new_derivative_subnet_list[{}] ={}".format(i,new_derivative_subnet_list[i].shape))
						# print(derivative_layer_list[i] .dot(function(layer_list[-(i+2)]).T))
					for i in range(len(self.subnet_list)):
						epoch_derivative_subnet_list[i]+=new_derivative_subnet_list[i]
					
					if change_learning_rate:
						# cap nhat learning_rate
						if len(derivative_subnet_list)!=0:
							for i in range(len(self.subnet_list)):
								epoch_learning_rate_negative_count[i]+= np.where(new_derivative_subnet_list[i]*derivative_subnet_list[i] <0,1,0)
					derivative_subnet_list= new_derivative_subnet_list
				# Cap nhat trong so
				if change_learning_rate:
					for i in range(len(learning_rate_subnet_list)):
						learning_rate_subnet_list[i]=np.where(epoch_learning_rate_negative_count[i]>0.5*10,\
								np.where( learning_rate_subnet_list[i]>0.0001,0.5*learning_rate_subnet_list[i],0.0),\
								np.where(epoch_learning_rate_negative_count[i]<0.5*10, np.where(learning_rate_subnet_list[i] <10.0,1.5*learning_rate_subnet_list[i],10.0),1.0) )
						self.subnet_list[i]=self.subnet_list[i]-learning_rate_subnet_list[i]/mini_batch_size*epoch_derivative_subnet_list[-(i+1)]
				else:
					for i in range(len(self.subnet_list)):
						self.subnet_list[i]=self.subnet_list[i]-3.0/mini_batch_size*epoch_derivative_subnet_list[-(i+1)]
			
			if epoch%1==0:
				count=0
				for label_input in test_label_input_list:
					label=label_input[1]
					input=label_input[0]
					layer_list=[]
					layer_list.append(input)
					#feedforward
					for i in range(num_layers-1):
						layer_list.append(self.subnet_list[i].dot(function(layer_list[i])) )
					if np.argmax(layer_list[-1])==np.argmax(label):
						count+=1
				accurancy=float(count)/len(test_label_input_list)*100
				if show_Train_Accurany:
					count=0
					for label_input in label_input_list:
						label=label_input[1]
						input=label_input[0]
						layer_list=[]
						layer_list.append(input)
						#feedforward
						for i in range(num_layers-1):
							layer_list.append(self.subnet_list[i].dot(function(layer_list[i])) )
						if np.argmax(layer_list[-1])==np.argmax(label):
							count+=1
					train_accurancy=float(count)/len(label_input_list)*100
				if epoch%1==0:
					if change_learning_rate:
						print('learning_rate= {}'.format(learning_rate_subnet_list[-1][0][0]))
					if show_Train_Accurany:
						print('Accurancy train data of epoch: {} = {}%'.format(epoch,train_accurancy))
						train_accurancy_list.append(train_accurancy)
					print('Accurancy data of epoch: {} = {}%'.format(epoch,accurancy))
				accurancy_list.append(accurancy)
				index_list.append(epoch)
				if accurancy >=97:
					break
		if show_Train_Accurany:
			plt.plot(index_list,accurancy_list,'b',index_list,train_accurancy_list,'r')
		else:
			plt.plot(index_list,accurancy_list,'b')
		plt.axis([0, epoch_size, 0, 100])
		plt.xlabel('epoch')
		plt.ylabel('accurancy')
		# plt.show()
		return accurancy_list[-1]
def perform_neron_net(label_input_list,subject_number,epoch_size,mini_batch_size,weight_file_path):
	#Cross-validation
	accurancy_list=[]
	for i in range(len(label_input_list)):
		test_label_input_list=label_input_list[i]
		train_label_input_list=[]
		for j in range(len(label_input_list)):
			if j !=i: train_label_input_list+=label_input_list[j]
		dimesion_number=len(train_label_input_list[0][0])
		size_layer_list=[dimesion_number,subject_number]
		
		
		network=Network(size_layer_list)
		accurancy=network.train(train_label_input_list,test_label_input_list,epoch_size,mini_batch_size,change_learning_rate=False,show_Train_Accurany=False)
		# accurancy=network.train(train_label_input_list,train_label_input_list,epoch_size,mini_batch_size,change_learning_rate=False,show_Train_Accurany=False)
		accurancy_list.append(accurancy)
		file=open(weight_file_path+str(i)+'.txt','wb')
		file.truncate()
		pickle.dump(network,file)
		file.close()
	print('Accurancy trung bình {} lần chạy = {}'.format( len(label_input_list),sum(accurancy_list)/len(label_input_list)))
if __name__ == '__main__':
	def tranform_from_dense_matrix_to_sparse_matrix(path,subject_number,file_number):
		source_file=open(path)
		input_list=[]
		label_list=[]
		files=source_file.read().split('\n')
		input_list=[]
		label_list=[]
		for file in files:
			if len(file)<2: break
			label=int(file[:2])
			if label>subject_number-1: break
			b=[]
			b+=label*[0.0]
			b.append(1.0)
			b+=(subject_number-1-label)*[0.0]
			b=np.array(b,dtype='float32').reshape(-1,1)
			label_list.append(b)
			couples=file[2:].split(' ')
			a=[]
			for couple in couples:
				if len(couple)<1 :
					continue
				pair=couple.split(':')
				index=int(pair[0])
				value=float(pair[1].strip(' '))
				number=index-len(a)
				a+=[0.0]*number
				a.append(value)
			a+=[0.0]*(file_number-len(a))
			array=np.array(a,dtype='float32').reshape(-1,1)
			input_list.append(array)
		label_input_list=list(zip(input_list,label_list))
		random.shuffle(label_input_list)
		denta=int(len(label_input_list)/5)
		final_list=[label_input_list[k:k+denta] for k in range(0,len(label_input_list),denta)]
		source_file.close()
		return final_list
	# Doc du lieu, chuyen ma tran day thanh ma tran thua, chia label_input_list thanh 5 phan
	subject_number=20
	file_number=17809
	epoch_size=1000
	mini_batch_size=10
	weight_file_path='D:/cntt-ky6/project2/btl/weight'
	source_file_path='D:/MLProject/machine_learning_btl2/preprocessing/data/data.txt'
	label_input_list=tranform_from_dense_matrix_to_sparse_matrix(source_file_path,subject_number,file_number)
	print('1')
	perform_neron_net(label_input_list,subject_number,epoch_size,mini_batch_size,weight_file_path)

	
	
	



