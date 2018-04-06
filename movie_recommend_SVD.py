#this is movie recommendation system
#use movielens data from grouplens
#us SVD method first created on 4/1/2018


#%reset clears everything

#########data cleaning###################
#define function load movie
import pandas as pd
import numpy as np
import time
from math import sqrt   

#load movie rating file
#format is:
#user id | item id | rating | timestamp.  
def load_rating_file(file_name):
    #file name is the file_name
    header_name=['user_ID', 'item_ID','rating','timestamp']
    rating_file=pd.read_csv(file_name, sep='\t', header=None, names=header_name)
    return rating_file
    
def load_movie_file(file_name):
    #file name is the file_name
    movie_file=pd.read_csv(file_name, sep='|', header=None)
    return movie_file
    
def load_occupation(file_name):
    #file name is the file name
    occupation_file=pd.read_csv(file_name, sep='\n', header=None)
    return occupation_file
    
def input_data(train_data_name, test_data_name):
    #create a numpy array containing test data
    #format row index from 1 to the largest user ID
    #column number from 1 to the largest item ID
    #create mask matrix
    #return train and test data rating matrix train_data_matrix & test_data_matrix
    train_data=load_rating_file(train_data_name)
    test_data=load_rating_file(test_data_name)
    
    #print test_data.user_ID.unique().size
    #print ('shape is: %d ' %test_data.user_ID.unique().shape())
    #create user-item matrix for test data
    num_user=943
    num_item=1682
    train_data_matrix=np.zeros((num_user, num_item))
    for line in train_data.itertuples():
        train_data_matrix[line[1]-1, line[2]-1]=line[3]
    
    test_data_matrix=np.zeros((num_user, num_item))
    for line in test_data.itertuples():
        test_data_matrix[line[1]-1, line[2]-1]=line[3]
        
    #print 'rating by user 1 on item 6 is %d: ' %test_data_matrix[0][5]
    #print 'rating by user 462 on item 682 is %d: ' %test_data_matrix[461][681]
    #rating_matrix=df.values  #(N_user, N_item)
    #mask_matrix=((df>0.1)*1).values
    #print '&&&&&&&&&&&'
    #print ('first elment is: %f ' %(train_data_matrix[0][0]))
    #print ('shape is: %d %d' %train_data_matrix.shape)
    return train_data_matrix, test_data_matrix
#*******************************************************  

def calculate_loss(input_data, pred_data):
    #calculate RMSE
    J_loss=0.0
    
    calc_diff=pred_data-input_data
    data_mask=((input_data>0.1)*1)
    number_rating=data_mask.sum()
    J_loss=np.multiply(np.power(calc_diff,2), data_mask).sum()/number_rating
    return sqrt(J_loss)

#use stochastic gradient descent method
def SGD(df_train_data,train_data_matrix, test_data_matrix): 
    num_user=943
    num_item=1682
    
    n_factors=8  #number of latent features
    alpha=0.0008 #learning rate
    num_run=3000
    num_cycles=0
    output_cycles=5  #every output_cycles, output results
    
    num_ratio=0.1
    start_time=time.time()
    
    user_feature=np.random.rand(num_user, n_factors)*num_ratio
    item_feature=np.random.rand(num_item, n_factors)*num_ratio
    
    file_loss_train=open('J_SVD_train.txt','w')
    file_loss_test=open('J_SVD_test.txt','w')
    user_feature_file='user_SVD_feature.out'
    item_feature_file='item_SVD_feature.out'
    
    train_mask=(train_data_matrix>0.1)*1
    
    #test_mask=(test_data_matrix>0.1)*1
    
    for num_cycles in range(num_run):
        pred_data=np.dot(user_feature,item_feature.T)
        
        diff_rating=np.multiply((pred_data-train_data_matrix), train_mask)  #(num_user, num_item)
        delta_sum_user=np.matmul(diff_rating, item_feature)  #(N_user,n_factors)
        delta_sum_item=np.matmul(diff_rating.T, user_feature) #(N_item, n_factors)
        user_feature=user_feature-alpha*delta_sum_user
        item_feature=item_feature-alpha*delta_sum_item
            
        if num_cycles%output_cycles==0:
                
                
            J_train_loss=calculate_loss(train_data_matrix, pred_data)
            J_test_loss=calculate_loss(test_data_matrix, pred_data)
            
            file_loss_train.write(str(J_train_loss)+'\n') 
            file_loss_test.write(str(J_test_loss)+'\n') 
            np.savetxt(user_feature_file, user_feature,fmt='%7.3f', delimiter=' ')
            np.savetxt(item_feature_file, item_feature,fmt='%7.3f', delimiter=' ')
            
    
    
    end_time=time.time()
    elapsed_time=end_time-start_time
    print ('elapsed time is:  %f' %elapsed_time)
    
#************************************
def main():
    print "main program begins"
    file_name='u.item'
    movie_file=load_movie_file(file_name)
    
    train_name='u1.base'
    test_name='u1.test'
    train_matrix, test_matrix=input_data(train_name, test_name)
    
    df_train_data=load_rating_file(train_name)
    
    SGD(df_train_data,train_matrix, test_matrix)
    
    
if __name__=='__main__':
    main()