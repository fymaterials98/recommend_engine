#this is movie recommendation system
#use movielens data from grouplens


#%reset clears everything

#########data cleaning###################
#define function load movie
import pandas as pd
import numpy as np
import time

#load movie rating file
#format is:
#user id | item id | rating | timestamp.  
def load_rating_file(file_name):
    #file name is the file_name
    rating_file=pd.read_csv(file_name, sep='\t', header=None)
    return rating_file
    
def load_movie_file(file_name):
    #file name is the file_name
    movie_file=pd.read_csv(file_name, sep='|', header=None)
    return movie_file
    
def load_occupation(file_name):
    #file name is the file name
    occupation_file=pd.read_csv(file_name, sep='\n', header=None)
    return occupation_file
    
def input_data(file_name):
    #create a matrix of size (N_user, N_item)
    #unrated entry is filled with value 0
    N_user=943
    N_item=1682
    df=load_rating_file(file_name)
    data_matrix=np.zeros((N_user, N_item))
    for line in df.itertuples():
        data_matrix[line[1]-1, line[2]-1]=line[3]
    
    mask_matrix=((data_matrix>0.1)*1)
    
    return data_matrix, mask_matrix
    
def input_test_data(file_name):
    #create a numpy array containing test data
    #format row index from 1 to the largest user ID
    #column number from 1 to the largest item ID
    #create mask matrix
    #return rating matrix and mask matrix based on data for test
    test_data=load_rating_file(file_name)
    df=test_data.pivot(index=0, columns=1, values=2)
    #print 'df shape is: '
    size_df_x=df.shape[0]
    size_df_y=df.shape[1]
    #print size_df_x
    #print size_df_y
    #print df.index[size_df_x-1]
    #print df.columns[size_df_y-1]
    
    df.fillna(0, inplace=True)
    #print df.iloc[0,0]
    
    
    
    
    #test rating matrix
    #row # is user ID from 0 to N_user-1  (total N_user)
    #column # is item IS from 0 to N_item-1 (total N_item)
    N_user=  df.index[size_df_x-1]
    N_item=  df.columns[size_df_y-1]
    #mask_matrix=np.zeros((N_user, N_item)) #1 if rated, 0 otherwise
    num=0
    #print "number_rating is:  "+ str(number_rating)
    
    
    start_time=time.time()
    #method 2
    index_number=0
    
    #print 'before shape[0] is:'
    #print df.shape[0]
    i=0
    num=1
    while i<size_df_x:
        if df.index[i]!=num:
            df.loc[num]=0
        else:
            i=i+1
        
        num=num+1
    
    
    #print 'after shape[0] is:'    
    #print df.shape[0]
    
    
            
    new_column=np.zeros((N_user,1))
    i=0
    num=1
    last_column_num=size_df_y
    #print 'before shape[1] is:'
    #print df.shape[1]
    
    while i<size_df_y:
        if df.columns[i]!=num:
            df.insert(last_column_num, column=num, value=new_column)
            last_column_num=last_column_num+1
        else:
            i=i+1
        num=num+1
        
    #print 'after shape[1] is:'
    #print df.shape[1]
        
    #sort index and column in ascending order
    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, inplace=True)
    
    #print df.iloc[df.shape[0]-1, 681]
    rating_matrix=df.values  #(N_user, N_item)
    mask_matrix=((df>0.1)*1).values
    
    return rating_matrix, mask_matrix
#*******************************************************   
 
    
#use DataFrame.as_matrix(Columns=None) to convert dataFrame into Numpy-array
#use Series.values to convert series into numpy array

#Movie_file from column 5 to 23 are genre relaed data
def Calc_Loss(rating_file, movie_file):
    #At first, extra information from rating_file
    #format is: user id | item id | rating | timestamp. 
    temp=rating_file.iloc[:,0:3]
    user_rating=temp.values
    number_rating=user_rating.shape[0]  #number of rating
    #print user_rating[0,:]
    #Secondly extract genre-related data
    temp=movie_file.iloc[0:, 5:]  #still dataFrame
    movie_genre=temp.values #convert DataFrame into numpy array
    #row different movies, column, which genres it belongs to
    
    #use transpose of user_rating matrix
    #user_rating=user_rating.T
    
   
    #numpy.random.seed(seed=None)
    np.random.seed(seed=2) #set random seed
    #create a user feature matrix of size (number_movie_feature, number_user)
    N_movie_features=movie_genre.shape[1]
    #print ("number of movie feature is: %d" %N_movie_features)  #this is a debug
    #print "movie_genre size is: "
    #print movie_genre.shape
    #print movie_genre[0,:]
    N_user=943  #total 943 users, can change using reading a file
    
    user_feature_matrix=np.random.rand(N_movie_features, N_user)
    
    
    
    #minimize loss function J using gradient descent method
    consider_regularization=False
    
    start_time=time.time()
    file_loss=open('J_loss.txt','w')
    #f.write(“Hello World”) 
    #now define learning rate alpha
    alpha=0.002
    if consider_regularization:
        #begin regularization
        print ("begin regularization")
    else: #no regularization
        #rating_calculate=np.matmul(movie_genre,user_feature_matrix)
        #print rating_calculate.shape
        #print rating_calculate[0,0]
        #how many cycles for gradient descent
        number_cycles=0
        while number_cycles<3000:
            rating_calculate=np.matmul(movie_genre,user_feature_matrix)
            #Calculate loss function
            if number_cycles%30==0:
                J_loss=0.0
                
                for num in range(number_rating):
                    J_loss=J_loss+(rating_calculate[user_rating[num,1]-1,user_rating[num,0]-1]-
                           user_rating[num,2])**2
                
                file_loss.write(str(J_loss)+'\n')                
                    
            #**********calculate gradient********        
            delta_sum=0
            #new_userID_row=0
            current_ID=user_rating[0,0]  #current user ID
            
            for num in range(number_rating):
                if user_rating[num,0]==current_ID:
                    #if user_rating[num,1]>=1682:
                    #   print "error!"
                    #    print user_rating[num,1]
                    delta_sum=delta_sum+(rating_calculate[user_rating[num,1]-1,user_rating[num,0]-1]-
                    user_rating[num,2])*movie_genre[user_rating[num,1]-1,:]
                else:
                    user_feature_matrix[:, current_ID-1]=user_feature_matrix[:, current_ID-1]-alpha*delta_sum
                    delta_sum=0
                    #new_userID_row=num
                    current_ID=user_rating[num,0]  #current user ID
                    delta_sum=delta_sum+(rating_calculate[user_rating[num,1]-1,user_rating[num,0]-1]-
                    user_rating[num,2])*movie_genre[user_rating[num,1]-1,:]
                    
            user_feature_matrix[:, current_ID-1]=user_feature_matrix[:, current_ID-1]-alpha*delta_sum
            #print user_feature_matrix[:, current_ID-1].shape
            #print delta_sum.shape
            #print ("current ID is: %d" %current_ID)
            
            
                
            
            number_cycles=number_cycles+1
    file_loss.close()       
    end_time=time.time()
    elapsed_time=end_time-start_time
    print ('elapsed time is:  %f' %elapsed_time)
    # numpy.matmul(a, b, out=None)  matrix multiplication
    #with open(“hello.txt”, “w”) as f: 
    #f.write(“Hello World”) 

from math import sqrt    
def Calc_loss(data_matrix, data_mask, rating_calculate):
    J_loss=0.0
    #print 'rating_calculate shape is: '
    #print rating_calculate.shape
    #print 'test_data_matrix.T shape is: '
    #print test_data_matrix.T.shape
    calc_diff=rating_calculate-data_matrix.T
    number_rating=data_mask.sum()
    J_loss=np.multiply(np.power(calc_diff,2), data_mask.T).sum()/number_rating
    return sqrt(J_loss)
      
    
    
#Movie_file from column 5 to 23 are genre relaed data
#calculate loss using matrix multiplication directly
def Calc_Loss_matrix(train_matrix,train_mask, test_matrix, test_mask, movie_file):
    
    
    
    start_time=time.time()
    #method 2
    num_train_data=train_mask.sum()
    num_test_data=test_mask.sum()
    
        
    
    
    
    #Extract genre-related data
    temp=movie_file.iloc[0:, 5:]  #still dataFrame
    movie_genre=temp.values #convert DataFrame into numpy array
    #row--item, column--genre
    #(N_item, N_feature)
    #print ('movie feature number is :%d ' %(movie_genre.shape[1]))
    
    #use transpose of user_rating matrix
    #user_rating=user_rating.T
    
   
    #numpy.random.seed(seed=None)
    np.random.seed(seed=3) #set random seed  2, 
    #create a user feature matrix of size (number_movie_feature, number_user)
    N_movie_features=movie_genre.shape[1]
    #print ("number of movie feature is: %d" %N_movie_features)  #this is a debug
    #print "movie_genre size is: "
    #print movie_genre.shape
    #print movie_genre[0,:]
    N_user=943  #total 943 users, can change using reading a file
    
    user_feature_matrix=np.random.rand(N_movie_features, N_user)
    
   
    #minimize loss function J using gradient descent method
    consider_regularization=False
    
    train_matrix_T=train_matrix.T
    train_mask_T=train_mask.T
    
    #start_time=time.time()
    file_loss=open('J_loss_matrix.txt','w')
    file_loss_test=open('J_loss_test.txt','w')
    user_feature_file='user_feature.out'
    #f.write(“Hello World”) 
    #now define learning rate alpha
    alpha=0.0035
    if consider_regularization:
        #begin regularization
        print ("begin regularization")
    else: #no regularization
        #rating_calculate=np.matmul(movie_genre,user_feature_matrix)
        #print rating_calculate.shape
        #print rating_calculate[0,0]
        #how many cycles for gradient descent
        number_cycles=0
        
        #(N_item, N_user)
        Number_run=100
        Num_cycles=5  #how many running cycle to save to J_loss file and user_feature file
        while number_cycles<Number_run:
            rating_calculate=np.matmul(movie_genre,user_feature_matrix)
            #Calculate loss function
            if number_cycles%Num_cycles==0:
                #J_loss=0.0
                #calc_diff=rating_calculate-train_matrix_T
                J_train_loss=Calc_loss(train_matrix, train_mask, rating_calculate)
                
                file_loss.write(str(J_train_loss)+'\n') 
                np.savetxt(user_feature_file, user_feature_matrix.T,fmt='%7.3f', delimiter=' ')
                

                
                #print 'user_feature_test shape is: '
                #print user_feature_test.shape
                #print 'movie_genre_test shape is: '
                #print movie_genre_test.shape
                
                J_test_loss=Calc_loss(test_matrix, test_mask, rating_calculate)
                file_loss_test.write(str(J_test_loss)+'\n')   
            #**********calculate gradient********        
            
            
            diff_rating=np.multiply((rating_calculate.T-train_matrix), train_mask)
            delta_sum=np.matmul(diff_rating, movie_genre)  #(N_user,N_feature)
            user_feature_matrix=user_feature_matrix-alpha*delta_sum.T
            
            #print ('delta_sum[0][0] is: %f ' %( delta_sum[0][0]))
            #print ('sum of delta_sum is: %f ' %(delta_sum.sum()))
                
                    
       
            
            
                
            
            number_cycles=number_cycles+1
    file_loss.close()       
    end_time=time.time()
    elapsed_time=end_time-start_time
    print ('elapsed time is:  %f' %elapsed_time)
    # numpy.matmul(a, b, out=None)  matrix multiplication
    #with open(“hello.txt”, “w”) as f: 
    #f.write(“Hello World”) 
    

        
    
    
        
    
    
    
def main():
    print "main program begins"
    file_name='u.item'
    movie_file=load_movie_file(file_name)
    
    file_name='u1.base'
    train_matrix, train_mask=input_data(file_name)
    
    file_name='u1.test'
    test_matrix, test_mask=input_data(file_name)
    
    #CF_user_model=CF_User_based()
    #print CF_user_model.show_model_name()    
    
    #print 'test data info: '
    #print test_data_matrix.shape
    #print test_data_mask.shape
    #print '**********'
    
    Calc_Loss_matrix(train_matrix,train_mask, test_matrix, test_mask, movie_file)
    
if __name__=='__main__':
    main()