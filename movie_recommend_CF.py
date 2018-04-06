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
 
    


      

from sklearn.metrics.pairwise import pairwise_distances
#use class for models
#define colloborative filter user based models
class CF_based:  
    model_name='Colloborative filter model'
    
    def __init__(self, train_matrix, model_type):
        print 'initialized'
        self.train_matrix=train_matrix
        self.model_type=model_type
    
    def show_model_name(self):
        return self.model_name
    
    #calculate similarity matrix    
    def calc_similarity(self):
        if self.model_type=='user': #user based model
            similarity_matrix=pairwise_distances(self.train_matrix, metric='cosine')
        elif self.model_type=='item': #item based model
            similarity_matrix=pairwise_distances(self.train_matrix.T, metric='cosine')
    
        #notice pairwise_distances when matric='cosine', it calculates 1-cosine_similarity
        return 1.0-similarity_matrix
        
    def predict_value(self, similarity):
        if self.model_type=='user': #user based model
            print 'run user based model'
            average_user_rating=np.ma.masked_where(self.train_matrix<1,self.train_matrix).mean(axis=1) #mask the blank rating
            #print('average_user_rating size is %d: ' %average_user_rating.size)
            self.train_matrix.shape
            rating_diff=(self.train_matrix-average_user_rating[:,np.newaxis])
            #print ('rating_diff shape is %d %d' %rating_diff.shape)
            pred=average_user_rating[:,np.newaxis]+np.matmul(similarity, rating_diff)/\
                 (np.sum(similarity, axis=1)[:,np.newaxis])
            #print ('pred shape is %d %d' %pred.shape)
            #print pred[0][0]
        elif self.model_type=='item':
            print 'run item based model'
            pred=np.matmul(self.train_matrix, similarity)/np.sum(similarity, axis=1)
    
        return pred

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, true_value):
    prediction=prediction[true_value.nonzero()].flatten()
    true_value=true_value[true_value.nonzero()].flatten()
    #print sqrt(3)
    return sqrt(mean_squared_error(prediction, true_value))
    
def main():
    print "main program begins"
    start_time=time.time()
    file_name='u.item'
    movie_file=load_movie_file(file_name)
    
    train_name='u1.base'
    test_name='u1.test'
    train_matrix, test_matrix=input_data(train_name, test_name)
    ##print ('******')
    ##print ('train_data_matrix shape is: %d %d' %train_matrix.shape)
    ##print ('test_data_matrix shape is: %d %d' %test_matrix.shape)
    ##print ('first train %f first test %f' %(train_matrix[0][0],test_matrix[0][0]))
    
    
    #model type either user based or contest based
    model_type='user'
    CF_user_model=CF_based(train_matrix, model_type)
    #print CF_user_model.show_model_name() 

    similarity_matrix=CF_user_model.calc_similarity()
    
    if model_type=='user':
        #print 'user based model'
        user_prediction=CF_user_model.predict_value(similarity_matrix)
    elif model_type=='item':
        item_prediction=CF_user_model.predict_value(similarity_matrix)   
        
    
    
    print 'User based model CF RMSE: '+str(rmse(user_prediction, test_matrix))
    
    
    model_type='item'
    CF_user_model=CF_based(train_matrix, model_type)
    #print CF_user_model.show_model_name() 

    similarity_matrix=CF_user_model.calc_similarity()
    
    if model_type=='user':
        #print 'user based model'
        user_prediction=CF_user_model.predict_value(similarity_matrix)
    elif model_type=='item':
        item_prediction=CF_user_model.predict_value(similarity_matrix)   
    
    print 'item based model CF RMSE: '+str(rmse(item_prediction, test_matrix))
    
    average_pred=(user_prediction+item_prediction)/2.0
    print 'average user and item CF RMSE: '+str(rmse(average_pred, test_matrix))
    #print 'test data info: '
    #print test_data_matrix.shape
    #print test_data_mask.shape
    #print '**********'
    end_time=time.time()
    elapsed_time=end_time-start_time
    print ('elapsed time is:  %f' %elapsed_time)
    
    #Calc_Loss_matrix(rating_file, movie_file,test_data_matrix, test_data_mask)
    
if __name__=='__main__':
    main()