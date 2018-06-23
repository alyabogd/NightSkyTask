*Sky_data.ipynb* contains notes with research activities for data exploration and different models evaluation. To reproduce the results, 
place train, unlabeled and validation datasets in the *data/* folder. 
  
Run *run.py* script to execute the solution. The script contains all the steps of the data processing, such as 
filling missing values, features selection, features creation and performs class prediction.    
You can run *run.py* from the command line as follows:  
  
      python3 run.py [path_to_train] [path_to_unlabeled_data] [path_to_test] [path_to_predictions]
        
 Train, unlabeled and test data need to be in CSV data format.  
 
 
 Run *check.py* to check the score of the prediction. Using macro-averaged f1-score as a measure of classification correctness.  
