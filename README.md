# 02445-Task-2
The code and data used for the individual assignment task 2.

# Files
HR_data.csv contains the dataset used for my report.  
final_errors.csv contains the validation errors each model obtained. They were obtained through cross-validation, where only data from seperate cohorts and rounds were used to train the models.
ConsistencyErrors.pkl contains a python dictionary with all the errors gotten from the consistency evaluation

ConsistencyTesterFunctions.py and BestModelFinders.py contain functions used in the notebooks. The difference between the two code files is only that the random state was removed in ConsistencyTesterFunctions.py. It was easier to do it like this.

Data analysis.ipynb (sorry about the space), contains the main data analysis for the report. 
ConsistencyTester.ipynb contains the code for consistency analysis of the models.



