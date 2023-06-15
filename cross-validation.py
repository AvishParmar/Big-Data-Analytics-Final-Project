from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd

"""
This code conducts hypothesis testing on the model to 
determine whether it is appropriate to use for our 
project
"""

# Load the dataset
train_data = pd.read_csv("/content/sample_data/preprocessed_Tweets.csv") 

# Drop unused columns
train_data = train_data.drop(["textID", "selected_text"], axis=1)
# Map sentiments to 0/1
train_data["sentiment"] = train_data["sentiment"].map({"positive": 1, "negative": 0})

# Convert train_data to list
dataset = train_data.values.tolist()

# Specify output directories to overcome output directory already exists error
output_files=["1", "2", "3", "4", "5"]
test_output_files=["t1", "t2", "t3", "t4", "t5"]
# Turn dataset into dataframe
train_data = pd.DataFrame(dataset)

# Count for what directory to use
count = 0
# Display the updated DataFrame
print(train_data.head())

# prepare cross validation
n=5
kf = KFold(n_splits=n, random_state=42, shuffle=True)

results = []

for train_index, val_index in kf.split(train_data):
    # splitting Dataframe (dataset not included)
    train_df = train_data.iloc[train_index]
    val_df = train_data.iloc[val_index]
    # Defining Model
    model = ClassificationModel('distilbert', "distilbert-base-uncased-finetuned-sst-2-english", )
    # train the model
    model.train_model(train_df, output_dir=output_files[count])
    # validate the model
    result, model_outputs, wrong_predictions = model.eval_model(val_df, acc=accuracy_score, output_dir=test_output_files[count])
    print(result['acc'])
    # append model score
    results.append(result['acc'])
    count += 1


print("results",results)
print(f"Mean-Precision: {sum(results) / len(results)}")