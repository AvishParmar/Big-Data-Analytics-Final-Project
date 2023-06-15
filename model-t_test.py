from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import scipy.stats as stats


# Load the dataset
train_data = pd.read_csv("/content/sample_data/Tweets.csv") 

train_data = train_data[train_data['sentiment'] != 'neutral']
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

# Accuracies obtained from 5-fold cross-validation
model_accuracies = np.array(results)

# Random classification accuracy (e.g., 33.3%)
random_accuracy = 0.333

# Calculate the mean and standard deviation
mean_accuracy = np.mean(model_accuracies)
std_dev_accuracy = np.std(model_accuracies)

# Calculate t-statistic and degrees of freedom
t_statistic = (mean_accuracy - random_accuracy) / (std_dev_accuracy / np.sqrt(len(model_accuracies)))
degrees_of_freedom = len(model_accuracies) - 1

# Set significance level (e.g., 0.05)
alpha = 0.05

# Find the critical value
critical_value = stats.t.ppf(1 - alpha, degrees_of_freedom)

# Null hypothesis (H0): There is no significant difference between the sentiment analysis model's accuracy and random classification.
# Alternative hypothesis (H1): The sentiment analysis model's accuracy is significantly different from random classification.

# Check whether the t-statistic is larger than the critical value
if t_statistic > critical_value:
    print("Reject the null hypothesis. The sentiment analysis model's accuracy is statistically significantly higher than random classification.")
else:
    print("Failed to reject the null hypothesis. The sentiment analysis model's accuracy is not statistically significantly different from random classification.")