import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#load the data
data = pd.read_csv('heart.csv')

#understanding the data
data.info()
data.shape
data.describe()

#exploratory data analysis
sns.countplot(x="target", data=data)
sns.countplot(x='sex', data=data)

#visualizing corelations
corr_matrix = data.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu")

#split data into training and test sets
from sklearn.model_selection import train_test_split

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training the model
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

#evaluating the model
y_preds = clf.predict(X_test)

#calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_preds)
print("Accuracy:", accuracy)

#loading the dataset
df = pd.read_csv('heart.csv')

#exploring the dataset
df.head()

#data visualization
sns.countplot(x="target", data=df, palette="bwr")
plt.show()

#creating a correlation matrix
corr_matrix = df.corr()
top_corr_features = corr_matrix.index
plt.figure(figsize=(10,10))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#data splitting
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

#model training
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)

#model evaluation
from sklearn.metrics import accuracy_score
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

# This code reads the heart disease csv file
heart_disease_df = pd.read_csv('heart.csv')

# This code gets the average age of the patients
age_mean = heart_disease_df['age'].mean()

# This code creates a graph to show the age distribution of the patients
heart_disease_df.hist(column='age', bins=10, range=[30, 80], grid=False, figsize=(10,8))
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Frequency')

# This code adds a line to show the average age of the patients
plt.axvline(age_mean, color='red', label = 'Average age')
plt.legend()

plt.show()

