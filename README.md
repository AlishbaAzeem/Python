# Titanic Survival Classification using Python 

Importing Dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Data Collection and processing
import pandas as pd
# load the data from csv file to Pandas Dataframe
titanic_data = pd.read_csv('/content/train.csv')

# printing the first 5 rows of the dataframe
titanic_data.head()

# number of rows and columns
titanic_data.shape

# getting some information about data
titanic_data.info()

# check the number of missing value in each column
titanic_data.isnull().sum()

Handling the Missing Values
# drop the cabin column from data table
if 'Cabin' in titanic_data.columns:
    titanic_data = titanic_data.drop(columns='Cabin' , axis=1)

# replacing the missing values in "Age" column with mean value
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())

# finding the mode value of "Embarked" column
print(titanic_data['Embarked'].mode())
print(titanic_data['Embarked'].mode()[0])

# replacing the missing values in "Embarked" column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# check the number of missing value in each column
titanic_data.isnull().sum()

Data Analysis

# getting some statistical measures about the data
titanic_data.describe()

# finding the number of people survived (1) and not survived (0)
titanic_data['Survived'].value_counts()

Data Visualization

sns.set()
# making a count plot for "Survived" column
sns.countplot(x='Survived',data=titanic_data)
titanic_data['Sex'].value_counts()

# making a count plot for "Sex" column
sns.countplot(x='Sex',data=titanic_data)

# number of survivors Genderwise
sns.countplot(x='Sex', hue='Survived', data=titanic_data)

# making a count plot for "Pclass" column
sns.countplot(x='Pclass',data=titanic_data)
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)

Encoding the Categorical Columns

titanic_data['Sex'].value_counts()
titanic_data['Embarked'].value_counts()

# converting categorical columns
titanic_data['Sex'] = titanic_data['Sex'].replace({'male':0,'female':1})
titanic_data['Embarked'] = titanic_data['Embarked'].replace({'S':0,'C':1,'Q':2})
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0]).astype(int)

titanic_data.head()

Separating features & Target

X = titanic_data.drop(columns=['PassengerId','Name','Ticket','Survived', 'Cabin'],axis=1)
Y = titanic_data['Survived']

print (X)
print (Y)

Splitting the data into training data and test data

from sklearn.model_selection import train_test_split

# Ensure 'Cabin' is dropped before splitting
if 'Cabin' in X.columns:
    X = X.drop(columns='Cabin', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

Model Training
[ ]

Start coding or generate with AI.
Importing Dependencies


[ ]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
Data Collection and processing


[ ]
import pandas as pd

# load the data from csv file to Pandas Dataframe
titanic_data = pd.read_csv('/content/train.csv')

[ ]
# printing the first 5 rows of the dataframe
titanic_data.head()


[ ]
# number of rows and columns
titanic_data.shape
(891, 12)

[ ]
# getting some information about data
titanic_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB

[ ]
# check the number of missing value in each column
titanic_data.isnull().sum()

Handling the Missing Values


[ ]
# drop the cabin column from data table
if 'Cabin' in titanic_data.columns:
    titanic_data = titanic_data.drop(columns='Cabin' , axis=1)

[ ]
# replacing the missing values in "Age" column with mean value
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())

[ ]
# finding the mode value of "Embarked" column
print(titanic_data['Embarked'].mode())
0    S
Name: Embarked, dtype: object

[ ]
print(titanic_data['Embarked'].mode()[0])
0

[ ]
# replacing the missing values in "Embarked" column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
/tmp/ipython-input-1194414775.py:2: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

[ ]
# check the number of missing value in each column
titanic_data.isnull().sum()

Data Analysis


[ ]
# getting some statistical measures about the data
titanic_data.describe()


[ ]
# finding the number of people survived (1) and not survived (0)
titanic_data['Survived'].value_counts()

Data Visualization


[ ]
sns.set()

[ ]
# making a count plot for "Survived" column
sns.countplot(x='Survived',data=titanic_data)


[ ]
titanic_data['Sex'].value_counts()


[ ]
# making a count plot for "Sex" column
sns.countplot(x='Sex',data=titanic_data)


[ ]
# number of survivors Genderwise
sns.countplot(x='Sex', hue='Survived', data=titanic_data)


[ ]
# making a count plot for "Pclass" column
sns.countplot(x='Pclass',data=titanic_data)


[ ]
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)

Encoding the Categorical Columns


[ ]
titanic_data['Sex'].value_counts()


[ ]
titanic_data['Embarked'].value_counts()


[1]
0s
# converting categorical columns
titanic_data['Sex'] = titanic_data['Sex'].replace({'male':0,'female':1})
titanic_data['Embarked'] = titanic_data['Embarked'].replace({'S':0,'C':1,'Q':2})
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0]).astype(int)

Next steps:

[2]
0s
titanic_data.head()

Next steps:
Separating features & Target


[ ]
X = titanic_data.drop(columns=['PassengerId','Name','Ticket','Survived', 'Cabin'],axis=1)
Y = titanic_data['Survived']

[ ]
print (X)
     Pclass  Sex   Age  SibSp  Parch     Fare Cabin  Embarked
0         3    0  22.0      1      0   7.2500   NaN       0.0
1         1    1  38.0      1      0  71.2833   C85       1.0
2         3    1  26.0      0      0   7.9250   NaN       0.0
3         1    1  35.0      1      0  53.1000  C123       0.0
4         3    0  35.0      0      0   8.0500   NaN       0.0
..      ...  ...   ...    ...    ...      ...   ...       ...
886       2    0  27.0      0      0  13.0000   NaN       0.0
887       1    1  19.0      0      0  30.0000   B42       0.0
888       3    1   NaN      1      2  23.4500   NaN       0.0
889       1    0  26.0      0      0  30.0000  C148       1.0
890       3    0  32.0      0      0   7.7500   NaN       2.0

[891 rows x 8 columns]

[ ]
print (Y)
0      0
1      1
2      1
3      1
4      0
      ..
886    0
887    1
888    0
889    1
890    0
Name: Survived, Length: 891, dtype: int64
Splitting the data into training data and test data


[ ]
from sklearn.model_selection import train_test_split

# Ensure 'Cabin' is dropped before splitting
if 'Cabin' in X.columns:
    X = X.drop(columns='Cabin', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

[ ]
print(X.shape, X_train.shape, X_test.shape)
(891, 7) (712, 8) (179, 8)

Model Training

Logistic Regression

model = LogisticRegression()
# training the Logistic Regression model with training data
model.fit(X_train, Y_train)

Model Evaluation
Accuracy Score

# accuracy on training data
X_train_prediction = model.predict(X_train)
print(X_train_prediction)

from sklearn.metrics import accuracy_score
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ',training_data_accuracy)

# accuracy on training data
X_test_prediction = model.predict(X_test)
print(X_test_prediction)

test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ',test_data_accuracy)






