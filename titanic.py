import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Basic exploration
print(train_data.head())
print(train_data.isnull().sum())

# Helper function for preprocessing
def preprocess(df, is_train=True):
    # Fill missing Age with median
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Fill missing Embarked with 'S'
    df['Embarked'] = df['Embarked'].fillna('S')

    # Drop Cabin (could also consider HasCabin = notnull)
    df = df.drop(columns=['Cabin'])

    # Extract Title from Name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # Convert Sex to numerical
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # Create FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch']

    # Drop unused columns
    df = df.drop(columns=['Ticket', 'Name'])

    return df

# Apply preprocessing
train_data = preprocess(train_data)
test_data = preprocess(test_data, is_train=False)

# Label Encoding for categorical features
le_embarked = LabelEncoder()
le_title = LabelEncoder()

train_data['Embarked'] = le_embarked.fit_transform(train_data['Embarked'])
test_data['Embarked'] = le_embarked.transform(test_data['Embarked'])

# Combine both title columns before fitting
all_titles = pd.concat([train_data['Title'], test_data['Title']])
le_title.fit(all_titles)

# Now transform
train_data['Title'] = le_title.transform(train_data['Title'])
test_data['Title'] = le_title.transform(test_data['Title'])

# Feature selection
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'Title']
X_train = train_data[features]
y_train = train_data['Survived']
X_test = test_data[features]

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Training accuracy
y_pred = model.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, y_pred))

# Prediction
test_predictions = model.predict(X_test)

# Submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('titanic_submission.csv', index=False)
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-validation accuracy: {:.4f} ± {:.4f}".format(cv_scores.mean(), cv_scores.std()))