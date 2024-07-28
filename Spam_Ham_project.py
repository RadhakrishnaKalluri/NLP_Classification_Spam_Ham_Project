
import pandas as pd
import matplotlib.pyplot as plt
data= pd.read_csv(r"\Users\rkkal\Downloads\NLP_Spam-Ham_Project\spam_ham_data.csv")
print(data['Category'])
print()

#To check null values in dataset
print('The null values count in dataset')
print(data.isnull().sum())
print()

# Check the size of each category
category_counts = data['Category'].value_counts()
# Print the size of each category
print("Category counts:\n", category_counts)
plt.figure(figsize=(8,6))
category_counts.plot(kind='bar',color=['blue','orange'])
plt.title('Target variable')
plt.xlabel('category')
plt.ylabel('Count')
plt.show()

#Need to change the Category column into numerical value ie.0,1
data.replace('ham', 0, inplace=True)
data.replace('spam', 1, inplace=True)
data['Category'] = data['Category'].astype('int8')

#print(data)

#Data Preprocessing 

import spacy
# Load the spaCy model
nlp = spacy.load('en_core_web_sm')
# Define a function to tokenize, remove stop words, and lemmatize the message using spaCy
def preprocess_message(message):
    doc = nlp(message)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens

data['preprocessed_message'] = data['Message'].apply(preprocess_message)
# Join the preprocessed messages back into strings for vectorization
data['preprocessed_message_str'] = data['preprocessed_message'].apply(lambda x: ' '.join(x))
print(data['preprocessed_message_str'])


#Now we need to apply Vectorization with sklearn vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['preprocessed_message_str'])
vectorizer.get_feature_names_out()
print(X.toarray())

Y = data['Category']
print(Y)


'Model devolapment'


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Support Vector Machine
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, svm_preds))
print("SVM Accuracy:", accuracy_score(y_test, svm_preds))

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test,svm_preds)
print('confusion matric values are')
print(cm)
sns.heatmap(cm, 
            annot=True,
            fmt='g', 
            xticklabels=['spam','Ham'],
            yticklabels=['spam','Ham'])
plt.xlabel('Prediction',fontsize=13)
plt.ylabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()


from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation scores
cv_scores = cross_val_score(svm_model, X, Y, cv=skf, scoring='accuracy')
print("Cross-validation accuracy scores:", cv_scores)
print("Mean cross-validation accuracy:", cv_scores.mean())

# Predict probabilities
y_pred_proba = svm_model.decision_function(X_test)

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC Score:", roc_auc)