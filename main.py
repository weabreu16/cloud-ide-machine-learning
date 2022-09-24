import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("diabetes.csv")
print("Head of Data")
print(df.head())

# Get Train and Test Data for our ML
X = df.drop(columns = ['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Transforms the data in such a manner that it has mean as 0 and standard deviation as 1
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model that estimates the probability of an event occurring
lr_model = LogisticRegression(solver='liblinear')
lr_model.fit(X_train, y_train)

y_pred_train = lr_model.predict(X_train)
y_pred_test = lr_model.predict(X_test)

print('Logistic Regression Predict Train Performance')
model_confusion_matrix = confusion_matrix(y_train, y_pred_train)
print(classification_report(y_train, y_pred_train))

# Confussion Matrix of Logistic Regression Model as Graph
class_names=[0,1] 
fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(model_confusion_matrix), annot=True, fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Matriz de Confusión del Modelo Logistic Regression', y=1.1)
plt.ylabel('Variable')
plt.xlabel('Etiqueta de predicción')
plt.savefig('confusion_matrix.png')

result = lr_model.predict([
  [1, 148, 64, 35, 0, 33.6, 0.627, 50]
])

print( "Has Diabetes: ", True if result[0] == 1 else False )