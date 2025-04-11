# Supervised-ML-Classification-Support Vector Machine

**SVM (Support Vector Machine)** is a supervised machine learning algorithm used primarily for classification tasks (but also for regression). 

The main idea behind SVM is to find the best decision boundary (or hyperplane) that separates classes of data.

🏞️ **Imagine This...**

Say you have two types of data points: red dots and blue triangles.

You want to draw a line (in 2D) or a plane (in 3D), or more generally a hyperplane, that separates the two groups as well as possible.

But SVM doesn’t just draw any line — it draws the one that maximizes the margin between the two groups.

📏 **Margin?**

The margin is the distance between the hyperplane and the nearest points from each class (called support vectors).

SVM finds the line that gives you the widest possible margin. This helps improve the model’s ability to generalize well on unseen data.

**Sklearn Pipeline**

A pipeline in scikit-learn is a way to chain multiple steps — like preprocessing, feature selection, and modeling — into a single object. 

It ensures that everything happens in the correct order, and makes your workflow cleaner and less error-prone.

Example:

  from sklearn.pipeline import Pipeline
  
  from sklearn.preprocessing import StandardScaler
  
  from sklearn.svm import SVC
  
  from sklearn.datasets import load_iris
  
  from sklearn.model_selection import train_test_split
  
  from sklearn.metrics import accuracy_score
  
  X, y = load_iris(return_X_y=True)
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  
  pipeline = Pipeline([
  
    ('scaler', StandardScaler()),
    
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
    
  ])
  
  pipeline.fit(X_train, y_train)
  
  y_pred = pipeline.predict(X_test)

  print("Accuracy:", accuracy_score(y_test, y_pred))
