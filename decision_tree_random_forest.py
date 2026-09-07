import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data=load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
cv=StratifiedKFold(5,shuffle=True,random_state=42)

depths=range(1,11); train=[]; test=[]; cv_scores=[]
for depth in depths:
    model=DecisionTreeClassifier(max_depth=depth,random_state=42)
    model.fit(X_train,y_train)
    train.append(model.score(X_train,y_train))
    test.append(model.score(X_test,y_test))
    cv_scores.append(cross_val_score(model,X_train,y_train,cv=cv).mean())

best_depth=list(depths)[cv_scores.index(max(cv_scores))]
tree=DecisionTreeClassifier(max_depth=best_depth,random_state=42)
tree.fit(X_train,y_train)
print("Decision Tree accuracy:",accuracy_score(y_test,tree.predict(X_test)))
print("Best depth:",best_depth)

plt.figure(figsize=(22,11))
plot_tree(tree,feature_names=data.feature_names,class_names=data.target_names,filled=True,rounded=True,fontsize=6)
plt.tight_layout(); plt.savefig("decision_tree.png",dpi=140); plt.show()

plt.figure(figsize=(9,6))
plt.plot(list(depths),train,marker="o",label="Training")
plt.plot(list(depths),test,marker="o",label="Test")
plt.plot(list(depths),cv_scores,marker="o",label="5-Fold CV")
plt.xlabel("Maximum Tree Depth"); plt.ylabel("Accuracy")
plt.title("Decision Tree Depth and Overfitting"); plt.legend()
plt.tight_layout(); plt.savefig("overfitting_analysis.png",dpi=140); plt.show()

forest=RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)
forest.fit(X_train,y_train)
print("Random Forest accuracy:",accuracy_score(y_test,forest.predict(X_test)))
print("Random Forest CV accuracy:",cross_val_score(forest,X_train,y_train,cv=cv).mean())

importance=pd.Series(forest.feature_importances_,index=data.feature_names).sort_values(ascending=False)
print("\nTop feature importances:")
print(importance.head(10))

plt.figure(figsize=(10,7))
importance.head(15).sort_values().plot(kind="barh")
plt.xlabel("Importance"); plt.title("Random Forest Feature Importance")
plt.tight_layout(); plt.savefig("feature_importance.png",dpi=140); plt.show()
