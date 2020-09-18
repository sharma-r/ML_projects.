# gender classification using height, weight, shoe size..
from sklearn import tree
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#[height, weight, shoe size]
X_train=[[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47],
[175, 64, 39], [177, 70, 40], [159, 60, 39], [171, 75, 42], [181, 85, 43]]

#labels
Y_train=['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']

X_test=[
        [188, 70, 38],[189, 94, 34],[180, 70, 42],[159, 61, 37],[165, 58, 39],
        [162, 54, 34],[171, 60, 40],[170, 70, 40],[143, 45, 37],[153, 48, 39], 
        [154, 60, 39]]

Y_test=['male','male','male','female','male',
            'female','male','male','female','female','female']


#initializing the classifier
clf=tree.DecisionTreeClassifier()
#training the dataset using fit method
clf=clf.fit(X_train, Y_train)
#predicting result
prediction =clf.predict(X_test);
  
#print (prediction)
acc=accuracy_score(Y_test,prediction)
print ("DecisionTree: ",acc*100)

#challenge
#logistic regression-1
clf1=LogisticRegression();
clf1=clf1.fit(X_train,Y_train)

pred1=clf1.predict(X_test)
acc1=accuracy_score(Y_test,pred1)
print ("LogisticRegression:  ",acc1*100)

#SVM-2
clf2=SVC();
clf2=clf2.fit(X_train,Y_train)

pred2=clf2.predict(X_test)
acc2=accuracy_score(Y_test,pred2)
print ("SVM: ", acc2*100)

#KNeighborsClassification-3
clf3=KNeighborsClassifier();
clf3=clf3.fit(X_train,Y_train)

pred3=clf3.predict(X_test)
acc3=accuracy_score(Y_test,pred3)
print ("KNeighbours: ",acc3*100)

most_acc=max(acc,acc1,acc2,acc3)
#print most_acc

if most_acc==acc:
  print "DecisionTree is most accurate";

if most_acc==acc1:
  print "LogisticRegression is most accurate";

if most_acc==acc2:
  print "SVM is most accurate";

if most_acc==acc3:
  print "KNeighbours is most accurate";






























