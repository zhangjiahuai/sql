from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, SimpleRNN, Activation
from keras import layers
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import seaborn as sns
filename = './data/Modified_SQL_Dataset.csv'
df = pd.read_csv(filename, encoding='utf-8')
vec = CountVectorizer(min_df=2, max_df=0.8, stop_words=stopwords.words('english'))

posts = vec.fit_transform(df['Query'].values.astype('U')).toarray()

trans = pd.DataFrame(posts)
df = pd.concat([df, trans], axis=1)
X = df[df.columns[2:]]
y = df['Label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
input_dim = X_train.shape[1]
def test():
    sns.set()
    model = Sequential()

    # Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
    # 在第一层必须指定所期望的输入数据尺寸：
    model.add(layers.Dense(64, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(32, activation='tanh'))
    model.add(layers.Dense(10, activation='tanh'))
    model.add(layers.Dense(1024, activation='relu'))

    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train,epochs=15,verbose=True,validation_data=(X_test, y_test),batch_size=10)
    y_pred_rate = model.predict(X_test)
    y_pred = y_pred_rate

    matrix_nn = confusion_matrix(y_test,y_pred)
    TN_nn, FP_nn, FN_nn, TP_nn = confusion_matrix(y_test,y_pred).ravel()
    acc_nn = accuracy_score(y_test,y_pred)
    pre_nn = TP_nn / (TP_nn+FP_nn)
    recall_nn = metrics.recall_score(y_test,y_pred)
    f1_nn = metrics.f1_score(y_test,y_pred)
    xtick = ['0','1']
    ytick = ['0','1']

    plt.figure(1)
    plt.title('NN')
    sns.heatmap(matrix_nn,fmt ='g',cmap='YlGnBu',annot=True,xticklabels=xtick,yticklabels=ytick)

    print('NN:')
    print(acc_nn,pre_nn,recall_nn,f1_nn)
    print(matrix_nn)
    # TPR_nn = TP_nn / (TP_nn + FN_nn)
    # FPR_nn = FP_nn / (TN_nn + FP_nn)
    # roc_nn = auc(FPR_nn,TPR_nn)
    # print(roc_nn)
    # fpr_nn, tpr_nn, threshold_nn = roc_curve(y_test, y_pred_rate)  ###计算真正率和假正率
    # roc_nn = auc(fpr_nn,tpr_nn)
    # print(roc_nn)
    #
    # plt.figure()
    # lw = 2
    # plt.figure()
    # plt.plot(fpr_nn, tpr_nn, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_nn)  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()


    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train,y_train)
    y_pred_knn = knn.predict(X_test)
    matrix_knn = confusion_matrix(y_test,y_pred_knn)
    TN_knn, FP_knn, FN_knn, TP_knn = confusion_matrix(y_test, y_pred_knn).ravel()
    print('KNN:')

    acc_knn = metrics.accuracy_score(y_test,y_pred_knn)
    pre_knn = metrics.precision_score(y_test,y_pred_knn)
    recall_knn = metrics.recall_score(y_test,y_pred_knn)
    f1_knn = metrics.f1_score(y_test,y_pred_knn)
    plt.figure(2)
    plt.title('knn')
    sns.heatmap(matrix_knn, fmt='g', cmap='YlGnBu', annot=True, xticklabels=xtick, yticklabels=ytick)

    print(acc_knn, pre_knn, recall_knn, f1_knn)
    print(matrix_knn)


    # svm = SVC(gamma='auto')
    # svm.fit(X_train,y_train)
    # y_pred_svm = svm.predict(X_test)
    # matrix_svm = confusion_matrix(y_test, y_pred_svm)
    # TN_svm, FP_svm, FN_svm, TP_svm = confusion_matrix(y_test, y_pred_svm).ravel()
    # print(matrix_svm)

    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(X_train,y_train)
    y_pred_dt = dt.predict(X_test)
    matrix_dt = confusion_matrix(y_test, y_pred_dt)
    TN_dt, FP_dt, FN_dt, TP_dt = confusion_matrix(y_test, y_pred_dt).ravel()
    acc_dt = metrics.accuracy_score(y_test, y_pred_dt)
    pre_dt = metrics.precision_score(y_test, y_pred_dt)
    recall_dt = metrics.recall_score(y_test, y_pred_dt)
    f1_dt = metrics.f1_score(y_test, y_pred_dt)
    plt.figure(3)
    plt.title('DT')
    sns.heatmap(matrix_dt, fmt='g', cmap='YlGnBu', annot=True, xticklabels=xtick, yticklabels=ytick)

    plt.show()
    print(acc_dt, pre_dt, recall_dt, f1_dt)
    print(matrix_dt)

def lstm():

    max_features = 30000
    model = Sequential()
    model.add(Embedding(max_features,32))
    # model.add(layers.Dense(20, input_dim=input_dim, activation='relu'))
    model.add(LSTM(32))
    model.add(Activation('relu'))
    # model.add(Dense(1024, activation='relu'))
    #
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X_train, y_train,epochs=15,verbose=True,validation_data=(X_test, y_test),batch_size=10)
    model.summary()
    y_pred_rate = model.predict(X_test)
    y_pred = y_pred_rate

    matrix_lstm = confusion_matrix(y_test, y_pred)
    print(matrix_lstm)
    acc_nn = accuracy_score(y_test, y_pred)
    pre_nn = TP_nn / (TP_nn + FP_nn)
    recall_nn = metrics.recall_score(y_test, y_pred)
    f1_nn = metrics.f1_score(y_test, y_pred)
    print(acc_nn)
    print(pre_nn)
    print(recall_nn)
    print(f1_nn)

if __name__ == '__main__':
    test()
    lstm()



