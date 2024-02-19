# Yazar: Habib Şako
# Oluşturulma Tarihi: 17/02/2024
# Açıklama: Bu kod, YZUP kapsamında 1. Modül projesidir. Verilen dataset ile data analizi, data ön işlemleri, model gerçeklemesi, model analizi ve streamlit entegrasyonu yapılması
# Dataset: Breast Cancer Wisconsin (Diagnostic) Data Set (Kaggle) https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download
# Öneri ve iletişim: habibsako@outlook.com veya habibsako@proton.me mail adreslerinden bana ulaşabilirisiniz.

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class App:
    def __init__(self):
        self.Streamlit_Page()
        self.clf = None
        self.X, self.y = None, None
        self.data = None
        self.model = None

    def run(self):
        self.get_dataset()
        self.generate()

    def Streamlit_Page(self):
        st.title("Habib Şako YZUP Proje 1")
        st.write("YZUP programı 1. Modül Projesidir")


    def get_dataset(self):
        dataset = st.sidebar.file_uploader("İşlem yapmak istediğiniz datasetini seçiniz", type=['csv'])
        if dataset is not None:
            # ilk satır ve dataframe tan.
            st.title("Görev 1")
            data_cancer = pd.read_csv(dataset)
            self.data = pd.DataFrame(data_cancer)
            st.write(self.data.head(10))

            # model seçimi
            self.model = st.sidebar.selectbox('Yöntem Seçiniz', ('KNN', 'SVM', 'Naive Bayes'))

        else:
            st.write("Veri yüklenmedi. Lütfen veri seti seçin ve yükleyin.")
    def generate(self):
        if self.data is not None:
            st.title("Görev 2")

            # NaN değerlerinin bulunduğu sutünları tespit et
            eksik_sutunlar = self.data.columns[self.data.isnull().any()]

            # id sutünunu ve NaN değerleri bulunan sutünları sil
            self.data.drop(eksik_sutunlar, axis = 1, inplace = True)
            self.data.drop(columns = ['id'], inplace = True)

            # diagnosis sutünunun 1 ve 0 olarak günceleme
            self.data['diagnosis'] = self.data['diagnosis'].map({'M': 1, 'B': 0})

            # Son 10 satırı göster
            st.write(self.data.tail(10))

            # X ve y verilerini tanımlama
            self.X = self.data.drop(columns = ['diagnosis'])
            self.y = self.data['diagnosis']

            # kolerasyon matrisinin çizdirilmesi
            st.title('Kolerasyon Matrisi')
            corr_matrix = self.X.corr()
            plt.figure(figsize=(15,15))
            sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = ".2f")
            plt.title('Kolerasyon Matrisi')
            fig= plt.show()
            st.pyplot(fig)

            # Maligant ve Benign
            # dataların ayrılması
            st.title('Maligant ve Benign')
            maligant_data = self.data[self.data['diagnosis'] == 1]
            benign_data = self.data[self.data['diagnosis'] == 0]

            # ekrana çizdirme
            plt.scatter(maligant_data['radius_mean'], maligant_data['texture_mean'], color='red', label='Maligant')
            plt.scatter(benign_data['radius_mean'], benign_data['texture_mean'], color='green', label='Benign')
            plt.title('Maligant ve Benign')
            plt.xlabel('radius_mean')
            plt.ylabel('texture_mean')
            plt.legend()
            fig= plt.show()
            st.pyplot(fig)

            # veriyi yüzde 80-20 olarak ayırma
            X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=44)

            self.method(X_train, X_test, Y_train, Y_test)


    def method(self, X_train, X_test, Y_train, Y_test ):
        st.title("Görev 3")
        if self.model == "KNN":
            # eğitim
            knn= KNeighborsClassifier()
            param_grid = {'n_neighbors': [3,5,7]}
            grid_search = GridSearchCV(knn, param_grid, cv=5)
            grid_search.fit(X_train, Y_train)
            best_params = grid_search.best_params_
            knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'])
            knn.fit(X_train, Y_train)
            Y_pred = knn.predict(X_test)

            # en iyi params yazdırma
            st.write("En iyi parametre değerleri:", best_params)

        elif self.model == "SVM":
            # eğitim
            svm = SVC()
            param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]}
            grid_search = GridSearchCV(svm, param_grid, cv=5)
            grid_search.fit(X_train, Y_train)
            best_params = grid_search.best_params_
            svm = SVC(C = best_params['C'], gamma = best_params['gamma'])
            svm.fit(X_train, Y_train)
            Y_pred = svm.predict(X_test)

            # en iyi params yazdırma
            st.write("En iyi parametreler:", best_params)

        else:       #navie bayes
            navie_bayes=GaussianNB()
            navie_bayes.fit(X_train,Y_train)
            Y_pred = navie_bayes.predict(X_test)

        # accuracy, precision, recall, f1, confusion matrixs değerlerini bulma
        accuracy = accuracy_score(Y_test, Y_pred)
        precision = precision_score(Y_test, Y_pred)
        recall = recall_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred)
        conf_matrix = confusion_matrix(Y_test, Y_pred)

        # Değerleri arayüzde Gösterme
        st.title("Görev 4")

        # diagnosis verilerini eski değerlerine döndürme
        pred = pd.Series(Y_pred).map({1 :'M', 0 : 'B'})
        test = pd.Series(Y_test).map({1 :'M', 0 : 'B'})


        st.write("Tahmin: M değeri 1, B değeri : 0 ", pred)
        st.write("Gerçek Sonuçlar: M değeri 1, B değeri : 0", test )
        st.write("Accuracy Değeri:", accuracy)
        st.write("Precision Değeri:", precision)
        st.write("Recall Değeri:", recall)
        st.write("F1 score Değeri:", f1)
        st.write("Confusion Matrix:")

        # Confusion matrix'i görselleştirme
        plt.figure(figsize=(8, 6))
        labels = ["0", "1"]
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Y_pred')
        plt.ylabel('Y_test')
        st.pyplot(plt)



