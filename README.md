
# Mini Proje Yapay Zekada Uzmanlık Programı 1. Bölüm Ödevi

Bu projede, veriyi lokalden yükleyip Streamlit arayüzünde görüntülemek, veri temizleme ve ön işleme adımlarını gerçekleştirmek, çeşitli makine öğrenimi modellerini uygulamak, en iyi parametreleri Gridsearch ile bulmak ve model sonuçlarını değerlendirmek gibi adımları içeren bir veri analizi ve makine öğrenimi uygulaması yapılacaktır. İşte proje adımları:


## Görev 1
- Veri seti lokalden seçilip yüklenir.
- Streamlit arayüzünde sidebar ile Veri Seti ismi seçilebilir. Seçilen veri seti Python tarafında yüklenir.
- Ana ekranda yüklenen verinin ilk 10 satırı ve sütunları gösterilir.

## Görev 2
- Veri temizleme ve ön işleme adımları gerçekleştirilir.
- Gereksiz sütunlar temizlenir.
- Verinin son 10 satırı gösterilir.
- 'diagnosis' sütunundaki 'M' değeri 1, 'B' değeri 0 olacak şekilde değiştirilir. Bu sütun Y etiket verisi olarak kullanılır, geri kalan sütunlar ise X öznitelik verisi olarak kullanılır.
- Seaborn kütüphanesi ile korelasyon matrisi çizdirilir.
- Veri, 'Malignant' ve 'Benign' olacak şekilde ayrılır ve belirli özellikler üzerinde çizdirilir.
- Veri, %80 eğitim ve %20 test verisi olarak ayrılır.

## Görev 3
- Makine öğrenimi modeli implementasyonu gerçekleştirilir.
- Streamlit sidebar üzerinden KNN, SVM veya Naïve Bayes yöntemlerinden biri seçilerek model seçimi yapılır.
- Gridsearch yöntemiyle en iyi parametreler bulunur.
- Optimum parametrelerle model X_train ve Y_train verileri ile eğitilir.

## Görev 4
- Model sonuçları, X_test ve Y_test verileri için Streamlit arayüzünde gösterilir.
- Accuracy, precision, recall, f1-score ve confusion matrix gibi metrikler sunulur.

## Görev 5
- Streamlit entegrasyonu sağlanır.



## Bilgisayarınızda Çalıştırın

Gerekli paketleri yükleyin

```bash
  pip install streamlit
  pip install scikit-learn
  pip install matplotlib
  pip install seaborn
  pip install numpy
  pip install pandas
```

Sunucuyu çalıştırın

```bash
  python -m streamlit run main.py --server.runOnSave true

```


  - (browse files) butonuna tıklayarak localde bulunan dataseti yükleyebilirsiniz.
  - Datasetinizi import ettikten sonra KNN, SVM, Navie Bayes seçenekleri çıkacak ve dilediğinizi seçin.



  
