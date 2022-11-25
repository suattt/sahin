# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:29:30 2022

@author: User
"""

import streamlit as st
import pandas as pd
import numpy as np


def main():
    st.title("ÖĞRENCİ AKADEMİK BAŞARI TAHMİN SİSTEMİ/STUDENT ACADEMIC SUCCESS PREDICTION SYSTEM")
    with st.beta_expander ("Sistem Açıklaması"):
        st.write("""Bu sistem, İstanbul üniversitesi Enformatik Bölümü tarafından yürütülen “Makine Öğrenmesi Yöntemleri İle Ortaokul Öğrenci Başarılarının Tespiti ve Bir Uygulama” adlı yüksek lisans tez çalışması kapsamında oluşturulmuştur. Tez çalışmasının amacı; ortaokul öğrencilerine ait sosyoekonomik, demografik özellikleri ve ders notları verisi kullanılarak sene sonu ağırlıklı not ortalamalarının makine öğrenmesi yöntemleri ile tahmin edilmesi ve başarıya en fazla etki eden niteliklerin belirlenmesidir. \n 
Bu kapsamda İstanbul ilindeki bir ortaokul öğrencilerine ait 24 bağımsız değişken kullanılarak bağımlı değişken sınıflandırma yöntemi ile tahmin edilmeye çalışılmıştır. Bağımlı değişkenin tahmini için özellik seçimi yöntemleri uygulanarak akademik başarıya en fazla etki eden öznitelikler seçilmiştir. Modelleme 7 farklı algoritma ile yapılmış olup en yüksek başarım oranlarını sağlayan Rastgele Orman modeli ile mevcut sistem oluşturulmuştur. \n
Sistemin işleyişi; formda bulunan öznitelik değerleri girilerek tahmin butonuna basıldığında özellikleri girilen öğrenciye ait sene sonu ağırlıklı not ortalaması tahmininin yapılması şeklindedir. Bağımlı değişkenin sınıf değerleri, Milli Eğitim Bakanlığı’nın not dönüşüm çizelgesine göre 100’lük not sisteminden 5’lik not sistemine dönüştürülmüş şeklidir. Not dönüşüm tablosu aşağıdaki gibidir.
""")
        st.write("""PUAN/POINT ----- NOT/GRADE""")
        st.write("""00-24 ---------------- 0""")
        st.write("""25-44 ---------------- 1""")
        st.write("""45-54 ---------------- 2""")
        st.write("""55-69 ---------------- 3""")
        st.write("""70-84 ---------------- 4""")
        st.write("""85-100 -------------- 5""")
    with st.beta_expander ("System Explanation"):
        st.write("""This system was created within the scope of the master's thesis named "Determination of Secondary School Students Achievement with Machine Learning Methods and An Application" conducted by the Department of Informatics at Istanbul University. The purpose of the thesis study; It is the prediction of the year-end weighted grade point averages using machine learning methods using the socioeconomic, demographic characteristics and course grades data of secondary school students and the determination of the qualities that affect the success the most.\n
In this context, using 24 independent variables belonging to a secondary school students in Istanbul province, the dependent variable was predicted by classification method. For the prediction of the dependent variable, feature selection methods were applied, and the features that had the most impact on academic achievement were selected. Modeling was done with 7 different algorithms and the existing system was created with the Random Forest model that provides the highest performance rates.\n
The functioning of the system; By entering the attribute values in the form and clicking the prediction button, it is to estimate the year-end weighted grade average of the student whose properties are entered. The class values of the dependent variable are transformed from a 100 grade system to a 5 grade system according to the grade conversion chart of the Ministry of Education. Note conversion table is as below.""")
        st.write("""PUAN/POINT ----- NOT/GRADE""")
        st.write("""00-24 ---------------- 0""")
        st.write("""25-44 ---------------- 1""")
        st.write("""45-54 ---------------- 2""")
        st.write("""55-69 ---------------- 3""")
        st.write("""70-84 ---------------- 4""")
        st.write("""85-100 -------------- 5""")
 
    
if __name__ == '__main__':
    main()

st.write("")
st.write("")
st.write("AKADEMİK BAŞARI TAHMİN FORMU/ACADEMİC SUCCESS PREDİCTİON FORM")
st.write("")

Sex_dict = {'female':0,'male':1}
Embarked_dict = {'C':0,'S':1,'Q':2}


PClass=st.number_input("Bilet sınıfı",1,3)
Sex=st.selectbox("Sex/Cinsiyet", tuple(Sex_dict.keys()))
Age=st.number_input("Age/Yaş",0,100)
SibSp=st.number_input("Titanic'deki kardeşsayısı", 0,10)
Parch=st.number_input("Ebeveyn çocuk sayısı",0,10)
Fare=st.number_input("Ücret",0.00,500.00)
Embarked=st.selectbox("Biniş limanı",tuple(Embarked_dict.keys()))



Sex=Sex_dict.get(Sex)
Embarked=Embarked_dict.get(Embarked)





res = pd.DataFrame(data =
        {'PClass':[PClass],'Sex':[Sex],'Age':[Age],
         'SibSp':[SibSp],'Parch':[Parch],'Fare':[Fare],
          'Embarked':[Embarked]
          })

import pickle
with open('RandomForestModel.pkl', 'rb') as f:
    model = pickle.load(f)
"""    
from joblib import dump, load
model=load('RandomForestModel.joblib')
"""
prediction = model.predict(res)
prediction = str(prediction).strip('[]')

if st.button('Tahmin/Predict'):
    st.write("Random Forest Modeli Tahmini/Random Forest Model Prediction: ",prediction)
    












    










    








    