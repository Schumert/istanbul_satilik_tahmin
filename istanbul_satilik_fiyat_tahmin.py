# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 04:29:25 2024

@author: Ben
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split,learning_curve, cross_val_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
import seaborn as sns
import re
import joblib
import os
from datetime import datetime

# %% Veri Yükleme ve Temizleme
df = pd.read_csv("istanbul.csv")

# Fiyat sütunundaki virgülleri kaldır ve sadece sayısal değeri al
df['fiyat'] = df['fiyat'].str.replace(',', '').str.extract(r'(\d+)').astype(float)

# Tarih bilgileri (update_date) içinden ay-yıl ve yıl ayrıştırma
df['update_month_year'] = df['update_date'].str.extract(r'(\w+\s+\d{4})')
df['yil'] = df['update_date'].str.extract(r'(\d{4})')

label_enc_update = LabelEncoder()
df['update_month_year_encoded'] = label_enc_update.fit_transform(df['update_month_year'])

# m2_brut ve m2_net sütunlarından sadece sayısal değerleri al
df['m2_brut'] = df['m2_brut'].str.extract(r'(\d+)').astype(float)
df['m2_net'] = df['m2_net'].str.extract(r'(\d+)').astype(float)

# KULLANIM DURUMU Label Encoding
label_enc_kullanim = LabelEncoder()
df['kullanim_durumu_encoded'] = label_enc_kullanim.fit_transform(df['kullanim_durumu'])

# oda_sayisi sütununu “oda+salon” vb. formattan ayırıp toplam sayı haline getirme
df['oda_sayisi_toplam'] = df['oda_sayisi'].apply(lambda x: sum(map(int, re.findall(r'\d+', x))))

# bulundugu_kat Label Encoding
label_enc_bulundugu_kat = LabelEncoder()
df['bulundugu_kat_encoded'] = label_enc_bulundugu_kat.fit_transform(df['bulundugu_kat'])

# site içinde mi değil mi
label_enc_site = LabelEncoder()
df['site_encoded'] = label_enc_site.fit_transform(df['site'])

# district Label Encoding
label_enc_district = LabelEncoder()
df['district_encoded'] = label_enc_district.fit_transform(df['district'])

# Gereksiz sütunları düşür
df.drop(['Unnamed: 3', 'Unnamed: 0'], axis=1, inplace=True)

# Banyo sayısı: "Yok" ifadesi varsa 0, aksi halde rakamları topla
df['banyo_sayisi'] = df['banyo_sayisi'].apply(
    lambda x: sum(map(int, re.findall(r'\d+', x))) if "Yok" not in x else 0
)

# (İsteğe bağlı) Korelasyon incelemesi
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
print("\n--- KORELASYON MATRİSİ (Özeti) ---\n", corr_matrix['fiyat'].sort_values(ascending=False))

# %% Aykırı Değer Temizleme (IQR yöntemi)
Q1 = df['fiyat'].quantile(0.2)
Q3 = df['fiyat'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['fiyat'] >= lower_bound) & (df['fiyat'] <= upper_bound)]

# # Temizlenmiş fiyat dağılımının histogramı
# sns.histplot(df['fiyat'], kde=True)
# plt.title("Temizlenmiş Fiyat Dağılımı (Histogram)")
# plt.show()

# %% Model Hazırlığı
# Hedef: 'fiyat' yerine log(fiyat) kullanıyoruz
df['log_fiyat'] = np.log1p(df['fiyat'])

feature_cols = [
    "banyo_sayisi", "m2_net", "m2_brut",
    "oda_sayisi_toplam", "kullanim_durumu_encoded",
    "bulundugu_kat_encoded", "district_encoded",
    "kat_sayisi"
]

MAX_VALUES = {
    'm2_net': df['m2_net'].max(),
    'm2_brut': df['m2_brut'].max(),
    'banyo_sayisi': df['banyo_sayisi'].max(),
    'oda_sayisi': df['oda_sayisi_toplam'].max(),
    'kat_sayisi': df['kat_sayisi'].max()
    }

X = df[feature_cols]
y = df['log_fiyat']

# Eğitim/Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# # %% XGBoost Model ve GridSearch
# xgb_estimator = xgb.XGBRegressor(
#     objective='reg:squarederror',
#     random_state=42
# )

# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [1, 2, 3, 5],
#     'learning_rate': [0.01, 0.1],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0]
# }

# grid_search = GridSearchCV(
#     estimator=xgb_estimator,
#     param_grid=param_grid,
#     scoring='neg_root_mean_squared_error',  # RMSE odaklı
#     cv=5,
#     verbose=1,
#     n_jobs=-1
# )

# grid_search.fit(X_train, y_train)

# print("Best Params:", grid_search.best_params_)
# print("Best CV Score (RMSE):", -grid_search.best_score_)

# best_model = grid_search.best_estimator_


# #%% modeli kaydetme

# joblib.dump(best_model, 'istanbul_satilik_tahmin_xgbregressor.pkl')


#%% modeli yükleme
if os.path.exists('istanbul_satilik_tahmin_xgbregressor.pkl'):
    best_model = joblib.load('istanbul_satilik_tahmin_xgbregressor.pkl')
    print("Mevcut model başarıyla yüklendi!")
else:
    print("Model bulunamadı!")
    exit()

#%% Değerlendirme 



#Seçilen en iyi modeli test setine uygula
# best_model = grid_search.best_estimator_
y_pred_log = best_model.predict(X_test)

y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)



# #Model değerlendirme
# rmse = mean_squared_error(y_test, y_pred, squared=False)  # squared=False => RMSE
# r2 = r2_score(y_test_orig, y_pred)

# print("Test RMSE:", rmse)
# print("Test R2:", r2)

# # Ardından eğitim setini tahmin edelim
# y_pred_train = best_model.predict(X_train)

# # Eğitim RMSE ve R²
# rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
# r2_train = r2_score(y_train, y_pred_train)
# print("Train RMSE:", rmse_train)
# print("Train R²:", r2_train)



#%%

# Tahmin Fonksiyonu
def predict_price_():
    print("\nYeni bir daire için özellikleri girin: (Veriler 2021-2022 için)")
    try:
        m2_net = float(input("Net m²: "))
        m2_brut = float(input("Brüt m²: "))
        banyo_sayisi = int(input("Banyo sayısı: "))
        bulundugu_kat = input("Bulunduğu kat (örn: '1. Kat', '2. Kat', 'Müstakil'): ")
        kullanim_durumu = int(input(("Kullanım durumu (örn: '0:Boş', '1:Kiracı Var', '2:Mal Sahibi'): ")))
        oda_sayisi_toplam = int(input("Oda+Salon toplam: "))
        
        
        district = input("İlçe (örn: 'kadikoy', 'besiktkas', 'adalar'): ")
        kat_sayisi = int(input("Bina kat sayısı: "))
        
        # Girdileri encode etmek
        #kullanim_durumu_encoded = label_enc_kullanim.transform([kullanim_durumu])[0]
        kullanim_durumu_encoded = kullanim_durumu
        bulundugu_kat_encoded = label_enc_bulundugu_kat.transform([bulundugu_kat])[0]
        district_encoded = label_enc_district.transform([district])[0]
        
        # Feature vektörünü oluşturma
        feature_vector = np.array([[banyo_sayisi, m2_net, m2_brut, oda_sayisi_toplam,
                                      kullanim_durumu_encoded, bulundugu_kat_encoded,
                                      district_encoded, kat_sayisi]])
        
        # Tahmin yapma
        predicted_log_price = best_model.predict(feature_vector)[0]
        predicted_price = np.expm1(predicted_log_price)  # Log dönüşümünden geri çevir
        
        print(f"\nTahmin edilen fiyat: {predicted_price:,.2f} TL")
    
    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        
        
# predict_price_()

#%%
def predict_price(m2_net, m2_brut, banyo_sayisi, bulundugu_kat,
kullanim_durumu, oda_sayisi_toplam, district, kat_sayisi):
    # Tahmin almak istediğiniz verileri DataFrame formatında oluşturun
    bulundugu_kat_encoded = label_enc_bulundugu_kat.transform([bulundugu_kat])[0]
    district_encoded = label_enc_district.transform([district])[0]
    
    # input_data = pd.DataFrame({
    #     'm2_net': [m2_net],
    #     'm2_brut': [m2_brut],
    #     'banyo_sayisi': [banyo_sayisi],
    #     'bulundugu_kat': [bulundugu_kat_encoded],
    #     'kullanim_durumu': [kullanim_durumu],
    #     'oda_sayisi':[oda_sayisi_toplam],
    #     'district_encoded': [district_encoded],
    #     'kat_sayisi': [kat_sayisi]
    #     })
    
    # Eğittiğimiz en iyi model (best_model) ile tahmin al
    feature_vector = np.array([[banyo_sayisi, m2_net, m2_brut, oda_sayisi_toplam,
                                  kullanim_durumu, bulundugu_kat_encoded,
                                  district_encoded, kat_sayisi]])
    
    # Tahmin yapma
    predicted_log_price = best_model.predict(feature_vector)[0]
    predicted_price = np.expm1(predicted_log_price)  # Log dönüşümünden geri çevir
    
    # predicted_price bir array dönecektir, ilk (ve tek) elemanını döndürüyoruz
    return predicted_price

predict_price(60, 65, 1, "1. Kat", 0, 5, "bagcilar", 5)


#%% grafikle değerlendirme

# # Gerçek vs Tahmin Edilen Değerler
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=y_test, y=y_pred)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
# plt.xlabel('Gerçek Değerler')
# plt.ylabel('Tahmin Edilen Değerler (log_fiyat)')
# plt.title('Gerçek vs Tahmin Edilen Değerler')
# plt.show()

# # Artıkların (Residuals) Dağılımı
# residuals = y_test - y_pred

# plt.figure(figsize=(10, 6))
# sns.histplot(residuals, kde=True, bins=30)
# plt.xlabel('Artıklar (Residuals)')
# plt.ylabel('Frekans')
# plt.title('Artıkların Dağılımı')
# plt.show()

# # Özellik Önem Skoru
# importances = best_model.feature_importances_
# features = X.columns

# plt.figure(figsize=(10, 6))
# sns.barplot(x=importances, y=features)
# plt.xlabel('Önem Skoru')
# plt.ylabel('Özellikler')
# plt.title('Özellik Önem Skoru')
# plt.show()



# train_sizes, train_scores, valid_scores = learning_curve(
#     best_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
# )

# train_scores_mean = -np.mean(train_scores, axis=1)
# valid_scores_mean = -np.mean(valid_scores, axis=1)

# plt.figure(figsize=(10, 6))
# plt.plot(train_sizes, train_scores_mean, label='Eğitim Hatası')
# plt.plot(train_sizes, valid_scores_mean, label='Doğrulama Hatası')
# plt.xlabel('Eğitim Seti Büyüklüğü')
# plt.ylabel('RMSE')
# plt.title('Öğrenme Eğrisi')
# plt.legend()
# plt.show()

# #%% fiyatla diğer özelliklerin ilişkisi




# from matplotlib.ticker import FuncFormatter

# # Fiyat ekseni için özel formatlayıcı
# def price_formatter(x, pos):
#     return f'{int(x/1000)}K' if x < 1000000 else f'{int(x/1000000)}M'

# # y_pred'yi pandas Series'e dönüştürerek indeksleri hizalayalım
# y_pred_series = pd.Series(y_pred, index=X_test.index)

# # Rastgele 1000 veri seçelim
# X_test_sample = X_test.sample(n=1000, random_state=42)
# y_pred_sample = y_pred_series.loc[X_test_sample.index]

# # Grafik 1: Banyo Sayısı ile Fiyat
# plt.figure(figsize=(10, 6))
# plt.scatter(X_test_sample["banyo_sayisi"], y_pred_sample, alpha=0.7)
# plt.title('Banyo Sayısı ile Fiyat Arasındaki İlişki (Rastgele 1000 Veri Noktası)')
# plt.xlabel('Banyo Sayısı')
# plt.ylabel('Fiyat (TL)')
# plt.gca().yaxis.set_major_formatter(FuncFormatter(price_formatter))  # Fiyat eksenini biçimlendir
# plt.show()

# # Grafik 2: Brüt m² ile Fiyat
# plt.figure(figsize=(10, 6))
# plt.scatter(X_test_sample["m2_brut"], y_pred_sample, alpha=0.7)
# plt.title('Brüt m² ile Fiyat Arasındaki İlişki (Rastgele 1000 Veri Noktası)')
# plt.xlabel('Brüt m²')
# plt.ylabel('Fiyat (TL)')
# plt.gca().yaxis.set_major_formatter(FuncFormatter(price_formatter))  # Fiyat eksenini biçimlendir
# plt.show()

# # Grafik 3: Kat Sayısı ile Fiyat
# plt.figure(figsize=(10, 6))
# plt.scatter(X_test_sample["kat_sayisi"], y_pred_sample, alpha=0.7)
# plt.title('Kat Sayısı ile Fiyat Arasındaki İlişki (Rastgele 1000 Veri Noktası)')
# plt.xlabel('Kat Sayısı')
# plt.ylabel('Fiyat (TL)')
# plt.gca().yaxis.set_major_formatter(FuncFormatter(price_formatter))  # Fiyat eksenini biçimlendir
# plt.show()


# #%%
# # District ve district_encoded sütunlarından bir harita oluşturun
# district_mapping = dict(zip(df["district_encoded"], df["district"]))
# # X_test içindeki district_encoded değerlerini district isimlerine çevirin
# X_test["district_name"] = X_test["district_encoded"].map(district_mapping)

# # Rastgele 1000 veri seçelim
# X_test_sample = X_test.sample(n=1000, random_state=42)
# y_pred_sample = y_pred_series.loc[X_test_sample.index]

# # Grafik: District'e Göre Fiyat Dağılımı
# plt.figure(figsize=(12, 8))
# sns.scatterplot(
#     data=X_test_sample,
#     x="district_name",
#     y=y_pred_sample,
#     hue="district_name",
#     palette="tab10",
#     alpha=0.7,
#     s=100
# )
# plt.title('District (Bölge) ile Fiyat Arasındaki İlişki (Rastgele 1000 Veri Noktası)')
# plt.xlabel('District (Bölge)')
# plt.ylabel('Fiyat (TL)')
# plt.xticks(rotation=45)  # Bölge isimlerini eğimli göster
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend'i dışarı al
# plt.show()




#%% tkinter



# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
import sys
import ctypes

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / "design" / "build" / "assets" / "frame0"


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Dosya yolunu doğru şekilde ayarlama
def resource_path(relative_path):
    """ PyInstaller ile paketlendiğinde dosya yolunu düzeltir """
    try:
        # PyInstaller temp klasöründe çalışıyorsa
        base_path = sys._MEIPASS
    except AttributeError:
        # Geliştirme ortamında çalışıyorsa
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

try:
    awareness = ctypes.windll.shcore.SetProcessDpiAwarenessContext
    awareness(-3)  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
except AttributeError:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # Windows 8.1 ve üzeri
    except Exception as e:
        print(f"DPI ayarı yapılamadı: {e}")


window = Tk()
window.tk.call('tk', 'scaling', window.winfo_fpixels('1i') / 72.0)
window.geometry("600x400")
window.configure(bg = "#FFFFFF")
window.title("İstanbul Satılık Emlak Tahmin Yapay Zekası")

canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 400,
    width = 600,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)


canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    300.0,
    400.0,
    fill="#552122",
    outline="")

canvas.create_text(
    13.0,
    18.0,
    anchor="nw",
    text="m2 net",
    fill="#FFFFFF",
    font=("Inter BoldItalic", 14 * -1)
)
entry_image_1 = PhotoImage(file=resource_path("design/build/assets/frame0/entry_1.png"))
entry_image_2 = PhotoImage(file=resource_path("design/build/assets/frame0/entry_2.png"))
entry_image_3 = PhotoImage(file=resource_path("design/build/assets/frame0/entry_3.png"))
entry_image_4 = PhotoImage(file=resource_path("design/build/assets/frame0/entry_4.png"))
entry_image_5 = PhotoImage(file=resource_path("design/build/assets/frame0/entry_5.png"))
entry_image_6 = PhotoImage(file=resource_path("design/build/assets/frame0/entry_6.png"))
entry_image_7 = PhotoImage(file=resource_path("design/build/assets/frame0/entry_7.png"))
entry_image_8 = PhotoImage(file=resource_path("design/build/assets/frame0/entry_8.png"))
entry_image_9 = PhotoImage(file=resource_path("design/build/assets/frame0/entry_9.png"))
button_image_1 = PhotoImage(file=resource_path("design/build/assets/frame0/button_1.png"))




entry_bg_1 = canvas.create_image(
    71.5,
    59.0,
    image=entry_image_1
)
entry_1 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)

entry_1.place(
    x=13.0,
    y=46.0,
    width=117.0,
    height=24.0
)

canvas.create_text(
    171.0,
    97.0,
    anchor="nw",
    text="toplam oda sayısı",
    fill="#FFFFFF",
    font=("Inter BoldItalic", 14 * -1)
)

entry_bg_2 = canvas.create_image(
    229.5,
    138.0,
    image=entry_image_2
)
entry_2 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)

entry_2.place(
    x=171.0,
    y=125.0,
    width=117.0,
    height=24.0
)

canvas.create_text(
    171.0,
    167.0,
    anchor="nw",
    text="               ilçe \n(kadikoy, besiktas)",
    fill="#FFFFFF",
    font=("Inter BoldItalic", 13 * -1)
)


entry_bg_3 = canvas.create_image(
    229.5,
    216.0,
    image=entry_image_3
)
entry_3 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_3.place(
    x=171.0,
    y=203.0,
    width=117.0,
    height=24.0
)

canvas.create_text(
    171.0,
    252.0,
    anchor="nw",
    text="kat sayısı\n",
    fill="#FFFFFF",
    font=("Inter BoldItalic", 14 * -1)
)


entry_bg_4 = canvas.create_image(
    229.5,
    293.0,
    image=entry_image_4
)
entry_4 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_4.place(
    x=171.0,
    y=280.0,
    width=117.0,
    height=24.0
)

canvas.create_text(
    13.0,
    96.0,
    anchor="nw",
    text="m2 brüt",
    fill="#FFFFFF",
    font=("Inter BoldItalic", 14 * -1)
)


entry_bg_5 = canvas.create_image(
    71.5,
    137.0,
    image=entry_image_5
)
entry_5 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_5.place(
    x=13.0,
    y=124.0,
    width=117.0,
    height=24.0
)

canvas.create_text(
    13.0,
    174.0,
    anchor="nw",
    text="banyo sayısı\n",
    fill="#FFFFFF",
    font=("Inter BoldItalic", 14 * -1)
)


entry_bg_6 = canvas.create_image(
    71.5,
    215.0,
    image=entry_image_6
)
entry_6 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_6.place(
    x=13.0,
    y=202.0,
    width=117.0,
    height=24.0
)

canvas.create_text(
    13.0,
    246.0,
    anchor="nw",
    text="bulunduğu kat\n'1. Kat', '2. Kat', 'Müstakil')",
    fill="#FFFFFF",
    font=("Inter BoldItalic", 12 * -1)
)


entry_bg_7 = canvas.create_image(
    71.5,
    293.0,
    image=entry_image_7
)
entry_7 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_7.place(
    x=13.0,
    y=280.0,
    width=117.0,
    height=24.0
)

canvas.create_text(
    171.0,
    0.0,
    anchor="nw",
    text="kullanim durumu \n('0:Boş', '1:Kiracı Var', \n'2:Mal Sahibi')",
    fill="#FFFFFF",
    font=("Inter BoldItalic", 12 * -1)
)


entry_bg_8 = canvas.create_image(
    229.5,
    60.0,
    image=entry_image_8
)
entry_8 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_8.place(
    x=171.0,
    y=47.0,
    width=117.0,
    height=24.0
)

button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_1 clicked"),
    relief="flat"
)
button_1.place(
    x=93.0,
    y=336.0,
    width=113.0,
    height=41.0
)


entry_bg_9 = canvas.create_image(
    450.0,
    214.5,
    image=entry_image_9
)
entry_9 = Text(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_9.place(
    x=329.0,
    y=55.0,
    width=242.0,
    height=317.0
)

canvas.create_text(
    340.0,
    30.0,
    anchor="nw",
    text="İstanbul Satılık Daire Tahmini (2021, 2022)",
    fill="#000000",
    font=("Inter BoldItalic", 12 * -1)
)
entry_1.insert(0, 60)
entry_5.insert(0, 65)
entry_6.insert(0, 1)
entry_7.insert(0, "1. Kat")
entry_8.insert(0, 0)
entry_2.insert(0, 2)
entry_3.insert(0, "kadikoy")
entry_4.insert(0, 4)


# --- Tahmin ve Loglama Fonksiyonu ---
def calculate_and_log():
    try:
        # Girişlerden veri al 
        m2_net = float(entry_1.get()) if entry_1.get() else None
        m2_brut = float(entry_5.get()) if entry_5.get() else None
        banyo_sayisi = float(entry_6.get()) if entry_6.get() else None
        bulundugu_kat = (entry_7.get()) if entry_7.get() else None
        district = (entry_3.get()) if entry_3.get() else None
        
        kullanim_durumu = int(entry_8.get()) if entry_8.get().isdigit() else None
        oda_sayisi = int(entry_2.get()) if entry_2.get().isdigit() else None
        
        kat_sayisi = int(entry_4.get()) if entry_4.get().isdigit() else None
        
        # Maksimum değer kontrolü
        if m2_net > MAX_VALUES['m2_net']:
            raise ValueError(f"m2 Net değeri maksimum {MAX_VALUES['m2_net']} olabilir.")
        if m2_brut > MAX_VALUES['m2_brut']:
            raise ValueError(f"m2 Brüt değeri maksimum {MAX_VALUES['m2_brut']} olabilir.")
        if banyo_sayisi > MAX_VALUES['banyo_sayisi']:
            raise ValueError(f"Banyo yaşı maksimum {MAX_VALUES['banyo_sayisi']} olabilir.")
        if oda_sayisi > MAX_VALUES['oda_sayisi']:
            raise ValueError(f"Oda Sayısı maksimum {MAX_VALUES['oda_sayisi']} olabilir.")
        if kat_sayisi > MAX_VALUES['kat_sayisi']:
            raise ValueError(f"Kat sayısı maksimum {MAX_VALUES['kat_sayisi']} olabilir.")
        
        # Yeni tahmin hesapla
        price = predict_price(
            m2_net, m2_brut, banyo_sayisi, bulundugu_kat,
            kullanim_durumu, oda_sayisi, district, kat_sayisi
        )
        
        
        
        # Yeni tahmin sonucunu entry_9'a ekle
        log_entry = "Tahmin edilen fiyat:\n" +f"{price:,.2f} TL\n"
        
        entry_9.insert('end', log_entry)
        entry_9.see('end')  # Scroll otomatik olarak en alta kaydır
        
    except ValueError as ve:
        entry_9.insert('end', f"[HATA] {ve}\n")
        entry_9.see('end')
    except Exception as e:
        entry_9.insert('end', f"[HATA] Bir sorun oluştu: {e}\n")
        entry_9.see('end')



# Butona Fonksiyonu Bağla
button_1.config(command=calculate_and_log)











window.resizable(False, False)
window.mainloop()





































