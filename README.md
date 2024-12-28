
<img src = "img1.png">
<h1 align="center">

🤖🤖 İstanbul Satılık Emlak Fiyat Tahmin Yapay Zekası (2021-2022) 🤖 🤖 </h1>

<h2 align="center">  Hakkında </h2> 

Yapay zeka XGBRegressor algoritmasıyla eğitildi. 

Kullandığım veri kümesi: https://www.kaggle.com/datasets/aselasel/house-price-dataset

<h1>Eğitim sonuçları şu şekildedir: </h1>
 <ul>                 
<li>Best CV Score (RMSE): 0.38024005097130564 </li>
<li>Test RMSE: 0.40 </li>
<li>Test R²: 0.6907 </li>
<li>Train RMSE: 0.34 </li>
<li>Train R²: 0.7733 </li>
</ul>

 <h2 align="center"> 💻 Programı Çalıştırma 💻 </h2>
 <h3> Exe'ye Çıkarma </h3>
 <ol>
        <li>CMD'yi açın.</li>
        <li><code>pip install pyinstaller</code> ile pyinstaller yoksa indirin.
        <li>Konumu proje dosyasına getirin. (istanbul_satilik_tahmin)</li>
        <li> XGBoost inili değilse <code> pip install xgboost </code> ile indirin. </li>
        <li><code>pyinstaller istanbul_satilik_fiyat_tahmin.py --onefile --windowed --add-data 
        "C:\Users\username\AppData\Local\Programs\Python\Python312\Lib\site-packages\xgboost\lib\xgboost.dll;xgboost/lib" 
        --add-data "C:\Users\username\AppData\Local\Programs\Python\Python312\Lib\site-packages\xgboost\VERSION;xgboost" --add-data "design;design"
        </code> kodu ile exe'ye çıkarmaya başlayın.</li>
        <li> Yukarıdaki xgboost.dll'i exe'ye dahil ettiğimiz bölüm için xgboost yolu sizde farklıysa yolu ona göre değiştirmelisiniz. </li>
        <li> Exe, ana klasörün içindeki <strong>"dist"<strong> adlı klasörün içinde belirecektir. Exe'yi oradan .csv, .pkl dosyalarının da bulunduğu ana klasöre taşıyınız.</li>
        <li> Exe'yi çalıştırarak programı başlatın </li>
    </ol>

<h3> IDE'den açma </h3>
Gerekli kütüphaneleri indirdikten sonra IDE içerisinden programı çalıştırabilirsiniz.

Bu proje, Yazılım Mühendisliği 3. sınıf final ödevi kapsamında hazırlanmıştır.
 
