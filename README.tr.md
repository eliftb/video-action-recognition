# Videoda Eylem Tanıma: SlowFast, VideoMAE, TimeSformer ve I3D

[![tests](https://github.com/eliftb/video-action-recognition/actions/workflows/tests.yml/badge.svg)](https://github.com/eliftb/video-action-recognition/actions/workflows/tests.yml)

[English](README.md)

Bir videoyu Kinetics-400 ile eğitilmiş dört hazır modelden geçirir. Her modelin
tahmin ettiği eylemi ve güven oranını gösterir, modellerin hızını ve bellek
kullanımını karşılaştırır. Ayrıca bir özet görseli ve tahminlerin üzerine
yazıldığı bir video kaydeder.

![Okçuluk örneği için özet görseli](docs/archery_results.png)

| Model | Mimari | Girdi |
|---|---|---|
| **SlowFast R50** | iki yollu 3B CNN (yavaş yol: görünüş, hızlı yol: hareket) | 32/8 kare, 256px |
| **VideoMAE-B** | maskeli otokodlayıcıyla ön eğitilmiş ViT | 16 kare, 224px |
| **TimeSformer-B** | zaman ve uzayda ayrık dikkat kullanan ViT | 8 kare, 224px |
| **I3D R50** | 2B filtreleri 3B'ye "şişiren" CNN | 8 kare, 224px |

## Çalıştırma

```bash
source env.sh                 # sanal ortamı açar, model önbelleklerini proje klasörüne yönlendirir
python demo.py                # okçuluk örneği (ilk çalıştırmada indirilir)
python demo.py --video assets/eating_spaghetti.mp4
python demo.py --video benim_videom.mp4
python demo.py --url "https://www.youtube.com/watch?v=..."   # yt-dlp gerekir
python demo.py --models videomae i3d --no-benchmark
```

Sıfırdan kurulum: `python -m venv .venv && source env.sh && pip install -r requirements.txt`.
İlk çalıştırma yaklaşık 1,5 GB model ağırlığı indirir.

Çıktılar `outputs/` klasörüne yazılır: özet görseli (`_results.png`), tahminlerin
yazılı olduğu video (`_annotated.mp4`) ve tüm sonuçları içeren JSON rapor (`_report.json`).

## Bu makinedeki sonuçlar

RTX 4060 Laptop GPU, fp32, batch boyutu 1, 10 çalıştırmanın ortalaması:

| Model | Parametre | Klip/sn | Ağırlık + aktivasyon belleği | Yayımlanan K400 top-1 |
|---|---|---|---|---|
| SlowFast R50 | 34.6M | 16.0 | 360 MB | 76.9% |
| VideoMAE-B | 86.5M | 9.8 | 601 MB | 80.9% |
| TimeSformer-B | 121.6M | 12.1 | 551 MB | 77.9% |
| I3D R50 | 28.0M | 44.1 | 266 MB | 73.3% |

Bu tablodaki doğruluk sütunu **yayımlanmış** değerlerdir; kaynakları [README.md](README.md) içindedir.

### Ölçülen doğruluk

`evaluate.py` ile 1000 Kinetics-400 doğrulama videosunda ölçüldü (20 resmî arşivin 0. parçası,
400 sınıfın 360'ı), her videodan tek merkez klip:

| Model | Ölçülen top-1 | Ölçülen top-5 | Yayımlanan top-1 |
|---|---|---|---|
| SlowFast R50 | 69.0% | 89.3% | 76.9% |
| VideoMAE-B | 77.0% | 92.9% | 80.9% |
| TimeSformer-B | 77.5% | 93.2% | 77.9% |
| I3D R50 | 64.3% | 86.0% | 73.3% |

1000 videoda hata payı yaklaşık ±3 puandır; yani VideoMAE ile TimeSformer burada berabere, ikisi de
SlowFast ve I3D'nin belirgin şekilde önündedir. Ölçülen değerler yayımlananlardan düşüktür, çünkü
yayımlanan değerler her videodan birden çok klip ve kırpmanın ortalamasıdır, bu proje ise tek klibe bakar.

Tekrarlamak için: `python tools/k400_val_subset.py` (1,5 GB indirir), ardından `python evaluate.py data/eval`.
Kendi etiketli videolarınızı `data/eval/<sınıf adı>/` altına koyarak da ölçebilirsiniz.

## Gerçek zamanlı kamera modu

```bash
python webcam.py                          # 0 numaralı kamera, beş model yan yana
python webcam.py --models videomae        # tek model: ilk 3 sınıf, güven çubuklarıyla
python webcam.py --source 1               # başka bir kamera
python webcam.py --source assets/archery.mp4   # kamera yoksa: bir videoyu döngüde oynatır
python webcam.py --record outputs/live.mp4     # ekrandaki görüntüyü de kaydeder
```

Çıkmak için `q` ya da `Esc`.

![Beş modelle canlı mod ekranı](docs/webcam_live.png)

*Canlı mod ekranı; burada kamera yerine okçuluk örnek videosu döngüde oynatılıyor (`--source assets/archery.mp4`).*

Kameradan gelen son birkaç saniyelik görüntü bir tamponda tutulur. Arka plandaki bir iş parçacığı
sürekli en yeni pencereyi alıp modelleri çalıştırır; görüntü ana iş parçacığında çizildiği için
modeller çalışırken de akıcı kalır. Her model kendi doğal süresine (yaklaşık 2,1 sn) bakar, bu süre
en fazla 3 saniyedir (`--max-window`). Etiket titremesin diye olasılıkların art arda pencereler
üzerinden ortalaması alınır (`--smoothing`). %30'un altındaki tahminler soluk yazılır.

Hareketi değiştirdikten sonra etiketin 2-3 saniye içinde değişmesi beklenir, çünkü pencerenin yeni
hareketle dolması gerekir. Dört karşılaştırma modeli yalnızca 400 Kinetics sınıfını bilir (örneğin su içme, alkışlama,
baş sallama, şınav, gitar çalma, diş fırçalama). Kamera modunda beşinci bir model daha çalışır:
Kinetics-600 ile eğitilmiş TimeSformer (`--models timesformer-k600`). 600 sınıfı arasında Kinetics-400'de
olmayan "waving hand" (el sallama) gibi gündelik hareketler vardır. Her model, hiçbir şey yapılmasa
bile kendi sınıflarından birini seçer.

## Testler

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

Testler model ağırlıklarına ve kameraya ihtiyaç duymaz. GitHub Actions bunları her push ve pull request'te
çalıştırır (Python 3.10 ve 3.12, `requirements-ci.txt` içindeki yalnızca CPU'lu PyTorch ile).

