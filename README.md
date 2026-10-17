# Forma Studio · PDF çeviri ve öğrenme

PDF’leri Türkçeye çeviren kişisel web uygulaması. Belge stüdyosu, yan yana kaynak/çıktı inceleme, yapılandırılabilir AI bağlantıları, kontrol raporları, çeviri belleği ve gerçek yerel LoRA eğitimi içerir.

## Başlatma

Python 3.12+ ve [uv](https://docs.astral.sh/uv/) gerekir:

```bash
cd /workspace/PDF-Translator
UV_CACHE_DIR=/workspace/.cache/uv uv sync --frozen --extra local
.venv/bin/python -m uvicorn pdf_translator.app:app --host 127.0.0.1 --port 8000
```

Arayüz sunucunun 8000 portunda çalışır. Kendi bilgisayarında depo ve önbellek yollarını değiştir. Mevcut çalışma kopyasını kullan; ayrı bir Git worktree gerekmez. `--extra local` PyTorch’un CPU sürümünü, Transformers ve PEFT’yi kurar. Yalnızca API/bellek kullanacaksan bu eki kaldırabilirsin; model eğitimi ve yerel üretim için gerekir. Kurulu bağımlılıklar temel model ağırlıklarını içermez.

## Üç çalışma biçimi

| Kip | Davranış |
| --- | --- |
| Akıllı · Bellek + seçili AI | Aynı kaynak metin, dil ve bağlam için onaylı eşleşmeleri kullanır. Yalnızca bilinmeyen metinleri seçili servise gönderir. |
| Tamamen yerel · API yok | Onaylı belleği, sonra indirilmiş yerel öğrenciyi kullanır. Dış API çağrısı yapmaz. Model yoksa bellekte olmayan metin için açık hata verir. |
| Yalnızca seçili AI | Belleği atlayarak tüm metinleri seçili servise gönderir. Servis bulut veya kendi Ollama/LM Studio sunucun olabilir. |

Bir PDF yükle, çalışma biçimini seç, gerekirse bağlantını tanımla ve **Türkçeye çevir** düğmesine bas. Kaynak ve çıktı önizlemesi ilk sayfayı gösterir; tam PDF ve JSON raporunu indirerek bütün sayfaları incele.

## Sağlayıcı listesiyle sınırlı olmayan bağlantılar

**Bağlantılar** ekranından API adresi, model kimliği, protokol, çıktı token sınırı ve JSON kipi ayarlanır. Sağlayıcı şablonları yalnızca başlangıç değerleridir. **Özel sağlayıcı** ile kendi adresini kullanabilirsin; backend de sabit bir sağlayıcı üyelik listesi gerektirmez.

| Protokol | Temel adres örneği | Örnek servisler |
| --- | --- | --- |
| OpenAI Chat Completions | `https://api.openai.com/v1` | OpenAI, OpenRouter, DeepSeek, Groq, Mistral, OpenCode Zen, LM Studio ve uyumlu diğer servisler |
| OpenAI Responses | `https://api.openai.com/v1` | Responses uyumlu sunucular |
| Anthropic Messages | `https://api.anthropic.com` | Anthropic ve uyumlu ağ geçitleri; `/v1` ile biten tabanlar da desteklenir |
| Gemini GenerateContent | `https://generativelanguage.googleapis.com/v1beta` | Gemini native API |
| Ollama | `http://127.0.0.1:11434` | Kendi Ollama sunucun |

OpenCode Zen şablonu `https://opencode.ai/zen/v1` adresini kullanır. Model kimliğini Zen panelinden al veya **Modelleri getir** ile listelemeyi dene. Zen’deki modelin protokolünü seç; her modelin Chat Completions protokolüyle çalıştığı varsayılmaz. **OpenCode CLI bir model API’si değildir; CLI veya oturum sunucusunu Zen ile aynı şey sayma.**

Model listesi ücretli çeviri isteği yapmadan sorgulanır; bazı servisler listelemeyi desteklemez, bu durumda kimliği elle gir. JSON çıktı kipini desteklemeyen servislerde kipi kapatabilirsin; modelin cevabı yine beklenen JSON yapısına uymalıdır. Çıktı token sınırını modelin desteklediği değere ayarla. Kesilmiş, boş, eksik veya yinelenen metin kimliği içeren yanıtlar PDF üretimini durdurur.

Uzak adreslerde HTTPS, yerel IP’lerde HTTP desteklenir. Adrese kullanıcı bilgisi, API anahtarı, sorgu parametresi veya fragment eklenmez. Özel kimlik doğrulama, farklı istek gövdesi veya farklı protokol isteyen servis için bir adaptör gerekir; birbirinden farklı bütün API’lerin tek istek biçimiyle uyumlu olduğu iddia edilmez. Adaptörler `pdf_translator/providers.py` içinde ayrılmıştır.

**Yerel adres, Forma’nın çalıştığı makinedir.** Buluttaki `127.0.0.1`, kendi bilgisayarındaki Ollama’ya ulaşmaz. Sunucuları aynı makinede çalıştır veya erişilebilir, uygun korumalı adresi kullan. Bulutta seçilen hedefin alan adını ortam ağ izinlerine ekle; adresi uygulamaya yazmak ağ politikasını değiştirmez.

## Öğrenme gerçekten ne yapar?

### 1. Aday örnekler

**Öğrenme adayı biriktir** açıkken API tarafından üretilen, anlam/sayı kontrolünden geçen ve PDF’e yerleşen metin çiftleri SQLite’a aday olarak kaydedilir. Adaylar otomatik olarak doğru kabul edilmez; çıkarım ve model eğitimi için kullanılmaz. API anahtarları kaydedilmez.

### 2. Onaylı çeviri belleği

Kontrol raporunu okuyup **Onayla ve belleğe öğret** düğmesini kullanabilir veya **Öğrenme merkezi** ekranında her çeviriyi düzelterek onaylayabilirsin. Sayı/bağlantı uyuşmazlığı olan kayıtlar onaylanmaz. PDF’te kaynak dilinde bırakılmış bölümler toplu onaya alınmaz.

Onaylı eşleşmeler aynı kaynak, kaynak dili ve blok bağlamında yeniden kullanılır. Aynı sözcük farklı bir bağlamda ise körlemesine eşleşmez. Bellek isabetleri çeviri ve kontrol için dış servis çağrısı yapmaz. Bu bir model eğitimi değildir; güvenilir tekrar kullanım katmanıdır. Bellekteki düzeltme, eski PDF’i değiştirmez; yeni çevirilerde kullanılır. **Eğitim verisini indir** yalnızca onaylı çiftleri JSONL olarak dışa aktarır.

### 3. Gerçek yerel öğrenci

Öğrenme merkezinden **Yerel modeli indir** ile `Qwen/Qwen3-0.6B` temel modelini indir. İlk indirme internet ve yaklaşık 1,5 GB disk alanı ister; CPU çalışma belleği birkaç GB olabilir. Modelin erişilebilirliği ve lisans koşulları geçerlidir. Kaynaklar, tokenizasyon ve kapasite sınırları nedeniyle 0.6B boyutunda bir modelin çeviri kalitesi güçlü bulut modellerine eşit varsayılmaz.

Model hazır olduktan sonra API olmadan **yeni metinler** de üretilebilir. En az **20 farklı onaylı kaynak metni** biriktiğinde gerçek LoRA eğitimi yapılabilir:

- Eğitim verileri yalnızca kullanıcının onayladığı çiftlerdir. Öğretmen API’den gelen hatalar onaysız eğitime sızdırılmaz.
- Aynı kaynak metin farklı bağlamlarda bulunsa bile eğitim ve test kümelerine bölünmez; kaynak metne göre gruplama ile test sızıntısı önlenir.
- Eğitimde kullanılmayan %20 örnek (en az 2, en fazla 10 kaynak grubu) önce mevcut modelle, sonra yeni modelle ölçülür.
- Eğitim, kaynak girdisinin loss değerini maskeleyerek Türkçe hedef üzerinde gerçek PyTorch geri yayılımı ve AdamW güncellemeleri yapar. LoRA `q_proj` / `v_proj` katmanlarında çalışır; ağırlıklar safetensors olarak yazılır.
- En az 8 eğitim örneği 768 token eğitim sınırına sığmalıdır. Uzun örnekler sessizce kırpılmaz; geçerli örnek sayısı raporlanır.
- Aday model ancak ayrılmış test kümesinde chrF skorunu artırır ve sayı/bağlantı koruma oranını düşürmezse etkinleştirilir. Aksi halde önceki model korunur. Eğitim kaybı ve değerlendirme skorları gerçek çalışmadan gelir.
- Her eğitim onaylı korpusu yeniden kullanarak temel modelden başlar; önceki adayın rastgele hataları üst üste biriktirilmez.

**Otomatik öğrenme** açıkken model hazırsa her 20 yeni onaylı kayıtta eğitim başlatılır. Manuel başlatma da vardır. Otomatik eğitim farklı kaynak sayısı koşulunu yine uygular. Temel model hiç indirilmediyse hiçbir eğitim veya model varmış gibi gösterilmez. Başarısız işlem kullanıcıya bildirilir; otomatik olarak sonsuz yeniden deneme yapılmaz.

Model dosyaları indirilince yükleme `local_files_only=True`, `trust_remote_code=False` ile yapılır; yerel üretim ve eğitim ağı kullanmaz. Mevcut uygulama CPU çalıştırır. Uzun süreli eğitimde işlem ilerlemesi ekranda takip edilir.

**API’dan tamamen bağımsız ve aynı kalitede bir çevirmen, belirli sayıda PDF yükleyince garanti olarak oluşmaz.** Bellek tekrar eden metinlerde hemen yarar sağlar; yerel model yeni metinler üretebilir; alanında yeterli, doğru ve çeşitli veri ile eğitim ve bağımsız kalite değerlendirmesi gerekir. Küçük test kümesi tüm gelecekteki belgeler için doğruluk kanıtı değildir. Bağımsız bulut anlam denetimi bulunmayan yerel öğrenci çıktıları açıkça insan incelemesi gerektirir.

## Çeviri ve PDF kontrolü

Metinler font, renk, punto ve blok bağlamıyla çıkarılır. Komut kaynağa sadık, eksiksiz Türkçe ister; özetleme istemez. Türkçe dil bilgisi nedeniyle sözcük sırası değişebilir; bire bir sözcük eşleştirmesi garanti edilmez.

Seçili servis çeviriden sonra ayrı bir istekte anlamı ve bütünlüğü kontrol eder. Önerdiği düzeltmeler tekrar denetlenir. Sayılar, URL’ler ve e-postalar kodla karşılaştırılır. Onaylı bellek eşleşmelerinde önceki kullanıcı onayı kullanılır. Yerel öğrencinin bağımsız anlam denetimi yapılmış gibi raporlanmaz; insan incelemesi notu eklenir.

Yerleşim önce deneme sayfasında ölçülür. Kaynak metin ancak yerleşim uygunsa kaldırılır; görsel ve vektörler korunur. Çıktı metni, sayfa sayısı ve boyutları tekrar doğrulanır. Sığmayan/eğik bölümler kaynak dilinde bırakılır, `rendered: false` ve sayfa/blok kimliğiyle raporlanır. Font yoksa veya Türkçe karakteri eksikse Noto Sans yedeği kullanılır. Özgün punto %65’e kadar küçülebilir; %10’dan fazla küçülme raporlanır. Birebir font, satır kırımı veya karmaşık PDF düzeni garantisi yoktur.

Sınırlar: **25 MB / 60 sayfa / 120.000 karakter**. Şifreli ve yalnızca görsel içeren sayfalar reddedilir; OCR yoktur. Görsellerin içindeki yazılar çevrilmez. İmzalar, PDF/A, form alanları veya erişilebilirlik etiketleri için koruma garantisi yoktur.

## Anahtarlar, saklama ve ağ

Arayüzden girilen anahtarlar bu sekmenin ve çalışan işin belleğinde tutulur; diske veya `localStorage`’a yazılmaz. Yerel tarayıcı depolaması sağlayıcı, temel adres, protokol, model ve token tercihlerini saklar. Sayfa yenilenince anahtar yeniden girilir. Sunucuya bağlı bir sağlayıcı anahtarı, özel URL ile başka bir hedefe taşınmaz. Anahtarsız yerel istemci mevcut `OPENAI_API_KEY` veya organizasyon bilgisini miras almaz.

İsteğe bağlı sunucu değişkenleri:

```text
TRANSLATOR_OPENAI_KEY          # OPENAI_API_KEY yedeği de desteklenir
TRANSLATOR_GEMINI_KEY
TRANSLATOR_DEEPSEEK_KEY
TRANSLATOR_OPENROUTER_KEY
TRANSLATOR_OPENCODE_KEY
TRANSLATOR_ANTHROPIC_KEY
TRANSLATOR_GROQ_KEY
TRANSLATOR_MISTRAL_KEY
TRANSLATOR_OLLAMA_KEY          # genellikle gerekmez
TRANSLATOR_LMSTUDIO_KEY        # genellikle gerekmez
```

Gerçek anahtarları sohbete veya Git’e yazma. `.env.example` isteğe bağlıdır; yerel `.env` oluşturursan Uvicorn’u `--env-file .env` ile başlat. Sohbet aboneliği API hesabı değildir. API kullanımına ve sağlayıcının veri koşullarına tabisin.

Varsayılan veri dizini Git tarafından yok sayılan `.data/` dizinidir; `PDF_TRANSLATOR_DATA_DIR` ile değiştirilebilir. Belge başına kaynak, önizleme, çıktı ve rapor saklanır. Kalıcı SQLite çeviri belleği ve yerel öğrenci `.data/learning/` altındadır. **PDF silme, öğrenme kayıtlarını silmez.** Kayıtlar Öğrenme merkezi ekranından ayrı silinir. Silinen bir kayıt eğitilmiş ağırlıklardan kendiliğinden unutulmaz; aktif ağırlıkların kaldırılması veya temiz korpustan yeniden eğitim ayrıca gerekir.

Dış AI ile işlenen metinler seçili sağlayıcıya gönderilir. Tamamen yerel kip dış API kullanmaz. Uygulama kişisel kullanımdır; kullanıcı hesapları ve çok kullanıcılı veri izolasyonu yoktur. Varsayılan bağlantı loopback’tir; internete açmak için kimlik doğrulama ve HTTPS katmanı gerekir. Tek Uvicorn worker kullan; iş kaydı ve eğitim koordinasyonu süreç içindedir.

Bulut ağ izinlerinde kullanacağın hedef gerekir. Hazır sağlayıcılar: `api.openai.com`, `generativelanguage.googleapis.com`, `api.deepseek.com`, `openrouter.ai`, `opencode.ai`, `api.anthropic.com`, `api.groq.com`, `api.mistral.ai`. İlk model indirmesi: `huggingface.co` ve indirme yönlendirmesinin kullandığı Hugging Face/Xet CDN alanları. Özel servisin için kendi alan adını ekle. Bir taslak kaydı, çalışan ağ politikasının değiştiğini veya yayınlandığını kanıtlamaz.

## Doğrulama

```bash
.venv/bin/python -m pytest -q
.venv/bin/python tests/browser_smoke.py --browser /usr/bin/chromium
```

Chromium yoksa `.venv/bin/python -m playwright install chromium` ile kurup browser komutunu `--browser` seçeneği olmadan çalıştır. Tarayıcı testleri geçici bir sunucu ve veri dizini kullanır; kullanıcının verilerini değiştirmez. Ekran görüntüleri varsayılan olarak `/tmp/forma-browser` altındadır.

Testler gerçek PDF işleme, koruma kontrolleri, 5 API protokolünün istek/yanıtları, özel adresler, kimlik bilgisi hedef sınırı, kalıcı bellek, onay/düzeltme, API’siz yeniden çeviri, eğitim/test ayrımı ve **gerçek küçük Qwen3 modelinde LoRA ağırlık güncellemesi, safetensors kaydı ve yeniden yüklemeyi** kapsar. Yerel ek bağımlılıklar yoksa gerçek ağırlık testi açıkça atlanır; bu durumda eğitim doğrulanmış sayılmaz.

Ücretli AI test sınırı kontrollü yanıtlarla değiştirilir; gerçek sağlayıcı kalitesi test edilmiş gibi sunulmaz. Küçük rastgele test modeli de tam boy pretrained Qwen3 modelinin Türkçe kalitesini kanıtlamaz. Hazır model dosyaları ve gerekli ağ/anahtarlarla kendi belgelerinde ayrıca canlı doğrulama yap.

Sağlık: `GET /api/health`. FastAPI/Uvicorn, PyMuPDF, SQLite, HTML/CSS/JS; isteğe bağlı PyTorch/Transformers/PEFT. MuPDF çağrıları iş parçacıkları arasında seri çalışır. Aynı anda iki çeviri işi, toplam sekiz bekleyen/çalışan belge ve tek eğitim işi desteklenir. Yalnızca kendi başlattığın servisi durdur veya yeniden başlat.

PyMuPDF, AGPL veya ticari lisansla dağıtılır; dağıtımında seçtiğin lisansın koşullarını değerlendir.
