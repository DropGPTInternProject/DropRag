import os
from huggingface_hub import hf_hub_download

# Senin belirttiğin değişkenler
repo_id = "tensorblock/Trendyol_Trendyol-LLM-8B-T1-GGUF"
filename = "Trendyol-LLM-8B-T1-Q5_K_M.gguf"
local_dir = "MY_LOCAL_DIR"

# Klasör yoksa oluşturur
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

print(f"{filename} dosyası {local_dir} klasörüne indiriliyor...")

try:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False  # Windows'ta dosya kopyasını doğrudan klasöre koyması için
    )
    print(f"Başarıyla indirildi: {path}")
except Exception as e:
    print(f"Hata oluştu: {e}")
    print("\nNot: Eğer 404 hatası alırsan, dosya adını 'Trendyol-LLM-8B-T1-Q5_K_M-GGUF.gguf' olarak güncellemeyi dene.")