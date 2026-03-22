import sys
import os

# GPU kullanımını zorla
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def interaktif_sohbet_baslat():
    print("=" * 60)
    print("RAG CHAT - TRENDYOL (GPU + Streaming)")
    print("=" * 60)

    print("\n[1/3] Embeddings yükleniyor...")
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    print("[2/3] FAISS yükleniyor...")
    try:
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"HATA: {e}")
        return

    print("[3/3] LLM hazırlanıyor (GPU)...")
    llm = Ollama(
        model="trendyol",
        num_predict=150,
        num_gpu=99  # Tüm katmanları GPU'da çalıştır
    )

    print("\nHAZIR! (Çıkış: q)")
    print("=" * 60)

    while True:
        try:
            soru = input("\nSen: ").strip()

            if soru.lower() in ['q', 'quit', 'exit', 'çıkış']:
                print("Çıkılıyor...")
                break

            if not soru:
                continue

            docs = vector_store.similarity_search(soru, k=2)
            context = "\n".join([d.page_content for d in docs])

            prompt = f"Bağlam: {context}\n\nSoru: {soru}\n\nKısa cevap:"

            print("\nBot: ", end="", flush=True)
            for chunk in llm.stream(prompt):
                print(chunk, end="", flush=True)

            print("\n" + "-" * 40)

        except KeyboardInterrupt:
            print("\nÇıkılıyor...")
            break
        except Exception as e:
            print(f"\nHATA: {str(e)}")

if __name__ == "__main__":
    interaktif_sohbet_baslat()
