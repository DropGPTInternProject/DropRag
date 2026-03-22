# -*- coding: utf-8 -*-
"""
TXT dosyasindaki sorulari 10'ar 10'ar yerel "serce" modeliyle test eder.
Kullanim:
  python run_txt_test.py                  # 0-10 arasi (ilk 10 soru)
  python run_txt_test.py 0 10             # 0-10 arasi
  python run_txt_test.py 10 20            # 10-20 arasi
"""
import sys
import io
import os
import json
import time
import argparse
from datetime import datetime

# LangChain Bilesenleri
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# -----------------------------------------------------------------------
# TXT DOSYASINDAN SORULARI OKU
# -----------------------------------------------------------------------
def load_questions_from_txt(filepath=None):
    if filepath is None:
        base = os.path.dirname(os.path.abspath(__file__))
        files = os.listdir(base)
        filepath = None
        for f in files:
            if f.endswith('.txt'): # Herhangi bir txt dosyasini bulur
                filepath = os.path.join(base, f)
                break
        if filepath is None:
            raise FileNotFoundError("Bulunabilir bir TXT dosyasi yok.")

    with open(filepath, 'r', encoding='utf-8') as fh:
        lines = fh.readlines()

    questions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit():
            dot_pos = line.find('.')
            if dot_pos != -1 and dot_pos < 6:
                line = line[dot_pos + 1:].strip()
        if line:
            questions.append(line)

    return questions, filepath

# -----------------------------------------------------------------------
# SONUCLARI KAYDET
# -----------------------------------------------------------------------
def save_results(results, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"    [Kaydedildi: {filename}]")

# -----------------------------------------------------------------------
# ANA TEST FONKSIYONU
# -----------------------------------------------------------------------
def run_txt_tests(start_idx=0, end_idx=10, txt_path=None):
    # Sorulari yukle
    all_questions, txt_file = load_questions_from_txt(txt_path)
    total_available = len(all_questions)

    end_idx = min(end_idx, total_available)
    questions = all_questions[start_idx:end_idx]
    total = len(questions)

    if total == 0:
        print(f"Test edilecek soru yok. Toplam soru: {total_available}")
        return []

    print("\n" + "=" * 70)
    print("RAG TEST - YEREL 'SERCE' MODELI TESTI")
    print("=" * 70)
    print(f"Dosya    : {os.path.basename(txt_file)}")
    print(f"Test     : #{start_idx + 1} - #{end_idx}  ({total} soru)")
    print("=" * 70)

    # 1. RAG Bilesenlerini Yukle (Cevirici veya Web yok, dogrudan lokal)
    print("\n[1/3] Vektor Yerlestirmeleri (Embeddings) Yukleniyor...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', encode_kwargs={'batch_size':32})

    print("[2/3] FAISS Vektor Veritabani Yukleniyor...")
    try:
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"HATA: 'faiss_index' klasoru bulunamadi! Once veritabanini olusturun. Detay: {e}")
        return []

    print("[3/3] 'Serce' Modeli Yukleniyor...")
    llm = Ollama(model="trendyol") # Ollama uzerindeki modelin

    # Prompt Sablonu (Modelin RAG ile nasil cevap verecegini belirler)
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""Aşağıda verilen bağlam bilgilerini kullanarak soruyu cevapla.
Eğer cevabı bağlamda bulamazsan, sadece bilmediğini söyle, kendi kendine bilgi uydurma.

Bağlam:
{context}

Soru: {question}

Cevap:"""
    )

    results = []

    print(f"\n{'='*70}")
    print(f"TEST BASLIYOR...")
    print(f"{'='*70}\n")

    for i, question in enumerate(questions):
        q_num = start_idx + i + 1
        print(f"\n[{q_num}/{end_idx}] {question[:70]}...")

        try:
            start_time = time.time()

            # 1. FAISS uzerinden benzer metinleri ara (En iyi 3 sonuc)
            docs = vector_store.similarity_search(question, k=3)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # 2. Eger sonuc bulunamazsa
            if not docs:
                answer = "[HATA] Vektör veritabanında alakalı kaynak bulunamadı."
                sources_info = []
            else:
                # 3. Prompt'u hazirla ve Modeli cagir
                final_prompt = prompt_template.format(context=context_text, question=question)
                answer = llm.invoke(final_prompt)
                
                # Kaynaklari kaydet (Metadata varsa ekler, yoksa sadece metni referans alir)
                sources_info = [{"source": doc.metadata.get("source", "PDF"), "page": doc.metadata.get("page", "Bilinmiyor")} for doc in docs]

            elapsed = time.time() - start_time

            result = {
                "id": q_num,
                "question": question,
                "answer": answer.strip(),
                "sources": sources_info,
                "elapsed_seconds": round(elapsed, 2),
                "status": "OK" if docs else "NO_SOURCE"
            }

            print(f"    OK - {elapsed:.1f}s")

        except Exception as e:
            result = {
                "id": q_num,
                "question": question,
                "answer": f"[HATA] {str(e)}",
                "sources": [],
                "elapsed_seconds": 0,
                "status": "ERROR"
            }
            print(f"    HATA: {str(e)[:80]}")

        results.append(result)

        # Her 10 soruda bir yedek kayit al
        if (i + 1) % 10 == 0:
            save_results(results, f"txt_test_partial_{start_idx}_{start_idx + i + 1}.json")

    # Final kayit
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"serce_test_results_{start_idx}_{end_idx}_{timestamp}.json"
    save_results(results, filename)

    ok_count = sum(1 for r in results if r['status'] == 'OK')
    err_count = sum(1 for r in results if r['status'] == 'ERROR')

    print(f"\n{'='*70}")
    print(f"TEST TAMAMLANDI!")
    print(f"Toplam   : {len(results)} soru")
    print(f"Basarili : {ok_count}")
    print(f"Hatali   : {err_count}")
    print(f"Sonuclar : {filename}")
    print(f"{'='*70}\n")

    return results

# -----------------------------------------------------------------------
# KOMUT SATIRI
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TXT dosyasindan sorulari yerel 'serce' modeliyle test eder")
    parser.add_argument("start", nargs="?", type=int, default=0, help="Baslangic indeksi (default: 0)")
    parser.add_argument("end", nargs="?", type=int, default=10, help="Bitis indeksi (default: 10, ilk 10 soru)")
    parser.add_argument("--file", type=str, default=None, help="TXT dosya yolu (varsayilan: otomatik bulur)")
    
    # Eskiden kalan --web argumanini gormezden gelmek uzere ekliyoruz (kodu bozmasin diye)
    parser.add_argument("--web", action="store_true", default=False, help="Web aramasi bu versiyonda devre disidir.")

    args = parser.parse_args()

    run_txt_tests(
        start_idx=args.start,
        end_idx=args.end,
        txt_path=args.file
    )