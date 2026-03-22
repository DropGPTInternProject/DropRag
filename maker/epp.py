from flask import Flask, render_template, request
from helper import helper
from langchain_classic.chains import RetrievalQA
from langchain_ollama import ChatOllama

app = Flask(__name__)

# --- RAG Hazırlık Aşaması ---
# Sadece bir kez yüklenir, her soruda tekrar çalışmaz
print("Sistem ayağa kaldırılıyor...")
embeddings = helper.download_hugging_face_embeddings()
vector_data = helper.loadVectors(embeddings)

# Senin Ollama modellerinP
#turkceLlm = ChatOllama(model="trendyol")
helperLlm = ChatOllama(model="mainBot\trendyol.gguf")

# RAG Zinciri
qa_chain = RetrievalQA.from_chain_type(
    llm=helperLlm,
    chain_type="stuff",
    retriever=vector_data.as_retriever(search_kwargs={"k": 3})
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    soru = msg
    
    # 1. Türkçeden İngilizceye çevir (Veya doğrudan sor)
    cevirim = helperLlm.invoke(soru)
    #ingceSoru = cevirim.content.strip()
    
    # 2. PDF'lerden cevabı bul
    yanit = qa_chain.invoke(cevirim)
    cevap = yanit["result"]
    
    # 3. Cevabı tekrar Türkçeye çevir
    #yaniTR = turkceLlm.invoke(cevap)
    
    return str(cevap)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)