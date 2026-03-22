import faiss
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
class helper:
    @staticmethod
    def load_pdf_file(data_path):
    # 1. PDF dosyalarını yükle
        pdf_loader = DirectoryLoader(
            data_path, 
            glob="*.pdf", 
            loader_cls=PyPDFLoader
        )
    
    # 2. TXT dosyalarını yükle (UTF-8 kodlaması ile)
        txt_loader = DirectoryLoader(
            data_path, 
            glob="*.txt", 
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
    
    # İki yükleyiciyi de çalıştır ve listeleri birleştir
        pdf_documents = pdf_loader.load()
        txt_documents = txt_loader.load()
    
        return pdf_documents + txt_documents
    @staticmethod
    def text_split(extracted_data):
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
        text_chunks=text_splitter.split_documents(extracted_data)
        filtered_chunks = [chunk for chunk in text_chunks if len(chunk.page_content) >= 80]
        return filtered_chunks
    @staticmethod
    def download_hugging_face_embeddings():
        embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', encode_kwargs={'batch_size':32})
        return embeddings
    @staticmethod
    def storeVectors(embeddings, text_chunks):
        index = faiss.IndexHNSWFlat(384, 32)
        index.hnsw.efConstruction = 200 
        index.hnsw.efSearch = 64        
        
        vector_db = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        
        vector_db.add_documents(text_chunks)
        
        vector_db.save_local("faiss_index")
        return vector_db
    @staticmethod
    def loadVectors(embeddings):
        vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return vector_db
