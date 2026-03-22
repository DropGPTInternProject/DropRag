from helper import helper

def trainAndStore(pdflocation: str):
    wherePdf = pdflocation
    extracted_data = helper.load_pdf_file(wherePdf)
    text_chunks = helper.text_split(extracted_data)
    embeddings = helper.download_hugging_face_embeddings()
    vector_data = helper.storeVectors(embeddings, text_chunks)
    
    
if __name__ == "__main__":
    # Fonksiyonu istediğin dizinle çağırabilirsin
    yol = (str(input("enter the destination of the pdf file")))
    trainAndStore(r'C:\Users\Sıla\Desktop\DropRag-main\maker')