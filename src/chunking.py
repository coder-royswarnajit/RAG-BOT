from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_chunks(docs, chunk_size=500, chunk_overlap=100):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_chunks = []
    global_chunk_id = 0
    
    for doc in docs:
        splits = text_splitter.split_text(doc["text"])
        
        for local_chunk_id, chunk_text in enumerate(splits):
            all_chunks.append({
                "doc_id": doc.get("doc_id", "unknown"),
                "source": doc.get("source", "unknown"),
                "page": doc.get("page", None),   
                "chunk_id": global_chunk_id,     
                "local_chunk_id": local_chunk_id,
                "text": chunk_text
            })
            
            global_chunk_id += 1
    
    print(f"Split {len(docs)} document(s) into {len(all_chunks)} chunks.")
    return all_chunks