"""แปลงทุกท่อนเป็นเวกเตอร์ (embedding) แล้วเก็บลง ChromaDB เพื่อใช้ค้นหาด้วยความหมาย"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
INDEX_DIR = Path("data/index")
COLLECTION_NAME = "wikipedia"

MODEL_NAME = "BAAI/bge-m3"  # ถ้าเครื่องช้า ลอง "intfloat/multilingual-e5-small"
MAX_TOKENS = 512            # ท่อนเรายาวไม่เกิน 500 ตัวอักษร จึงไม่ต้องใช้ถึง 8,192 token
BATCH_SIZE = 16             # แปลงทีละกี่ท่อน (ถ้าแรมไม่พอให้ลดลง)


def read_chunks(path: Path) -> list[dict]:
    """อ่านไฟล์ JSONL ทีละบรรทัด แต่ละบรรทัดคือ 1 ท่อน"""
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            chunks.append(json.loads(line))
    return chunks


def text_for_embedding(chunk: dict) -> str:
    """เติมชื่อบทความและหัวข้อไว้หน้าข้อความ ให้เวกเตอร์รู้ว่าท่อนนี้พูดถึงอะไร

    ตัวอย่าง: "ธนาคารแห่งประเทศไทย > ประวัติ: ก่อตั้งขึ้นเมื่อ พ.ศ. 2485..."
    """
    if chunk["section"] == "บทนำ":
        return f"{chunk['title']}: {chunk['text']}"
    return f"{chunk['title']} > {chunk['section']}: {chunk['text']}"


def main():
    # ① อ่านท่อนทั้งหมด
    chunks = read_chunks(CHUNKS_PATH)
    print(f"อ่านได้ {len(chunks)} ท่อน จาก {CHUNKS_PATH}")

    # ② เตรียมข้อมูล 4 อย่างที่ ChromaDB ต้องการ (ลำดับต้องตรงกันทุก list)
    ids = []          # รหัสไม่ซ้ำของแต่ละท่อน
    documents = []    # ข้อความต้นฉบับ ไว้แสดงผลตอนค้นเจอ
    metadatas = []    # ข้อมูลที่มา
    embed_texts = []  # ข้อความที่เติมบริบทแล้ว ใช้ทำเวกเตอร์

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        metadatas.append({
            "article_id": chunk["article_id"],
            "title": chunk["title"],
            "section": chunk["section"],
            "url": chunk["url"],
        })
        embed_texts.append(text_for_embedding(chunk))

    # ③ โหลดโมเดล (ครั้งแรกจะดาวน์โหลดประมาณ 2 GB)
    print(f"กำลังโหลดโมเดล {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_TOKENS
    print(f"ใช้อุปกรณ์: {model.device}")  # cuda = GPU, cpu = CPU

    # ④ แปลงข้อความเป็นเวกเตอร์
    print("กำลังแปลงข้อความเป็นเวกเตอร์ ...")
    vectors = model.encode(
        embed_texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,  # ปรับความยาวเวกเตอร์เป็น 1 ใช้เทียบแบบ cosine ได้
        show_progress_bar=True,
    )
    print(f"ได้เวกเตอร์ {vectors.shape[0]} ตัว ตัวละ {vectors.shape[1]} มิติ")

    # ChromaDB ต้องการ list ธรรมดา ไม่ใช่ numpy array
    embeddings = vectors.tolist()

    # ⑤ สร้างฐานข้อมูลใหม่ (ลบของเก่าทิ้งก่อน กันข้อมูลเก่าค้างปน)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
        print("ลบดัชนีเก่าแล้ว")
    except Exception:
        pass  # ยังไม่เคยสร้าง ไม่ต้องลบ

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",         # วัดความใกล้ด้วย cosine
            "embedding_model": MODEL_NAME,  # จดไว้ ตอนค้นหาต้องใช้โมเดลเดียวกัน
        },
    )

    # ⑥ ใส่ข้อมูลทีละชุด (ใส่ทีเดียวเยอะๆ อาจเกินขีดจำกัดของ ChromaDB)
    add_batch_size = 100
    for start in range(0, len(ids), add_batch_size):
        end = start + add_batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
    print(f"เก็บลง ChromaDB แล้ว {collection.count()} รายการ ที่ {INDEX_DIR}")

    # ⑦ ทดสอบค้นหา 1 คำถาม
    question = "ธนาคารแห่งประเทศไทยก่อตั้งเมื่อไร"
    print(f"\nทดสอบค้นหา: {question}")

    question_vector = model.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=question_vector, n_results=3)

    # results มีโครงสร้างแบบ list ซ้อน list เพราะค้นได้หลายคำถามพร้อมกัน
    # เราค้นคำถามเดียว จึงใช้ [0]
    found_documents = results["documents"][0]
    found_metadatas = results["metadatas"][0]
    found_distances = results["distances"][0]

    for i in range(len(found_documents)):
        similarity = 1 - found_distances[i]  # ChromaDB ให้ระยะห่าง (distance) = 1 - cosine
        meta = found_metadatas[i]
        print(f"\n[{i + 1}] {meta['title']} > {meta['section']}  (ความคล้าย {similarity:.3f})")
        print(f"    {found_documents[i][:150]}...")


if __name__ == "__main__":
    main()
