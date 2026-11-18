"""แปลงทุกท่อนเป็นเวกเตอร์ (embedding) แล้วเก็บลง ChromaDB เพื่อใช้ค้นหาด้วยความหมาย

ตัวอย่างการรัน:
    python scripts/build_index.py
    python scripts/build_index.py --model pretrained_models/bge-m3
    python scripts/build_index.py --model intfloat/multilingual-e5-small --batch-size 32
    python scripts/build_index.py --device cpu
    python scripts/build_index.py --device cuda:0 --batch-size 8
    python scripts/build_index.py --help
"""

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from common import decide_device, describe_device

CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
INDEX_DIR = Path("data/index")
COLLECTION_NAME = "wikipedia"

DEFAULT_MODEL = "BAAI/bge-m3"


def parse_args() -> argparse.Namespace:
    """อ่านค่าที่ผู้ใช้พิมพ์ต่อท้ายคำสั่งตอนรัน"""
    parser = argparse.ArgumentParser(
        description="สร้างดัชนีเวกเตอร์จาก chunks.jsonl แล้วเก็บลง ChromaDB"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "ชื่อโมเดลบน Hugging Face (เช่น BAAI/bge-m3) "
            "หรือ path ของโฟลเดอร์โมเดลในเครื่อง (เช่น pretrained_models/bge-m3) "
            f"ค่าเริ่มต้น: {DEFAULT_MODEL}"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "อุปกรณ์ที่ใช้ประมวลผล: auto (ใช้ GPU ถ้ามี ไม่มีก็ใช้ CPU), "
            "cpu, cuda (GPU ตัวแรก) หรือ cuda:0, cuda:1 ... เพื่อเลือก GPU ตามลำดับ "
            "(ค่าเริ่มต้น: auto)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="แปลงทีละกี่ท่อน ถ้าแรมไม่พอให้ลดลง (ค่าเริ่มต้น: 16)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="ความยาวข้อความสูงสุดที่ให้โมเดลอ่าน หน่วยเป็น token (ค่าเริ่มต้น: 512)",
    )
    parser.add_argument(
        "--query-prefix",
        default=None,
        help="ข้อความที่ต้องเติมหน้าคำถาม (ถ้าไม่ระบุ จะเลือกให้อัตโนมัติตามโมเดล)",
    )
    parser.add_argument(
        "--passage-prefix",
        default=None,
        help="ข้อความที่ต้องเติมหน้าเนื้อหา (ถ้าไม่ระบุ จะเลือกให้อัตโนมัติตามโมเดล)",
    )
    return parser.parse_args()


def decide_prefixes(model: str, query_prefix, passage_prefix) -> tuple[str, str]:
    """บางโมเดลต้องเติมคำนำหน้าข้อความถึงจะทำงานได้ดี

    - โมเดลตระกูล e5 ถูกฝึกมาให้เห็น "query: " หน้าคำถาม และ "passage: " หน้าเนื้อหา
    - bge-m3 ไม่ต้องเติมอะไร
    ถ้าผู้ใช้ระบุเองผ่าน --query-prefix / --passage-prefix จะใช้ค่าที่ระบุแทน
    """
    is_e5 = "e5" in model.lower()

    if query_prefix is None:
        if is_e5:
            query_prefix = "query: "
        else:
            query_prefix = ""

    if passage_prefix is None:
        if is_e5:
            passage_prefix = "passage: "
        else:
            passage_prefix = ""

    return query_prefix, passage_prefix


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
    args = parse_args()
    device = decide_device(args.device)
    query_prefix, passage_prefix = decide_prefixes(
        args.model, args.query_prefix, args.passage_prefix
    )

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
        embed_texts.append(passage_prefix + text_for_embedding(chunk))

    # ③ โหลดโมเดล
    if Path(args.model).exists():
        print(f"กำลังโหลดโมเดลจากโฟลเดอร์ในเครื่อง: {args.model}")
    else:
        print(f"ไม่พบโฟลเดอร์ '{args.model}' ในเครื่อง จะโหลดจาก Hugging Face แทน")
        print("(ครั้งแรกอาจใช้เวลานาน ครั้งต่อไปจะใช้ไฟล์ที่เก็บไว้)")

    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_tokens

    print(f"ใช้อุปกรณ์: {describe_device(device)}")
    if query_prefix != "" or passage_prefix != "":
        print(f"คำนำหน้า: query='{query_prefix}' passage='{passage_prefix}'")

    # ④ แปลงข้อความเป็นเวกเตอร์
    print("กำลังแปลงข้อความเป็นเวกเตอร์ ...")
    vectors = model.encode(
        embed_texts,
        batch_size=args.batch_size,
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
            "hnsw:space": "cosine",  # วัดความใกล้ด้วย cosine
            # จดไว้ ตอนค้นหา (search.py) ต้องใช้โมเดลและคำนำหน้าเดียวกัน
            "embedding_model": args.model,
            "query_prefix": query_prefix,
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

    question_vector = model.encode(
        [query_prefix + question], normalize_embeddings=True
    ).tolist()
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
