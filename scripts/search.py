"""ค้นหาท่อนข้อความด้วยความหมาย จากดัชนีที่ build_index.py สร้างไว้

ตัวอย่างการรัน:
    python scripts/search.py                                  # โหมดวนถาม
    python scripts/search.py --query "แบงก์ชาติมีหน้าที่อะไร"      # ถามครั้งเดียวแล้วจบ
    python scripts/search.py --top-k 3 --min-score 0.5
    python scripts/search.py --title "ประเทศไทย"                 # ค้นเฉพาะบทความนี้
    python scripts/search.py --help
"""

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import chromadb
from sentence_transformers import SentenceTransformer

from common import decide_device, describe_device

INDEX_DIR = Path("data/index")
COLLECTION_NAME = "wikipedia"
SNIPPET_CHARS = 200  # แสดงข้อความแต่ละผลลัพธ์ยาวกี่ตัวอักษร


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ค้นหาท่อนข้อความด้วยความหมาย")
    parser.add_argument(
        "--query",
        default=None,
        help="คำถามที่จะค้น (ถ้าไม่ใส่ จะเข้าโหมดวนถามทีละข้อ)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="แสดงผลกี่อันดับ (ค่าเริ่มต้น: 5)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="ซ่อนผลที่ความคล้ายต่ำกว่าค่านี้ ช่วง 0-1 (ค่าเริ่มต้น: 0 คือแสดงทั้งหมด)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="ค้นเฉพาะในบทความที่ชื่อตรงกับค่านี้ เช่น \"ประเทศไทย\"",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, ... (ค่าเริ่มต้น: auto)",
    )
    return parser.parse_args()


def open_collection():
    """เปิดดัชนีที่ build_index.py สร้างไว้ ถ้ายังไม่มีให้บอกวิธีแก้แล้วจบโปรแกรม"""
    if not INDEX_DIR.exists():
        print(f"✘ ไม่พบดัชนีที่ {INDEX_DIR}")
        print("  ให้รัน python scripts\\build_index.py ก่อน")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"✘ ไม่พบ collection '{COLLECTION_NAME}' ใน {INDEX_DIR}")
        print("  ให้รัน python scripts\\build_index.py ก่อน")
        sys.exit(1)


def search(collection, model, query_prefix: str, question: str, top_k: int, title) -> list[dict]:
    """แปลงคำถามเป็นเวกเตอร์ แล้วหาท่อนที่ใกล้ที่สุด top_k อันดับ"""
    question_vector = model.encode(
        [query_prefix + question],
        normalize_embeddings=True,  # ต้องเหมือนตอนสร้างดัชนี
    ).tolist()

    # กรองด้วย metadata ถ้าผู้ใช้ระบุ --title
    where = None
    if title is not None:
        where = {"title": title}

    raw = collection.query(
        query_embeddings=question_vector,
        n_results=top_k,
        where=where,
    )

    # ChromaDB คืนค่าเป็น list ซ้อน list (เผื่อค้นหลายคำถามพร้อมกัน)
    # เราค้นคำถามเดียว จึงใช้ [0] แล้วจัดใหม่ให้เป็น list ของ dict อ่านง่ายกว่า
    results = []
    for i in range(len(raw["ids"][0])):
        meta = raw["metadatas"][0][i]
        results.append({
            "chunk_id": raw["ids"][0][i],
            "score": 1 - raw["distances"][0][i],  # distance → ความคล้าย (cosine)
            "title": meta["title"],
            "section": meta["section"],
            "url": meta["url"],
            "text": raw["documents"][0][i],
        })
    return results


def print_results(results: list[dict], min_score: float):
    """แสดงผลลัพธ์ ข้ามรายการที่ความคล้ายต่ำกว่า min_score"""
    shown = 0
    for result in results:
        if result["score"] < min_score:
            continue
        shown += 1

        snippet = result["text"].replace("\n", " ")
        if len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS] + "..."

        print(f"\n[{shown}] {result['score']:.3f}  {result['title']} > {result['section']}  ({result['chunk_id']})")
        print(f"    {snippet}")
        print(f"    {unquote(result['url'])}")  # unquote แปลง %E0%B8... กลับเป็นภาษาไทย

    if shown == 0:
        print(f"\nไม่พบข้อมูลที่เกี่ยวข้อง (ไม่มีผลที่ความคล้าย ≥ {min_score})")

    hidden = len(results) - shown
    if hidden > 0 and shown > 0:
        print(f"\n(ซ่อน {hidden} ผลที่ความคล้ายต่ำกว่า {min_score})")


def run_one(collection, model, query_prefix: str, question: str, args):
    """ค้น 1 คำถาม พร้อมจับเวลา แล้วแสดงผล"""
    start = time.perf_counter()
    results = search(collection, model, query_prefix, question, args.top_k, args.title)
    elapsed = time.perf_counter() - start

    print(f"ค้นเสร็จใน {elapsed:.3f} วินาที")
    print_results(results, args.min_score)


def main():
    args = parse_args()
    device = decide_device(args.device)

    # ① เปิดดัชนี แล้วอ่านการตั้งค่าที่ build_index.py จดไว้
    collection = open_collection()
    settings = collection.metadata or {}
    model_name = settings.get("embedding_model")
    query_prefix = settings.get("query_prefix", "")

    if model_name is None:
        print("✘ ดัชนีนี้ไม่ได้บันทึกชื่อโมเดลไว้ ให้รัน build_index.py ใหม่")
        sys.exit(1)

    print(f"เปิดดัชนี {INDEX_DIR}: {collection.count()} ท่อน")
    if args.title is not None:
        print(f"กรองเฉพาะบทความ: {args.title}")

    # ② โหลดโมเดลเดียวกับตอนสร้างดัชนี (ครั้งเดียว)
    print(f"กำลังโหลดโมเดล {model_name} ...")
    model = SentenceTransformer(model_name, device=device)
    print(f"โมเดล: {model_name} บน {describe_device(device)}")

    # ③ ถามครั้งเดียวแล้วจบ
    if args.query is not None:
        print(f"\nคำถาม> {args.query}")
        run_one(collection, model, query_prefix, args.query, args)
        return

    # ④ โหมดวนถาม
    print("\nพิมพ์คำถาม (Enter ว่าง หรือ exit เพื่อออก)")
    while True:
        try:
            question = input("\nคำถาม> ").strip()
        except (KeyboardInterrupt, EOFError):
            # กด Ctrl+C หรือ Ctrl+Z → ออกแบบสุภาพ ไม่ขึ้น error ยาวๆ
            print()
            break

        if question == "" or question.lower() in ("exit", "quit"):
            break

        run_one(collection, model, query_prefix, question, args)

    print("จบการค้นหา")


if __name__ == "__main__":
    main()
