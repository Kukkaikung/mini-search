import json
import time
from pathlib import Path

import requests

API_URL = "https://th.wikipedia.org/w/api.php"  # ภาษาอังกฤษใช้ en.wikipedia.org
HEADERS = {
    # Wikimedia บังคับให้ระบุ User-Agent ถ้าไม่ใส่อาจโดนบล็อก
    "User-Agent": "mini-search/0.1 (learning project)"
}

DELAY_SECONDS = 1.0  # เวลาพักหลังทุกคำขอ
MAX_RETRIES = 3      # ลองใหม่กี่ครั้งเมื่อโดน 429 (ยิงถี่เกินไป)

# ใช้ Session เพื่อใช้การเชื่อมต่อเดิมซ้ำ ทำให้ยิงหลายคำขอได้เร็วขึ้น
session = requests.Session()
session.headers.update(HEADERS)


def call_api(params: dict) -> dict:
    """ยิงคำขอไปที่ API ถ้าโดน 429 จะรอแล้วลองใหม่"""
    for attempt in range(1, MAX_RETRIES + 1):
        response = session.get(API_URL, params=params, timeout=30)
        time.sleep(DELAY_SECONDS)

        if response.status_code == 429:
            # server อาจบอกมาว่าให้รอกี่วินาที (Retry-After) ถ้าไม่บอกก็รอนานขึ้นทุกรอบ
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None and retry_after.isdigit():
                wait = int(retry_after)
            else:
                wait = 5 * attempt
            print(f"  ... โดนจำกัดความเร็ว รอ {wait} วินาที (ครั้งที่ {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response.json()

    # ลองครบทุกรอบแล้วยังโดน 429
    response.raise_for_status()
    return response.json()


def search_titles(query: str, limit: int = 10) -> list[str]:
    """ค้นหาชื่อบทความจากคำค้น"""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }
    data = call_api(params)
    results = data["query"]["search"]

    titles = []
    for item in results:
        titles.append(item["title"])
    return titles


def get_article(title: str) -> dict | None:
    """ดึงเนื้อหาบทความเป็นข้อความล้วน (ไม่มี HTML)"""
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,  # ได้ plain text
        "inprop": "url",   # ได้ URL ของบทความ
        "titles": title,
        "redirects": 1,
        "format": "json",
    }
    data = call_api(params)
    pages = data["query"]["pages"]

    # pages ใช้ pageid เป็น key ซึ่งเราไม่รู้ล่วงหน้า
    # เลยแปลงค่าทั้งหมดเป็น list แล้วหยิบตัวแรก (เราขอแค่บทความเดียว)
    page_list = list(pages.values())
    page = page_list[0]

    if "missing" in page:
        return None

    return {
        "id": page["pageid"],
        "title": page["title"],
        "url": page["fullurl"],
        "text": page.get("extract", ""),
    }


def main():
    topics = ["ปัญญาประดิษฐ์", "การเรียนรู้ของเครื่อง", "ประเทศไทย"]
    out_path = Path("data/raw/wikipedia.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    saved_count = 0

    with out_path.open("w", encoding="utf-8") as f:
        for topic in topics:
            print(f"ค้นหา: {topic}")

            try:
                titles = search_titles(topic, limit=5)
            except requests.RequestException as e:
                print(f"  ✘ ค้นหา '{topic}' ไม่สำเร็จ: {e}")
                continue

            for title in titles:
                if title in seen:
                    continue
                seen.add(title)

                try:
                    article = get_article(title)
                except requests.RequestException as e:
                    print(f"  ✘ ดึง '{title}' ไม่สำเร็จ: {e}")
                    continue

                if article is None or article["text"] == "":
                    print(f"  - ข้าม '{title}' (ไม่มีเนื้อหา)")
                    continue

                line = json.dumps(article, ensure_ascii=False)
                f.write(line + "\n")
                saved_count += 1
                print(f"  ✔ {title} ({len(article['text'])} chars)")

    print(f"บันทึก {saved_count} บทความที่: {out_path}")


if __name__ == "__main__":
    main()
