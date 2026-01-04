import os
import json
import requests
import time
import sys
import re
import difflib
from xml.etree import ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

# --- Configuration ---
MAX_FEED_ITEMS = 100  # <--- NEW: Limit before spilling to overflow file

URLS = [
    "https://evilgodfahim.github.io/bdlb/final.xml",
    "https://evilgodfahim.github.io/fp/final.xml",
    "https://evilgodfahim.github.io/bdl/final.xml",
    "https://evilgodfahim.github.io/int/final.xml",
    "https://evilgodfahim.github.io/gpd/daily_feed.xml",
    "https://evilgodfahim.github.io/daily/daily_master.xml",
    "https://evilgodfahim.github.io/bdit/daily_feed_2.xml",
    "https://evilgodfahim.github.io/bdit/daily_feed.xml",
    "https://evilgodfahim.github.io/edit/daily_feed.xml"
]

# Optimized batch sizes (25) to prevent JSON cutoffs/413 errors
MODELS = [
    {"name": "llama-3.3-70b-versatile", "display": "Llama-3.3-70B", "batch_size": 25},
    {"name": "qwen/qwen3-32b", "display": "Qwen-3-32B", "batch_size": 25},
    {"name": "openai/gpt-oss-120b", "display": "GPT-OSS-120B", "batch_size": 25}
]

GROQ_API_KEY = os.environ.get("GEM")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are a Chief Information Filter.
Your task is to select headlines with structural and lasting significance.
You do not evaluate importance by popularity, novelty, or emotion.
You evaluate how information explains or alters systems.

TWO INFORMATION TYPES:
1. STRUCTURAL (Select these): Explains how power, institutions, or economies operate/change.
2. EPISODIC (Ignore these): Isolated events, crime, sports, individual actions.

OUTPUT SPEC:
Return ONLY a JSON array. 
Each item must contain exactly: 
- id (integer from input)
- category (Governance, Economics, Power Relations, or Ideas)
- reason (one concise sentence)

Start with [ and end with ]. No markdown formatting.
"""

def save_xml(data, filename, error_message=None):
    """
    Saves a list of articles to an RSS XML file.
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    
    # Dynamic title based on filename
    feed_title = "Elite News Feed"
    if "overflow" in filename:
        feed_title += " (Overflow)"
        
    ET.SubElement(channel, "title").text = feed_title
    ET.SubElement(channel, "lastBuildDate").text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0600")
    ET.SubElement(channel, "link").text = "https://github.com/evilgodfahim"
    ET.SubElement(channel, "description").text = "AI-curated structural news feed"

    if error_message:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = "System Error"
        ET.SubElement(item, "description").text = f"Script failed: {error_message}"
        ET.SubElement(item, "pubDate").text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0600")
    elif not data:
        # If empty (overflow or no news), add a placeholder so RSS readers don't complain
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = "End of Feed"
        ET.SubElement(item, "description").text = "No additional articles in this feed."
        ET.SubElement(item, "pubDate").text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0600")
    else:
        for art in data:
            item = ET.SubElement(channel, "item")
            ET.SubElement(item, "title").text = art['title']
            ET.SubElement(item, "link").text = art['link']
            ET.SubElement(item, "pubDate").text = art['pubDate']
            
            models_str = ", ".join(art.get('selected_by', ['Unknown']))
            category_info = art.get('category', 'News')
            reason_info = art.get('reason', 'Selected')
            
            html_desc = f"<p><b>[{category_info}]</b></p>"
            html_desc += f"<p><i>{reason_info}</i></p>"
            html_desc += f"<p><small>Selected by: {models_str}</small></p>"
            html_desc += f"<hr/><p>{art['description']}</p>"
            
            ET.SubElement(item, "description").text = html_desc

    try:
        tree = ET.ElementTree(rss)
        ET.indent(tree, space="  ", level=0)
        tree.write(filename, encoding="utf-8", xml_declaration=True)
        print(f"   Saved {len(data) if data else 0} items to {filename}", flush=True)
    except Exception as e:
        print(f"::error::Failed to write XML {filename}: {e}", flush=True)

def fetch_titles_only():
    all_articles = []
    seen_links = set()
    now = datetime.now(timezone.utc)
    cutoff_time = now - timedelta(hours=26)
    
    print(f"Time Filter: Articles after {cutoff_time.strftime('%Y-%m-%d %H:%M UTC')}", flush=True)
    headers = {'User-Agent': 'BCS-Curator/3.0-Ensemble'}

    for url in URLS:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200: continue
            
            try:
                root = ET.fromstring(r.content)
            except: continue

            for item in root.findall('.//item'):
                pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
                if not pub_date: continue
                
                try:
                    dt = parsedate_to_datetime(pub_date)
                    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
                    else: dt = dt.astimezone(timezone.utc)
                    if dt < cutoff_time: continue
                except: continue

                link = item.find('link').text or ""
                if not link or link in seen_links: continue
                
                title = item.find('title').text or "No Title"
                title = title.strip()
                seen_links.add(link)
                
                desc = item.find('description')
                desc_text = desc.text if desc is not None else ""

                all_articles.append({
                    "id": len(all_articles),
                    "title": title,
                    "link": link,
                    "description": desc_text or title,
                    "pubDate": pub_date
                })
        except Exception: continue

    print(f"Loaded {len(all_articles)} unique headlines", flush=True)
    return all_articles

def extract_json_from_text(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        match = re.search(r'(\[.*\])', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except json.JSONDecodeError:
        pass
        
    return None

def call_model(model_info, batch):
    prompt_list = [f"{a['id']}: {a['title']}" for a in batch]
    prompt_text = "\n".join(prompt_list)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_info["name"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here are the headlines:\n{prompt_text}"}
        ],
        "temperature": 0.1,
        "max_tokens": 4096
    }

    max_retries = 5
    base_wait = 30 

    for attempt in range(max_retries):
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=90)
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                if content.startswith("```"):
                    content = content.replace("```json", "").replace("```", "").strip()

                parsed_data = extract_json_from_text(content)
                if parsed_data is not None and isinstance(parsed_data, list):
                    return parsed_data
                else:
                    print(f"    [{model_info['display']}] JSON error (Attempt {attempt+1})", flush=True)
            
            elif response.status_code == 429:
                wait_time = base_wait * (2 ** attempt)
                print(f"    [{model_info['display']}] Rate Limit (429). Cooling down {wait_time}s...", flush=True)
                time.sleep(wait_time)
                continue
            
            elif response.status_code >= 500:
                print(f"    [{model_info['display']}] Server Error {response.status_code}. Retrying...", flush=True)
                time.sleep(10)
                continue
            
        except requests.exceptions.RequestException as e:
            print(f"    [{model_info['display']}] Net Error. Retrying...", flush=True)
            time.sleep(5)
            
        time.sleep(2)
    
    print(f"    [{model_info['display']}] Failed after {max_retries} attempts.", flush=True)
    return []

# --- DEDUPLICATION LOGIC ---
def normalize_text(text):
    text = re.sub(r'[।,:;\-\(\)\"\'\?]', ' ', text)
    return text.lower().strip()

def extract_key_terms(text):
    text = normalize_text(text)
    bangla_stops = {'এ', 'এর', 'ও', 'তে', 'না', 'কে', 'যে', 'হয়', 'এবং', 'করে', 'থেকে', 'নিয়ে', 'জন্য', 'বলে', 'করা'}
    english_stops = {'the', 'a', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'has', 'have', 'be'}
    words = re.split(r'\s+', text)
    return {w for w in words if len(w) > 1 and w not in (bangla_stops | english_stops)}

def smart_similarity(text1, text2):
    terms1 = extract_key_terms(text1)
    terms2 = extract_key_terms(text2)
    
    if not terms1 or not terms2:
        token_score = 0.0
    else:
        token_score = len(terms1 & terms2) / min(len(terms1), len(terms2))

    seq_score = difflib.SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()
    return max(token_score, seq_score)

def semantic_deduplication(articles, similarity_threshold=0.55):
    if not articles or len(articles) < 2: return articles
    print(f"\nSemantic Deduplication (Smart Topic Mode, threshold={similarity_threshold})...", flush=True)
    
    sorted_indices = sorted(range(len(articles)), key=lambda k: len(articles[k]['title']), reverse=True)
    keep_mask = [True] * len(articles)
    duplicates = 0
    
    for i in range(len(sorted_indices)):
        idx_a = sorted_indices[i]
        if not keep_mask[idx_a]: continue
            
        for j in range(i + 1, len(sorted_indices)):
            idx_b = sorted_indices[j]
            if not keep_mask[idx_b]: continue
            
            if smart_similarity(articles[idx_a]['title'], articles[idx_b]['title']) >= similarity_threshold:
                keep_mask[idx_b] = False
                duplicates += 1

    result = [articles[i] for i in range(len(articles)) if keep_mask[i]]
    print(f"   Removed {duplicates} topic duplicates", flush=True)
    return result

def main():
    print("=" * 60, flush=True)
    print("Elite News Curator - BULLETPROOF EDITION + OVERFLOW", flush=True)
    print("=" * 60, flush=True)

    if not GROQ_API_KEY:
        print("::error::GEM environment variable is missing!", flush=True)
        sys.exit(1)
    
    articles = fetch_titles_only()
    if not articles:
        print("No articles found.", flush=True)
        save_xml([], "filtered_feed.xml")
        save_xml([], "filtered_feed_overflow.xml")
        return

    # Process batches
    model_batches = {}
    for model_info in MODELS:
        bs = model_info['batch_size']
        model_batches[model_info['name']] = [articles[i:i + bs] for i in range(0, len(articles), bs)]
    
    max_batch_count = max(len(batches) for batches in model_batches.values())
    MAX_BATCHES_LIMIT = 20
    selections_map = {}
    
    print(f"\nProcessing {min(max_batch_count, MAX_BATCHES_LIMIT)} Batch Groups...", flush=True)

    for batch_idx in range(min(MAX_BATCHES_LIMIT, max_batch_count)):
        print(f"  Batch Group {batch_idx+1}...", flush=True)
        
        for model_info in MODELS:
            m_name = model_info['name']
            if batch_idx >= len(model_batches[m_name]): continue
            
            decisions = call_model(model_info, model_batches[m_name][batch_idx])
            
            if decisions:
                print(f"    [{model_info['display']}] Selected {len(decisions)} articles", flush=True)
                for d in decisions:
                    aid = d.get('id')
                    if aid is not None and isinstance(aid, int) and aid < len(articles):
                        if aid not in selections_map:
                            selections_map[aid] = {'models': [], 'decisions': []}
                        selections_map[aid]['models'].append(model_info['display'])
                        selections_map[aid]['decisions'].append(d)
            else:
                 print(f"    [{model_info['display']}] No selections", flush=True)
            
            time.sleep(3) 
        
        time.sleep(5)

    # Merging
    final_articles = []
    print(f"\nMerging...", flush=True)
    for aid, info in selections_map.items():
        original = articles[aid].copy()
        first_dec = info['decisions'][0]
        original['category'] = first_dec.get('category', 'Priority')
        original['reason'] = first_dec.get('reason', 'Systemic Significance')
        original['selected_by'] = info['models']
        final_articles.append(original)

    # Deduplication
    final_articles = semantic_deduplication(final_articles)
    
    # --- NEW: SPLIT OUTPUT LOGIC ---
    print(f"\nRESULTS:", flush=True)
    print(f"   Analyzed: {len(articles)} headlines", flush=True)
    print(f"   Selected: {len(final_articles)} unique articles", flush=True)
    
    if len(final_articles) > MAX_FEED_ITEMS:
        print(f"   [!] Feed Limit Exceeded ({len(final_articles)} > {MAX_FEED_ITEMS}). Splitting output.", flush=True)
        
        # Primary File (Top 100)
        save_xml(final_articles[:MAX_FEED_ITEMS], "filtered_feed.xml")
        
        # Overflow File (The Rest)
        save_xml(final_articles[MAX_FEED_ITEMS:], "filtered_feed_overflow.xml")
    else:
        # Normal Case
        save_xml(final_articles, "filtered_feed.xml")
        # Ensure overflow file is cleared/empty
        save_xml([], "filtered_feed_overflow.xml")

if __name__ == "__main__":
    main()
