import os
import json
import requests
import time
import sys
import re
from xml.etree import ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

# --- Configuration ---
MAX_FEED_ITEMS = 100

URLS = [
    "https://evilgodfahim.github.io/gpd/daily_feed.xml",
    "https://evilgodfahim.github.io/daily/daily_master.xml",
    "https://evilgodfahim.github.io/bdit/daily_feed_2.xml",
    "https://evilgodfahim.github.io/bdit/daily_feed.xml",
    "https://evilgodfahim.github.io/edit/daily_feed.xml"
]

MODELS = [
    {
        "name": "kimi-k2-instruct-0905",
        "display": "Kimi-K2-Instruct",
        "batch_size": 50,
        "api": "fyra"
    },
    {
        "name": "meta-llama/llama-3.3-70b-instruct",
        "display": "Llama-3.3-70B",
        "batch_size": 50,
        "api": "openrouter"
    },
    {
        "name": "qwen/qwen3-32b",
        "display": "Qwen-3-32B",
        "batch_size": 25,
        "api": "groq"
    },
    {
        "name": "openai/gpt-oss-120b",
        "display": "GPT-OSS-120B",
        "batch_size": 25,
        "api": "groq"
    },
    {
        "name": "mistral-small-latest",
        "display": "Mistral-Small",
        "batch_size": 40,
        "api": "mistral"
    },
    {
        "name": "gemini-2.5-flash-lite",
        "display": "Gemini-2.5-Flash-Lite",
        "batch_size": 100,
        "api": "google"
    }
]

# API Keys and URLs
GROQ_API_KEY = os.environ.get("GEM")
OPENROUTER_API_KEY = os.environ.get("OP")
FYRA_API_KEY = os.environ.get("FRY")
MISTRAL_API_KEY = os.environ.get("GEM2")
GOOGLE_API_KEY = os.environ.get("LAM")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
FYRA_API_URL = "https://fyra.im/v1/chat/completions"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
GOOGLE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are a Chief Information Filter.
Your task is to select headlines with structural and lasting significance.
You do not evaluate importance by popularity, novelty, or emotion.
You evaluate how information explains or alters systems.
Judgment must rely only on linguistic structure, implied scope, and systemic consequence.
TWO INFORMATION TYPES (internal use)
STRUCTURAL
— Explains how power, institutions, economies, or long-term social/strategic forces operate or change.
EPISODIC
— Describes isolated events, individual actions, or short-lived situations without system impact.
Select only STRUCTURAL.
FOUR STRUCTURAL LENSES (exclusive)
GOVERNANCE & CONTROL
Rules, enforcement, institutional balance, authority transfer, administrative or judicial change.
ECONOMIC & RESOURCE FLOWS
Capital movement, trade structure, production capacity, fiscal or monetary direction, systemic risk.
POWER RELATIONS & STRATEGY
Strategic alignment, coercion, deterrence, security posture, long-term rivalry or cooperation.
IDEAS, ARGUMENTS & LONG-TERM TRENDS
Editorial reasoning, policy debate, scientific or technological trajectories, demographic or climate forces.
CONTEXTUAL GRAVITY RULE (KEY)
When two or more headlines show equal structural strength, favor the one that:
• Operates closer to the decision-making center of a society
• Directly affects national policy formation or institutional practice
• Originates from internal analytical or editorial discourse, not external observation
This rule applies universally, regardless of language or country.
SINGLE DECISION TEST (mandatory)
Ask only:
"Does this headline clarify how a system functions or how its future direction is being shaped, in a way that remains relevant after time passes?"
• Yes or plausibly yes → SELECT
• No → SKIP
No secondary tests.
AUTOMATIC EXCLUSIONS
Skip always: • Crime, accidents, or scandals without institutional consequence
• Sports, entertainment, lifestyle
• Personal narratives without systemic implication
• Repetition of already-settled facts
OUTPUT SPEC (strict)
Return only a JSON array of selected IDs.
Example: [0, 5, 12, 23]
No markdown.
No commentary.
No text outside JSON."""

def save_xml(data, filename, error_message=None):
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")

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
                if not link:
                    # Fallback to guid if link is missing or empty
                    guid = item.find('guid')
                    link = guid.text if guid is not None else ""
                
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
        match = re.search(r'(\[[\d,\s]*\])', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except json.JSONDecodeError:
        pass
    return None

def call_model(model_info, batch):
    prompt_list = [f"{a['id']}: {a['title']}" for a in batch]
    prompt_text = "\n".join(prompt_list)

    # Select API based on model config
    api_type = model_info.get("api", "groq")

    if api_type == "openrouter":
        api_url = OPENROUTER_API_URL
        api_key = OPENROUTER_API_KEY
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/evilgodfahim",
            "X-Title": "Elite News Curator"
        }
        payload = {
            "model": model_info["name"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.3
        }
    elif api_type == "fyra":
        api_url = FYRA_API_URL
        api_key = FYRA_API_KEY
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_info["name"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.3
        }
    elif api_type == "mistral":
        api_url = MISTRAL_API_URL
        api_key = MISTRAL_API_KEY
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_info["name"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.3
        }
    elif api_type == "google":
        api_url = f"{GOOGLE_API_URL}/{model_info['name']}:generateContent?key={GOOGLE_API_KEY}"
        api_key = GOOGLE_API_KEY
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{SYSTEM_PROMPT}\n\n{prompt_text}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.3
            }
        }
    else:  # groq
        api_url = GROQ_API_URL
        api_key = GROQ_API_KEY
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_info["name"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.3
        }

    max_retries = 5
    base_wait = 30 

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=90)

            if response.status_code == 200:
                try:
                    response_data = response.json()
                    
                    if 'error' in response_data:
                        print(f"    [{model_info['display']}] API Error: {response_data.get('error', 'Unknown error')}", flush=True)
                        continue
                    
                    if api_type == "google":
                        content = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
                    else:
                        content = response_data['choices'][0]['message']['content'].strip()
                    
                except (KeyError, IndexError) as e:
                    print(f"    [{model_info['display']}] Response parse error: {e}", flush=True)
                    continue

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

def main():
    print("=" * 60, flush=True)
    print("Elite News Curator - Multi-API Ensemble", flush=True)
    print("=" * 60, flush=True)

    # Validate API keys
    if not GROQ_API_KEY:
        print("::error::GEM environment variable is missing!", flush=True)
        sys.exit(1)

    needs_openrouter = any(m.get("api") == "openrouter" for m in MODELS)
    if needs_openrouter and not OPENROUTER_API_KEY:
        print("::error::OP environment variable is missing!", flush=True)
        sys.exit(1)

    needs_fyra = any(m.get("api") == "fyra" for m in MODELS)
    if needs_fyra and not FYRA_API_KEY:
        print("::error::FRY environment variable is missing!", flush=True)
        sys.exit(1)

    needs_mistral = any(m.get("api") == "mistral" for m in MODELS)
    if needs_mistral and not MISTRAL_API_KEY:
        print("::error::GEM2 environment variable is missing!", flush=True)
        sys.exit(1)

    needs_google = any(m.get("api") == "google" for m in MODELS)
    if needs_google and not GOOGLE_API_KEY:
        print("::error::LAM environment variable is missing!", flush=True)
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
                for aid in decisions:
                    if isinstance(aid, int) and aid < len(articles):
                        if aid not in selections_map:
                            selections_map[aid] = {'models': [], 'count': 0}
                        selections_map[aid]['models'].append(model_info['display'])
                        selections_map[aid]['count'] += 1
            else:
                print(f"    [{model_info['display']}] No selections", flush=True)

            time.sleep(15)  # Delay between models

        time.sleep(30)  # Delay between batch groups

    # Merging - only keep articles selected by at least 2 models
    final_articles = []
    print(f"\nMerging (2+ model consensus required)...", flush=True)
    for aid, info in selections_map.items():
        if len(info['models']) >= 2:  # At least 2 models must agree
            original = articles[aid].copy()
            original['category'] = 'Priority'
            original['reason'] = 'Systemic Significance'
            original['selected_by'] = info['models']
            final_articles.append(original)

    print(f"   ✅ {len(final_articles)} articles passed 2+ model consensus from {len(selections_map)} total selections", flush=True)

    # Results
    print(f"\nRESULTS:", flush=True)
    print(f"   Analyzed: {len(articles)} headlines", flush=True)
    print(f"   Selected: {len(final_articles)} unique articles", flush=True)

    if len(final_articles) > MAX_FEED_ITEMS:
        print(f"   [!] Feed Limit Exceeded ({len(final_articles)} > {MAX_FEED_ITEMS}). Splitting output.", flush=True)
        save_xml(final_articles[:MAX_FEED_ITEMS], "filtered_feed.xml")
        save_xml(final_articles[MAX_FEED_ITEMS:], "filtered_feed_overflow.xml")
    else:
        save_xml(final_articles, "filtered_feed.xml")
        save_xml([], "filtered_feed_overflow.xml")

if __name__ == "__main__":
    main()