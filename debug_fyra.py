import os
import requests
import json
import time

# --- CONFIGURATION ---
# Correct URL from your PDF (No 'api.' subdomain)
BASE_URL = "https://fyra.im"
CHAT_ENDPOINT = f"{BASE_URL}/v1/chat/completions"
MODELS_ENDPOINT = f"{BASE_URL}/v1/models"

# The specific ID you asked for
TARGET_MODEL = "deepseek-v3.1"
API_KEY = os.environ.get("FRY")

def debug_fyra():
    print(f"--- FYRA.IM DIAGNOSTIC ---")
    print(f"Target URL:   {CHAT_ENDPOINT}")
    print(f"Target Model: {TARGET_MODEL}")
    
    if not API_KEY:
        print("::error:: FRY environment variable is missing!")
        return

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # --- TEST 1: Direct Chat Attempt ---
    print(f"\n[1] Testing Model '{TARGET_MODEL}'...")
    payload = {
        "model": TARGET_MODEL,
        "messages": [{"role": "user", "content": "Hello, simply say 'Online'."}],
        "temperature": 0.5
    }

    try:
        start_time = time.time()
        # Note: PDF implies standard OpenAI format
        response = requests.post(CHAT_ENDPOINT, headers=headers, json=payload, timeout=30)
        elapsed = time.time() - start_time
        
        print(f"   Status: {response.status_code}")
        print(f"   Time:   {elapsed:.2f}s")
        
        if response.status_code == 200:
            try:
                data = response.json()
                content = data['choices'][0]['message']['content']
                print(f"   ✅ SUCCESS: {content}")
                return # Exit if successful
            except Exception as e:
                print(f"   ⚠️ Response 200 but parse failed: {e}")
                print(f"   Raw: {response.text[:200]}")
        else:
            print(f"   ❌ Failed. HTTP {response.status_code}")
            print(f"   Raw Error: {response.text}")

    except Exception as e:
        print(f"   ❌ Connection Error: {e}")

    # --- TEST 2: List Models (Fallback) ---
    # If the above failed, let's see what the API actually calls this model
    print(f"\n[2] Fetching available model IDs...")
    try:
        r = requests.get(MODELS_ENDPOINT, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # Try to find deepseek related IDs
            all_ids = [m['id'] for m in data.get('data', [])]
            deepseek_ids = [mid for mid in all_ids if 'deepseek' in mid.lower()]
            
            print(f"   Found {len(all_ids)} total models.")
            print(f"   DeepSeek variants found: {deepseek_ids}")
            
            if TARGET_MODEL not in all_ids:
                print(f"   ⚠️ NOTICE: '{TARGET_MODEL}' was not found in the list. Check spelling above.")
        else:
            print(f"   ❌ Could not list models: {r.status_code}")
    except Exception as e:
        print(f"   ❌ Model list failed: {e}")

if __name__ == "__main__":
    debug_fyra()
