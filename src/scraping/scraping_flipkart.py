"""
Flipkart Reviews Scraper (API-based)
Saves to: ../.. / data / raw / flipkart_reviews.csv
"""

import requests
import time
import random
import os
import pandas as pd
from urllib.parse import urlencode

# ------------------------------
# Helpers
# ------------------------------

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.flipkart.com/",
    "Origin": "https://www.flipkart.com"
}

def safe_get(dct, keys, default=""):
    """Try a list of keys (or nested keys) and return first found."""
    for key in keys:
        # support nested keys given as tuples
        if isinstance(key, (list, tuple)):
            cur = dct
            found = True
            for k in key:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    found = False
                    break
            if found and cur is not None:
                return cur
        else:
            if key in dct and dct[key] is not None:
                return dct[key]
    return default

# ------------------------------
# Core: Fetch reviews from Flipkart internal API
# ------------------------------

def fetch_reviews_page(pid, page_num, session=None, headers=None):
    """
    Fetch a page of reviews for a product PID using Flipkart's internal reviews endpoint.
    Returns parsed JSON (dict) or None on failure.
    """
    if session is None:
        session = requests.Session()
    if headers is None:
        headers = DEFAULT_HEADERS

    # Known Flipkart internal review API pattern (works for many products)
    # We include query params which Flipkart's frontend uses; the API variant may vary.
    # We'll attempt two common endpoints: 'api/3' and 'api/3/product/reviews' patterns.
    endpoints = [
        f"https://www.flipkart.com/api/3/product/reviews",
        f"https://www.flipkart.com/api/3/search/top-reviews"  # fallback pattern (rare)
    ]

    params = {
        "pid": pid,
        "page": page_num,
        "count": 10  # number per page; Flipkart typically returns ~10
    }

    for base in endpoints:
        try:
            url = base + "?" + urlencode(params)
            resp = session.get(url, headers=headers, timeout=20)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    # not JSON
                    continue
            # try slight variant: sometimes endpoint requires /?params or different path
            time.sleep(0.2)
        except requests.RequestException:
            time.sleep(0.5)
            continue

    # If the simple endpoints fail, attempt a more generic endpoint pattern used by some products
    try:
        alt_url = f"https://www.flipkart.com/api/3/product/reviews?{urlencode(params)}"
        resp = session.get(alt_url, headers=headers, timeout=20)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass

    return None

# ------------------------------
# Parse the Flipkart review JSON
# ------------------------------

def parse_reviews_from_json(js):
    """
    Given JSON returned from the Flipkart reviews API, return a list of review dicts
    with keys: name, rating, title, review, date, review_id
    This function uses multiple fallbacks because Flipkart's JSON shape varies.
    """
    reviews_out = []

    if not js:
        return reviews_out

    # Flipkart often nests review lists in different keys; try common ones
    candidates = []
    # Example shapes observed across products:
    # {'reviews': {... 'data': [...]}} or {'productReviews': [...]} or {'RESPONSE': {'reviews': [...]}}
    if isinstance(js, dict):
        # common direct lists
        for k in ("reviews", "reviewList", "productReviews", "data", "reviewData", "results"):
            if k in js and isinstance(js[k], (list, tuple)):
                candidates.append(js[k])

        # nested shapes
        # look for dict values that themselves contain review lists
        for v in js.values():
            if isinstance(v, dict):
                for k2 in ("reviews", "data", "reviewList", "result", "results"):
                    if k2 in v and isinstance(v[k2], (list, tuple)):
                        candidates.append(v[k2])

        # some responses have deep nesting with keys like 'RESPONSE' or 'payload'
        if "RESPONSE" in js and isinstance(js["RESPONSE"], dict):
            for k in ("reviews", "data", "reviewList"):
                if k in js["RESPONSE"] and isinstance(js["RESPONSE"][k], (list, tuple)):
                    candidates.append(js["RESPONSE"][k])

    # Flatten unique candidate lists
    final_lists = []
    for c in candidates:
        if isinstance(c, (list, tuple)) and c not in final_lists:
            final_lists.append(c)

    # If no lists found, maybe the top-level object already is a list of reviews
    if not final_lists and isinstance(js, (list, tuple)):
        final_lists = [js]

    # If still empty, try to search recursively for likely review objects (heuristic)
    if not final_lists:
        # find lists inside nested dicts
        def find_lists(obj):
            out = []
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, (list, tuple)):
                        out.append(v)
                    elif isinstance(v, dict):
                        out.extend(find_lists(v))
            return out
        found = find_lists(js)
        final_lists = found

    # Now attempt to parse each candidate list for review objects
    for lst in final_lists:
        for item in lst:
            if not isinstance(item, dict):
                continue

            # rating: try several possible keys
            rating = safe_get(item, ["rating", "ratingValue", ("rating", "value"), ("reviewRating", "value")], "")
            # review text: various keys
            review_text = safe_get(item, ["reviewText", "review", "comment", ("review", "text"), "text", "reviewTextV2"], "")
            # title
            title = safe_get(item, ["title", "reviewTitle", "headline"], "")
            # author / name
            name = safe_get(item, ["author", "name", "reviewerName", ("author", "name")], "")
            # date
            date = safe_get(item, ["created", "createdAt", "date", "publishedDate", "createdOn"], "")
            # review id
            review_id = safe_get(item, ["id", "reviewId", "review_id", ("id", "reviewId")], "")

            # Some JSON shape: nested object 'review' containing these fields
            if not review_text and "review" in item and isinstance(item["review"], dict):
                review_text = safe_get(item["review"], ["reviewText", "text"], review_text)

            # Clean up types
            rating = str(rating).strip()
            title = str(title).strip()
            name = str(name).strip()
            review_text = str(review_text).strip()
            date = str(date).strip()
            review_id = str(review_id).strip()

            # Heuristic: ensure there is some review text or title
            if review_text or title:
                reviews_out.append({
                    "name": name,
                    "rating": rating,
                    "title": title,
                    "review": review_text,
                    "date": date,
                    "review_id": review_id
                })

    # remove duplicates by review_id + beginning of review text
    seen = set()
    unique = []
    for r in reviews_out:
        key = f"{r.get('review_id','')}_{r.get('review','')[:60]}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    return unique

# ------------------------------
# Public wrapper: paginate and collect
# ------------------------------

def scrape_flipkart_by_pid(pid, max_reviews=500, wait_min=0.3, wait_max=1.0):
    """
    Scrape reviews for a Flipkart product pid using the internal API.
    Returns a list of review dicts.
    """
    session = requests.Session()
    headers = DEFAULT_HEADERS.copy()
    collected = []
    seen_keys = set()
    page = 1
    consecutive_empty = 0

    while len(collected) < max_reviews and consecutive_empty < 3:
        js = fetch_reviews_page(pid, page, session=session, headers=headers)
        if not js:
            consecutive_empty += 1
            print(f"⚠️ Page {page} returned no JSON or failed. (attempt).")
            page += 1
            time.sleep(random.uniform(wait_min, wait_max))
            continue

        parsed = parse_reviews_from_json(js)
        if not parsed:
            consecutive_empty += 1
            print(f"⚠️ Page {page} parsed 0 reviews.")
            page += 1
            time.sleep(random.uniform(wait_min, wait_max))
            continue

        consecutive_empty = 0
        new_count = 0

        for r in parsed:
            key = f'{r.get("review_id","")}_{r.get("review","")[:60]}'
            if key in seen_keys:
                continue
            seen_keys.add(key)
            collected.append(r)
            new_count += 1
            if len(collected) >= max_reviews:
                break

        print(f"📄 Page {page} => {new_count} new reviews (total {len(collected)})")
        page += 1
        time.sleep(random.uniform(wait_min, wait_max))

    return collected

# ------------------------------
# Save to project data/raw
# ------------------------------

def save_reviews_to_csv(reviews, filename=None):
    if filename is None:
        filename = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "flipkart_reviews.csv")
        )
    else:
        filename = os.path.abspath(filename)

    df = pd.DataFrame(reviews)
    if len(df) > 0:
        # Drop internal duplicates, keep first
        df = df.drop_duplicates(subset=["review"], keep="first")
        # Export (don't include review_id column in exported file if you don't want it)
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"\n✅ Saved {len(df)} reviews to: {filename}")
    else:
        print("\n⚠️ No reviews to save.")

# ------------------------------
# CLI
# ------------------------------

if __name__ == "__main__":
    # Product PID from your link:
    PID = "COMGRHJUCYY63HAA"

    print("=" * 80)
    print("FLIPKART REVIEWS SCRAPER (API)")
    print("=" * 80 + "\n")

    # how many reviews you want
    MAX_REVIEWS = 500

    reviews = scrape_flipkart_by_pid(PID, max_reviews=MAX_REVIEWS)

    print(f"\n📊 Total reviews collected: {len(reviews)}")

    save_reviews_to_csv(reviews)
