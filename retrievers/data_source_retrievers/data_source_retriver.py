import requests


def wikipedia_search(query, limit=1):
    search_url = "https://en.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "Mozilla/5.0 (educational-script)"
    }
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1,
        "srlimit": limit
    }
    response = requests.get(search_url, params=params, headers=headers, timeout=10)

    # ✅ SAFETY CHECK
    if response.status_code != 200:
        print("❌ Wikipedia API request failed")
        print("Status:", response.status_code)
        print(response.text[:200])
        return []

    try:
        data = response.json()
    except Exception:
        print("❌ Response is not valid JSON")
        print(response.text[:200])
        return []

    results = []

    if "query" not in data or "search" not in data["query"]:
        print("❌ No search results found")
        return []

    for item in data["query"]["search"]:
        title = item["title"]

        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

        page_response = requests.get(summary_url, headers=headers, timeout=10)

        if page_response.status_code != 200:
            continue

        try:
            page_data = page_response.json()
        except Exception:
            continue

        extract = page_data.get("extract", "")
        results.append(extract)

    return results


# ✅ TEST IT
if __name__ == "__main__":
    docs = wikipedia_search("Virat Kohli")

    for i, d in enumerate(docs, 1):
        print(f"\n--- RESULT {i} ---")
        print(d[:500])
