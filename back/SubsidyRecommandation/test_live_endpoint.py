"""Debug test - hits the live Django endpoint on port 8001 and prints FULL response or error."""
import urllib.request
import json

data = json.dumps({
    "farmer_type": "small farmer",
    "crop_type": "wheat",
    "season": "rabi",
    "soil_type": "alluvial",
    "water_sources": ["canal"],
    "state": "Punjab",
    "district": "Ludhiana",
    "rainfall_region": "moderate",
    "temperature_zone": "temperate",
    "income": "80000",
    "land_size": "2"
}).encode()

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/subsidy-recommendations/recommend/",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
        parsed = json.loads(raw)

        print("=" * 60)
        print("SUCCESS:", parsed.get("success"))
        print("TOTAL FOUND:", parsed.get("total_found"))
        print("=" * 60)

        recs = parsed.get("recommendations", [])
        if recs:
            print("\nTOP RECOMMENDATIONS (from real DB):")
            for rec in recs:
                title = rec.get("title", "").encode("ascii", "replace").decode()
                print(f"  #{rec['rank']}  score={rec['relevance_score']}%  {title}")
        else:
            print("\nWARNING: No recommendations returned!")
            print("Full response:")
            print(json.dumps(parsed, indent=2, ensure_ascii=True)[:2000])

        print("\nLLM ANALYSIS:")
        llm = parsed.get("llm_analysis", "(none)").encode("ascii", "replace").decode()
        print(llm[:800])

except urllib.error.HTTPError as e:
    print("HTTP ERROR:", e.code, e.reason)
    body = e.read().decode("utf-8", errors="replace")
    print("RESPONSE BODY:")
    print(body[:3000])
except urllib.error.URLError as e:
    print("CONNECTION ERROR:", e.reason)
    print(">>> Is the Django server running? Start it with: python manage.py runserver 8001")
except Exception as e:
    print("UNEXPECTED ERROR:", type(e).__name__, str(e))
