"""
test_recommendation_pipeline.py
---------------------------------
Standalone test for the SubsidyRecommander RAG pipeline.
Run from the `back/` directory:
    python SubsidyRecommandation/test_recommendation_pipeline.py

Requirements: GROQ_API_KEY must be set in .env or environment.
No Django DB used -- subsidies are mocked inline so the test is self-contained.
"""

import json
import os
import sys
from pathlib import Path

# Load .env so GROQ_API_KEY is available without running Django
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

sys.path.insert(0, str(Path(__file__).parent.parent))

from SubsidyRecommandation.SubsidyRecommander import SubsidyRecommander  # noqa: E402

SEP = "=" * 70
THIN = "-" * 70

# ---------------------------------------------------------------------------
# Mock subsidy data (realistic, covering multiple categories)
# ---------------------------------------------------------------------------
MOCK_SUBSIDIES = [
    {
        "id": 1,
        "title": "PM-KISAN Samman Nidhi",
        "description": "Direct income support of Rs 6000/year to small and marginal farmers owning cultivable land.",
        "category": "Credit",
        "amount": 6000.0,
        "eligibility_criteria": ["small farmer", "marginal farmer", "land owner", "income below 2 lakh"],
        "documents_required": ["Aadhaar", "land record", "bank account"],
        "application_start_date": "2025-01-01",
        "application_end_date": "2025-12-31",
    },
    {
        "id": 2,
        "title": "Pradhan Mantri Fasal Bima Yojana",
        "description": "Crop insurance scheme for kharif and rabi crops against natural calamities, pest attacks, and disease.",
        "category": "Insurance",
        "amount": 15000.0,
        "eligibility_criteria": ["all farmers", "kharif crop", "rabi crop", "loanee and non-loanee"],
        "documents_required": ["land record", "bank account", "Aadhaar"],
        "application_start_date": "2025-06-01",
        "application_end_date": "2025-07-31",
    },
    {
        "id": 3,
        "title": "PM Krishi Sinchai Yojana - Drip Irrigation Subsidy",
        "description": "Subsidy for drip and sprinkler irrigation systems to promote water use efficiency in farming.",
        "category": "Irrigation",
        "amount": 50000.0,
        "eligibility_criteria": ["all farmers", "horticultural crops", "vegetable growers", "water scarce region"],
        "documents_required": ["land record", "Aadhaar", "quotation from supplier"],
        "application_start_date": "2025-03-01",
        "application_end_date": "2025-09-30",
    },
    {
        "id": 4,
        "title": "Soil Health Card Scheme and Fertilizer Subsidy",
        "description": "Free soil testing and subsidized fertilizer supply based on soil health card recommendations.",
        "category": "Crop Input",
        "amount": 3000.0,
        "eligibility_criteria": ["all farmers", "wheat grower", "rice grower", "black soil", "red soil"],
        "documents_required": ["Aadhaar", "land record"],
        "application_start_date": "2025-01-01",
        "application_end_date": "2025-12-31",
    },
    {
        "id": 5,
        "title": "Kisan Credit Card (KCC) Scheme",
        "description": "Short-term credit facility for purchase of seeds, fertilizers, pesticides and allied activities.",
        "category": "Credit",
        "amount": 300000.0,
        "eligibility_criteria": ["farmer", "sharecropper", "tenant farmer", "marginal farmer", "small farmer"],
        "documents_required": ["Aadhaar", "land record", "bank account", "passport photo"],
        "application_start_date": "2025-01-01",
        "application_end_date": "2025-12-31",
    },
    {
        "id": 6,
        "title": "Solar Pump Subsidy Scheme",
        "description": "Subsidy on solar-powered water pumps for irrigation to reduce dependency on grid electricity.",
        "category": "Energy",
        "amount": 75000.0,
        "eligibility_criteria": ["all farmers", "dryland farmer", "irrigation need", "Gujarat", "Rajasthan", "Maharashtra"],
        "documents_required": ["Aadhaar", "land record", "electricity bill"],
        "application_start_date": "2025-04-01",
        "application_end_date": "2025-10-31",
    },
    {
        "id": 7,
        "title": "National Food Security Mission - Cotton Seed Subsidy",
        "description": "Subsidized high-yielding cotton seeds and pesticide kits for cotton farmers in dryland areas.",
        "category": "Crop Input",
        "amount": 8000.0,
        "eligibility_criteria": ["cotton farmer", "dryland", "Gujarat", "Maharashtra", "Telangana"],
        "documents_required": ["Aadhaar", "land record"],
        "application_start_date": "2025-05-01",
        "application_end_date": "2025-06-30",
    },
    {
        "id": 8,
        "title": "PM Kisan Tractor Subsidy Scheme",
        "description": "Subsidy up to 50% on purchase of tractors for small and medium farmers.",
        "category": "Equipment",
        "amount": 125000.0,
        "eligibility_criteria": ["small farmer", "medium farmer", "land owner", "no previous tractor subsidy"],
        "documents_required": ["Aadhaar", "land record", "bank account", "quotation"],
        "application_start_date": "2025-02-01",
        "application_end_date": "2025-11-30",
    },
]

# ---------------------------------------------------------------------------
# 3 realistic farmer test profiles
# ---------------------------------------------------------------------------
TEST_PROFILES = [
    {
        "_name": "Profile 1 -- Small wheat farmer, Punjab",
        "farmer_type": "marginal farmer",
        "crop_type": "wheat",
        "season": "rabi",
        "soil_type": "alluvial",
        "water_sources": ["canal", "tube well"],
        "state": "Punjab",
        "district": "Ludhiana",
        "rainfall_region": "moderate",
        "temperature_zone": "temperate",
        "income": "80000",
        "land_size": "2",
    },
    {
        "_name": "Profile 2 -- Cotton farmer, Gujarat, dryland",
        "farmer_type": "small farmer",
        "crop_type": "cotton",
        "season": "kharif",
        "soil_type": "black soil",
        "water_sources": ["rainwater"],
        "state": "Gujarat",
        "district": "Saurashtra",
        "rainfall_region": "semi-arid",
        "temperature_zone": "hot",
        "income": "120000",
        "land_size": "5",
    },
    {
        "_name": "Profile 3 -- Vegetable farmer, Maharashtra, drip irrigation",
        "farmer_type": "small farmer",
        "crop_type": "tomato vegetables",
        "season": "kharif",
        "soil_type": "red laterite",
        "water_sources": ["drip irrigation", "well"],
        "state": "Maharashtra",
        "district": "Nashik",
        "rainfall_region": "moderate",
        "temperature_zone": "tropical",
        "income": "200000",
        "land_size": "3",
    },
]


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
def run_tests():
    print("\n" + SEP)
    print("  KrushiSetu RAG Pipeline Test -- SubsidyRecommander")
    print(SEP)

    recommender = SubsidyRecommander()

    for profile in TEST_PROFILES:
        name = profile.pop("_name")
        print("\n" + THIN)
        print(f"  {name}")
        print(THIN)
        print(f"  Profile: {json.dumps(profile, indent=4)}")

        result = recommender.recommend_subsidies(profile, MOCK_SUBSIDIES)

        recs = result.get("recommended_subsidies", [])
        print(f"\n  [TF-IDF] Top {len(recs)} retrieved candidates:")
        for rec in recs:
            print(
                f"    #{rec['rank']}  score={rec['relevance_score']:.1f}%  "
                f"'{rec['title']}'  (Category: {rec['category']}, Amount: Rs {rec['amount']:,.0f})"
            )
            print(f"           Why: {rec['why_recommended']}")

        print("\n  [LLM Analysis]")
        llm = result.get("llm_analysis", "(no analysis)")
        for line in llm.splitlines():
            print(f"    {line}")

        # Restore key for safety
        profile["_name"] = name

    print("\n" + SEP)
    print("  All 3 profiles tested. Check output above for correctness.")
    print(SEP + "\n")


if __name__ == "__main__":
    run_tests()
