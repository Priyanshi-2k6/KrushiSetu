import os
import json
import time
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


class SubsidyRecommander:
    """
    Same pipeline as before (evaluate -> generate), but implemented as
    plain sequential Python methods instead of a LangGraph StateGraph.

    Why drop LangGraph: this pipeline is linear — no branching, no
    conditional routing, no loops. LangGraph's actual value is modeling
    a GRAPH (multiple possible paths through nodes). A straight line of
    steps doesn't need a graph engine; it just needs functions called
    in order. Plain LangChain (or even plain Python) expresses that
    with less machinery and one less dependency to justify.
    """

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.model = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.2,
            max_tokens=4000,
            timeout=45
        )

    # ---------------------- Step 1: evaluate ---------------------- #
    def _evaluate_subsidies(self, farmer_profile: Dict[str, Any], all_subsidies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        start = time.time()

        if not all_subsidies:
            return []

        subsidy_brief = [
            {
                "id": s.get("id"),
                "title": s.get("title"),
                "description": s.get("description", "")[:300],
                "amount": s.get("amount"),
                "eligibility_criteria": s.get("eligibility_criteria", []),
            }
            for s in all_subsidies
        ]

        user_prompt = f"""You are evaluating subsidies for ONE farmer against a list of subsidies.

Farmer Profile:
- Income: {farmer_profile.get('income')}
- Land Size: {farmer_profile.get('land_size')} acres
- Farmer Type: {farmer_profile.get('farmer_type')}
- Crop: {farmer_profile.get('crop_type')}
- Season: {farmer_profile.get('season')}
- Soil Type: {farmer_profile.get('soil_type')}
- State: {farmer_profile.get('state')}
- District: {farmer_profile.get('district')}

Subsidies (JSON array):
{json.dumps(subsidy_brief)}

For EVERY subsidy in the list above, decide:
1. "eligible": true/false — does the farmer meet the eligibility_criteria (if the criteria list is empty, treat as eligible)?
2. "score": 0-100 relevance score, ONLY meaningful if eligible=true (use 0 if not eligible).
   Score based on: crop match (40pts), income/land fit (30pts), region relevance (20pts), timing (10pts)
3. "reasoning": one short sentence.
4. "key_benefits": up to 2 short bullet phrases (empty list if not eligible).

Return ONLY valid JSON, no markdown, no explanation outside the JSON, in this exact shape:
{{"results": [{{"id": <subsidy_id>, "eligible": true, "score": 85, "reasoning": "...", "key_benefits": ["...", "..."]}}, ...]}}
Include one entry per subsidy id from the input list, in any order."""

        messages = [
            SystemMessage(content="You are a subsidy eligibility and scoring engine. Respond only with valid JSON, matching every subsidy id given."),
            HumanMessage(content=user_prompt)
        ]

        evaluated = []
        try:
            response = self.model.invoke(messages)
            parsed = json.loads(response.content)
            results_by_id = {r.get("id"): r for r in parsed.get("results", [])}

            for subsidy in all_subsidies:
                result = results_by_id.get(subsidy.get("id"))
                if result is None:
                    subsidy["eligible"] = True
                    subsidy["score"] = 0
                    subsidy["scoring_reasoning"] = "Not evaluated by AI (missing from model response)."
                    subsidy["key_benefits"] = []
                else:
                    subsidy["eligible"] = result.get("eligible", True)
                    subsidy["score"] = result.get("score", 0)
                    subsidy["scoring_reasoning"] = result.get("reasoning", "")
                    subsidy["key_benefits"] = result.get("key_benefits", [])
                evaluated.append(subsidy)

        except Exception as e:
            print(f"evaluate_subsidies fallback triggered: {e}")
            for subsidy in all_subsidies:
                subsidy["eligible"] = True
                subsidy["score"] = 0
                subsidy["scoring_reasoning"] = "AI evaluation unavailable, showing all subsidies."
                subsidy["key_benefits"] = []
                evaluated.append(subsidy)

        eligible_only = [s for s in evaluated if s.get("eligible")]
        eligible_only.sort(key=lambda x: x.get("score", 0), reverse=True)

        print(f"Evaluate: {time.time()-start:.1f}s → {len(eligible_only)} eligible (1 LLM call total)")
        return eligible_only

    # ---------------------- Step 2: generate ---------------------- #
    def _generate_recommendations(self, evaluated_subsidies: List[Dict[str, Any]]) -> Dict[str, Any]:
        top_subsidies = evaluated_subsidies[:5]
        recommended_subsidies = []

        for i, subsidy in enumerate(top_subsidies, 1):
            recommended_subsidies.append({
                "rank": i,
                "subsidy_id": subsidy.get("id"),
                "title": subsidy.get("title"),
                "description": subsidy.get("description", ""),
                "amount": subsidy.get("amount", 0),
                "relevance_score": subsidy.get("score", 0),
                "why_recommended": subsidy.get("scoring_reasoning", ""),
                "key_benefits": subsidy.get("key_benefits", []),
                "application_dates": {
                    "start": subsidy.get("application_start_date", "N/A"),
                    "end": subsidy.get("application_end_date", "N/A")
                },
                "documents_required": subsidy.get("documents_required", [])
            })

        return {
            "recommended_subsidies": recommended_subsidies,
            "total_recommended": len(evaluated_subsidies)
        }

    # ---------------------- Entry point (same signature as before) ---------------------- #
    def recommend_subsidies(self, farmer_profile: Dict[str, Any], all_subsidies: List[Dict[str, Any]]) -> Dict[str, Any]:
        overall_start = time.time()

        evaluated_subsidies = self._evaluate_subsidies(farmer_profile, all_subsidies)
        final_recommendations = self._generate_recommendations(evaluated_subsidies)

        total_time = time.time() - overall_start
        print(f"{'='*60}")
        print(f" TOTAL TIME: {total_time:.1f}s")
        print(f"{'='*60}\n")

        return final_recommendations


if __name__ == "__main__":
    recommander = SubsidyRecommander()