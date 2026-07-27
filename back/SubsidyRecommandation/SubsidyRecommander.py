import json
import math
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph


class RecommendationState(TypedDict):
    farmer_profile: Dict[str, Any]
    all_subsidies: List[Dict[str, Any]]
    eligible_subsidies: List[Dict[str, Any]]
    scored_subsidies: List[Dict[str, Any]]
    retrieval_context: str
    recommended_subsidies: List[Dict[str, Any]]
    analysis: str
    final_recommendations: Dict[str, Any]


class SubsidyRecommander:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.model = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1500,
            timeout=30,
        )
        self.graph = self.build_graph()

    def build_graph(self) -> StateGraph:
        graph = StateGraph(RecommendationState)

        graph.add_node("retrieve_relevant_subsidies", self._retrieve_relevant_subsidies)
        graph.add_node("format_retrieval_context", self._format_retrieval_context)
        graph.add_node("generate_recommendations", self._generate_recommendations)

        graph.add_edge(START, "retrieve_relevant_subsidies")
        graph.add_edge("retrieve_relevant_subsidies", "format_retrieval_context")
        graph.add_edge("format_retrieval_context", "generate_recommendations")
        graph.add_edge("generate_recommendations", END)

        return graph.compile()

    # ---------------------- retrieve_relevant_subsidies Node ---------------------- #
    def _retrieve_relevant_subsidies(
        self, state: RecommendationState, top_k: int = 5
    ) -> RecommendationState:
        start = time.time()
        farmer_profile = state["farmer_profile"]
        all_subsidies = state["all_subsidies"]

        if not all_subsidies:
            state["eligible_subsidies"] = []
            state["scored_subsidies"] = []
            print("TF-IDF retrieval: 0.0s -> 0 candidates")
            return state

        farmer_text = self._farmer_profile_text(farmer_profile)
        subsidy_texts = [self._subsidy_search_text(subsidy) for subsidy in all_subsidies]
        scores = self._tfidf_cosine_scores(farmer_text, subsidy_texts)

        scored_subsidies = []
        for subsidy, score in zip(all_subsidies, scores):
            ranked_subsidy = dict(subsidy)
            ranked_subsidy["score"] = round(score * 100, 2)
            ranked_subsidy["scoring_reasoning"] = self._retrieval_reason(
                farmer_profile, subsidy, score
            )
            ranked_subsidy["key_benefits"] = self._extract_key_benefits(subsidy)
            scored_subsidies.append(ranked_subsidy)

        scored_subsidies.sort(key=lambda item: item.get("score", 0), reverse=True)
        top_subsidies = scored_subsidies[:top_k]

        state["eligible_subsidies"] = top_subsidies
        state["scored_subsidies"] = top_subsidies
        print(f"TF-IDF retrieval: {time.time() - start:.1f}s -> {len(top_subsidies)} candidates")
        return state

    # Compatibility wrappers for older tests or imports that call the old nodes.
    def _score_subsidies(self, state: RecommendationState) -> RecommendationState:
        return self._retrieve_relevant_subsidies(state)

    def _filter_eligibility(self, state: RecommendationState) -> RecommendationState:
        state["eligible_subsidies"] = state.get("all_subsidies", [])
        return state

    # ---------------------- format_retrieval_context Node ---------------------- #
    def _format_retrieval_context(self, state: RecommendationState) -> RecommendationState:
        context_blocks = []

        for index, subsidy in enumerate(state["scored_subsidies"][:5], 1):
            context_blocks.append(
                "\n".join(
                    [
                        f"Candidate {index}",
                        f"ID: {subsidy.get('id')}",
                        f"Title: {subsidy.get('title', 'N/A')}",
                        f"Category: {subsidy.get('category', 'N/A')}",
                        f"Description: {subsidy.get('description', 'N/A')}",
                        "Eligibility: "
                        f"{json.dumps(subsidy.get('eligibility_criteria', []), ensure_ascii=False)}",
                        f"Benefit Amount: {subsidy.get('amount', 0)}",
                        f"TF-IDF Score: {subsidy.get('score', 0)}",
                    ]
                )
            )

        state["retrieval_context"] = "\n\n".join(context_blocks)
        return state

    # ---------------------- generate_recommendations Node ---------------------- #
    def _generate_recommendations(self, state: RecommendationState) -> RecommendationState:
        top_subsidies = state["scored_subsidies"][:5]
        recommended_subsidies = []
        llm_analysis = (
            self._generate_grounded_llm_analysis(state)
            if top_subsidies
            else "No matching subsidy candidates were retrieved."
        )

        for index, subsidy in enumerate(top_subsidies, 1):
            recommended_subsidies.append(
                {
                    "rank": index,
                    "subsidy_id": subsidy.get("id"),
                    "title": subsidy.get("title"),
                    "description": subsidy.get("description", ""),
                    "amount": subsidy.get("amount", 0),
                    "category": subsidy.get("category", "N/A"),
                    "relevance_score": subsidy.get("score", 0),
                    "why_recommended": subsidy.get("scoring_reasoning", ""),
                    "key_benefits": subsidy.get("key_benefits", []),
                    "application_dates": {
                        "start": subsidy.get("application_start_date", "N/A"),
                        "end": subsidy.get("application_end_date", "N/A"),
                    },
                    "documents_required": subsidy.get("documents_required", []),
                }
            )

        state["final_recommendations"] = {
            "recommended_subsidies": recommended_subsidies,
            "total_recommended": len(recommended_subsidies),
            "llm_analysis": llm_analysis,
            "retrieval_context": state.get("retrieval_context", ""),
        }
        return state

    def _generate_grounded_llm_analysis(self, state: RecommendationState) -> str:
        allowed_ids = [str(subsidy.get("id")) for subsidy in state["scored_subsidies"][:5]]
        user_prompt = f"""Farmer Profile:
{json.dumps(state["farmer_profile"], indent=2, ensure_ascii=False)}

Retrieved subsidy candidates:
{state.get("retrieval_context", "")}

Recommend subsidies only from the retrieved candidate list above.
Do not use outside schemes, general knowledge, or assumptions not present in the candidate list.
If a scheme is not in the candidate list, do not mention it.
Explain briefly why each recommended candidate fits the farmer profile.
Use the exact candidate title and ID from the list."""

        messages = [
            SystemMessage(
                content=(
                    "You are KrushiSetu's subsidy recommendation assistant. "
                    "You must ground your answer only in the provided retrieved candidates. "
                    f"Allowed candidate IDs: {', '.join(allowed_ids)}."
                )
            ),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.model.invoke(messages)
            return response.content
        except Exception as exc:
            return f"LLM generation unavailable for the retrieved candidates: {exc}"

    def _farmer_profile_text(self, farmer_profile: Dict[str, Any]) -> str:
        values = [
            farmer_profile.get("farmer_type", ""),
            farmer_profile.get("crop_type", ""),
            farmer_profile.get("season", ""),
            farmer_profile.get("soil_type", ""),
            " ".join(map(str, farmer_profile.get("water_sources", []))),
            farmer_profile.get("state", ""),
            farmer_profile.get("district", ""),
            farmer_profile.get("rainfall_region", ""),
            farmer_profile.get("temperature_zone", ""),
            str(farmer_profile.get("income", "")),
            str(farmer_profile.get("land_size", "")),
        ]
        return " ".join(str(value) for value in values if value)

    def _subsidy_search_text(self, subsidy: Dict[str, Any]) -> str:
        values = [
            subsidy.get("title", ""),
            subsidy.get("description", ""),
            subsidy.get("category", ""),
            json.dumps(subsidy.get("eligibility_criteria", []), ensure_ascii=False),
            json.dumps(subsidy.get("documents_required", []), ensure_ascii=False),
            str(subsidy.get("amount", "")),
        ]
        return " ".join(str(value) for value in values if value)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", str(text).lower())

    def _tfidf_cosine_scores(self, query: str, documents: List[str]) -> List[float]:
        tokenized_docs = [self._tokenize(document) for document in documents]
        query_tokens = self._tokenize(query)
        vocabulary = sorted(set(query_tokens).union(*(set(tokens) for tokens in tokenized_docs)))

        if not vocabulary:
            return [0.0 for _ in documents]

        doc_count = len(tokenized_docs)
        document_frequencies = {
            term: sum(1 for tokens in tokenized_docs if term in tokens)
            for term in vocabulary
        }

        def vectorize(tokens: List[str]) -> List[float]:
            counts = Counter(tokens)
            total = len(tokens) or 1
            vector = []
            for term in vocabulary:
                tf = counts.get(term, 0) / total
                idf = math.log((1 + doc_count) / (1 + document_frequencies.get(term, 0))) + 1
                vector.append(tf * idf)
            return vector

        query_vector = vectorize(query_tokens)
        query_norm = math.sqrt(sum(value * value for value in query_vector))

        scores = []
        for tokens in tokenized_docs:
            doc_vector = vectorize(tokens)
            doc_norm = math.sqrt(sum(value * value for value in doc_vector))
            if not query_norm or not doc_norm:
                scores.append(0.0)
                continue

            dot_product = sum(q * d for q, d in zip(query_vector, doc_vector))
            scores.append(dot_product / (query_norm * doc_norm))

        return scores

    def _retrieval_reason(
        self, farmer_profile: Dict[str, Any], subsidy: Dict[str, Any], score: float
    ) -> str:
        farmer_terms = set(self._tokenize(self._farmer_profile_text(farmer_profile)))
        subsidy_terms = set(self._tokenize(self._subsidy_search_text(subsidy)))
        matches = sorted(farmer_terms.intersection(subsidy_terms))

        if matches:
            return (
                f"Retrieved by TF-IDF cosine similarity ({score:.2f}) using matching "
                f"profile terms: {', '.join(matches[:6])}."
            )

        return (
            f"Retrieved by TF-IDF cosine similarity ({score:.2f}) as one of the "
            "closest available subsidy matches."
        )

    def _extract_key_benefits(self, subsidy: Dict[str, Any]) -> List[str]:
        benefits = []
        amount = subsidy.get("amount")
        if amount not in (None, ""):
            benefits.append(f"Benefit amount: {amount}")

        documents = subsidy.get("documents_required") or []
        if documents:
            benefits.append(f"Documents required: {', '.join(map(str, documents[:3]))}")

        return benefits

    # ---------------------- End of Nodes --------------------- #
    def recommend_subsidies(
        self, farmer_profile: Dict[str, Any], all_subsidies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        overall_start = time.time()

        initial_state: RecommendationState = {
            "farmer_profile": farmer_profile,
            "all_subsidies": all_subsidies,
            "eligible_subsidies": [],
            "scored_subsidies": [],
            "retrieval_context": "",
            "recommended_subsidies": [],
            "analysis": "",
            "final_recommendations": {},
        }

        result = self.graph.invoke(initial_state)

        total_time = time.time() - overall_start
        print(f"{'=' * 60}")
        print(f" TOTAL TIME: {total_time:.1f}s")
        print(f"{'=' * 60}\n")

        return result["final_recommendations"]


if __name__ == "__main__":
    recommander = SubsidyRecommander()
