import os
import json
import time
from typing import TypedDict, List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


# load_dotenv()

class RecommendationState(TypedDict) :
    farmer_profile : Dict[str, Any]
    all_subsidies : List[Dict[str,Any]]
    eligible_subsidies : List[Dict[str,Any]]
    scored_subsidies : List[Dict[str,Any]]
    recommended_subsidies : List[Dict[str,Any]]
    analysis : str
    final_recommendations : Dict[str, Any]
    
class SubsidyRecommander:
    
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.model = ChatGroq(
            model = "openai/gpt-oss-120b", 
            temperature=0.3,
            max_tokens=1500,  
            timeout=30
        )
        self.graph = self.build_graph()
    
    def build_graph(self) -> StateGraph:
        
        graph = StateGraph(RecommendationState)
        
        # --------------------- Define Nodes --------------------- #
        graph.add_node("filter_eligibility", self._filter_eligibility)
        graph.add_node("score_subsidies", self._score_subsidies)
        graph.add_node("generate_recommendations", self._generate_recommendations)
        
        # --------------------- Define Edges --------------------- #
        graph.add_edge(START,"filter_eligibility")
        graph.add_edge("filter_eligibility","score_subsidies")
        graph.add_edge("score_subsidies","generate_recommendations")
        graph.add_edge("generate_recommendations",END)
        
        return graph.compile()

    # ---------------------- filter_eligibility Node ---------------------- #
    def _filter_eligibility(self, state: RecommendationState) -> RecommendationState:
        start = time.time()
        farmer_profile = state['farmer_profile']
        all_subsidies = state['all_subsidies']
        eligible_subsidies = []
        
        for subsidy in all_subsidies : 
            eligibility = subsidy.get('eligibility_criteria', [])
            
            # if eligibility is defined that time it goes to the ai model for checking
            if eligibility :
                user_prompt =  f"""Does this farmer meet the eligibility criteria for this subsidy?
                                Farmer Profile:
                                - Income: {farmer_profile.get('income')}
                                - Land Size: {farmer_profile.get('land_size')} acres
                                - Farmer Type: {farmer_profile.get('farmer_type')}
                                - Crop: {farmer_profile.get('crop_type')}
                                - State: {farmer_profile.get('state')}

                                Subsidy: {subsidy.get('title')}
                                Eligibility Criteria: {json.dumps(eligibility)}

                                Answer with JSON:
                                {{"eligible": true/false, "reason": "brief explanation"}}"""

                try:
                    messages = [
                        SystemMessage(content = "You are an eligibility checker. Respond only with valid JSON."),
                        HumanMessage(content = user_prompt)
                    ]
                    
                    response = self.model.invoke(messages)
                    result = json.loads(response.content)
                    
                    if result.get('eligible', False):
                        eligible_subsidies.append(subsidy)
                
                except:
                    eligible_subsidies.append(subsidy)
                    
            else :
                eligible_subsidies.append(subsidy)
        
        state['eligible_subsidies'] = eligible_subsidies
        print(f"Filter: {time.time()-start:.1f}s → {len(eligible_subsidies)} eligible")
        return state

    # ---------------------- score_subsidies Node ---------------------- #
    def _score_subsidies(self, state: RecommendationState) -> RecommendationState:
        start = time.time()
        farmer_profile = state['farmer_profile']
        eligible_subsidies = state['eligible_subsidies']
        scored_subsidies = []
        
        for subsidy in eligible_subsidies : 
            user_prompt = f"""Score this subsidy's relevance (0-100) for this farmer.
                            Farmer: {farmer_profile.get('farmer_type')} with {farmer_profile.get('land_size')} acres, growing {farmer_profile.get('crop_type')}, income ₹{farmer_profile.get('income')}, from {farmer_profile.get('district')}, {farmer_profile.get('state')}
                            Subsidy: {subsidy.get('title')} - {subsidy.get('description', 'N/A')} (Amount: ₹{subsidy.get('amount')})
                            Score based on: crop match (40pts), income/land fit (30pts), region relevance (20pts), timing (10pts)
                            Return ONLY this JSON format, no markdown, no explanation:
                            {{"score": 85, "reasoning": "Brief reason for score", "key_benefits": ["benefit1", "benefit2"]}}"""
        
            messages = [
                SystemMessage(content="You are a subsidy scorer. Return ONLY valid JSON, no markdown formatting."),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.model.invoke(messages)
            result = json.loads(response.content)
            subsidy['score'] = result.get('score', 0)
            subsidy['scoring_reasoning'] = result.get('reasoning', '')
            subsidy['key_benefits'] = result.get('key_benefits', [])
            scored_subsidies.append(subsidy)
        
        scored_subsidies.sort(key=lambda x: x.get('score', 0), reverse=True)
        state['scored_subsidies'] = scored_subsidies
        print(f"Score: {time.time()-start:.1f}s → {len(scored_subsidies)} scored")
        return state

    # ---------------------- generate_recommendations Node ---------------------- #
    def _generate_recommendations(self, state: RecommendationState) -> RecommendationState:
        top_subsidies = state['scored_subsidies'][:5]
        recommendate_subsidies = []
        
        for i, subsidy in enumerate(top_subsidies,1):
            recommendate_subsidies.append({
                "rank": i,
                "subsidy_id": subsidy.get('id'),
                "title": subsidy.get('title'),
                "description": subsidy.get('description',""),
                "amount": subsidy.get('amount',0),
                "relevance_score": subsidy.get('score',0),
                "why_recommended": subsidy.get('scoring_reasoning',""),
                "key_benefits": subsidy.get('key_benefits',[]),
                "application_dates": {
                    "start": subsidy.get('application_start_date',"N/A"),
                    "end": subsidy.get('application_end_date',"N/A")
                },
                "documents_required": subsidy.get('documents_required', [])
            })
            
        state['final_recommendations'] = {
            "recommended_subsidies": recommendate_subsidies,
            "total_recommended": len(state['eligible_subsidies'])
        }
        return state

    # ---------------------- End of Nodes --------------------- #
    def recommend_subsidies(self, farmer_profile: Dict[str,Any], all_subsidies: List[Dict[str,Any]]) -> Dict[str,Any]:
        overall_start = time.time()
        
        initial_state : RecommendationState = {
            "farmer_profile" : farmer_profile,
            "all_subsidies" : all_subsidies,
            "eligible_subsidies" : [],
            "scored_subsidies" : [],
            "recommended_subsidies" : [],
            "analysis" : "",
            "final_recommendations" : {}
        }
        
        result = self.graph.invoke(initial_state)
        
        # Print total time taken
        total_time = time.time() - overall_start
        print(f"{'='*60}")
        print(f" TOTAL TIME: {total_time:.1f}s")
        print(f"{'='*60}\n")
        
        return result['final_recommendations']

if __name__ == "__main__":
    recommander = SubsidyRecommander()