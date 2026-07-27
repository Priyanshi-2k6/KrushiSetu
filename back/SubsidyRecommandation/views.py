from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache
from .SubsidyRecommander import SubsidyRecommander
from app.models import Subsidy
import os
import hashlib
import json


def _derive_category(title: str, description: str) -> str:
    """Derive a category label from title/description since the model has no category field."""
    text = f"{title} {description}".lower()
    if any(w in text for w in ['drip', 'irrigation', 'water', 'pump', 'sprinkler']):
        return 'Irrigation'
    if any(w in text for w in ['seed', 'crop', 'fertilizer', 'pesticide', 'organic', 'soil']):
        return 'Crop Input'
    if any(w in text for w in ['equipment', 'tractor', 'machine', 'tool', 'implement']):
        return 'Equipment'
    if any(w in text for w in ['insurance', 'fasal bima', 'loss', 'disaster', 'flood', 'drought']):
        return 'Insurance'
    if any(w in text for w in ['loan', 'credit', 'kisan', 'finance', 'bank']):
        return 'Credit'
    if any(w in text for w in ['solar', 'energy', 'power', 'electricity']):
        return 'Energy'
    if any(w in text for w in ['storage', 'warehouse', 'cold chain', 'market']):
        return 'Post-Harvest'
    return 'General'


@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def recommend_subsidies(request):
    try:
        # Extract farmer_profile from request
        request_data = request.data.get('farmer_profile', request.data)

        farmer_profile = {
            "income": request_data.get("income", ""),
            "farmer_type": request_data.get("farmer_type", ""),
            "land_size": request_data.get("land_size", ""),
            "crop_type": request_data.get("crop_type", ""),
            "season": request_data.get("season", ""),
            "soil_type": request_data.get("soil_type", ""),
            "water_sources": request_data.get("water_sources", []),
            "state": request_data.get("state", ""),
            "district": request_data.get("district", ""),
            "rainfall_region": request_data.get("rainfall_region", ""),
            "temperature_zone": request_data.get("temperature_zone", ""),
            "past_subsidies": request_data.get("past_subsidies", []),
        }

        required_field = ["income", "farmer_type", "land_size", "crop_type", "state"]
        missing_fields = [field for field in required_field if not farmer_profile.get(field)]

        if missing_fields:
            return Response({
                "success": False,
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }, status=status.HTTP_400_BAD_REQUEST)

        # ------------------------ Load Subsidies From DB (with caching) ---------------------
        subsidies_cache_key = "all_subsidies_data_v2"
        subsidies = cache.get(subsidies_cache_key)

        if subsidies is None:
            raw_subsidies = Subsidy.objects.all().values(
                'id', 'title', 'description', 'amount', 'eligibility',
                'documents_required', 'application_start_date', 'application_end_date', 'rating'
            )
            subsidies = []
            for s in raw_subsidies:
                subsidies.append({
                    'id': s['id'],
                    'title': s['title'],
                    'description': s['description'],
                    'amount': float(s['amount']),
                    'category': _derive_category(s['title'], s['description']),
                    'rating': float(s.get('rating') or 0),
                    'eligibility_criteria': s['eligibility'] if s['eligibility'] else [],
                    'documents_required': s['documents_required'] if s['documents_required'] else [],
                    'application_start_date': s['application_start_date'].isoformat() if s['application_start_date'] else None,
                    'application_end_date': s['application_end_date'].isoformat() if s['application_end_date'] else None,
                })
            # Cache formatted list for 30 minutes (subsidies change infrequently)
            cache.set(subsidies_cache_key, subsidies, 1800)
            print("Loaded subsidies from database")
        else:
            print("Loaded subsidies from cache")

        if not subsidies:
            return Response({
                "success": False,
                "error": "No subsidies available in the system."
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Create a cache key for this specific farmer profile + subsidy set
        cache_key_data = {
            'farmer_profile': farmer_profile,
            'subsidy_count': len(subsidies)
        }
        cache_key = f"subsidy_rec_{hashlib.md5(json.dumps(cache_key_data, sort_keys=True).encode()).hexdigest()}"

        # Try recommendation cache first (5-minute TTL)
        recommendation_result = cache.get(cache_key)

        if recommendation_result is None:
            try:
                recommender = SubsidyRecommander()
                recommendation_result = recommender.recommend_subsidies(farmer_profile, subsidies)
                cache.set(cache_key, recommendation_result, 300)
                print("Generated new recommendations for farmer profile")
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise
        else:
            print("Retrieved recommendations from cache")

        formatted_response = {
            "success": True,
            "recommendations": recommendation_result.get("recommended_subsidies", []),
            "total_found": recommendation_result.get("total_recommended", 0),
            "llm_analysis": recommendation_result.get("llm_analysis", ""),
            "retrieval_context": recommendation_result.get("retrieval_context", ""),
            "summary": (
                f"Based on your profile as a {farmer_profile.get('farmer_type', 'farmer')} "
                f"with {farmer_profile.get('land_size', 'unknown')} acres growing "
                f"{farmer_profile.get('crop_type', 'crops')} in "
                f"{farmer_profile.get('district', 'your area')}, {farmer_profile.get('state', '')}, "
                f"we found {recommendation_result.get('total_recommended', 0)} eligible subsidies "
                "tailored to your needs."
            ),
        }

        return Response(formatted_response, status=status.HTTP_200_OK)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({
            "success": False,
            "error": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([AllowAny])
def recommendation_status(request):
    return Response({
        "success": True,
        "message": "Subsidy Recommendation Service is operational."
    }, status=status.HTTP_200_OK)
