"""
Simple test script for SPARK Analytics endpoints
Run this after starting your FastAPI server
"""
import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000/api/v1"

def print_response(title: str, response: requests.Response):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")

def test_health_check():
    """Test health check endpoint"""
    print("\n🔍 Testing Health Check...")
    response = requests.get(f"{BASE_URL.replace('/api/v1', '')}/health")
    print_response("Health Check", response)
    return response.status_code == 200

def test_get_teams():
    """Get list of teams (need team IDs for predictions)"""
    print("\n🔍 Testing Get Teams...")
    response = requests.get(f"{BASE_URL}/teams/")
    print_response("Get Teams", response)
    
    if response.status_code == 200:
        teams = response.json()
        if teams:
            print(f"\n✅ Found {len(teams)} teams")
            print("\nFirst few teams:")
            for team in teams[:3]:
                print(f"  - ID: {team.get('team_id')}, Name: {team.get('name')}")
            # Return first two team IDs for match prediction testing
            if len(teams) >= 2:
                return (teams[0].get('team_id'), teams[1].get('team_id'))
            return (teams[0].get('team_id'), None) if teams else (None, None)
        else:
            print("\n⚠️  No teams found. You may need to create teams first.")
            return (None, None)
    return (None, None)

def test_match_prediction(home_team_id: int = 1, away_team_id: int = 2):
    """Test match outcome prediction"""
    print("\n🔍 Testing Match Outcome Prediction...")
    
    # Example match data - adjust based on your needs
    match_data = {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_points": 45,
        "home_wins": 13,
        "home_draws": 6,
        "home_losses": 5,
        "home_gf": 38,  # goals for
        "home_ga": 22,  # goals against
        "home_gd": 16,  # goal difference
        "away_points": 38,
        "away_wins": 11,
        "away_draws": 5,
        "away_losses": 8,
        "away_gf": 32,
        "away_ga": 28,
        "away_gd": 4,
        # Optional xG data
        "home_xg": 1.8,
        "away_xg": 1.5
    }
    
    response = requests.post(
        f"{BASE_URL}/analytics/predict/match",
        json=match_data
    )
    
    print_response("Match Prediction", response)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Prediction Summary:")
        print(f"   Home Win Probability: {data.get('home_win_probability'):.1%}")
        print(f"   Draw Probability: {data.get('draw_probability'):.1%}")
        print(f"   Away Win Probability: {data.get('away_win_probability'):.1%}")
        print(f"   Predicted Outcome: {data.get('predicted_outcome')}")
        return True
    return False

def test_season_prediction(team_id: int = 1):
    """Test season performance prediction"""
    print("\n🔍 Testing Season Performance Prediction...")
    
    # Example season data - adjust based on your needs
    season_data = {
        "team_id": team_id,
        "wins": 13,
        "draws": 6,
        "losses": 5,
        "goals_for": 38,
        "goals_against": 22,
        "goal_diff": 16,
        "points": 45
    }
    
    response = requests.post(
        f"{BASE_URL}/analytics/predict/season",
        json=season_data
    )
    
    print_response("Season Prediction", response)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Prediction Summary:")
        print(f"   Predicted Final Points: {data.get('predicted_points'):.1f}")
        print(f"   Predicted Final Position: {data.get('predicted_position'):.1f}")
        return True
    return False

def test_error_handling():
    """Test error handling with invalid data"""
    print("\n🔍 Testing Error Handling...")
    
    # Test with invalid team ID
    print("\n1. Testing with invalid team ID...")
    invalid_match = {
        "home_team_id": 99999,
        "away_team_id": 99998,
        "home_points": 45,
        "home_wins": 13,
        "home_draws": 6,
        "home_losses": 5,
        "home_gf": 38,
        "home_ga": 22,
        "home_gd": 16,
        "away_points": 38,
        "away_wins": 11,
        "away_draws": 5,
        "away_losses": 8,
        "away_gf": 32,
        "away_ga": 28,
        "away_gd": 4
    }
    
    response = requests.post(
        f"{BASE_URL}/analytics/predict/match",
        json=invalid_match
    )
    print(f"   Status: {response.status_code} (Expected: 404)")
    
    # Test with missing required fields
    print("\n2. Testing with missing required fields...")
    incomplete_data = {"team_id": 1}
    response = requests.post(
        f"{BASE_URL}/analytics/predict/season",
        json=incomplete_data
    )
    print(f"   Status: {response.status_code} (Expected: 422)")

def main():
    """Run all tests"""
    print("="*60)
    print("SPARK Analytics API Test Suite")
    print("="*60)
    print("\n⚠️  Make sure your FastAPI server is running on http://localhost:8000")
    print("   Start it with: python -m uvicorn src.main:app --reload")
    
    input("\nPress Enter to continue...")
    
    # Test health check
    if not test_health_check():
        print("\n❌ Health check failed. Is the server running?")
        return
    
    # Get teams (to get valid team IDs)
    team_ids = test_get_teams()
    
    if team_ids and team_ids[0]:
        home_team_id, away_team_id = team_ids
        # Use second team ID if available, otherwise use first team ID + 1
        if not away_team_id and team_ids[0]:
            away_team_id = team_ids[0] + 1
        
        # Test match prediction with different teams
        if away_team_id:
            test_match_prediction(home_team_id=home_team_id, away_team_id=away_team_id)
        else:
            print("\n⚠️  Only one team found, using it for both home and away (not ideal)")
            test_match_prediction(home_team_id=home_team_id, away_team_id=home_team_id)
        
        # Test season prediction
        test_season_prediction(team_id=home_team_id)
    else:
        print("\n⚠️  Skipping prediction tests - no teams found")
        print("   You can still test manually using the example data in TESTING_GUIDE.md")
    
    # Test error handling
    test_error_handling()
    
    print("\n" + "="*60)
    print("✅ Testing Complete!")
    print("="*60)
    print("\n💡 Tip: Use Swagger UI at http://localhost:8000/docs for interactive testing")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the server.")
        print("   Make sure FastAPI is running on http://localhost:8000")
        print("   Start it with: python -m uvicorn src.main:app --reload")
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

