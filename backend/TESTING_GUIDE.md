# FastAPI Testing Guide for SPARK Analytics Endpoints

## 🚀 Starting the Server

1. **Make sure your database is running:**
   ```bash
   docker-compose up -d postgres
   ```

2. **Start the FastAPI server:**
   ```bash
   cd backend
   python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```
   
   Or if you have a script:
   ```bash
   cd backend
   python -m src.main
   ```

3. **Access the interactive API documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## 📋 Endpoints to Test

### 1. Health Check (Verify Server is Running)
- **GET** `http://localhost:8000/health`
- **GET** `http://localhost:8000/`

### 2. Get Teams (You'll need team IDs for predictions)
- **GET** `http://localhost:8000/api/v1/teams/`
- Returns list of teams with their IDs

### 3. Match Outcome Prediction
- **POST** `http://localhost:8000/api/v1/analytics/predict/match`
- Predicts Win/Draw/Loss probabilities for a match

### 4. Season Performance Prediction
- **POST** `http://localhost:8000/api/v1/analytics/predict/season`
- Predicts final points and league position for a team

## 🧪 Example Test Requests

### Test 1: Match Outcome Prediction

**Endpoint:** `POST /api/v1/analytics/predict/match`

**Request Body:**
```json
{
  "home_team_id": 1,
  "away_team_id": 2,
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
  "away_gd": 4,
  "home_xg": 1.8,
  "away_xg": 1.5
}
```

**Expected Response:**
```json
{
  "home_team_id": 1,
  "away_team_id": 2,
  "home_win_probability": 0.45,
  "draw_probability": 0.28,
  "away_win_probability": 0.27,
  "predicted_outcome": "H"
}
```

### Test 2: Season Performance Prediction

**Endpoint:** `POST /api/v1/analytics/predict/season`

**Request Body:**
```json
{
  "team_id": 1,
  "wins": 13,
  "draws": 6,
  "losses": 5,
  "goals_for": 38,
  "goals_against": 22,
  "goal_diff": 16,
  "points": 45
}
```

**Expected Response:**
```json
{
  "team_id": 1,
  "predicted_points": 68.5,
  "predicted_position": 5.2
}
```

## 🧪 Testing via Swagger UI (Easiest Method)

1. Open http://localhost:8000/docs in your browser
2. Find the **Analytics** section
3. Click on `/api/v1/analytics/predict/match` or `/api/v1/analytics/predict/season`
4. Click **"Try it out"**
5. Paste the example JSON into the request body
6. Click **"Execute"**
7. View the response below

## 🧪 Testing via cURL

### Match Prediction:
```bash
curl -X POST "http://localhost:8000/api/v1/analytics/predict/match" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team_id": 1,
    "away_team_id": 2,
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
  }'
```

### Season Prediction:
```bash
curl -X POST "http://localhost:8000/api/v1/analytics/predict/season" \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": 1,
    "wins": 13,
    "draws": 6,
    "losses": 5,
    "goals_for": 38,
    "goals_against": 22,
    "goal_diff": 16,
    "points": 45
  }'
```

## 🧪 Testing via Python

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000/api/v1"

# Test Match Prediction
match_prediction = {
    "home_team_id": 1,
    "away_team_id": 2,
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

response = requests.post(f"{BASE_URL}/analytics/predict/match", json=match_prediction)
print("Match Prediction:", response.json())

# Test Season Prediction
season_prediction = {
    "team_id": 1,
    "wins": 13,
    "draws": 6,
    "losses": 5,
    "goals_for": 38,
    "goals_against": 22,
    "goal_diff": 16,
    "points": 45
}

response = requests.post(f"{BASE_URL}/analytics/predict/season", json=season_prediction)
print("Season Prediction:", response.json())
```

## ✅ What to Check

1. **Health Check:**
   - Server responds with status 200
   - Database connection works

2. **Match Prediction:**
   - Returns probabilities between 0 and 1
   - All three probabilities (home, draw, away) sum to ~1.0
   - Predicted outcome is one of: "H", "D", "A"
   - Validates team IDs exist in database

3. **Season Prediction:**
   - Returns predicted_points as a float
   - Returns predicted_position as a float (typically 1-20 for a league)
   - Validates team ID exists in database

4. **Error Handling:**
   - Invalid team ID returns 404
   - Missing model files returns 503
   - Invalid request format returns 422

## 🐛 Troubleshooting

1. **Models not found:**
   - Ensure `match_outcome_model.pkl` and `season_performance_model.pkl` are in the project root
   - Check file paths in the error message

2. **Database connection error:**
   - Ensure PostgreSQL is running: `docker-compose up -d postgres`
   - Check database credentials in `backend/src/core/config.py`

3. **Team not found:**
   - First create teams or check existing teams: `GET /api/v1/teams/`
   - Use valid team IDs from your database

4. **Import errors:**
   - Install dependencies: `pip install -r backend/requirements.txt`
   - Ensure you're in the correct directory when running the server

