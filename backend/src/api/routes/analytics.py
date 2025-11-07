"""
Analytics API endpoints - ML model predictions
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.core.database import get_db
from src.models import models
from src.schemas import schemas
from src.api.dependencies import get_team_or_404

# Add parent directory to path to import ML models
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.ml_models.match_predictor01 import MatchOutcomePredictor
from src.ml_models.season_performance_predictor03 import SeasonPerformancePredictor

router = APIRouter(prefix="/analytics", tags=["Analytics"])

# Model paths (adjust if models are stored elsewhere)
# analytics.py location: backend/src/api/routes/analytics.py
# Project root (where pkl files are): 5 levels up from this file
_project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
MATCH_MODEL_PATH = _project_root / "match_outcome_model.pkl"
SEASON_MODEL_PATH = _project_root / "season_performance_model.pkl"

# Global model instances (loaded on first use)
_match_predictor = None
_season_predictor = None


def get_match_predictor():
    """Lazy load match outcome predictor"""
    global _match_predictor
    if _match_predictor is None:
        if not MATCH_MODEL_PATH.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Match prediction model not found at {MATCH_MODEL_PATH}"
            )
        _match_predictor = MatchOutcomePredictor()
        _match_predictor.load_model(str(MATCH_MODEL_PATH))
    return _match_predictor


def get_season_predictor():
    """Lazy load season performance predictor"""
    global _season_predictor
    if _season_predictor is None:
        if not SEASON_MODEL_PATH.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Season prediction model not found at {SEASON_MODEL_PATH}"
            )
        _season_predictor = SeasonPerformancePredictor()
        _season_predictor.load_model(str(SEASON_MODEL_PATH))
    return _season_predictor


def prepare_match_features(request: schemas.MatchPredictionRequest, predictor: MatchOutcomePredictor) -> pd.DataFrame:
    """Prepare features for match outcome prediction"""
    # Create a DataFrame with the required features
    data = {
        'home_advantage': [1],
        'home_points': [request.home_points],
        'away_points': [request.away_points],
        'home_wins': [request.home_wins],
        'home_draws': [request.home_draws],
        'home_losses': [request.home_losses],
        'away_wins': [request.away_wins],
        'away_draws': [request.away_draws],
        'away_losses': [request.away_losses],
        'home_gf': [request.home_gf],
        'home_ga': [request.home_ga],
        'home_gd': [request.home_gd],
        'away_gf': [request.away_gf],
        'away_ga': [request.away_ga],
        'away_gd': [request.away_gd],
    }
    
    # Calculate derived features
    home_matches = request.home_wins + request.home_draws + request.home_losses
    away_matches = request.away_wins + request.away_draws + request.away_losses
    
    data['home_form'] = [request.home_points / (home_matches + 1)]
    data['away_form'] = [request.away_points / (away_matches + 1)]
    data['home_attack'] = [request.home_gf / (home_matches + 1)]
    data['away_attack'] = [request.away_gf / (away_matches + 1)]
    data['home_defense'] = [request.home_ga / (home_matches + 1)]
    data['away_defense'] = [request.away_ga / (away_matches + 1)]
    data['home_gd_per_game'] = [request.home_gd / (home_matches + 1)]
    data['away_gd_per_game'] = [request.away_gd / (away_matches + 1)]
    data['home_win_rate'] = [request.home_wins / (home_matches + 1)]
    data['away_win_rate'] = [request.away_wins / (away_matches + 1)]
    data['points_diff'] = [request.home_points - request.away_points]
    data['gd_diff'] = [request.home_gd - request.away_gd]
    data['form_diff'] = [data['home_form'][0] - data['away_form'][0]]
    data['attack_diff'] = [data['home_attack'][0] - data['away_attack'][0]]
    data['defense_diff'] = [data['home_defense'][0] - data['away_defense'][0]]
    
    # Add xG features if provided
    if request.home_xg is not None and request.away_xg is not None:
        data['xg_diff'] = [request.home_xg - request.away_xg]
        data['home_xg_per_match'] = [request.home_xg]
        data['away_xg_per_match'] = [request.away_xg]
    
    df = pd.DataFrame(data)
    
    # Ensure all required feature columns are present
    required_features = predictor.feature_names
    
    # Add missing features with default values
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only the required features in the correct order
    df = df[required_features]
    
    return df


def prepare_season_features(request: schemas.SeasonPredictionRequest, predictor: SeasonPerformancePredictor) -> pd.DataFrame:
    """Prepare features for season performance prediction"""
    # Create a DataFrame with the required features
    total_matches = request.wins + request.draws + request.losses
    
    data = {
        'total_matches': [total_matches],
        'win_rate': [request.wins / (total_matches + 1)],
        'draw_rate': [request.draws / (total_matches + 1)],
        'loss_rate': [request.losses / (total_matches + 1)],
        'goals_per_match': [request.goals_for / (total_matches + 1)],
        'goals_conceded_per_match': [request.goals_against / (total_matches + 1)],
        'goal_diff_per_match': [request.goal_diff / (total_matches + 1)],
        'points_per_match': [request.points / (total_matches + 1)],
        'win_to_goal_ratio': [request.wins / (request.goals_for + 1)],
        'attack_defense_ratio': [request.goals_for / (request.goals_against + 1)],
        'wins': [request.wins],
        'draws': [request.draws],
        'losses': [request.losses],
        'goals_for': [request.goals_for],
        'goals_against': [request.goals_against],
        'goal_diff': [request.goal_diff],
    }
    
    df = pd.DataFrame(data)
    
    # Ensure all required feature columns are present
    required_features = predictor.feature_names
    
    # Add missing features with default values
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select only the required features in the correct order
    df = df[required_features]
    
    return df


@router.post("/predict/match", response_model=schemas.MatchPredictionResponse)
def predict_match_outcome(
    request: schemas.MatchPredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Predict match outcome (Win/Draw/Loss) probabilities
    
    This endpoint uses the trained ML model to predict the outcome of a match
    based on team statistics. The prediction can be compared with user predictions
    in the frontend.
    
    - **home_team_id**: ID of the home team
    - **away_team_id**: ID of the away team
    - **home/away statistics**: Current season statistics for both teams
    """
    # Verify teams exist
    home_team = get_team_or_404(request.home_team_id, db)
    away_team = get_team_or_404(request.away_team_id, db)
    
    # Get predictor first (needed for feature preparation)
    predictor = get_match_predictor()
    
    # Prepare features
    features_df = prepare_match_features(request, predictor)
    
    # Make prediction
    predictions = predictor.predict(features_df)
    
    if not predictions:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate prediction"
        )
    
    prediction = predictions[0]
    
    # Determine predicted outcome
    home_prob = prediction.get('home_win_prob', 0)
    draw_prob = prediction.get('draw_prob', 0)
    away_prob = prediction.get('away_win_prob', 0)
    
    if home_prob >= draw_prob and home_prob >= away_prob:
        predicted_outcome = 'H'
    elif away_prob >= draw_prob:
        predicted_outcome = 'A'
    else:
        predicted_outcome = 'D'
    
    return schemas.MatchPredictionResponse(
        home_team_id=request.home_team_id,
        away_team_id=request.away_team_id,
        home_win_probability=float(home_prob),
        draw_probability=float(draw_prob),
        away_win_probability=float(away_prob),
        predicted_outcome=predicted_outcome
    )


@router.post("/predict/season", response_model=schemas.SeasonPredictionResponse)
def predict_season_performance(
    request: schemas.SeasonPredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Predict season performance (final points and position) for a team
    
    This endpoint uses the trained ML model to predict the final league position
    and points total for a team based on their current season statistics.
    The prediction can be compared with user predictions in the frontend.
    
    - **team_id**: ID of the team
    - **season statistics**: Current wins, draws, losses, goals, and points
    """
    # Verify team exists
    team = get_team_or_404(request.team_id, db)
    
    # Get predictor first (needed for feature preparation)
    predictor = get_season_predictor()
    
    # Prepare features
    features_df = prepare_season_features(request, predictor)
    
    # Make prediction
    predictions = predictor.predict(features_df)
    
    if predictions is None or 'predicted_points' not in predictions:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate prediction"
        )
    
    # Extract predictions (handles both single and array outputs)
    points_pred = predictions['predicted_points']
    position_pred = predictions['predicted_position']
    
    # Handle numpy array outputs
    if isinstance(points_pred, np.ndarray):
        points_pred = float(points_pred[0])
    else:
        points_pred = float(points_pred)
    
    if isinstance(position_pred, np.ndarray):
        position_pred = float(position_pred[0])
    else:
        position_pred = float(position_pred)
    
    return schemas.SeasonPredictionResponse(
        team_id=request.team_id,
        predicted_points=points_pred,
        predicted_position=position_pred
    )

