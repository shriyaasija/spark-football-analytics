"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import date, datetime
from decimal import Decimal


# ============================================
# TEAM SCHEMAS
# ============================================

class TeamBase(BaseModel):
    name: str = Field(..., max_length=100)
    founded_year: Optional[int] = None
    stadium_name: Optional[str] = Field(None, max_length=100)
    city: Optional[str] = Field(None, max_length=100)
    logo_url: Optional[str] = None


class TeamCreate(TeamBase):
    """Schema for creating a team"""
    pass


class TeamUpdate(BaseModel):
    """Schema for updating a team (all fields optional)"""
    name: Optional[str] = Field(None, max_length=100)
    founded_year: Optional[int] = None
    stadium_name: Optional[str] = Field(None, max_length=100)
    city: Optional[str] = Field(None, max_length=100)
    logo_url: Optional[str] = None


class Team(TeamBase):
    """Schema for team response"""
    team_id: int
    fbref_id: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


# ============================================
# PLAYER SCHEMAS
# ============================================

class PlayerBase(BaseModel):
    full_name: str = Field(..., max_length=150)
    dob: Optional[date] = None
    nationality: Optional[str] = Field(None, max_length=50)
    position: Optional[str] = Field(None, max_length=20)
    height_cm: Optional[int] = Field(None, gt=0, lt=250)
    weight_kg: Optional[int] = Field(None, gt=0, lt=200)
    shirt_number: Optional[int] = Field(None, gt=0, le=99)
    team_id: Optional[int] = None


class PlayerCreate(PlayerBase):
    """Schema for creating a player"""
    pass


class PlayerUpdate(BaseModel):
    """Schema for updating a player (all fields optional)"""
    full_name: Optional[str] = Field(None, max_length=150)
    dob: Optional[date] = None
    nationality: Optional[str] = Field(None, max_length=50)
    position: Optional[str] = Field(None, max_length=20)
    height_cm: Optional[int] = Field(None, gt=0, lt=250)
    weight_kg: Optional[int] = Field(None, gt=0, lt=200)
    shirt_number: Optional[int] = Field(None, gt=0, le=99)
    team_id: Optional[int] = None


class Player(PlayerBase):
    """Schema for player response"""
    player_id: int
    fbref_id: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class PlayerWithTeam(Player):
    """Player with team information"""
    team: Optional[Team] = None
    
    model_config = ConfigDict(from_attributes=True)


# ============================================
# MATCH SCHEMAS
# ============================================

class MatchBase(BaseModel):
    match_date: datetime
    venue: Optional[str] = Field(None, max_length=100)
    home_score_final: Optional[int] = Field(None, ge=0)
    away_score_final: Optional[int] = Field(None, ge=0)
    home_xg: Optional[Decimal] = None
    away_xg: Optional[Decimal] = None
    attendance: Optional[int] = Field(None, ge=0)
    season_id: int
    home_team_id: int
    away_team_id: int


class MatchCreate(MatchBase):
    """Schema for creating a match"""
    pass


class MatchUpdate(BaseModel):
    """Schema for updating a match"""
    match_date: Optional[datetime] = None
    venue: Optional[str] = Field(None, max_length=100)
    home_score_final: Optional[int] = Field(None, ge=0)
    away_score_final: Optional[int] = Field(None, ge=0)
    home_xg: Optional[Decimal] = None
    away_xg: Optional[Decimal] = None
    attendance: Optional[int] = Field(None, ge=0)


class Match(MatchBase):
    """Schema for match response"""
    match_id: int
    fbref_id: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class MatchWithTeams(Match):
    """Match with team names"""
    home_team: Team
    away_team: Team
    
    model_config = ConfigDict(from_attributes=True)


# ============================================
# ANALYTICS SCHEMAS
# ============================================

class TopScorer(BaseModel):
    """Top scorer statistics"""
    player_name: str
    team_name: str
    goals: int
    matches_played: int


class TeamStandings(BaseModel):
    """Team standings"""
    position: int
    team_name: str
    played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    goal_diff: int
    points: int


class MatchStatistics(BaseModel):
    """Detailed match statistics"""
    match_id: int
    home_team: str
    away_team: str
    score: str
    date: datetime
    home_xg: Optional[Decimal] = None
    away_xg: Optional[Decimal] = None
    total_goals: int
    total_cards: int


# ============================================
# PREDICTION SCHEMAS
# ============================================

class MatchPredictionRequest(BaseModel):
    """Request schema for match outcome prediction"""
    home_team_id: int
    away_team_id: int
    # Home team statistics
    home_points: int
    home_wins: int
    home_draws: int
    home_losses: int
    home_gf: int  # goals for
    home_ga: int  # goals against
    home_gd: int  # goal difference
    # Away team statistics
    away_points: int
    away_wins: int
    away_draws: int
    away_losses: int
    away_gf: int
    away_ga: int
    away_gd: int
    # Optional xG data
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None


class MatchPredictionResponse(BaseModel):
    """Response schema for match outcome prediction"""
    home_team_id: int
    away_team_id: int
    home_win_probability: float = Field(..., ge=0, le=1, description="Probability of home team winning")
    draw_probability: float = Field(..., ge=0, le=1, description="Probability of a draw")
    away_win_probability: float = Field(..., ge=0, le=1, description="Probability of away team winning")
    predicted_outcome: str = Field(..., description="Predicted outcome: H (home win), D (draw), A (away win)")


class SeasonPredictionRequest(BaseModel):
    """Request schema for season performance prediction"""
    team_id: int
    # Current season statistics
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    goal_diff: int
    points: int


class SeasonPredictionResponse(BaseModel):
    """Response schema for season performance prediction"""
    team_id: int
    predicted_points: float = Field(..., description="Predicted final points total")
    predicted_position: float = Field(..., description="Predicted final league position")


# ============================================
# GENERIC RESPONSES
# ============================================

class MessageResponse(BaseModel):
    """Generic message response"""
    message: str
    detail: Optional[str] = None


class HealthCheck(BaseModel):
    """API health check response"""
    status: str
    version: str
    database: str