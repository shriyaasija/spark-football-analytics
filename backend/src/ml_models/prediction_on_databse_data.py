"""
SPARK Database Prediction Pipeline
Integrates ML models with PostgreSQL database
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import joblib
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import model classes (ensure they're in the same directory)
from ml_models.match_predictor01 import MatchOutcomePredictor
from ml_models.goal_scorer_predictor02 import GoalScorerPredictor
from ml_models.season_performance_predictor03 import SeasonPerformancePredictor

load_dotenv()

class SPARKPredictionPipeline:
    def __init__(self):
        self.conn = None
        self.match_predictor = MatchOutcomePredictor()
        self.goal_predictor = GoalScorerPredictor()
        self.season_predictor = SeasonPerformancePredictor()
        
        # Load pre-trained models
        self.load_models()
    
    def connect_db(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=os.getenv('POSTGRES_PORT', '5432'),
                database=os.getenv('POSTGRES_DB', 'spark_db'),
                user=os.getenv('POSTGRES_USER', 'spark_user'),
                password=os.getenv('POSTGRES_PASSWORD', 'spark_password_2024'),
                connect_timeout=10
            )
            print("✅ Connected to database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
    
    def close_db(self):
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")
    
    def load_models(self):
        """Load pre-trained ML models"""
        try:
            self.match_predictor.load_model('match_outcome_model.pkl')
            self.goal_predictor.load_model('goal_scorer_model.pkl')
            self.season_predictor.load_model('season_performance_model.pkl')
            print("✅ All models loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load models: {e}")
            print("   Run training scripts first to generate model files")
    
    def get_upcoming_matches(self, days_ahead=7):
        """
        Get upcoming matches from database
        """
        query = """
        SELECT 
            m.match_id,
            m.match_date,
            ht.team_id as home_team_id,
            ht.name as home_team,
            at.team_id as away_team_id,
            at.name as away_team,
            m.venue,
            s.season_id,
            s.year as season
        FROM MATCHES m
        JOIN TEAMS ht ON m.home_team_id = ht.team_id
        JOIN TEAMS at ON m.away_team_id = at.team_id
        JOIN SEASONS s ON m.season_id = s.season_id
        WHERE m.match_date >= CURRENT_DATE
        AND m.match_date <= CURRENT_DATE + INTERVAL '%s days'
        AND m.home_score_final IS NULL
        ORDER BY m.match_date
        """
        
        df = pd.read_sql(query, self.conn, params=(days_ahead,))
        return df
    
    def get_team_form(self, team_id, season_id):
        """
        Get team's current season statistics
        """
        query = """
        SELECT 
            ts.points,
            ts.wins,
            ts.draws,
            ts.losses,
            ts.goals_for,
            ts.goals_against,
            ts.goal_diff,
            ts.position
        FROM TEAM_SEASONS ts
        WHERE ts.team_id = %s AND ts.season_id = %s
        """
        
        cur = self.conn.cursor()
        cur.execute(query, (team_id, season_id))
        result = cur.fetchone()
        cur.close()
        
        if result:
            return {
                'points': result[0] or 0,
                'wins': result[1] or 0,
                'draws': result[2] or 0,
                'losses': result[3] or 0,
                'goals_for': result[4] or 0,
                'goals_against': result[5] or 0,
                'goal_diff': result[6] or 0,
                'position': result[7] or 10
            }
        return None
    
    def prepare_match_features(self, match_row):
        """
        Prepare features for match outcome prediction
        """
        home_form = self.get_team_form(match_row['home_team_id'], match_row['season_id'])
        away_form = self.get_team_form(match_row['away_team_id'], match_row['season_id'])
        
        if not home_form or not away_form:
            return None
        
        # Create feature dictionary
        features = {
            'home_points': home_form['points'],
            'away_points': away_form['points'],
            'home_wins': home_form['wins'],
            'away_wins': away_form['wins'],
            'home_draws': home_form['draws'],
            'away_draws': away_form['draws'],
            'home_losses': home_form['losses'],
            'away_losses': away_form['losses'],
            'home_gf': home_form['goals_for'],
            'away_gf': away_form['goals_for'],
            'home_ga': home_form['goals_against'],
            'away_ga': away_form['goals_against'],
            'home_gd': home_form['goal_diff'],
            'away_gd': away_form['goal_diff']
        }
        
        # Create features using predictor's method
        df = pd.DataFrame([features])
        features_df = self.match_predictor.create_features(df)
        
        return features_df[self.match_predictor.feature_names]
    
    def get_match_squad(self, team_id):
        """
        Get current squad for a team with statistics
        """
        query = """
        SELECT 
            p.player_id,
            p.full_name,
            p.position,
            p.shirt_number,
            COUNT(DISTINCT ml.match_id) as matches_played,
            AVG(ml.minutes_played) as avg_minutes,
            COUNT(CASE WHEN ml.is_starter = true THEN 1 END) as starts,
            COUNT(CASE WHEN me.event_type = 'Goal' THEN 1 END) as goals_scored
        FROM PLAYERS p
        LEFT JOIN MATCH_LINEUPS ml ON p.player_id = ml.player_id
        LEFT JOIN MATCH_EVENT me ON p.player_id = me.player_id AND me.event_type = 'Goal'
        WHERE p.team_id = %s
        GROUP BY p.player_id, p.full_name, p.position, p.shirt_number
        HAVING COUNT(DISTINCT ml.match_id) > 0
        ORDER BY starts DESC, goals_scored DESC
        """
        
        df = pd.read_sql(query, self.conn, params=(team_id,))
        return df
    
    def predict_match_outcome(self, match_id):
        """
        Predict outcome for a specific match
        """
        # Get match details
        query = """
        SELECT 
            m.match_id,
            m.match_date,
            ht.team_id as home_team_id,
            ht.name as home_team,
            at.team_id as away_team_id,
            at.name as away_team,
            m.season_id
        FROM MATCHES m
        JOIN TEAMS ht ON m.home_team_id = ht.team_id
        JOIN TEAMS at ON m.away_team_id = at.team_id
        WHERE m.match_id = %s
        """
        
        df = pd.read_sql(query, self.conn, params=(match_id,))
        
        if df.empty:
            return None
        
        match_row = df.iloc[0]
        
        # Prepare features
        features = self.prepare_match_features(match_row)
        
        if features is None:
            return None
        
        # Predict
        prediction = self.match_predictor.predict(features)[0]
        
        return {
            'match_id': match_id,
            'home_team': match_row['home_team'],
            'away_team': match_row['away_team'],
            'match_date': match_row['match_date'],
            'home_win_prob': prediction['home_win_prob'],
            'draw_prob': prediction['draw_prob'],
            'away_win_prob': prediction['away_win_prob']
        }
    
    def predict_goal_scorers(self, match_id, top_n=5):
        """
        Predict likely goal scorers for a match
        """
        # Get match details
        query = """
        SELECT 
            m.match_id,
            ht.team_id as home_team_id,
            ht.name as home_team,
            at.team_id as away_team_id,
            at.name as away_team
        FROM MATCHES m
        JOIN TEAMS ht ON m.home_team_id = ht.team_id
        JOIN TEAMS at ON m.away_team_id = at.team_id
        WHERE m.match_id = %s
        """
        
        df = pd.read_sql(query, self.conn, params=(match_id,))
        
        if df.empty:
            return None
        
        match_row = df.iloc[0]
        
        # Get squads for both teams
        home_squad = self.get_match_squad(match_row['home_team_id'])
        away_squad = self.get_match_squad(match_row['away_team_id'])
        
        # Add team and home/away context
        home_squad['player_team'] = match_row['home_team']
        home_squad['is_home'] = 1
        home_squad['is_starter'] = True  # Assume starters
        home_squad['minutes_played'] = 90
        
        away_squad['player_team'] = match_row['away_team']
        away_squad['is_home'] = 0
        away_squad['is_starter'] = True
        away_squad['minutes_played'] = 90
        
        # Combine squads
        all_players = pd.concat([home_squad, away_squad], ignore_index=True)
        
        # Rename for compatibility
        all_players = all_players.rename(columns={'full_name': 'player_name'})
        
        # Create features
        features = self.goal_predictor.create_features(all_players)
        X = features[self.goal_predictor.feature_names]
        
        # Predict
        probabilities = self.goal_predictor.predict(X)
        
        all_players['goal_probability'] = probabilities
        
        # Get top scorers
        top_scorers = all_players.nlargest(top_n, 'goal_probability')[
            ['player_name', 'position', 'player_team', 'goals_scored', 'goal_probability']
        ]
        
        return top_scorers
    
    def save_prediction_to_db(self, prediction):
        """
        Save match prediction to MATCH_PREDICTIONS table
        """
        cur = self.conn.cursor()
        
        try:
            cur.execute("""
                INSERT INTO MATCH_PREDICTIONS 
                (match_id, prediction_date, win_probability_home, draw_probability, win_probability_away)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (match_id) DO UPDATE
                SET prediction_date = EXCLUDED.prediction_date,
                    win_probability_home = EXCLUDED.win_probability_home,
                    draw_probability = EXCLUDED.draw_probability,
                    win_probability_away = EXCLUDED.win_probability_away
                RETURNING prediction_id
            """, (
                prediction['match_id'],
                datetime.now(),
                prediction['home_win_prob'],
                prediction['draw_prob'],
                prediction['away_win_prob']
            ))
            
            prediction_id = cur.fetchone()[0]
            self.conn.commit()
            
            return prediction_id
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error saving prediction: {e}")
            return None
        finally:
            cur.close()
    
    def run_predictions_for_upcoming_matches(self, days_ahead=7):
        """
        Run predictions for all upcoming matches and save to database
        """
        print(f"\n🔮 Running predictions for next {days_ahead} days...")
        
        upcoming = self.get_upcoming_matches(days_ahead)
        
        if upcoming.empty:
            print("   No upcoming matches found")
            return
        
        print(f"   Found {len(upcoming)} upcoming matches")
        
        predictions = []
        
        for _, match in upcoming.iterrows():
            print(f"\n   Processing: {match['home_team']} vs {match['away_team']}")
            
            # Match outcome prediction
            outcome_pred = self.predict_match_outcome(match['match_id'])
            
            if outcome_pred:
                # Save to database
                pred_id = self.save_prediction_to_db(outcome_pred)
                
                if pred_id:
                    print(f"   ✅ Saved prediction (ID: {pred_id})")
                    print(f"      Home Win: {outcome_pred['home_win_prob']:.1%}")
                    print(f"      Draw: {outcome_pred['draw_prob']:.1%}")
                    print(f"      Away Win: {outcome_pred['away_win_prob']:.1%}")
                
                # Goal scorers prediction
                scorers = self.predict_goal_scorers(match['match_id'], top_n=3)
                
                if scorers is not None and not scorers.empty:
                    print(f"\n      Top 3 Likely Scorers:")
                    for idx, row in scorers.iterrows():
                        print(f"      {idx+1}. {row['player_name']} ({row['player_team']}) - {row['goal_probability']:.1%}")
                
                predictions.append(outcome_pred)
        
        return predictions
    
    def generate_match_report(self, match_id):
        """
        Generate comprehensive prediction report for a match
        """
        print("\n" + "="*60)
        print("SPARK MATCH PREDICTION REPORT")
        print("="*60)
        
        # Match outcome
        outcome = self.predict_match_outcome(match_id)
        
        if not outcome:
            print("❌ Could not generate prediction")
            return
        
        print(f"\n🏆 {outcome['home_team']} vs {outcome['away_team']}")
        print(f"📅 {outcome['match_date']}")
        print(f"\n📊 Win Probabilities:")
        print(f"   {outcome['home_team']}: {outcome['home_win_prob']:.1%}")
        print(f"   Draw: {outcome['draw_prob']:.1%}")
        print(f"   {outcome['away_team']}: {outcome['away_win_prob']:.1%}")
        
        # Likely scorers
        scorers = self.predict_goal_scorers(match_id, top_n=5)
        
        if scorers is not None and not scorers.empty:
            print(f"\n⚽ Top 5 Likely Goal Scorers:")
            for idx, row in scorers.iterrows():
                print(f"   {idx+1}. {row['player_name']:25s} ({row['player_team']:15s}) - {row['goal_probability']:.1%}")
        
        print("\n" + "="*60)
        
        return outcome


def main():
    pipeline = SPARKPredictionPipeline()
    
    try:
        pipeline.connect_db()
        
        # Run predictions for upcoming matches
        predictions = pipeline.run_predictions_for_upcoming_matches(days_ahead=14)
        
        print(f"\n✅ Generated {len(predictions) if predictions else 0} predictions")
        
        # Example: Generate detailed report for first upcoming match
        upcoming = pipeline.get_upcoming_matches(days_ahead=7)
        if not upcoming.empty:
            first_match_id = upcoming.iloc[0]['match_id']
            pipeline.generate_match_report(first_match_id)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.close_db()


if __name__ == '__main__':
    main()