"""
SPARK ML Model #2: Goal Scorer Predictor
Predicts: Which players will score in upcoming matches
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class GoalScorerPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_player_match_data(self, matches_csv, lineups_csv, events_csv, players_csv):
        """
        Create player-match level dataset with goal scoring labels
        """
        print("📊 Loading data files...")
        
        matches_df = pd.read_csv(matches_csv)
        lineups_df = pd.read_csv(lineups_csv)
        events_df = pd.read_csv(events_csv)
        players_df = pd.read_csv(players_csv)
        
        print(f"   - {len(matches_df)} matches")
        print(f"   - {len(lineups_df)} lineup records")
        print(f"   - {len(events_df)} events")
        print(f"   - {len(players_df)} players")
        
        # Create goal scoring labels
        goal_events = events_df[events_df['event_type'] == 'Goal'].copy()
        goal_events['scored_goal'] = 1
        
        # Merge lineups with player stats
        player_matches = lineups_df.merge(
            players_df[['full_name', 'position', 'goals_scored', 'matches_played', 'avg_minutes', 'starts']],
            left_on='player_name',
            right_on='full_name',
            how='left'
        )
        
        # Add goal scoring label
        player_matches = player_matches.merge(
            goal_events[['match_id', 'player_name', 'scored_goal']],
            on=['match_id', 'player_name'],
            how='left'
        )
        player_matches['scored_goal'] = player_matches['scored_goal'].fillna(0).astype(int)
        
        # Add match context
        player_matches = player_matches.merge(
            matches_df[['match_id', 'home_team', 'away_team']],
            on='match_id',
            how='left'
        )
        
        # Feature: is home team
        player_matches['is_home'] = (player_matches['player_team'] == player_matches['home_team']).astype(int)
        
        print(f"\n✅ Prepared {len(player_matches)} player-match records")
        print(f"   - Players who scored: {player_matches['scored_goal'].sum()}")
        print(f"   - Players who didn't score: {(player_matches['scored_goal'] == 0).sum()}")
        
        return player_matches
    
    def create_features(self, df):
        """
        Engineer features for goal scoring prediction
        """
        features = df.copy()
        
        # Player statistics
        features['goals_per_match'] = features['goals_scored'] / (features['matches_played'] + 1)
        features['goals_per_90'] = features['goals_scored'] / ((features['avg_minutes'] / 90) + 0.01)
        features['start_rate'] = features['starts'] / (features['matches_played'] + 1)
        
        # Position encoding (Forward = highest scoring probability)
        position_map = {'Forward': 3, 'Midfielder': 2, 'Defender': 1, 'Goalkeeper': 0}
        features['position_encoded'] = features['position'].map(position_map).fillna(2)
        
        # Match context
        features['is_starter'] = features['is_starter'].astype(int)
        features['minutes_ratio'] = features['minutes_played'] / 90
        
        # Interaction features
        features['forward_and_starter'] = (features['position_encoded'] == 3) * features['is_starter']
        features['high_scorer'] = (features['goals_per_match'] > 0.3).astype(int)
        
        # Fill NaN
        features = features.fillna(0)
        
        return features
    
    def train(self, matches_csv, lineups_csv, events_csv, players_csv, test_size=0.2):
        """
        Train goal scorer prediction model
        """
        print("\n🎯 Training Goal Scorer Predictor...")
        
        # Prepare data
        df = self.prepare_player_match_data(matches_csv, lineups_csv, events_csv, players_csv)
        features = self.create_features(df)
        
        # Select features
        feature_cols = [
            'goals_per_match', 'goals_per_90', 'start_rate',
            'position_encoded', 'is_starter', 'minutes_ratio',
            'is_home', 'forward_and_starter', 'high_scorer',
            'goals_scored', 'matches_played', 'avg_minutes'
        ]
        
        X = features[feature_cols]
        y = features['scored_goal']
        
        self.feature_names = feature_cols
        
        # Handle class imbalance
        class_weights = {0: 1, 1: (len(y) - y.sum()) / y.sum()}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost with class weights
        print("   Training XGBoost classifier...")
        
        scale_pos_weight = class_weights[1]
        
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print(f"\n✅ Model trained successfully!")
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Goal', 'Goal']))
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\n   ROC-AUC Score: {roc_auc:.3f}")
        
        # Feature importance
        self.plot_feature_importance()
        
        # Top predicted scorers
        self.show_top_predictions(X_test, y_test, features)
        
        return roc_auc
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance for Goal Scoring Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('goal_scorer_feature_importance.png', dpi=300, bbox_inches='tight')
        print("   Saved feature importance to: goal_scorer_feature_importance.png")
    
    def show_top_predictions(self, X_test, y_test, features_df):
        """Show players with highest goal scoring probabilities"""
        X_test_scaled = self.scaler.transform(X_test)
        probs = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Get indices
        test_indices = X_test.index
        
        # Create results dataframe
        results = pd.DataFrame({
            'player': features_df.loc[test_indices, 'player_name'].values,
            'position': features_df.loc[test_indices, 'position'].values,
            'team': features_df.loc[test_indices, 'player_team'].values,
            'is_starter': features_df.loc[test_indices, 'is_starter'].values,
            'actual_goal': y_test.values,
            'goal_probability': probs
        })
        
        # Top 10 predictions
        top_predictions = results.nlargest(10, 'goal_probability')
        
        print("\n🎯 Top 10 Predicted Goal Scorers (Test Set):")
        print(top_predictions.to_string(index=False))
    
    def predict(self, player_features):
        """
        Predict goal scoring probability for players
        Returns: Array of probabilities
        """
        X_scaled = self.scaler.transform(player_features)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return probabilities
    
    def predict_match_scorers(self, match_lineups_df, top_n=5):
        """
        Predict most likely goal scorers for a specific match
        
        Args:
            match_lineups_df: DataFrame with player stats for a match
            top_n: Number of top scorers to return
        
        Returns:
            DataFrame with predicted scorers
        """
        features = self.create_features(match_lineups_df)
        X = features[self.feature_names]
        
        probabilities = self.predict(X)
        
        results = match_lineups_df.copy()
        results['goal_probability'] = probabilities
        
        top_scorers = results.nlargest(top_n, 'goal_probability')[
            ['player_name', 'position', 'player_team', 'goal_probability']
        ]
        
        return top_scorers
    
    def save_model(self, filename='goal_scorer_model.pkl'):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filename)
        print(f"✅ Model saved to: {filename}")
    
    def load_model(self, filename='goal_scorer_model.pkl'):
        """Load trained model"""
        data = joblib.load(filename)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        print(f"✅ Model loaded from: {filename}")


def main():
    # Initialize predictor
    predictor = GoalScorerPredictor()
    
    # Train model
    roc_auc = predictor.train(
        matches_csv='ml_data/match_outcomes.csv',
        lineups_csv='ml_data/match_lineups.csv',
        events_csv='ml_data/match_events.csv',
        players_csv='ml_data/player_goal_stats.csv'
    )
    
    # Save model
    predictor.save_model()
    
    print("\n✅ Goal Scorer Predictor training complete!")


if __name__ == '__main__':
    main()