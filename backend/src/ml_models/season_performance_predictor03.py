"""
SPARK ML Model #3: Season Performance Predictor
Predicts: Final league positions and points totals
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SeasonPerformancePredictor:
    def __init__(self):
        self.points_model = None
        self.position_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def create_features(self, df):
        """
        Engineer features for season performance prediction
        """
        features = df.copy()
        
        # Basic statistics
        features['total_matches'] = features['wins'] + features['draws'] + features['losses']
        features['win_rate'] = features['wins'] / (features['total_matches'] + 1)
        features['draw_rate'] = features['draws'] / (features['total_matches'] + 1)
        features['loss_rate'] = features['losses'] / (features['total_matches'] + 1)
        
        # Goal statistics
        features['goals_per_match'] = features['goals_for'] / (features['total_matches'] + 1)
        features['goals_conceded_per_match'] = features['goals_against'] / (features['total_matches'] + 1)
        features['goal_diff_per_match'] = features['goal_diff'] / (features['total_matches'] + 1)
        
        # Points per match
        features['points_per_match'] = features['points'] / (features['total_matches'] + 1)
        
        # Efficiency metrics
        features['win_to_goal_ratio'] = features['wins'] / (features['goals_for'] + 1)
        features['clean_sheets_estimate'] = (features['wins'] * 0.4)  # Rough estimate
        
        # Attack/Defense balance
        features['attack_defense_ratio'] = features['goals_for'] / (features['goals_against'] + 1)
        
        # Historical performance indicators
        features['is_top_6'] = (features['position'] <= 6).astype(int) if 'position' in features else 0
        features['is_relegation_zone'] = (features['position'] >= 18).astype(int) if 'position' in features else 0
        
        # Fill NaN
        features = features.fillna(0)
        
        return features
    
    def prepare_data(self, csv_path):
        """
        Load and prepare season performance data
        """
        print("📊 Loading season performance data...")
        df = pd.read_csv(csv_path)
        
        print(f"✅ Loaded {len(df)} team-season records")
        print(f"   - Seasons: {df['season'].nunique()}")
        print(f"   - Teams: {df['team'].nunique()}")
        
        # Create features
        features = self.create_features(df)
        
        return features, df
    
    def train(self, csv_path, test_size=0.2):
        """
        Train season performance prediction models
        """
        print("\n🎯 Training Season Performance Predictor...")
        
        # Prepare data
        features, df = self.prepare_data(csv_path)
        
        # Select features (excluding target variables)
        feature_cols = [
            'total_matches', 'win_rate', 'draw_rate', 'loss_rate',
            'goals_per_match', 'goals_conceded_per_match', 'goal_diff_per_match',
            'points_per_match', 'win_to_goal_ratio', 'attack_defense_ratio',
            'wins', 'draws', 'losses', 'goals_for', 'goals_against', 'goal_diff'
        ]
        
        X = features[feature_cols]
        y_points = features['points']
        y_position = features['position']
        
        self.feature_names = feature_cols
        
        # Split data
        X_train, X_test, y_points_train, y_points_test = train_test_split(
            X, y_points, test_size=test_size, random_state=42
        )
        
        _, _, y_pos_train, y_pos_test = train_test_split(
            X, y_position, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Points Predictor
        print("\n   Training Points Predictor...")
        self.points_model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.points_model.fit(X_train_scaled, y_points_train)
        
        # Evaluate points model
        points_pred = self.points_model.predict(X_test_scaled)
        points_mae = mean_absolute_error(y_points_test, points_pred)
        points_rmse = np.sqrt(mean_squared_error(y_points_test, points_pred))
        points_r2 = r2_score(y_points_test, points_pred)
        
        print(f"   Points Model Performance:")
        print(f"   - MAE: {points_mae:.2f} points")
        print(f"   - RMSE: {points_rmse:.2f} points")
        print(f"   - R²: {points_r2:.3f}")
        
        # Train Position Predictor
        print("\n   Training Position Predictor...")
        self.position_model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.position_model.fit(X_train_scaled, y_pos_train)
        
        # Evaluate position model
        position_pred = self.position_model.predict(X_test_scaled)
        position_mae = mean_absolute_error(y_pos_test, position_pred)
        position_rmse = np.sqrt(mean_squared_error(y_pos_test, position_pred))
        position_r2 = r2_score(y_pos_test, position_pred)
        
        print(f"   Position Model Performance:")
        print(f"   - MAE: {position_mae:.2f} positions")
        print(f"   - RMSE: {position_rmse:.2f} positions")
        print(f"   - R²: {position_r2:.3f}")
        
        # Visualizations
        self.plot_predictions(y_points_test, points_pred, y_pos_test, position_pred)
        self.plot_feature_importance()
        
        # Example predictions
        self.show_example_predictions(X_test, y_points_test, y_pos_test, df)
        
        return {
            'points_mae': points_mae,
            'points_r2': points_r2,
            'position_mae': position_mae,
            'position_r2': position_r2
        }
    
    def plot_predictions(self, y_points_true, y_points_pred, y_pos_true, y_pos_pred):
        """Plot actual vs predicted values"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Points prediction
        axes[0].scatter(y_points_true, y_points_pred, alpha=0.6)
        axes[0].plot([y_points_true.min(), y_points_true.max()],
                    [y_points_true.min(), y_points_true.max()],
                    'r--', lw=2)
        axes[0].set_xlabel('Actual Points')
        axes[0].set_ylabel('Predicted Points')
        axes[0].set_title('Points Prediction')
        axes[0].grid(True, alpha=0.3)
        
        # Position prediction
        axes[1].scatter(y_pos_true, y_pos_pred, alpha=0.6)
        axes[1].plot([y_pos_true.min(), y_pos_true.max()],
                    [y_pos_true.min(), y_pos_true.max()],
                    'r--', lw=2)
        axes[1].set_xlabel('Actual Position')
        axes[1].set_ylabel('Predicted Position')
        axes[1].set_title('League Position Prediction')
        axes[1].grid(True, alpha=0.3)
        axes[1].invert_yaxis()
        axes[1].invert_xaxis()
        
        plt.tight_layout()
        plt.savefig('season_performance_predictions.png', dpi=300, bbox_inches='tight')
        print("   Saved predictions plot to: season_performance_predictions.png")
    
    def plot_feature_importance(self):
        """Plot feature importance for both models"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Points model
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.points_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        axes[0].barh(importance_df['feature'][:10], importance_df['importance'][:10])
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Top 10 Features - Points Prediction')
        axes[0].invert_yaxis()
        
        # Position model
        importance_df2 = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.position_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        axes[1].barh(importance_df2['feature'][:10], importance_df2['importance'][:10])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Top 10 Features - Position Prediction')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('season_performance_feature_importance.png', dpi=300, bbox_inches='tight')
        print("   Saved feature importance to: season_performance_feature_importance.png")
    
    def show_example_predictions(self, X_test, y_points_test, y_pos_test, df):
        """Show example predictions"""
        X_test_scaled = self.scaler.transform(X_test)
        points_pred = self.points_model.predict(X_test_scaled)
        position_pred = self.position_model.predict(X_test_scaled)
        
        # Get team names
        test_indices = X_test.index
        teams = df.loc[test_indices, 'team'].values
        
        # Create results
        results = pd.DataFrame({
            'Team': teams,
            'Actual Points': y_points_test.values,
            'Predicted Points': np.round(points_pred, 1),
            'Points Diff': np.abs(y_points_test.values - points_pred),
            'Actual Position': y_pos_test.values,
            'Predicted Position': np.round(position_pred, 1),
            'Position Diff': np.abs(y_pos_test.values - position_pred)
        })
        
        # Show best and worst predictions
        print("\n🎯 Best Position Predictions (Top 5):")
        best = results.nsmallest(5, 'Position Diff')
        print(best.to_string(index=False))
        
        print("\n❌ Worst Position Predictions (Top 5):")
        worst = results.nlargest(5, 'Position Diff')
        print(worst.to_string(index=False))
    
    def predict(self, team_features):
        """
        Predict points and position for team(s)
        
        Returns: Dict with points and position predictions
        """
        X_scaled = self.scaler.transform(team_features)
        
        points_pred = self.points_model.predict(X_scaled)
        position_pred = self.position_model.predict(X_scaled)
        
        return {
            'predicted_points': points_pred,
            'predicted_position': position_pred
        }
    
    def predict_season_table(self, teams_df):
        """
        Predict final season standings for multiple teams
        
        Args:
            teams_df: DataFrame with team statistics
        
        Returns:
            DataFrame with predicted final table
        """
        features = self.create_features(teams_df)
        X = features[self.feature_names]
        
        predictions = self.predict(X)
        
        results = teams_df.copy()
        results['predicted_points'] = predictions['predicted_points']
        results['predicted_position'] = predictions['predicted_position']
        
        # Sort by predicted points
        final_table = results.sort_values('predicted_points', ascending=False).reset_index(drop=True)
        final_table['predicted_position'] = range(1, len(final_table) + 1)
        
        return final_table[['team', 'predicted_points', 'predicted_position']]
    
    def save_model(self, filename='season_performance_model.pkl'):
        """Save trained models"""
        joblib.dump({
            'points_model': self.points_model,
            'position_model': self.position_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filename)
        print(f"✅ Model saved to: {filename}")
    
    def load_model(self, filename='season_performance_model.pkl'):
        """Load trained models"""
        data = joblib.load(filename)
        self.points_model = data['points_model']
        self.position_model = data['position_model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        print(f"✅ Model loaded from: {filename}")


def main():
    # Initialize predictor
    predictor = SeasonPerformancePredictor()
    
    # Train models
    metrics = predictor.train('ml_data/team_season_performance.csv')
    
    # Save models
    predictor.save_model()
    
    print("\n✅ Season Performance Predictor training complete!")
    print(f"\n📊 Final Metrics:")
    print(f"   Points Prediction - MAE: {metrics['points_mae']:.2f}, R²: {metrics['points_r2']:.3f}")
    print(f"   Position Prediction - MAE: {metrics['position_mae']:.2f}, R²: {metrics['position_r2']:.3f}")


if __name__ == '__main__':
    main()