# """
# SPARK ML Model #1: Match Outcome Predictor
# Predicts: Win/Draw/Loss probabilities
# """

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')

# class MatchOutcomePredictor:
#     def __init__(self):
#         self.model = None
#         self.scaler = StandardScaler()
#         self.label_encoder = LabelEncoder()
#         self.feature_names = None
        
#     def create_features(self, df):
#         """
#         Engineer features for match outcome prediction
#         """
#         features = df.copy()
        
#         # Basic features
#         features['home_advantage'] = 1  # Home team indicator
        
#         # Form features (points, wins, etc.)
#         features['home_form'] = features['home_points'] / (features['home_wins'] + features['home_draws'] + features['home_losses'] + 1)
#         features['away_form'] = features['away_points'] / (features['away_wins'] + features['away_draws'] + features['away_losses'] + 1)
        
#         # Goal scoring ability
#         features['home_attack'] = features['home_gf'] / (features['home_wins'] + features['home_draws'] + features['home_losses'] + 1)
#         features['away_attack'] = features['away_gf'] / (features['away_wins'] + features['away_draws'] + features['away_losses'] + 1)
        
#         # Defensive strength
#         features['home_defense'] = features['home_ga'] / (features['home_wins'] + features['home_draws'] + features['home_losses'] + 1)
#         features['away_defense'] = features['away_ga'] / (features['away_wins'] + features['away_draws'] + features['away_losses'] + 1)
        
#         # Goal difference per game
#         features['home_gd_per_game'] = features['home_gd'] / (features['home_wins'] + features['home_draws'] + features['home_losses'] + 1)
#         features['away_gd_per_game'] = features['away_gd'] / (features['away_wins'] + features['away_draws'] + features['away_losses'] + 1)
        
#         # Win rate
#         features['home_win_rate'] = features['home_wins'] / (features['home_wins'] + features['home_draws'] + features['home_losses'] + 1)
#         features['away_win_rate'] = features['away_wins'] / (features['away_wins'] + features['away_draws'] + features['away_losses'] + 1)
        
#         # Comparative features
#         features['points_diff'] = features['home_points'] - features['away_points']
#         features['gd_diff'] = features['home_gd'] - features['away_gd']
#         features['form_diff'] = features['home_form'] - features['away_form']
#         features['attack_diff'] = features['home_attack'] - features['away_attack']
#         features['defense_diff'] = features['home_defense'] - features['away_defense']
        
#         # xG features (if available)
#         if 'home_xg' in features.columns and 'away_xg' in features.columns:
#             features['xg_diff'] = features['home_xg'].fillna(0) - features['away_xg'].fillna(0)
#             features['home_xg_per_match'] = features['home_xg'].fillna(0)
#             features['away_xg_per_match'] = features['away_xg'].fillna(0)
        
#         # Fill NaN values
#         features = features.fillna(0)
        
#         return features
    
#     def prepare_data(self, csv_path):
#         """
#         Load and prepare data for training
#         """
#         print("📊 Loading match data...")
#         df = pd.read_csv(csv_path)
        
#         print(f"✅ Loaded {len(df)} matches")
#         print(f"   - Home wins: {(df['result'] == 'H').sum()}")
#         print(f"   - Draws: {(df['result'] == 'D').sum()}")
#         print(f"   - Away wins: {(df['result'] == 'A').sum()}")
        
#         # Create features
#         features = self.create_features(df)
        
#         # Select feature columns
#         feature_cols = [
#             'home_advantage', 'home_form', 'away_form',
#             'home_attack', 'away_attack', 'home_defense', 'away_defense',
#             'home_gd_per_game', 'away_gd_per_game',
#             'home_win_rate', 'away_win_rate',
#             'points_diff', 'gd_diff', 'form_diff', 'attack_diff', 'defense_diff',
#             'home_points', 'away_points', 'home_gd', 'away_gd'
#         ]
        
#         # Add xG features if available
#         if 'xg_diff' in features.columns:
#             feature_cols.extend(['xg_diff', 'home_xg_per_match', 'away_xg_per_match'])
        
#         X = features[feature_cols]
#         y = df['result']
        
#         self.feature_names = feature_cols
        
#         return X, y, df
    
#     def train(self, X, y, test_size=0.2):
#         """
#         Train the match outcome prediction model
#         """
#         print("\n🎯 Training Match Outcome Predictor...")
        
#         # Encode labels
#         y_encoded = self.label_encoder.fit_transform(y)
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
#         )
        
#         # Scale features
#         X_train_scaled = self.scaler.fit_transform(X_train)
#         X_test_scaled = self.scaler.transform(X_test)
        
#         # Train XGBoost model (best for this task)
#         print("   Training XGBoost classifier...")
#         self.model = XGBClassifier(
#             n_estimators=200,
#             max_depth=6,
#             learning_rate=0.1,
#             random_state=42,
#             eval_metric='mlogloss'
#         )
        
#         self.model.fit(X_train_scaled, y_train)
        
#         # Evaluate
#         y_pred = self.model.predict(X_test_scaled)
#         accuracy = accuracy_score(y_test, y_pred)
        
#         print(f"\n✅ Model trained successfully!")
#         print(f"   Accuracy: {accuracy:.2%}")
        
#         # Detailed classification report
#         print("\n📊 Classification Report:")
#         print(classification_report(
#             y_test, y_pred,
#             target_names=self.label_encoder.classes_
#         ))
        
#         # Confusion matrix
#         cm = confusion_matrix(y_test, y_pred)
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                    xticklabels=self.label_encoder.classes_,
#                    yticklabels=self.label_encoder.classes_)
#         plt.title('Match Outcome Prediction - Confusion Matrix')
#         plt.ylabel('Actual')
#         plt.xlabel('Predicted')
#         plt.savefig('match_outcome_confusion_matrix.png', dpi=300, bbox_inches='tight')
#         print("   Saved confusion matrix to: match_outcome_confusion_matrix.png")
        
#         # Feature importance
#         self.plot_feature_importance()
        
#         return accuracy
    
#     def plot_feature_importance(self):
#         """Plot feature importance"""
#         importance_df = pd.DataFrame({
#             'feature': self.feature_names,
#             'importance': self.model.feature_importances_
#         }).sort_values('importance', ascending=False)
        
#         plt.figure(figsize=(10, 8))
#         plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
#         plt.xlabel('Importance')
#         plt.title('Top 15 Features for Match Outcome Prediction')
#         plt.gca().invert_yaxis()
#         plt.tight_layout()
#         plt.savefig('match_outcome_feature_importance.png', dpi=300, bbox_inches='tight')
#         print("   Saved feature importance to: match_outcome_feature_importance.png")
    
#     def predict(self, X):
#         """
#         Predict match outcome probabilities
#         Returns: Dictionary with probabilities for each outcome
#         """
#         X_scaled = self.scaler.transform(X)
#         probabilities = self.model.predict_proba(X_scaled)
        
#         results = []
#         for probs in probabilities:
#             result = {}
#             for idx, outcome in enumerate(self.label_encoder.classes_):
#                 if outcome == 'H':
#                     result['home_win_prob'] = probs[idx]
#                 elif outcome == 'D':
#                     result['draw_prob'] = probs[idx]
#                 elif outcome == 'A':
#                     result['away_win_prob'] = probs[idx]
#             results.append(result)
        
#         return results
    
#     def save_model(self, filename='match_outcome_model.pkl'):
#         """Save trained model"""
#         joblib.dump({
#             'model': self.model,
#             'scaler': self.scaler,
#             'label_encoder': self.label_encoder,
#             'feature_names': self.feature_names
#         }, filename)
#         print(f"✅ Model saved to: {filename}")
    
#     def load_model(self, filename='match_outcome_model.pkl'):
#         """Load trained model"""
#         data = joblib.load(filename)
#         self.model = data['model']
#         self.scaler = data['scaler']
#         self.label_encoder = data['label_encoder']
#         self.feature_names = data['feature_names']
#         print(f"✅ Model loaded from: {filename}")


# def main():
#     # Initialize predictor
#     predictor = MatchOutcomePredictor()
    
#     # Prepare data
#     X, y, df = predictor.prepare_data('ml_data/match_outcomes.csv')
    
#     # Train model
#     accuracy = predictor.train(X, y)
    
#     # Save model
#     predictor.save_model()
    
#     # Example prediction
#     print("\n🔮 Example Prediction:")
#     print("   Using first match from dataset...")
    
#     sample_features = X.iloc[[0]]
#     sample_match = df.iloc[0]
    
#     prediction = predictor.predict(sample_features)[0]
    
#     print(f"\n   Match: {sample_match['home_team']} vs {sample_match['away_team']}")
#     print(f"   Actual result: {sample_match['result']}")
#     print(f"\n   Predictions:")
#     print(f"   - Home Win: {prediction['home_win_prob']:.1%}")
#     print(f"   - Draw: {prediction['draw_prob']:.1%}")
#     print(f"   - Away Win: {prediction['away_win_prob']:.1%}")
    
#     print("\n✅ Match Outcome Predictor training complete!")


# if __name__ == '__main__':
#     main()

"""
SPARK ML Model #1: Match Outcome Predictor
Predicts: Win/Draw/Loss probabilities
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # <--- Added tqdm
import warnings
warnings.filterwarnings('ignore')

class MatchOutcomePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def create_features(self, df):
        """
        Engineer features for match outcome prediction
        """
        features = df.copy()
        
        # Basic features
        features['home_advantage'] = 1  # Home team indicator
        
        # Form features (points, wins, etc.)
        features['home_form'] = features['home_points'] / (features['home_wins'] + features['home_draws'] + features['home_losses'] + 1)
        features['away_form'] = features['away_points'] / (features['away_wins'] + features['away_draws'] + features['away_losses'] + 1)
        
        # Goal scoring ability
        features['home_attack'] = features['home_gf'] / (features['home_wins'] + features['home_draws'] + features['home_losses'] + 1)
        features['away_attack'] = features['away_gf'] / (features['away_wins'] + features['away_draws'] + features['away_losses'] + 1)
        
        # Defensive strength
        features['home_defense'] = features['home_ga'] / (features['home_wins'] + features['home_draws'] + features['home_losses'] + 1)
        features['away_defense'] = features['away_ga'] / (features['away_wins'] + features['away_draws'] + features['away_losses'] + 1)
        
        # Goal difference per game
        features['home_gd_per_game'] = features['home_gd'] / (features['home_wins'] + features['home_draws'] + features['home_losses'] + 1)
        features['away_gd_per_game'] = features['away_gd'] / (features['away_wins'] + features['away_draws'] + features['away_losses'] + 1)
        
        # Win rate
        features['home_win_rate'] = features['home_wins'] / (features['home_wins'] + features['home_draws'] + features['home_losses'] + 1)
        features['away_win_rate'] = features['away_wins'] / (features['away_wins'] + features['away_draws'] + features['away_losses'] + 1)
        
        # Comparative features
        features['points_diff'] = features['home_points'] - features['away_points']
        features['gd_diff'] = features['home_gd'] - features['away_gd']
        features['form_diff'] = features['home_form'] - features['away_form']
        features['attack_diff'] = features['home_attack'] - features['away_attack']
        features['defense_diff'] = features['home_defense'] - features['away_defense']
        
        # xG features (if available)
        if 'home_xg' in features.columns and 'away_xg' in features.columns:
            features['xg_diff'] = features['home_xg'].fillna(0) - features['away_xg'].fillna(0)
            features['home_xg_per_match'] = features['home_xg'].fillna(0)
            features['away_xg_per_match'] = features['away_xg'].fillna(0)
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features
    
    def prepare_data(self, csv_path):
        """
        Load and prepare data for training
        """
        print("📊 Loading match data...")
        df = pd.read_csv(csv_path)
        
        print(f"✅ Loaded {len(df)} matches")
        print(f"   - Home wins: {(df['result'] == 'H').sum()}")
        print(f"   - Draws: {(df['result'] == 'D').sum()}")
        print(f"   - Away wins: {(df['result'] == 'A').sum()}")
        
        # Create features
        features = self.create_features(df)
        
        # Select feature columns
        feature_cols = [
            'home_advantage', 'home_form', 'away_form',
            'home_attack', 'away_attack', 'home_defense', 'away_defense',
            'home_gd_per_game', 'away_gd_per_game',
            'home_win_rate', 'away_win_rate',
            'points_diff', 'gd_diff', 'form_diff', 'attack_diff', 'defense_diff',
            'home_points', 'away_points', 'home_gd', 'away_gd'
        ]
        
        # Add xG features if available
        if 'xg_diff' in features.columns:
            feature_cols.extend(['xg_diff', 'home_xg_per_match', 'away_xg_per_match'])
        
        X = features[feature_cols]
        y = df['result']
        
        self.feature_names = feature_cols
        
        return X, y, df
    
    def train(self, X, y, test_size=0.2):
        """
        Train the match outcome prediction model
        """
        print("\n🎯 Training Match Outcome Predictor...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model (best for this task)
        print("   Training XGBoost classifier...")
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Model trained successfully!")
        print(f"   Accuracy: {accuracy:.2%}")
        
        # Detailed classification report
        print("\n📊 Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Match Outcome Prediction - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('match_outcome_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("   Saved confusion matrix to: match_outcome_confusion_matrix.png")
        
        # Feature importance
        self.plot_feature_importance()
        
        return accuracy
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
        plt.xlabel('Importance')
        plt.title('Top 15 Features for Match Outcome Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('match_outcome_feature_importance.png', dpi=300, bbox_inches='tight')
        print("   Saved feature importance to: match_outcome_feature_importance.png")
    
    def predict(self, X):
        """
        Predict match outcome probabilities
        Returns: Dictionary with probabilities for each outcome
        """
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        results = []
        # Wrap the loop processing predictions with tqdm
        for probs in tqdm(probabilities, desc="Generating Predictions"): # <--- Changed
            result = {}
            for idx, outcome in enumerate(self.label_encoder.classes_):
                if outcome == 'H':
                    result['home_win_prob'] = probs[idx]
                elif outcome == 'D':
                    result['draw_prob'] = probs[idx]
                elif outcome == 'A':
                    result['away_win_prob'] = probs[idx]
            results.append(result)
        
        return results
    
    def save_model(self, filename='match_outcome_model.pkl'):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }, filename)
        print(f"✅ Model saved to: {filename}")
    
    def load_model(self, filename='match_outcome_model.pkl'):
        """Load trained model"""
        data = joblib.load(filename)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        self.feature_names = data['feature_names']
        print(f"✅ Model loaded from: {filename}")


def main():
    # Initialize predictor
    predictor = MatchOutcomePredictor()
    
    # Prepare data
    X, y, df = predictor.prepare_data('ml_data/match_outcomes.csv')
    
    # Train model
    accuracy = predictor.train(X, y)
    
    # Save model
    predictor.save_model()
    
    # Example prediction
    print("\n🔮 Example Prediction:")
    print("   Using first match from dataset...")
    
    sample_features = X.iloc[[0]]
    sample_match = df.iloc[0]
    
    prediction = predictor.predict(sample_features)[0]
    
    print(f"\n   Match: {sample_match['home_team']} vs {sample_match['away_team']}")
    print(f"   Actual result: {sample_match['result']}")
    print(f"\n   Predictions:")
    print(f"   - Home Win: {prediction['home_win_prob']:.1%}")
    print(f"   - Draw: {prediction['draw_prob']:.1%}")
    print(f"   - Away Win: {prediction['away_win_prob']:.1%}")
    
    print("\n✅ Match Outcome Predictor training complete!")


if __name__ == '__main__':
    main()