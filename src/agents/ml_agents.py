"""
Enhanced ML-Powered Agents
Agents that learn from synthetic data to make better decisions
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import json

from .core_agents import BaseAgent, UserContext, CauseRecommendation

logger = logging.getLogger(__name__)

class MLDonationLikelyScoreAgent(BaseAgent):
    """ML-enhanced donation likelihood scoring agent"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("MLDonationLikelyScoreAgent")
        self.model = None
        self.scaler = None
        self.encoders = None
        self.feature_columns = [
            'transaction_amount', 'wallet_balance', 'days_since_last_donation',
            'days_since_last_prompt', 'total_lifetime_donations', 'average_donation_amount',
            'is_established_donor', 'transaction_to_wallet_ratio', 'wallet_health_score',
            'transaction_category_encoded', 'age_range_encoded', 'income_level_encoded',
            'day_of_week', 'hour_of_day', 'is_weekend', 'is_payday_week'
        ]
        
        if model_path:
            self.load_model(model_path)
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Train the donation likelihood model"""
        
        self.logger.info("Training donation likelihood model...")
        
        # Train Gradient Boosting Classifier
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test)
        
        self.logger.info(f"Model trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def process(self, context: UserContext) -> Dict[str, Any]:
        """Process user context with ML model"""
        
        if not self.model:
            # Fallback to rule-based logic if no model
            return self._fallback_process(context)
        
        try:
            # Prepare features
            features = self._prepare_features(context)
            features_df = pd.DataFrame([features])
            
            # Ensure all required columns are present
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            features_df = features_df[self.feature_columns]
            
            # Scale features if scaler is available
            if self.scaler:
                numerical_features = ['transaction_amount', 'wallet_balance', 'days_since_last_donation',
                                    'days_since_last_prompt', 'total_lifetime_donations', 'average_donation_amount']
                features_df[numerical_features] = self.scaler.transform(features_df[numerical_features])
            
            # Get prediction and probability
            likelihood_prob = self.model.predict_proba(features_df)[0, 1]  # Probability of donation
            likelihood_score = int(likelihood_prob * 100)  # Convert to 0-100 scale
            
            # Decision logic based on ML prediction
            if likelihood_prob >= 0.7:
                recommendation = "prompt_now"
            elif likelihood_prob >= 0.4:
                recommendation = "prompt_with_caution"
            else:
                recommendation = "skip_prompt"
            
            # Get feature importance for interpretability
            feature_importance = self._get_feature_importance(features_df)
            
            decision = {
                'likelihood_score': likelihood_score,
                'likelihood_probability': likelihood_prob,
                'recommendation': recommendation,
                'ml_confidence': self._calculate_ml_confidence(likelihood_prob),
                'top_factors': feature_importance[:3],
                'model_version': 'ml_v1.0',
                'next_check_hours': self._calculate_next_check(likelihood_score)
            }
            
            self.log_decision(context, decision, f"ML Score: {likelihood_score}, Prob: {likelihood_prob:.3f}")
            return decision
            
        except Exception as e:
            self.logger.error(f"ML processing failed: {str(e)}, falling back to rules")
            return self._fallback_process(context)
    
    def _prepare_features(self, context: UserContext) -> Dict[str, Any]:
        """Prepare features for ML model"""
        
        # Encode categorical variables
        transaction_category_encoded = self._encode_category(context.transaction_category)
        age_range_encoded = self._encode_age_range(context.demographic_hints.get('age_range', '25-34'))
        income_level_encoded = self._encode_income_level(context.demographic_hints.get('income_level', 'middle'))
        
        # Time-based features
        now = datetime.now()
        
        features = {
            'transaction_amount': context.transaction_amount,
            'wallet_balance': context.wallet_balance,
            'days_since_last_donation': context.days_since_last_donation,
            'days_since_last_prompt': context.days_since_last_prompt,
            'total_lifetime_donations': context.total_lifetime_donations,
            'average_donation_amount': context.average_donation_amount,
            'is_established_donor': int(context.total_lifetime_donations > 0),
            'transaction_to_wallet_ratio': context.transaction_amount / max(1, context.wallet_balance),
            'wallet_health_score': min(5, context.wallet_balance / 1000),
            'transaction_category_encoded': transaction_category_encoded,
            'age_range_encoded': age_range_encoded,
            'income_level_encoded': income_level_encoded,
            'day_of_week': now.weekday(),
            'hour_of_day': now.hour,
            'is_weekend': int(now.weekday() >= 5),
            'is_payday_week': int(now.day <= 7 or now.day >= 25)
        }
        
        return features
    
    def _get_feature_importance(self, features_df: pd.DataFrame) -> List[str]:
        """Get top important features for this prediction"""
        
        if not hasattr(self.model, 'feature_importances_'):
            return ['ml_prediction', 'historical_pattern', 'transaction_context']
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_names = self.feature_columns
        
        # Sort by importance
        importance_pairs = list(zip(feature_names, importances))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top features
        return [name for name, _ in importance_pairs[:5]]
    
    def _calculate_ml_confidence(self, probability: float) -> str:
        """Calculate confidence in ML prediction"""
        
        # Confidence based on how far probability is from 0.5 (uncertainty)
        distance_from_uncertain = abs(probability - 0.5)
        
        if distance_from_uncertain >= 0.4:
            return 'high'
        elif distance_from_uncertain >= 0.2:
            return 'medium'
        else:
            return 'low'
    
    def _fallback_process(self, context: UserContext) -> Dict[str, Any]:
        """Fallback to rule-based processing when ML fails"""
        
        # Import the original rule-based agent
        from .core_agents import DonationLikelyScoreAgent
        
        rule_agent = DonationLikelyScoreAgent()
        result = rule_agent.process(context)
        result['model_version'] = 'fallback_rules'
        
        return result
    
    def _encode_category(self, category: str) -> int:
        """Encode transaction category"""
        if self.encoders and 'transaction_category' in self.encoders:
            try:
                return self.encoders['transaction_category'].transform([category])[0]
            except:
                pass
        return hash(category) % 100
    
    def _encode_age_range(self, age_range: str) -> int:
        """Encode age range"""
        if self.encoders and 'age_range' in self.encoders:
            try:
                return self.encoders['age_range'].transform([age_range])[0]
            except:
                pass
        age_mapping = {'18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55-64': 5, '65+': 6}
        return age_mapping.get(age_range, 2)
    
    def _encode_income_level(self, income_level: str) -> int:
        """Encode income level"""
        if self.encoders and 'income_level' in self.encoders:
            try:
                return self.encoders['income_level'].transform([income_level])[0]
            except:
                pass
        income_mapping = {'low': 1, 'lower_middle': 2, 'middle': 3, 'upper_middle': 4, 'high': 5}
        return income_mapping.get(income_level, 3)
    
    def _calculate_next_check(self, score: float) -> int:
        """Calculate when to next evaluate this user"""
        if score >= 70:
            return 24 * 7  # 1 week
        elif score >= 40:
            return 24 * 3  # 3 days
        else:
            return 24 * 1  # 1 day
    
    def save_model(self, path: str):
        """Save trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model and preprocessors"""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.encoders = model_data.get('encoders')
            self.feature_columns = model_data.get('feature_columns', self.feature_columns)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load model from {path}: {str(e)}")


class MLDonationAmountOptimizerAgent(BaseAgent):
    """ML-enhanced donation amount optimization agent"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("MLDonationAmountOptimizerAgent")
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'transaction_amount', 'wallet_balance', 'user_avg_donation', 'user_total_donations',
            'transaction_amount_log', 'wallet_to_transaction_ratio', 'is_first_time_donor',
            'is_essential_purchase', 'is_luxury_purchase'
        ]
        
        if model_path:
            self.load_model(model_path)
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Train the donation amount optimization model"""
        
        self.logger.info("Training donation amount model...")
        
        # Train Random Forest Regressor
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        self.logger.info(f"Model trained - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}, RMSE: {rmse:.2f}")
        
        return {
            'train_r2': train_score,
            'test_r2': test_score,
            'rmse': rmse,
            'mse': mse
        }
    
    def process(self, context: UserContext) -> Dict[str, Any]:
        """Process user context with ML model"""
        
        if not self.model:
            return self._fallback_process(context)
        
        try:
            # Prepare features
            features = self._prepare_features(context)
            features_df = pd.DataFrame([features])
            
            # Ensure all required columns are present
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            features_df = features_df[self.feature_columns]
            
            # Scale features if scaler is available
            if self.scaler:
                features_df = pd.DataFrame(
                    self.scaler.transform(features_df),
                    columns=features_df.columns
                )
            
            # Get prediction
            predicted_amount = self.model.predict(features_df)[0]
            predicted_amount = max(1.0, predicted_amount)  # Minimum ₱1
            
            # Generate alternative amounts around prediction
            alternatives = self._generate_alternatives(predicted_amount, context)
            
            # Round to sensible values
            primary_amount = self._round_to_sensible(predicted_amount)
            alternatives = [self._round_to_sensible(amt) for amt in alternatives]
            
            decision = {
                'primary_amount': primary_amount,
                'alternative_amounts': alternatives,
                'ml_prediction': predicted_amount,
                'reasoning': self._generate_ml_reasoning(predicted_amount, context),
                'confidence_score': self._calculate_prediction_confidence(features_df),
                'model_version': 'ml_v1.0'
            }
            
            self.log_decision(context, decision, f"ML predicted: ₱{predicted_amount:.2f}, suggested: ₱{primary_amount}")
            return decision
            
        except Exception as e:
            self.logger.error(f"ML processing failed: {str(e)}, falling back to rules")
            return self._fallback_process(context)
    
    def _prepare_features(self, context: UserContext) -> Dict[str, Any]:
        """Prepare features for ML model"""
        
        features = {
            'transaction_amount': context.transaction_amount,
            'wallet_balance': context.wallet_balance,
            'user_avg_donation': context.average_donation_amount,
            'user_total_donations': context.total_lifetime_donations,
            'transaction_amount_log': np.log1p(context.transaction_amount),
            'wallet_to_transaction_ratio': context.wallet_balance / max(1, context.transaction_amount),
            'is_first_time_donor': int(context.total_lifetime_donations == 0),
            'is_essential_purchase': int(context.transaction_category in ['groceries', 'utilities', 'food']),
            'is_luxury_purchase': int(context.transaction_category in ['entertainment', 'shopping', 'electronics'])
        }
        
        return features
    
    def _generate_alternatives(self, predicted_amount: float, context: UserContext) -> List[float]:
        """Generate alternative amounts around prediction"""
        
        # Generate amounts 50% and 150% of prediction
        lower_alt = predicted_amount * 0.7
        higher_alt = predicted_amount * 1.3
        
        # Ensure alternatives are reasonable
        lower_alt = max(1.0, lower_alt)
        higher_alt = min(100.0, higher_alt)  # Cap at ₱100
        
        # Also consider wallet constraints
        wallet_limit = context.wallet_balance * 0.01  # Max 1% of wallet
        higher_alt = min(higher_alt, wallet_limit)
        
        return [lower_alt, higher_alt]
    
    def _round_to_sensible(self, amount: float) -> float:
        """Round amount to sensible values"""
        
        if amount < 2:
            return 1.0
        elif amount < 5:
            return round(amount)
        elif amount < 10:
            return round(amount * 2) / 2  # Round to 0.5
        elif amount < 50:
            return round(amount)
        else:
            return round(amount / 5) * 5  # Round to nearest 5
    
    def _generate_ml_reasoning(self, predicted_amount: float, context: UserContext) -> str:
        """Generate reasoning for ML prediction"""
        
        if context.total_lifetime_donations == 0:
            return f"AI suggests ₱{predicted_amount:.0f} as perfect starter amount"
        elif predicted_amount <= 5:
            return f"AI optimized for your giving pattern: small but meaningful"
        elif predicted_amount <= 20:
            return f"AI matched to your transaction size and history"
        else:
            return f"AI suggests generous amount based on your capacity"
    
    def _calculate_prediction_confidence(self, features_df: pd.DataFrame) -> float:
        """Calculate confidence in prediction"""
        
        if hasattr(self.model, 'estimators_'):
            # For Random Forest, use prediction variance
            predictions = np.array([tree.predict(features_df)[0] for tree in self.model.estimators_])
            variance = np.var(predictions)
            
            # Convert variance to confidence (lower variance = higher confidence)
            confidence = 1.0 / (1.0 + variance)
            return min(1.0, confidence)
        
        return 0.8  # Default confidence
    
    def _fallback_process(self, context: UserContext) -> Dict[str, Any]:
        """Fallback to rule-based processing when ML fails"""
        
        from .core_agents import DonationAmountOptimizerAgent
        
        rule_agent = DonationAmountOptimizerAgent()
        result = rule_agent.process(context)
        result['model_version'] = 'fallback_rules'
        
        return result
    
    def save_model(self, path: str):
        """Save trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model and preprocessors"""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.feature_columns = model_data.get('feature_columns', self.feature_columns)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load model from {path}: {str(e)}")


class MLEnhancedOrchestrator:
    """Orchestrator that uses ML-enhanced agents"""
    
    def __init__(self, ngo_database, model_dir: str = 'models'):
        self.likelihood_agent = MLDonationLikelyScoreAgent(f"{model_dir}/likelihood_model.joblib")
        self.amount_agent = MLDonationAmountOptimizerAgent(f"{model_dir}/amount_model.joblib")
        
        # Keep rule-based cause agent for now
        from .core_agents import LocalCauseRecommenderAgent
        self.cause_agent = LocalCauseRecommenderAgent(ngo_database)
        
        self.logger = logging.getLogger("ImpactSense.MLOrchestrator")
    
    def generate_recommendation(self, context: UserContext) -> Optional[Dict[str, Any]]:
        """Generate ML-enhanced recommendation"""
        
        try:
            # Step 1: ML likelihood assessment
            likelihood_result = self.likelihood_agent.process(context)
            
            if likelihood_result['recommendation'] == 'skip_prompt':
                self.logger.info(f"User {context.user_id}: ML recommends skipping - score {likelihood_result['likelihood_score']}")
                return None
            
            # Step 2: Cause recommendation (rule-based for now)
            cause_result = self.cause_agent.process(context)
            
            if not cause_result.get('primary_recommendation'):
                self.logger.info(f"User {context.user_id}: No suitable causes found")
                return None
            
            # Step 3: ML amount optimization
            amount_result = self.amount_agent.process(context)
            
            # Step 4: Combine results
            recommendation = {
                'primary_amount': amount_result['primary_amount'],
                'alternative_amounts': amount_result['alternative_amounts'],
                'cause': cause_result['primary_recommendation'],
                'likelihood_score': likelihood_result['likelihood_score'],
                'ml_confidence': {
                    'likelihood': likelihood_result.get('ml_confidence', 'medium'),
                    'amount': amount_result.get('confidence_score', 0.8)
                },
                'reasoning': {
                    'likelihood': likelihood_result.get('top_factors', []),
                    'amount': amount_result.get('reasoning', ''),
                    'cause': f"Matched to {cause_result['primary_recommendation'].cause_category} in {cause_result['primary_recommendation'].region_focus}"
                },
                'model_versions': {
                    'likelihood': likelihood_result.get('model_version', 'unknown'),
                    'amount': amount_result.get('model_version', 'unknown')
                },
                'timing_optimal': likelihood_result['recommendation'] == 'prompt_now'
            }
            
            self.logger.info(f"User {context.user_id}: ML recommendation generated")
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error in ML orchestrator for user {context.user_id}: {str(e)}")
            return None


def train_all_models():
    """Train all ML models with synthetic data"""
    
    print("🤖 Training ML models for ImpactSense agents...")
    
    # Load preprocessed training data
    try:
        X_train_lik = pd.read_csv('data/X_train_likelihood.csv')
        y_train_lik = pd.read_csv('data/y_train_likelihood.csv').iloc[:, 0]
        X_test_lik = pd.read_csv('data/X_test_likelihood.csv')
        y_test_lik = pd.read_csv('data/y_test_likelihood.csv').iloc[:, 0]
        
        X_train_amt = pd.read_csv('data/X_train_amount.csv')
        y_train_amt = pd.read_csv('data/y_train_amount.csv').iloc[:, 0]
        X_test_amt = pd.read_csv('data/X_test_amount.csv')
        y_test_amt = pd.read_csv('data/y_test_amount.csv').iloc[:, 0]
        
        # Load preprocessors
        scalers = joblib.load('data/scalers.joblib')
        encoders = joblib.load('data/encoders.joblib')
        
    except Exception as e:
        print(f"❌ Error loading training data: {str(e)}")
        print("💡 Please run generate_synthetic_data.py and preprocess_training_data.py first")
        return
    
    # Create models directory
    import os
    os.makedirs('models', exist_ok=True)
    
    # Train likelihood model
    print("🎯 Training donation likelihood model...")
    likelihood_agent = MLDonationLikelyScoreAgent()
    likelihood_agent.scaler = scalers.get('likelihood')
    likelihood_agent.encoders = encoders
    
    likelihood_metrics = likelihood_agent.train_model(X_train_lik, y_train_lik, X_test_lik, y_test_lik)
    likelihood_agent.save_model('models/likelihood_model.joblib')
    
    print(f"✅ Likelihood model trained - Accuracy: {likelihood_metrics['test_accuracy']:.3f}")
    
    # Train amount model
    print("💰 Training donation amount model...")
    amount_agent = MLDonationAmountOptimizerAgent()
    amount_agent.scaler = scalers.get('amount')
    
    amount_metrics = amount_agent.train_model(X_train_amt, y_train_amt, X_test_amt, y_test_amt)
    amount_agent.save_model('models/amount_model.joblib')
    
    print(f"✅ Amount model trained - R²: {amount_metrics['test_r2']:.3f}, RMSE: ₱{amount_metrics['rmse']:.2f}")
    
    # Save training metrics
    training_summary = {
        'training_date': datetime.now().isoformat(),
        'likelihood_model': likelihood_metrics,
        'amount_model': amount_metrics,
        'training_samples': {
            'likelihood': len(X_train_lik),
            'amount': len(X_train_amt)
        }
    }
    
    with open('models/training_summary.json', 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print(f"\n🎉 All models trained successfully!")
    print(f"📁 Models saved in /models/ directory")
    print(f"📊 Training summary saved to models/training_summary.json")


if __name__ == "__main__":
    train_all_models()
