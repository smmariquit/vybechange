"""
Agent Training Data Preprocessor
Converts raw synthetic data into agent-ready formats for machine learning
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from typing import Dict, List, Tuple, Any

class AgentDataPreprocessor:
    """Preprocesses synthetic data for agent training and inference"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
    
    def prepare_donation_likelihood_data(self, users_df: pd.DataFrame, 
                                       transactions_df: pd.DataFrame,
                                       donations_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare training data for DonationLikelyScoreAgent"""
        
        # Merge transaction and user data
        transaction_features = transactions_df.merge(users_df, on='user_id', how='left')
        
        # Add donation outcomes
        transaction_features['donated'] = transaction_features['transaction_id'].isin(
            donations_df['transaction_id']
        ).astype(int)
        
        # Feature engineering for likelihood scoring
        features = []
        
        for _, row in transaction_features.iterrows():
            feature_row = {
                'user_id': row['user_id'],
                'transaction_id': row['transaction_id'],
                
                # Core features that agents use
                'transaction_amount': row['amount'],
                'wallet_balance': row['wallet_balance'],
                'days_since_last_donation': row['days_since_last_donation'],
                'days_since_last_prompt': row['days_since_last_prompt'],
                'total_lifetime_donations': row['total_lifetime_donations'],
                'average_donation_amount': row['average_donation_amount'],
                
                # Derived features
                'is_established_donor': int(row['total_lifetime_donations'] > 0),
                'transaction_to_wallet_ratio': row['amount'] / max(1, row['wallet_balance']),
                'wallet_health_score': min(5, row['wallet_balance'] / 1000),
                
                # Categorical features (encoded)
                'transaction_category': row['category'],
                'age_range': row['age_range'],
                'income_level': row['income_level'],
                'region': row['region'],
                'donor_segment': row['donor_segment'],
                
                # Time-based features
                'day_of_week': row['transaction_date'].weekday(),
                'hour_of_day': row['transaction_date'].hour,
                'is_weekend': int(row['transaction_date'].weekday() >= 5),
                'is_payday_week': int(row['transaction_date'].day <= 7 or row['transaction_date'].day >= 25),
                
                # Target variable
                'donated': row['donated']
            }
            features.append(feature_row)
        
        features_df = pd.DataFrame(features)
        
        # Encode categorical variables
        categorical_columns = ['transaction_category', 'age_range', 'income_level', 'region', 'donor_segment']
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                features_df[f'{col}_encoded'] = self.encoders[col].fit_transform(features_df[col])
            else:
                features_df[f'{col}_encoded'] = self.encoders[col].transform(features_df[col])
        
        return features_df
    
    def prepare_cause_recommendation_data(self, users_df: pd.DataFrame,
                                        donations_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare training data for LocalCauseRecommenderAgent"""
        
        # Create positive examples (successful cause matches)
        positive_examples = []
        
        for _, donation in donations_df.iterrows():
            user = users_df[users_df['user_id'] == donation['user_id']].iloc[0]
            
            example = {
                'user_id': donation['user_id'],
                'ngo_id': donation['ngo_id'],
                'cause_category': donation['cause_category'],
                'user_region': user['region'],
                'user_preferred_causes': user['preferred_causes'],
                'donation_amount': donation['amount'],
                'user_lifetime_donations': user['total_lifetime_donations'],
                'match_success': 1  # Positive example
            }
            positive_examples.append(example)
        
        # Create negative examples (unsuccessful cause matches)
        from constants.ngos import NGOS
        negative_examples = []
        
        # For each user who donated, create examples of NGOs they didn't choose
        donor_users = users_df[users_df['is_active_donor'] == True]
        
        for _, user in donor_users.sample(min(1000, len(donor_users))).iterrows():
            user_donations = donations_df[donations_df['user_id'] == user['user_id']]
            chosen_ngos = set(user_donations['ngo_id'].values)
            
            # Sample NGOs they didn't choose
            available_ngos = [ngo for ngo in NGOS if ngo['ngo_id'] not in chosen_ngos]
            
            for ngo in np.random.choice(available_ngos, min(3, len(available_ngos)), replace=False):
                example = {
                    'user_id': user['user_id'],
                    'ngo_id': ngo['ngo_id'],
                    'cause_category': ngo['category'],
                    'user_region': user['region'],
                    'user_preferred_causes': user['preferred_causes'],
                    'donation_amount': 0,  # No donation made
                    'user_lifetime_donations': user['total_lifetime_donations'],
                    'match_success': 0  # Negative example
                }
                negative_examples.append(example)
        
        # Combine positive and negative examples
        all_examples = positive_examples + negative_examples
        recommendation_df = pd.DataFrame(all_examples)
        
        # Feature engineering for cause matching
        recommendation_df['user_preferred_causes_list'] = recommendation_df['user_preferred_causes'].apply(
            lambda x: json.loads(x) if x else []
        )
        
        recommendation_df['cause_in_preferences'] = recommendation_df.apply(
            lambda row: int(row['cause_category'] in row['user_preferred_causes_list']), axis=1
        )
        
        recommendation_df['region_match'] = recommendation_df.apply(
            lambda row: int(self._check_region_match(row['user_region'], row['ngo_id'])), axis=1
        )
        
        return recommendation_df
    
    def prepare_amount_optimization_data(self, donations_df: pd.DataFrame,
                                       transactions_df: pd.DataFrame,
                                       users_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare training data for DonationAmountOptimizerAgent"""
        
        # Merge donation data with transaction and user context
        amount_data = donations_df.merge(transactions_df, on='transaction_id', how='left')
        amount_data = amount_data.merge(users_df, on='user_id', how='left')
        
        # Feature engineering for amount optimization
        features = []
        
        for _, row in amount_data.iterrows():
            feature_row = {
                'user_id': row['user_id'],
                'donation_id': row['donation_id'],
                
                # Context features
                'transaction_amount': row['amount_x'],  # Transaction amount
                'wallet_balance': row['wallet_balance'],
                'user_avg_donation': row['average_donation_amount'],
                'user_total_donations': row['total_lifetime_donations'],
                'transaction_category': row['category'],
                'cause_category': row['cause_category'],
                
                # Derived features
                'transaction_amount_log': np.log1p(row['amount_x']),
                'wallet_to_transaction_ratio': row['wallet_balance'] / max(1, row['amount_x']),
                'is_first_time_donor': int(row['total_lifetime_donations'] == row['amount_y']),  # Donation amount
                
                # Psychological factors
                'is_essential_purchase': int(row['category'] in ['groceries', 'utilities', 'food']),
                'is_luxury_purchase': int(row['category'] in ['entertainment', 'shopping', 'electronics']),
                
                # Target variable
                'donation_amount': row['amount_y']  # Actual donation amount
            }
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def prepare_user_segmentation_data(self, users_df: pd.DataFrame,
                                     engagement_df: pd.DataFrame,
                                     donations_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for user segmentation and targeting"""
        
        # Merge user data with engagement metrics
        segmentation_data = users_df.merge(engagement_df, on='user_id', how='left')
        
        # Add donation aggregates
        donation_aggs = donations_df.groupby('user_id').agg({
            'amount': ['sum', 'mean', 'count', 'std'],
            'donation_date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        donation_aggs.columns = ['user_id', 'total_donated', 'avg_donation', 'donation_count', 
                               'donation_std', 'first_donation', 'last_donation']
        
        segmentation_data = segmentation_data.merge(donation_aggs, on='user_id', how='left')
        
        # Fill NAs for non-donors
        segmentation_data[['total_donated', 'avg_donation', 'donation_count', 'donation_std']] = \
            segmentation_data[['total_donated', 'avg_donation', 'donation_count', 'donation_std']].fillna(0)
        
        # Feature engineering for segmentation
        segmentation_data['days_since_first_donation'] = (
            datetime.now() - pd.to_datetime(segmentation_data['first_donation'])
        ).dt.days.fillna(999)
        
        segmentation_data['days_since_last_donation'] = (
            datetime.now() - pd.to_datetime(segmentation_data['last_donation'])
        ).dt.days.fillna(999)
        
        segmentation_data['donation_frequency'] = segmentation_data['donation_count'] / \
            np.maximum(1, segmentation_data['days_since_first_donation'] / 30)  # Per month
        
        segmentation_data['engagement_score'] = (
            segmentation_data['donation_opt_in_rate'] * 0.3 +
            segmentation_data['retention_score'] * 0.3 +
            segmentation_data['impact_updates_opened'] / 10 * 0.2 +
            segmentation_data['impact_stories_shared'] * 0.2
        )
        
        return segmentation_data
    
    def create_agent_training_sets(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create train/test splits for all agents"""
        
        print("📊 Loading synthetic datasets...")
        
        # Load all datasets
        users_df = pd.read_csv('data/user_profiles.csv')
        transactions_df = pd.read_csv('data/transaction_history.csv')
        donations_df = pd.read_csv('data/donation_history.csv')
        engagement_df = pd.read_csv('data/user_engagement.csv')
        
        # Convert date columns
        date_columns = ['transaction_date', 'donation_date', 'created_at', 'last_activity_date']
        for df, cols in [(transactions_df, ['transaction_date']), 
                        (donations_df, ['donation_date']),
                        (users_df, ['created_at']),
                        (engagement_df, ['last_activity_date'])]:
            for col in cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        
        training_sets = {}
        
        # 1. Donation Likelihood Agent Data
        print("🎯 Preparing donation likelihood training data...")
        likelihood_data = self.prepare_donation_likelihood_data(users_df, transactions_df, donations_df)
        
        # Features for likelihood model
        likelihood_features = [
            'transaction_amount', 'wallet_balance', 'days_since_last_donation',
            'days_since_last_prompt', 'total_lifetime_donations', 'average_donation_amount',
            'is_established_donor', 'transaction_to_wallet_ratio', 'wallet_health_score',
            'transaction_category_encoded', 'age_range_encoded', 'income_level_encoded',
            'day_of_week', 'hour_of_day', 'is_weekend', 'is_payday_week'
        ]
        
        X_likelihood = likelihood_data[likelihood_features]
        y_likelihood = likelihood_data['donated']
        
        X_train_lik, X_test_lik, y_train_lik, y_test_lik = train_test_split(
            X_likelihood, y_likelihood, test_size=0.2, random_state=42, stratify=y_likelihood
        )
        
        # Scale numerical features
        numerical_features = ['transaction_amount', 'wallet_balance', 'days_since_last_donation',
                            'days_since_last_prompt', 'total_lifetime_donations', 'average_donation_amount']
        
        scaler_likelihood = StandardScaler()
        X_train_lik[numerical_features] = scaler_likelihood.fit_transform(X_train_lik[numerical_features])
        X_test_lik[numerical_features] = scaler_likelihood.transform(X_test_lik[numerical_features])
        
        self.scalers['likelihood'] = scaler_likelihood
        training_sets['likelihood'] = (
            (X_train_lik, y_train_lik),
            (X_test_lik, y_test_lik)
        )
        
        # 2. Cause Recommendation Agent Data
        print("🎨 Preparing cause recommendation training data...")
        recommendation_data = self.prepare_cause_recommendation_data(users_df, donations_df)
        
        recommendation_features = [
            'user_lifetime_donations', 'cause_in_preferences', 'region_match'
        ]
        
        X_recommendation = recommendation_data[recommendation_features]
        y_recommendation = recommendation_data['match_success']
        
        X_train_rec, X_test_rec, y_train_rec, y_test_rec = train_test_split(
            X_recommendation, y_recommendation, test_size=0.2, random_state=42, stratify=y_recommendation
        )
        
        training_sets['recommendation'] = (
            (X_train_rec, y_train_rec),
            (X_test_rec, y_test_rec)
        )
        
        # 3. Amount Optimization Agent Data
        print("💰 Preparing amount optimization training data...")
        amount_data = self.prepare_amount_optimization_data(donations_df, transactions_df, users_df)
        
        amount_features = [
            'transaction_amount', 'wallet_balance', 'user_avg_donation', 'user_total_donations',
            'transaction_amount_log', 'wallet_to_transaction_ratio', 'is_first_time_donor',
            'is_essential_purchase', 'is_luxury_purchase'
        ]
        
        X_amount = amount_data[amount_features]
        y_amount = amount_data['donation_amount']
        
        X_train_amt, X_test_amt, y_train_amt, y_test_amt = train_test_split(
            X_amount, y_amount, test_size=0.2, random_state=42
        )
        
        # Scale features for amount optimization
        scaler_amount = StandardScaler()
        X_train_amt = pd.DataFrame(
            scaler_amount.fit_transform(X_train_amt), 
            columns=X_train_amt.columns, 
            index=X_train_amt.index
        )
        X_test_amt = pd.DataFrame(
            scaler_amount.transform(X_test_amt), 
            columns=X_test_amt.columns, 
            index=X_test_amt.index
        )
        
        self.scalers['amount'] = scaler_amount
        training_sets['amount'] = (
            (X_train_amt, y_train_amt),
            (X_test_amt, y_test_amt)
        )
        
        # 4. User Segmentation Data
        print("👥 Preparing user segmentation training data...")
        segmentation_data = self.prepare_user_segmentation_data(users_df, engagement_df, donations_df)
        
        # Create segmentation targets
        segmentation_data['engagement_segment'] = pd.cut(
            segmentation_data['engagement_score'], 
            bins=[-np.inf, 0.2, 0.5, 0.8, np.inf],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        segmentation_features = [
            'total_sessions', 'avg_session_duration_minutes', 'donation_opt_in_rate',
            'retention_score', 'lifetime_value_score', 'donation_frequency',
            'total_donated', 'avg_donation', 'donation_count'
        ]
        
        X_segmentation = segmentation_data[segmentation_features].fillna(0)
        y_segmentation = segmentation_data['engagement_segment']
        
        # Remove rows with NaN targets
        valid_indices = ~y_segmentation.isna()
        X_segmentation = X_segmentation[valid_indices]
        y_segmentation = y_segmentation[valid_indices]
        
        if len(X_segmentation) > 0:
            X_train_seg, X_test_seg, y_train_seg, y_test_seg = train_test_split(
                X_segmentation, y_segmentation, test_size=0.2, random_state=42, stratify=y_segmentation
            )
            
            training_sets['segmentation'] = (
                (X_train_seg, y_train_seg),
                (X_test_seg, y_test_seg)
            )
        
        return training_sets
    
    def save_training_data(self, training_sets: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]):
        """Save processed training data and preprocessors"""
        
        print("💾 Saving training datasets...")
        
        for agent_name, (train_data, test_data) in training_sets.items():
            X_train, y_train = train_data
            X_test, y_test = test_data
            
            # Save training data
            X_train.to_csv(f'data/X_train_{agent_name}.csv', index=False)
            y_train.to_csv(f'data/y_train_{agent_name}.csv', index=False)
            
            # Save test data
            X_test.to_csv(f'data/X_test_{agent_name}.csv', index=False)
            y_test.to_csv(f'data/y_test_{agent_name}.csv', index=False)
        
        # Save preprocessors
        joblib.dump(self.scalers, 'data/scalers.joblib')
        joblib.dump(self.encoders, 'data/encoders.joblib')
        
        print("✅ Training data saved successfully!")
    
    def _check_region_match(self, user_region: str, ngo_id: str) -> bool:
        """Check if NGO region matches user region"""
        from constants.ngos import NGOS
        ngo = next((ngo for ngo in NGOS if ngo['ngo_id'] == ngo_id), None)
        
        if not ngo:
            return False
        
        ngo_region = ngo['region_focus']
        
        # Exact match
        if ngo_region == user_region:
            return True
        
        # Nationwide coverage
        if ngo_region == 'Nationwide':
            return True
        
        # Regional matches (simplified)
        region_groups = {
            'Luzon': ['NCR', 'CAR', 'Region I', 'Region II', 'Region III', 'Region IV-A', 'Region IV-B', 'Region V'],
            'Visayas': ['Region VI', 'Region VII', 'Region VIII'],
            'Mindanao': ['Region IX', 'Region X', 'Region XI', 'Region XII', 'CARAGA', 'BARMM']
        }
        
        for region_group, regions in region_groups.items():
            if user_region in regions and region_group in ngo_region:
                return True
        
        return False


def main():
    """Main preprocessing pipeline"""
    print("🔄 Starting agent training data preprocessing...")
    
    preprocessor = AgentDataPreprocessor()
    
    # Create training sets for all agents
    training_sets = preprocessor.create_agent_training_sets()
    
    # Save processed data
    preprocessor.save_training_data(training_sets)
    
    # Print summary statistics
    print("\n📋 Training Data Summary:")
    for agent_name, (train_data, test_data) in training_sets.items():
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        print(f"\n{agent_name.upper()} Agent:")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Test samples: {len(X_test):,}")
        print(f"  Features: {len(X_train.columns)}")
        
        # Additional statistics for classification tasks
        if agent_name in ['likelihood', 'recommendation', 'segmentation']:
            if len(y_train.unique()) <= 10:  # Categorical target
                print(f"  Class distribution (train): {dict(y_train.value_counts())}")
    
    print("\n✅ Preprocessing complete!")
    print("📁 Training data saved in /data/ directory")


if __name__ == "__main__":
    main()
