"""
Synthetic Data Generator for ImpactSense
Generates realistic datasets for testing and development
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random
import uuid

# Set random seed for reproducible data
np.random.seed(42)
random.seed(42)

class ImpactSenseDataGenerator:
    """Generate synthetic datasets for all ImpactSense components"""
    
    def __init__(self):
        self.regions = [
            "NCR", "CAR", "Region I", "Region II", "Region III", "Region IV-A", 
            "Region IV-B", "Region V", "Region VI", "Region VII", "Region VIII",
            "Region IX", "Region X", "Region XI", "Region XII", "CARAGA", "BARMM"
        ]
        
        self.transaction_categories = [
            'groceries', 'food', 'utilities', 'transportation', 'healthcare', 
            'education', 'entertainment', 'shopping', 'bills', 'fuel',
            'restaurants', 'pharmacy', 'clothing', 'electronics', 'services'
        ]
        
        self.cause_categories = [
            'Health', 'Education', 'Poverty Alleviation', 'Environment', 
            'Disaster Relief', 'Women\'s Health', 'Nutrition', 'Street Children',
            'Animal Welfare', 'Human Rights', 'Marine Conservation', 'Housing'
        ]
        
        self.age_ranges = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        self.income_levels = ['low', 'lower_middle', 'middle', 'upper_middle', 'high']
    
    def generate_user_profiles(self, num_users: int = 10000) -> pd.DataFrame:
        """Generate synthetic user profiles with behavioral patterns"""
        
        users = []
        for i in range(num_users):
            # Generate basic profile
            user_id = f"user_{str(uuid.uuid4())[:8]}"
            age_range = np.random.choice(self.age_ranges, p=[0.15, 0.35, 0.25, 0.15, 0.08, 0.02])
            income_level = np.random.choice(self.income_levels, p=[0.15, 0.25, 0.35, 0.20, 0.05])
            region = np.random.choice(self.regions, p=[0.35, 0.05, 0.08, 0.06, 0.12, 0.10, 0.03, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01])
            
            # Wallet balance based on income level
            wallet_multipliers = {'low': 500, 'lower_middle': 1500, 'middle': 3000, 'upper_middle': 8000, 'high': 20000}
            base_wallet = wallet_multipliers[income_level]
            wallet_balance = max(50, np.random.normal(base_wallet, base_wallet * 0.3))
            
            # Donation behavior patterns
            donation_probability = self._calculate_donation_probability(age_range, income_level)
            is_donor = np.random.random() < donation_probability
            
            if is_donor:
                # Established donors
                total_donations = max(0, np.random.exponential(100))
                num_donations = max(1, int(np.random.poisson(8)))
                avg_donation = total_donations / num_donations if num_donations > 0 else 0
                last_donation_days = np.random.randint(1, 90)
                
                # Preferred causes (donors have 1-3 preferences)
                num_preferences = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                preferred_causes = list(np.random.choice(self.cause_categories, num_preferences, replace=False))
            else:
                # Non-donors or new users
                total_donations = 0
                avg_donation = 0
                last_donation_days = 999  # Never donated
                preferred_causes = []
            
            # Prompt history
            last_prompt_days = np.random.randint(0, 30) if np.random.random() < 0.7 else 999
            
            # Notification preferences
            notification_prefs = {
                'impact_updates': np.random.random() < 0.8,
                'social_sharing': np.random.random() < 0.3,
                'monthly_summary': np.random.random() < 0.6,
                'milestone_alerts': np.random.random() < 0.7
            }
            
            user = {
                'user_id': user_id,
                'age_range': age_range,
                'income_level': income_level,
                'region': region,
                'wallet_balance': round(wallet_balance, 2),
                'total_lifetime_donations': round(total_donations, 2),
                'average_donation_amount': round(avg_donation, 2),
                'days_since_last_donation': last_donation_days,
                'days_since_last_prompt': last_prompt_days,
                'preferred_causes': json.dumps(preferred_causes),
                'notification_preferences': json.dumps(notification_prefs),
                'is_active_donor': is_donor,
                'donor_segment': self._classify_donor_segment(total_donations, avg_donation),
                'created_at': datetime.now() - timedelta(days=np.random.randint(30, 365))
            }
            users.append(user)
        
        return pd.DataFrame(users)
    
    def generate_transaction_history(self, users_df: pd.DataFrame, num_transactions: int = 50000) -> pd.DataFrame:
        """Generate synthetic transaction history for users"""
        
        transactions = []
        
        for _, user in users_df.iterrows():
            # Number of transactions per user (based on income level)
            income_multipliers = {'low': 15, 'lower_middle': 25, 'middle': 40, 'upper_middle': 60, 'high': 100}
            num_user_transactions = np.random.poisson(income_multipliers[user['income_level']])
            
            for _ in range(num_user_transactions):
                # Transaction amount based on category and user income
                category = np.random.choice(self.transaction_categories)
                amount = self._generate_transaction_amount(category, user['income_level'])
                
                # Transaction timing (more recent transactions weighted higher)
                days_ago = int(np.random.exponential(30))  # Exponential distribution favoring recent
                transaction_date = datetime.now() - timedelta(days=days_ago)
                
                # Geographic context
                lat, lng = self._generate_coordinates(user['region'])
                
                transaction = {
                    'transaction_id': str(uuid.uuid4()),
                    'user_id': user['user_id'],
                    'amount': round(amount, 2),
                    'category': category,
                    'merchant_name': self._generate_merchant_name(category),
                    'transaction_date': transaction_date,
                    'region': user['region'],
                    'latitude': lat,
                    'longitude': lng,
                    'payment_method': np.random.choice(['vibe_wallet', 'linked_card'], p=[0.8, 0.2]),
                    'is_recurring': np.random.random() < 0.15  # 15% recurring transactions
                }
                transactions.append(transaction)
        
        return pd.DataFrame(transactions[:num_transactions])  # Limit to requested number
    
    def generate_donation_history(self, users_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic donation history linked to transactions"""
        
        donations = []
        
        # Filter to only active donors
        active_donors = users_df[users_df['is_active_donor'] == True]
        
        for _, donor in active_donors.iterrows():
            # Get user's transactions
            user_transactions = transactions_df[transactions_df['user_id'] == donor['user_id']]
            
            # Determine number of donations (some percentage of transactions)
            donation_rate = self._calculate_user_donation_rate(donor['donor_segment'])
            num_donations = int(len(user_transactions) * donation_rate)
            
            if num_donations > 0:
                # Select random transactions that resulted in donations
                donation_transactions = user_transactions.sample(min(num_donations, len(user_transactions)))
                
                for _, transaction in donation_transactions.iterrows():
                    # Select NGO based on user preferences and transaction context
                    ngo_id = self._select_ngo_for_donation(donor, transaction)
                    
                    # Calculate donation amount
                    donation_amount = self._calculate_donation_amount(
                        transaction['amount'], donor['average_donation_amount']
                    )
                    
                    donation = {
                        'donation_id': str(uuid.uuid4()),
                        'user_id': donor['user_id'],
                        'transaction_id': transaction['transaction_id'],
                        'ngo_id': ngo_id,
                        'amount': round(donation_amount, 2),
                        'donation_date': transaction['transaction_date'],
                        'cause_category': self._get_ngo_category(ngo_id),
                        'region': transaction['region'],
                        'impact_proof_received': np.random.random() < 0.85,  # 85% receive proof
                        'proof_received_date': transaction['transaction_date'] + timedelta(
                            days=np.random.randint(1, 14)
                        ) if np.random.random() < 0.85 else None,
                        'user_satisfaction_score': np.random.choice([3, 4, 5], p=[0.1, 0.3, 0.6]),
                        'shared_impact': np.random.random() < 0.2  # 20% share their impact
                    }
                    donations.append(donation)
        
        return pd.DataFrame(donations)
    
    def generate_ngo_performance_data(self) -> pd.DataFrame:
        """Generate synthetic NGO performance and reliability data"""
        
        from constants.ngos import NGOS
        
        performance_data = []
        
        for ngo in NGOS:
            # Base performance varies by NGO type and size
            base_reliability = np.random.uniform(0.7, 0.95)
            
            # Simulate 6 months of performance data
            for month in range(6):
                month_date = datetime.now() - timedelta(days=30 * month)
                
                # Monthly metrics with some variance
                performance = {
                    'ngo_id': ngo['ngo_id'],
                    'ngo_name': ngo['ngo_name'],
                    'month': month_date.strftime('%Y-%m'),
                    'donations_received': np.random.poisson(20),
                    'total_amount_received': round(np.random.exponential(5000), 2),
                    'proof_submissions': np.random.poisson(18),
                    'avg_proof_delay_days': max(1, np.random.normal(5, 2)),
                    'proof_quality_score': np.random.uniform(0.8, 1.0),
                    'user_satisfaction_avg': np.random.uniform(3.5, 5.0),
                    'response_time_hours': max(1, np.random.exponential(12)),
                    'communication_quality': np.random.uniform(0.7, 1.0),
                    'impact_stories_submitted': np.random.poisson(3),
                    'reliability_score': min(1.0, max(0.0, base_reliability + np.random.normal(0, 0.1))),
                    'active_projects': np.random.poisson(5),
                    'geographic_coverage_score': np.random.uniform(0.6, 1.0)
                }
                performance_data.append(performance)
        
        return pd.DataFrame(performance_data)
    
    def generate_impact_proof_data(self, donations_df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic impact proof submissions from NGOs"""
        
        proofs = []
        
        # Generate proofs for donations that received them
        proven_donations = donations_df[donations_df['impact_proof_received'] == True]
        
        for _, donation in proven_donations.iterrows():
            proof_types = ['photo', 'video', 'milestone', 'completion', 'beneficiary_story']
            proof_type = np.random.choice(proof_types, p=[0.4, 0.2, 0.2, 0.1, 0.1])
            
            # Generate realistic impact metrics based on amount and cause
            impact_metrics = self._generate_impact_metrics(
                donation['amount'], donation['cause_category']
            )
            
            proof = {
                'proof_id': str(uuid.uuid4()),
                'donation_id': donation['donation_id'],
                'ngo_id': donation['ngo_id'],
                'proof_type': proof_type,
                'submission_date': donation['proof_received_date'],
                'validation_status': np.random.choice(['pending', 'approved', 'rejected'], p=[0.1, 0.85, 0.05]),
                'gps_latitude': np.random.uniform(4.0, 21.0),  # Philippines bounds
                'gps_longitude': np.random.uniform(116.0, 127.0),
                'impact_metrics': json.dumps(impact_metrics),
                'media_url': f"https://impactsense-media.s3.amazonaws.com/{proof['proof_id']}.jpg",
                'description': self._generate_impact_description(donation['cause_category'], impact_metrics),
                'beneficiaries_count': np.random.randint(1, 50),
                'validation_score': np.random.uniform(0.7, 1.0),
                'user_engagement_score': np.random.uniform(0.5, 1.0)  # How much users liked this proof
            }
            proofs.append(proof)
        
        return pd.DataFrame(proofs)
    
    def generate_user_engagement_metrics(self, users_df: pd.DataFrame, donations_df: pd.DataFrame) -> pd.DataFrame:
        """Generate user engagement and behavior metrics"""
        
        engagement_data = []
        
        for _, user in users_df.iterrows():
            user_donations = donations_df[donations_df['user_id'] == user['user_id']]
            
            # Calculate engagement metrics
            metrics = {
                'user_id': user['user_id'],
                'total_sessions': np.random.poisson(50),
                'avg_session_duration_minutes': np.random.exponential(8),
                'donation_prompts_shown': np.random.poisson(15),
                'donation_prompts_accepted': len(user_donations),
                'donation_opt_in_rate': len(user_donations) / max(1, np.random.poisson(15)),
                'impact_updates_opened': np.random.poisson(8),
                'impact_stories_shared': user_donations['shared_impact'].sum() if len(user_donations) > 0 else 0,
                'avg_time_between_donations_days': self._calculate_avg_donation_interval(user_donations),
                'preferred_donation_amount': user['average_donation_amount'],
                'streak_months': self._calculate_donation_streak(user_donations),
                'retention_score': np.random.uniform(0.3, 1.0),
                'lifetime_value_score': user['total_lifetime_donations'] / 100,  # Normalized LTV
                'churn_risk_score': np.random.uniform(0.0, 0.8),
                'last_activity_date': datetime.now() - timedelta(days=np.random.randint(0, 30))
            }
            engagement_data.append(metrics)
        
        return pd.DataFrame(engagement_data)
    
    def generate_ml_training_data(self, users_df: pd.DataFrame, transactions_df: pd.DataFrame, 
                                donations_df: pd.DataFrame) -> pd.DataFrame:
        """Generate training data for ML models"""
        
        training_data = []
        
        # For each transaction, determine if it resulted in a donation
        for _, transaction in transactions_df.iterrows():
            user = users_df[users_df['user_id'] == transaction['user_id']].iloc[0]
            
            # Check if this transaction resulted in a donation
            donation_made = len(donations_df[
                donations_df['transaction_id'] == transaction['transaction_id']
            ]) > 0
            
            # Feature engineering
            features = {
                'user_id': transaction['user_id'],
                'transaction_id': transaction['transaction_id'],
                
                # Transaction features
                'transaction_amount': transaction['amount'],
                'transaction_category_encoded': self._encode_category(transaction['category']),
                'day_of_week': transaction['transaction_date'].weekday(),
                'hour_of_day': transaction['transaction_date'].hour,
                'is_weekend': transaction['transaction_date'].weekday() >= 5,
                
                # User features
                'wallet_balance': user['wallet_balance'],
                'user_age_range_encoded': self._encode_age_range(user['age_range']),
                'user_income_level_encoded': self._encode_income_level(user['income_level']),
                'days_since_last_donation': user['days_since_last_donation'],
                'days_since_last_prompt': user['days_since_last_prompt'],
                'total_lifetime_donations': user['total_lifetime_donations'],
                'average_donation_amount': user['average_donation_amount'],
                'is_established_donor': user['total_lifetime_donations'] > 0,
                
                # Derived features
                'transaction_to_wallet_ratio': transaction['amount'] / max(1, user['wallet_balance']),
                'transaction_amount_log': np.log1p(transaction['amount']),
                'donation_propensity_score': self._calculate_propensity_score(user, transaction),
                
                # Target variable
                'donated': donation_made
            }
            training_data.append(features)
        
        return pd.DataFrame(training_data)
    
    # Helper methods
    def _calculate_donation_probability(self, age_range: str, income_level: str) -> float:
        """Calculate probability of being a donor based on demographics"""
        age_probs = {'18-24': 0.2, '25-34': 0.4, '35-44': 0.5, '45-54': 0.6, '55-64': 0.7, '65+': 0.8}
        income_probs = {'low': 0.2, 'lower_middle': 0.3, 'middle': 0.5, 'upper_middle': 0.7, 'high': 0.8}
        return (age_probs[age_range] + income_probs[income_level]) / 2
    
    def _classify_donor_segment(self, total_donations: float, avg_donation: float) -> str:
        """Classify donor into segments"""
        if total_donations == 0:
            return 'non_donor'
        elif total_donations < 50:
            return 'new_donor'
        elif avg_donation < 5:
            return 'micro_donor'
        elif avg_donation < 20:
            return 'regular_donor'
        else:
            return 'major_donor'
    
    def _generate_transaction_amount(self, category: str, income_level: str) -> float:
        """Generate realistic transaction amounts"""
        # Base amounts by category
        category_bases = {
            'groceries': 500, 'food': 200, 'utilities': 800, 'transportation': 150,
            'healthcare': 1000, 'education': 2000, 'entertainment': 300, 'shopping': 800,
            'bills': 600, 'fuel': 400, 'restaurants': 350, 'pharmacy': 250,
            'clothing': 600, 'electronics': 3000, 'services': 500
        }
        
        # Income level multipliers
        income_multipliers = {'low': 0.5, 'lower_middle': 0.8, 'middle': 1.0, 'upper_middle': 1.5, 'high': 2.5}
        
        base_amount = category_bases.get(category, 300)
        multiplier = income_multipliers[income_level]
        
        # Add some randomness
        amount = base_amount * multiplier * np.random.lognormal(0, 0.3)
        return max(10, amount)  # Minimum transaction
    
    def _generate_coordinates(self, region: str) -> tuple:
        """Generate GPS coordinates within region bounds"""
        # Simplified coordinate ranges for major regions
        region_bounds = {
            'NCR': (14.5, 14.7, 120.9, 121.1),
            'Region III': (14.8, 15.8, 120.2, 121.2),
            'Region IV-A': (13.5, 14.8, 120.5, 122.0),
            # Add more regions as needed
        }
        
        if region in region_bounds:
            lat_min, lat_max, lng_min, lng_max = region_bounds[region]
        else:
            # Default Philippines bounds
            lat_min, lat_max, lng_min, lng_max = 4.0, 21.0, 116.0, 127.0
        
        lat = np.random.uniform(lat_min, lat_max)
        lng = np.random.uniform(lng_min, lng_max)
        return round(lat, 6), round(lng, 6)
    
    def _generate_merchant_name(self, category: str) -> str:
        """Generate realistic merchant names"""
        merchant_names = {
            'groceries': ['SM Supermarket', 'Robinsons Supermarket', 'Puregold', 'Mercury Drug', 'Shopwise'],
            'food': ['Jollibee', 'McDonald\'s', 'KFC', 'Chowking', 'Max\'s Restaurant'],
            'utilities': ['Meralco', 'Manila Water', 'PLDT', 'Globe Telecom', 'Smart Communications'],
            'transportation': ['Grab', 'Uber', 'LTFRB', 'MRT', 'LRT'],
            'healthcare': ['St. Luke\'s Hospital', 'Philippine General Hospital', 'Makati Medical Center'],
            'fuel': ['Petron', 'Shell', 'Caltex', 'Total']
        }
        
        names = merchant_names.get(category, ['Generic Store', 'Local Business', 'Online Store'])
        return np.random.choice(names)
    
    def _calculate_user_donation_rate(self, donor_segment: str) -> float:
        """Calculate what percentage of transactions result in donations"""
        rates = {
            'non_donor': 0.0,
            'new_donor': 0.05,
            'micro_donor': 0.15,
            'regular_donor': 0.25,
            'major_donor': 0.35
        }
        return rates.get(donor_segment, 0.1)
    
    def _select_ngo_for_donation(self, donor: pd.Series, transaction: pd.Series) -> str:
        """Select appropriate NGO based on donor preferences and context"""
        from constants.ngos import NGOS
        
        # Filter NGOs by region and preferences
        preferred_causes = json.loads(donor['preferred_causes']) if donor['preferred_causes'] else []
        
        relevant_ngos = [
            ngo for ngo in NGOS 
            if (ngo['region_focus'] == donor['region'] or ngo['region_focus'] == 'Nationwide')
            and (not preferred_causes or ngo['category'] in preferred_causes)
        ]
        
        if not relevant_ngos:
            relevant_ngos = [ngo for ngo in NGOS if ngo['region_focus'] == 'Nationwide']
        
        return np.random.choice([ngo['ngo_id'] for ngo in relevant_ngos])
    
    def _get_ngo_category(self, ngo_id: str) -> str:
        """Get category for NGO ID"""
        from constants.ngos import NGOS
        ngo = next((ngo for ngo in NGOS if ngo['ngo_id'] == ngo_id), None)
        return ngo['category'] if ngo else 'Unknown'
    
    def _calculate_donation_amount(self, transaction_amount: float, user_avg: float) -> float:
        """Calculate realistic donation amount"""
        if user_avg > 0:
            # Existing donors: use their pattern with some variance
            amount = max(1, np.random.normal(user_avg, user_avg * 0.3))
        else:
            # New donors: conservative amounts
            if transaction_amount >= 1000:
                amount = np.random.choice([1, 2, 5, 10], p=[0.4, 0.3, 0.2, 0.1])
            else:
                amount = np.random.choice([1, 2, 5], p=[0.6, 0.3, 0.1])
        
        return round(amount, 2)
    
    def _generate_impact_metrics(self, donation_amount: float, cause_category: str) -> Dict:
        """Generate realistic impact metrics based on donation and cause"""
        metrics = {}
        
        impact_rates = {
            'Health': {'consultations': donation_amount * 2, 'medicines': donation_amount / 5},
            'Education': {'supplies': donation_amount / 3, 'books': donation_amount / 10},
            'Nutrition': {'meals': donation_amount * 3, 'vitamins': donation_amount / 2},
            'Environment': {'trees': donation_amount / 10, 'cleanup_kg': donation_amount / 2},
            'Poverty Alleviation': {'livelihood_hours': donation_amount / 2, 'training_hours': donation_amount / 3}
        }
        
        if cause_category in impact_rates:
            for metric, value in impact_rates[cause_category].items():
                metrics[metric] = max(1, int(value))
        else:
            metrics['general_impact'] = f"₱{donation_amount} in direct support"
        
        return metrics
    
    def _generate_impact_description(self, cause_category: str, metrics: Dict) -> str:
        """Generate human-readable impact description"""
        descriptions = {
            'Health': f"Funded {metrics.get('consultations', 0)} medical consultations",
            'Education': f"Provided {metrics.get('supplies', 0)} school supply sets",
            'Nutrition': f"Funded {metrics.get('meals', 0)} nutritious meals",
            'Environment': f"Supported planting of {metrics.get('trees', 0)} trees",
            'Poverty Alleviation': f"Provided {metrics.get('livelihood_hours', 0)} hours of livelihood training"
        }
        
        return descriptions.get(cause_category, f"Direct support for {cause_category.lower()} initiatives")
    
    def _calculate_avg_donation_interval(self, donations_df: pd.DataFrame) -> float:
        """Calculate average days between donations"""
        if len(donations_df) < 2:
            return 0
        
        dates = pd.to_datetime(donations_df['donation_date']).sort_values()
        intervals = dates.diff().dt.days.dropna()
        return intervals.mean() if len(intervals) > 0 else 0
    
    def _calculate_donation_streak(self, donations_df: pd.DataFrame) -> int:
        """Calculate current donation streak in months"""
        if len(donations_df) == 0:
            return 0
        
        # Simplified: assume regular monthly donations
        recent_months = len(donations_df[donations_df['donation_date'] >= datetime.now() - timedelta(days=180)])
        return min(6, recent_months)  # Cap at 6 months
    
    def _encode_category(self, category: str) -> int:
        """Encode transaction category as integer"""
        return hash(category) % 100
    
    def _encode_age_range(self, age_range: str) -> int:
        """Encode age range as integer"""
        age_mapping = {'18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55-64': 5, '65+': 6}
        return age_mapping.get(age_range, 0)
    
    def _encode_income_level(self, income_level: str) -> int:
        """Encode income level as integer"""
        income_mapping = {'low': 1, 'lower_middle': 2, 'middle': 3, 'upper_middle': 4, 'high': 5}
        return income_mapping.get(income_level, 0)
    
    def _calculate_propensity_score(self, user: pd.Series, transaction: pd.Series) -> float:
        """Calculate donation propensity score"""
        score = 0
        
        # Historical donor bonus
        if user['total_lifetime_donations'] > 0:
            score += 0.3
        
        # Transaction size factor
        if transaction['amount'] > 500:
            score += 0.2
        
        # Wallet balance factor
        if user['wallet_balance'] > 1000:
            score += 0.2
        
        # Category factor
        if transaction['category'] in ['groceries', 'food', 'utilities']:
            score += 0.15
        
        return min(1.0, score)


def main():
    """Generate all synthetic datasets"""
    print("🔄 Generating ImpactSense synthetic datasets...")
    
    generator = ImpactSenseDataGenerator()
    
    # Generate datasets
    print("📊 Generating user profiles...")
    users_df = generator.generate_user_profiles(10000)
    
    print("💳 Generating transaction history...")
    transactions_df = generator.generate_transaction_history(users_df, 50000)
    
    print("💝 Generating donation history...")
    donations_df = generator.generate_donation_history(users_df, transactions_df)
    
    print("🏢 Generating NGO performance data...")
    ngo_performance_df = generator.generate_ngo_performance_data()
    
    print("📸 Generating impact proof data...")
    impact_proofs_df = generator.generate_impact_proof_data(donations_df)
    
    print("📈 Generating user engagement metrics...")
    engagement_df = generator.generate_user_engagement_metrics(users_df, donations_df)
    
    print("🤖 Generating ML training data...")
    ml_training_df = generator.generate_ml_training_data(users_df, transactions_df, donations_df)
    
    # Save datasets
    print("💾 Saving datasets...")
    users_df.to_csv('data/user_profiles.csv', index=False)
    transactions_df.to_csv('data/transaction_history.csv', index=False)
    donations_df.to_csv('data/donation_history.csv', index=False)
    ngo_performance_df.to_csv('data/ngo_performance.csv', index=False)
    impact_proofs_df.to_csv('data/impact_proofs.csv', index=False)
    engagement_df.to_csv('data/user_engagement.csv', index=False)
    ml_training_df.to_csv('data/ml_training_data.csv', index=False)
    
    # Generate summary statistics
    print("\n📋 Dataset Summary:")
    print(f"Users: {len(users_df):,}")
    print(f"Transactions: {len(transactions_df):,}")
    print(f"Donations: {len(donations_df):,}")
    print(f"Active Donors: {len(users_df[users_df['is_active_donor'] == True]):,}")
    print(f"Total Donated: ₱{donations_df['amount'].sum():,.2f}")
    print(f"Average Donation: ₱{donations_df['amount'].mean():.2f}")
    print(f"Donation Rate: {len(donations_df) / len(transactions_df) * 100:.1f}%")
    
    print("\n✅ All datasets generated successfully!")
    print("📁 Files saved in /data/ directory")


if __name__ == "__main__":
    main()
