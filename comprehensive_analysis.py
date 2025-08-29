import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class ComprehensiveAnalyzer:
    def __init__(self, file_path: str):
        """Initialize the analyzer with CSV file path"""
        self.file_path = file_path
        self.df = None
        self.cohorts = None
        self.overall_insights = {}
        self.feature_insights = {}
        
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean the CSV data"""
        print("📊 Loading and cleaning data...")
        
        # Load the CSV
        self.df = pd.read_csv(self.file_path)
        
        # Get cohort names (columns 2-14)
        self.cohorts = self.df.columns[1:].tolist()
        print(f"👥 Found {len(self.cohorts)} cohorts: {self.cohorts}")
        
        # Clean the data
        self._clean_data()
        
        return self.df
    
    def _clean_data(self):
        """Clean and standardize the data"""
        # Remove completely empty rows
        self.df = self.df.dropna(how='all')
        
        # Replace dashes with NaN
        self.df = self.df.replace('-', np.nan)
        
        # Clean percentage values and time values
        for col in self.cohorts:
            if col in self.df.columns:
                # Convert to string first
                self.df[col] = self.df[col].astype(str)
                # Remove percentage signs, time units, and clean up
                self.df[col] = self.df[col].str.replace('%', '').str.replace('secs', '').str.replace('mins', '').str.replace('sec', '')
                # Convert to numeric, errors='coerce' will convert invalid values to NaN
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def analyze_overall_metrics(self):
        """Analyze overall engagement metrics"""
        print("📈 Analyzing overall metrics...")
        
        # Extract overall section (rows 2-13)
        overall_data = self.df.iloc[2:13].copy()
        overall_data = overall_data.reset_index(drop=True)
        overall_data.iloc[:, 0] = overall_data.iloc[:, 0].str.strip()
        
        insights = {}
        
        # Total Users
        total_users_row = overall_data[overall_data.iloc[:, 0] == 'Total Users']
        if not total_users_row.empty:
            user_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(total_users_row.columns) - 1:
                    value = total_users_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        user_data[cohort] = float(value)
            insights['total_users'] = user_data
        
        # DAU and DTU
        dau_row = overall_data[overall_data.iloc[:, 0].str.contains('DAU', na=False)]
        dtu_row = overall_data[overall_data.iloc[:, 0].str.contains('DTU', na=False)]
        
        if not dau_row.empty:
            dau_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(dau_row.columns) - 1:
                    value = dau_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        dau_data[cohort] = float(value)
            insights['dau'] = dau_data
        
        if not dtu_row.empty:
            dtu_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(dtu_row.columns) - 1:
                    value = dtu_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        dtu_data[cohort] = float(value)
            insights['dtu'] = dtu_data
        
        # Session metrics
        session_time_row = overall_data[overall_data.iloc[:, 0].str.contains('Time Spent per session', na=False)]
        if not session_time_row.empty:
            session_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(session_time_row.columns) - 1:
                    value = session_time_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        session_data[cohort] = float(value)
            insights['session_time'] = session_data
        
        # Scanning behavior
        scanning_row = overall_data[overall_data.iloc[:, 0].str.contains('scanning', na=False)]
        if not scanning_row.empty:
            scanning_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(scanning_row.columns) - 1:
                    value = scanning_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        scanning_data[cohort] = float(value)
            insights['scanning'] = scanning_data
        
        self.overall_insights = insights
        return insights
    
    def analyze_feature_metrics(self):
        """Analyze feature-specific metrics"""
        print("🔍 Analyzing feature metrics...")
        
        features = {}
        
        # Spotlight analysis
        spotlight_data = self.df.iloc[16:28].copy()
        spotlight_data = spotlight_data.reset_index(drop=True)
        spotlight_data.iloc[:, 0] = spotlight_data.iloc[:, 0].str.strip()
        
        spotlight_insights = {}
        
        # Search efficiency
        search_row = spotlight_data[spotlight_data.iloc[:, 0].str.contains('search efficiency', na=False, case=False)]
        if not search_row.empty:
            search_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(search_row.columns) - 1:
                    value = search_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        search_data[cohort] = float(value)
            spotlight_insights['search_efficiency'] = search_data
        
        # Quick actions usage
        quick_actions_row = spotlight_data[spotlight_data.iloc[:, 0].str.contains('quick actions usage', na=False, case=False)]
        if not quick_actions_row.empty:
            quick_actions_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(quick_actions_row.columns) - 1:
                    value = quick_actions_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        quick_actions_data[cohort] = float(value)
            spotlight_insights['quick_actions'] = quick_actions_data
        
        features['spotlight'] = spotlight_insights
        
        # DM analysis
        dm_data = self.df.iloc[36:55].copy()
        dm_data = dm_data.reset_index(drop=True)
        dm_data.iloc[:, 0] = dm_data.iloc[:, 0].str.strip()
        
        dm_insights = {}
        
        # Re-engagement
        reengagement_row = dm_data[dm_data.iloc[:, 0].str.contains('repeat opens', na=False, case=False)]
        if not reengagement_row.empty:
            reengagement_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(reengagement_row.columns) - 1:
                    value = reengagement_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        reengagement_data[cohort] = float(value)
            dm_insights['reengagement'] = reengagement_data
        
        # Message sending
        message_row = dm_data[dm_data.iloc[:, 0].str.contains('Messages sent per session', na=False, case=False)]
        if not message_row.empty:
            message_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(message_row.columns) - 1:
                    value = message_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        message_data[cohort] = float(value)
            dm_insights['message_sending'] = message_data
        
        features['dm'] = dm_insights
        
        self.feature_insights = features
        return features
    
    def identify_key_patterns(self):
        """Identify key patterns and insights"""
        print("🔍 Identifying key patterns...")
        
        patterns = {}
        
        # 1. Age-based patterns
        if 'dau' in self.overall_insights and '18+' in self.overall_insights['dau'] and '18-' in self.overall_insights['dau']:
            dau_18_plus = self.overall_insights['dau']['18+']
            dau_18_minus = self.overall_insights['dau']['18-']
            patterns['age_engagement_gap'] = {
                'description': f'18- users show {dau_18_minus/dau_18_plus:.1f}x higher DAU than 18+ users',
                'value': dau_18_minus/dau_18_plus
            }
        
        # 2. Platform patterns
        if 'scanning' in self.overall_insights and 'IOS ' in self.overall_insights['scanning'] and 'Android' in self.overall_insights['scanning']:
            ios_scanning = self.overall_insights['scanning']['IOS ']
            android_scanning = self.overall_insights['scanning']['Android']
            patterns['platform_scanning_gap'] = {
                'description': f'iOS users show {ios_scanning/android_scanning:.1f}x higher scanning rate than Android',
                'value': ios_scanning/android_scanning
            }
        
        # 3. Feature adoption patterns
        if 'spotlight' in self.feature_insights and 'quick_actions' in self.feature_insights['spotlight']:
            quick_actions = self.feature_insights['spotlight']['quick_actions']
            if quick_actions:
                max_adoption = max(quick_actions.values())
                min_adoption = min(quick_actions.values())
                patterns['feature_adoption_variance'] = {
                    'description': f'Quick actions adoption varies by {max_adoption/min_adoption:.1f}x across cohorts',
                    'value': max_adoption/min_adoption
                }
        
        # 4. Engagement conversion patterns
        if 'dau' in self.overall_insights and 'dtu' in self.overall_insights:
            conversion_rates = {}
            for cohort in self.overall_insights['dau'].keys():
                if cohort in self.overall_insights['dtu']:
                    dau = self.overall_insights['dau'][cohort]
                    dtu = self.overall_insights['dtu'][cohort]
                    conversion_rate = (dtu / dau * 100) if dau > 0 else 0
                    conversion_rates[cohort] = conversion_rate
            
            if conversion_rates:
                max_conversion = max(conversion_rates.values())
                min_conversion = min(conversion_rates.values())
                patterns['conversion_variance'] = {
                    'description': f'DAU to DTU conversion varies by {max_conversion/min_conversion:.1f}x across cohorts',
                    'value': max_conversion/min_conversion
                }
        
        return patterns
    
    def generate_business_recommendations(self):
        """Generate business recommendations based on analysis"""
        print("💡 Generating business recommendations...")
        
        recommendations = []
        
        # 1. Age-based recommendations
        if 'dau' in self.overall_insights and '18+' in self.overall_insights['dau'] and '18-' in self.overall_insights['dau']:
            dau_18_plus = self.overall_insights['dau']['18+']
            dau_18_minus = self.overall_insights['dau']['18-']
            
            if dau_18_minus > dau_18_plus * 2:
                recommendations.append({
                    'category': 'User Engagement',
                    'priority': 'High',
                    'recommendation': 'Focus on improving engagement for 18+ users through targeted features and content',
                    'rationale': f'18- users show {dau_18_minus/dau_18_plus:.1f}x higher engagement than 18+ users'
                })
        
        # 2. Platform optimization
        if 'scanning' in self.overall_insights and 'IOS ' in self.overall_insights['scanning'] and 'Android' in self.overall_insights['scanning']:
            ios_scanning = self.overall_insights['scanning']['IOS ']
            android_scanning = self.overall_insights['scanning']['Android']
            
            if ios_scanning > android_scanning * 5:
                recommendations.append({
                    'category': 'Platform Optimization',
                    'priority': 'Medium',
                    'recommendation': 'Investigate and optimize Android scanning experience',
                    'rationale': f'iOS users show {ios_scanning/android_scanning:.1f}x higher scanning adoption'
                })
        
        # 3. Feature optimization
        if 'spotlight' in self.feature_insights and 'search_efficiency' in self.feature_insights['spotlight']:
            search_efficiency = self.feature_insights['spotlight']['search_efficiency']
            if search_efficiency:
                max_time = max(search_efficiency.values())
                min_time = min(search_efficiency.values())
                
                if max_time > min_time * 2:
                    slowest_cohort = max(search_efficiency, key=search_efficiency.get)
                    recommendations.append({
                        'category': 'Feature Optimization',
                        'priority': 'Medium',
                        'recommendation': f'Optimize search efficiency for {slowest_cohort} cohort',
                        'rationale': f'{slowest_cohort} takes {max_time/min_time:.1f}x longer to complete searches'
                    })
        
        # 4. Engagement optimization
        if 'dm' in self.feature_insights and 'reengagement' in self.feature_insights['dm']:
            reengagement = self.feature_insights['dm']['reengagement']
            if reengagement:
                min_reengagement = min(reengagement.values())
                max_reengagement = max(reengagement.values())
                
                if max_reengagement > min_reengagement * 1.5:
                    lowest_cohort = min(reengagement, key=reengagement.get)
                    recommendations.append({
                        'category': 'User Retention',
                        'priority': 'High',
                        'recommendation': f'Improve DM re-engagement for {lowest_cohort} cohort',
                        'rationale': f'{lowest_cohort} shows {min_reengagement:.1f}% re-engagement vs {max_reengagement:.1f}% for best cohort'
                    })
        
        return recommendations
    
    def create_comprehensive_visualization(self):
        """Create comprehensive visualization dashboard"""
        print("📊 Creating comprehensive visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Cohort Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. User Distribution
        if 'total_users' in self.overall_insights:
            ax = axes[0, 0]
            user_data = self.overall_insights['total_users']
            cohorts = list(user_data.keys())
            users = list(user_data.values())
            
            bars = ax.bar(cohorts, users, color='skyblue', alpha=0.7)
            ax.set_title('Total Users by Cohort', fontweight='bold')
            ax.set_xlabel('Cohorts')
            ax.set_ylabel('Number of Users')
            ax.tick_params(axis='x', rotation=45)
            plt.setp(ax.get_xticklabels(), ha='right')
            
            for bar, user_count in zip(bars, users):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(users)*0.01,
                       f'{user_count:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # 2. DAU vs DTU
        if 'dau' in self.overall_insights and 'dtu' in self.overall_insights:
            ax = axes[0, 1]
            dau_data = self.overall_insights['dau']
            dtu_data = self.overall_insights['dtu']
            
            common_cohorts = set(dau_data.keys()) & set(dtu_data.keys())
            if common_cohorts:
                cohorts = list(common_cohorts)
                dau_values = [dau_data[c] for c in cohorts]
                dtu_values = [dtu_data[c] for c in cohorts]
                
                x = np.arange(len(cohorts))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, dau_values, width, label='DAU %', color='lightcoral', alpha=0.7)
                bars2 = ax.bar(x + width/2, dtu_values, width, label='DTU %', color='lightgreen', alpha=0.7)
                
                ax.set_title('DAU vs DTU by Cohort', fontweight='bold')
                ax.set_xlabel('Cohorts')
                ax.set_ylabel('Percentage (%)')
                ax.set_xticks(x)
                ax.set_xticklabels(cohorts, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 3. Scanning Behavior
        if 'scanning' in self.overall_insights:
            ax = axes[0, 2]
            scanning_data = self.overall_insights['scanning']
            cohorts = list(scanning_data.keys())
            values = list(scanning_data.values())
            
            bars = ax.bar(cohorts, values, color='gold', alpha=0.7)
            ax.set_title('User Scanning Rate', fontweight='bold')
            ax.set_xlabel('Cohorts')
            ax.set_ylabel('Scanning Rate (%)')
            ax.tick_params(axis='x', rotation=45)
            plt.setp(ax.get_xticklabels(), ha='right')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 4. Search Efficiency
        if 'spotlight' in self.feature_insights and 'search_efficiency' in self.feature_insights['spotlight']:
            ax = axes[1, 0]
            search_data = self.feature_insights['spotlight']['search_efficiency']
            cohorts = list(search_data.keys())
            values = list(search_data.values())
            
            bars = ax.bar(cohorts, values, color='lightblue', alpha=0.7)
            ax.set_title('Search Efficiency (seconds)', fontweight='bold')
            ax.set_xlabel('Cohorts')
            ax.set_ylabel('Time (seconds)')
            ax.tick_params(axis='x', rotation=45)
            plt.setp(ax.get_xticklabels(), ha='right')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       f'{value:.1f}s', ha='center', va='bottom', fontsize=9)
        
        # 5. DM Re-engagement
        if 'dm' in self.feature_insights and 'reengagement' in self.feature_insights['dm']:
            ax = axes[1, 1]
            reengagement_data = self.feature_insights['dm']['reengagement']
            cohorts = list(reengagement_data.keys())
            values = list(reengagement_data.values())
            
            bars = ax.bar(cohorts, values, color='lightcoral', alpha=0.7)
            ax.set_title('DM Re-engagement Rate', fontweight='bold')
            ax.set_xlabel('Cohorts')
            ax.set_ylabel('Re-engagement Rate (%)')
            ax.tick_params(axis='x', rotation=45)
            plt.setp(ax.get_xticklabels(), ha='right')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 6. Quick Actions Usage
        if 'spotlight' in self.feature_insights and 'quick_actions' in self.feature_insights['spotlight']:
            ax = axes[1, 2]
            quick_actions_data = self.feature_insights['spotlight']['quick_actions']
            cohorts = list(quick_actions_data.keys())
            values = list(quick_actions_data.values())
            
            bars = ax.bar(cohorts, values, color='lightgreen', alpha=0.7)
            ax.set_title('Quick Actions Usage', fontweight='bold')
            ax.set_xlabel('Cohorts')
            ax.set_ylabel('Usage Rate (%)')
            ax.tick_params(axis='x', rotation=45)
            plt.setp(ax.get_xticklabels(), ha='right')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Comprehensive visualization saved as 'comprehensive_analysis.png'")
    
    def print_comprehensive_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*80)
        
        # Overall insights
        print("\n🎯 KEY INSIGHTS:")
        patterns = self.identify_key_patterns()
        for pattern_name, pattern_data in patterns.items():
            print(f"   • {pattern_data['description']}")
        
        # Business recommendations
        print("\n💡 BUSINESS RECOMMENDATIONS:")
        recommendations = self.generate_business_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"\n   {i}. [{rec['priority']}] {rec['category']}:")
            print(f"      {rec['recommendation']}")
            print(f"      Rationale: {rec['rationale']}")
        
        # Cohort performance summary
        print("\n👥 COHORT PERFORMANCE SUMMARY:")
        if 'dau' in self.overall_insights:
            dau_data = self.overall_insights['dau']
            print("   📈 DAU Performance (Top 5):")
            sorted_dau = sorted(dau_data.items(), key=lambda x: x[1], reverse=True)[:5]
            for cohort, dau in sorted_dau:
                print(f"      - {cohort}: {dau:.1f}%")
        
        if 'scanning' in self.overall_insights:
            scanning_data = self.overall_insights['scanning']
            print("   🔍 Scanning Performance (Top 5):")
            sorted_scanning = sorted(scanning_data.items(), key=lambda x: x[1], reverse=True)[:5]
            for cohort, scanning in sorted_scanning:
                print(f"      - {cohort}: {scanning:.1f}%")
        
        print("\n✅ Analysis completed! Check 'comprehensive_analysis.png' for visualizations")

def main():
    """Main function to run comprehensive analysis"""
    print("🚀 Starting Comprehensive Analysis...")
    
    # Initialize analyzer
    analyzer = ComprehensiveAnalyzer('Cohort Wise Analysis Fam 2.0 - Sheet1.csv')
    
    # Load and clean data
    df = analyzer.load_and_clean_data()
    
    # Analyze overall metrics
    overall_insights = analyzer.analyze_overall_metrics()
    
    # Analyze feature metrics
    feature_insights = analyzer.analyze_feature_metrics()
    
    # Create comprehensive visualization
    analyzer.create_comprehensive_visualization()
    
    # Print comprehensive summary
    analyzer.print_comprehensive_summary()
    
    print("\n✅ Comprehensive analysis completed!")

if __name__ == "__main__":
    main()
