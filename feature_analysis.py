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

class FeatureAnalyzer:
    def __init__(self, file_path: str):
        """Initialize the analyzer with CSV file path"""
        self.file_path = file_path
        self.df = None
        self.cohorts = None
        
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
    
    def extract_feature_data(self, feature_name: str) -> pd.DataFrame:
        """Extract data for a specific feature"""
        feature_data = None
        
        if feature_name.lower() == 'spotlight':
            # Spotlight rows: 17-28
            feature_data = self.df.iloc[16:28].copy()
        elif feature_name.lower() == 'home':
            # Home rows: 32-33
            feature_data = self.df.iloc[31:33].copy()
        elif feature_name.lower() == 'dm':
            # DM rows: 37-55
            feature_data = self.df.iloc[36:55].copy()
        elif feature_name.lower() == 'bubble':
            # Bubble rows: 59-67
            feature_data = self.df.iloc[58:67].copy()
        
        if feature_data is not None:
            feature_data = feature_data.reset_index(drop=True)
            # Clean metric names
            feature_data.iloc[:, 0] = feature_data.iloc[:, 0].str.strip()
        
        return feature_data
    
    def analyze_spotlight_feature(self):
        """Analyze Spotlight feature metrics"""
        print("\n🔍 Analyzing Spotlight Feature...")
        
        spotlight_data = self.extract_feature_data('spotlight')
        if spotlight_data is None:
            print("❌ No Spotlight data found")
            return {}
        
        insights = {}
        
        # 1. Daily interaction metrics
        swipe_row = spotlight_data[spotlight_data.iloc[:, 0].str.contains('swipe', na=False, case=False)]
        tap_row = spotlight_data[spotlight_data.iloc[:, 0].str.contains('tap', na=False, case=False)]
        
        if not swipe_row.empty:
            swipe_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(swipe_row.columns) - 1:
                    value = swipe_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        swipe_data[cohort] = float(value)
            insights['swipe_data'] = swipe_data
        
        if not tap_row.empty:
            tap_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(tap_row.columns) - 1:
                    value = tap_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        tap_data[cohort] = float(value)
            insights['tap_data'] = tap_data
        
        # 2. Search efficiency
        search_row = spotlight_data[spotlight_data.iloc[:, 0].str.contains('search efficiency', na=False, case=False)]
        if not search_row.empty:
            search_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(search_row.columns) - 1:
                    value = search_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        search_data[cohort] = float(value)
            insights['search_efficiency'] = search_data
        
        # 3. Button adoption rates
        paste_row = spotlight_data[spotlight_data.iloc[:, 0].str.contains('paste button', na=False, case=False)]
        if not paste_row.empty:
            paste_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(paste_row.columns) - 1:
                    value = paste_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        paste_data[cohort] = float(value)
            insights['paste_adoption'] = paste_data
        
        # 4. Quick actions usage
        quick_actions_row = spotlight_data[spotlight_data.iloc[:, 0].str.contains('quick actions usage', na=False, case=False)]
        if not quick_actions_row.empty:
            quick_actions_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(quick_actions_row.columns) - 1:
                    value = quick_actions_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        quick_actions_data[cohort] = float(value)
            insights['quick_actions_usage'] = quick_actions_data
        
        return insights
    
    def analyze_dm_feature(self):
        """Analyze Direct Messaging feature metrics"""
        print("\n💬 Analyzing Direct Messaging Feature...")
        
        dm_data = self.extract_feature_data('dm')
        if dm_data is None:
            print("❌ No DM data found")
            return {}
        
        insights = {}
        
        # 1. Session engagement
        session_time_row = dm_data[dm_data.iloc[:, 0].str.contains('time spent per DM session', na=False, case=False)]
        if not session_time_row.empty:
            session_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(session_time_row.columns) - 1:
                    value = session_time_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        session_data[cohort] = float(value)
            insights['session_time'] = session_data
        
        # 2. Re-engagement
        reengagement_row = dm_data[dm_data.iloc[:, 0].str.contains('repeat opens', na=False, case=False)]
        if not reengagement_row.empty:
            reengagement_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(reengagement_row.columns) - 1:
                    value = reengagement_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        reengagement_data[cohort] = float(value)
            insights['reengagement'] = reengagement_data
        
        # 3. Pin usage
        pin_row = dm_data[dm_data.iloc[:, 0].str.contains('pins at least once', na=False, case=False)]
        if not pin_row.empty:
            pin_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(pin_row.columns) - 1:
                    value = pin_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        pin_data[cohort] = float(value)
            insights['pin_usage'] = pin_data
        
        # 4. Message sending
        message_row = dm_data[dm_data.iloc[:, 0].str.contains('Messages sent per session', na=False, case=False)]
        if not message_row.empty:
            message_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(message_row.columns) - 1:
                    value = message_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        message_data[cohort] = float(value)
            insights['message_sending'] = message_data
        
        return insights
    
    def analyze_bubble_feature(self):
        """Analyze Bubble feature metrics"""
        print("\n🫧 Analyzing Bubble Feature...")
        
        bubble_data = self.extract_feature_data('bubble')
        if bubble_data is None:
            print("❌ No Bubble data found")
            return {}
        
        insights = {}
        
        # 1. Recent ticket size
        ticket_row = bubble_data[bubble_data.iloc[:, 0].str.contains('Recent Ticket Size', na=False, case=False)]
        if not ticket_row.empty:
            ticket_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(ticket_row.columns) - 1:
                    value = ticket_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        ticket_data[cohort] = float(value)
            insights['ticket_size'] = ticket_data
        
        # 2. Recent payment conversion
        conversion_row = bubble_data[bubble_data.iloc[:, 0].str.contains('recent bubble txn', na=False, case=False)]
        if not conversion_row.empty:
            conversion_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(conversion_row.columns) - 1:
                    value = conversion_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        conversion_data[cohort] = float(value)
            insights['conversion_rate'] = conversion_data
        
        # 3. User adoption
        adoption_row = bubble_data[bubble_data.iloc[:, 0].str.contains('users using bubbles', na=False, case=False)]
        if not adoption_row.empty:
            adoption_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(adoption_row.columns) - 1:
                    value = adoption_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        adoption_data[cohort] = float(value)
            insights['user_adoption'] = adoption_data
        
        return insights
    
    def analyze_home_feature(self):
        """Analyze Home feature metrics"""
        print("\n🏠 Analyzing Home Feature...")
        
        home_data = self.extract_feature_data('home')
        if home_data is None:
            print("❌ No Home data found")
            return {}
        
        insights = {}
        
        # Scanner usage
        scanner_row = home_data[home_data.iloc[:, 0].str.contains('Scanner', na=False, case=False)]
        if not scanner_row.empty:
            scanner_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(scanner_row.columns) - 1:
                    value = scanner_row.iloc[0, i + 1]
                    if pd.notna(value) and value > 0:
                        scanner_data[cohort] = float(value)
            insights['scanner_usage'] = scanner_data
        
        return insights
    
    def create_feature_visualizations(self, feature_insights: Dict, feature_name: str):
        """Create visualizations for feature analysis"""
        print(f"📊 Creating {feature_name} visualizations...")
        
        if not feature_insights:
            print(f"❌ No data available for {feature_name}")
            return
        
        # Create subplots based on available data
        num_metrics = len(feature_insights)
        if num_metrics == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{feature_name.upper()} Feature Analysis', fontsize=16, fontweight='bold')
        
        plot_count = 0
        
        for metric_name, metric_data in feature_insights.items():
            if plot_count >= 4:  # Max 4 plots
                break
                
            row = plot_count // 2
            col = plot_count % 2
            ax = axes[row, col]
            
            if metric_data:
                cohorts = list(metric_data.keys())
                values = list(metric_data.values())
                
                bars = ax.bar(cohorts, values, color='lightblue', alpha=0.7)
                ax.set_title(f'{metric_name.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel('Cohorts')
                ax.set_ylabel('Value')
                ax.tick_params(axis='x', rotation=45)
                plt.setp(ax.get_xticklabels(), ha='right')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=9)
            
            plot_count += 1
        
        # Hide empty subplots
        for i in range(plot_count, 4):
            row = i // 2
            col = i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{feature_name.lower()}_analysis.png', dpi=300, bbox_inches='tight')
        print(f"📊 {feature_name} visualizations saved as '{feature_name.lower()}_analysis.png'")
    
    def print_feature_summary(self, feature_insights: Dict, feature_name: str):
        """Print summary for a specific feature"""
        print(f"\n{'='*60}")
        print(f"📊 {feature_name.upper()} FEATURE SUMMARY")
        print(f"{'='*60}")
        
        if not feature_insights:
            print(f"❌ No data available for {feature_name}")
            return
        
        for metric_name, metric_data in feature_insights.items():
            if metric_data:
                print(f"\n📈 {metric_name.replace('_', ' ').title()}:")
                max_cohort = max(metric_data, key=metric_data.get)
                min_cohort = min(metric_data, key=metric_data.get)
                print(f"   • Highest: {max_cohort} ({metric_data[max_cohort]:.2f})")
                print(f"   • Lowest: {min_cohort} ({metric_data[min_cohort]:.2f})")
                
                # Show all values
                for cohort, value in sorted(metric_data.items(), key=lambda x: x[1], reverse=True):
                    print(f"      - {cohort}: {value:.2f}")

def main():
    """Main function to run feature analysis"""
    print("🚀 Starting Feature-Wise Analysis...")
    
    # Initialize analyzer
    analyzer = FeatureAnalyzer('Cohort Wise Analysis Fam 2.0 - Sheet1.csv')
    
    # Load and clean data
    df = analyzer.load_and_clean_data()
    
    # Analyze each feature
    features = ['Spotlight', 'DM', 'Bubble', 'Home']
    
    for feature in features:
        print(f"\n{'='*50}")
        print(f"🔍 ANALYZING {feature.upper()} FEATURE")
        print(f"{'='*50}")
        
        if feature == 'Spotlight':
            insights = analyzer.analyze_spotlight_feature()
        elif feature == 'DM':
            insights = analyzer.analyze_dm_feature()
        elif feature == 'Bubble':
            insights = analyzer.analyze_bubble_feature()
        elif feature == 'Home':
            insights = analyzer.analyze_home_feature()
        
        # Create visualizations
        analyzer.create_feature_visualizations(insights, feature)
        
        # Print summary
        analyzer.print_feature_summary(insights, feature)
    
    print("\n✅ Feature-wise analysis completed!")
    print("📊 Check the generated PNG files for visualizations")

if __name__ == "__main__":
    main()
