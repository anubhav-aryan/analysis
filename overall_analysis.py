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

class CohortAnalyzer:
    def __init__(self, file_path: str):
        """Initialize the analyzer with CSV file path"""
        self.file_path = file_path
        self.df = None
        self.cohorts = None
        self.overall_metrics = None
        
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
    
    def extract_overall_metrics(self) -> pd.DataFrame:
        """Extract overall engagement metrics"""
        print("📈 Extracting overall engagement metrics...")
        
        # Define the overall metrics rows (rows 2-13)
        overall_rows = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        
        # Extract overall section
        overall_data = self.df.iloc[overall_rows].copy()
        overall_data = overall_data.reset_index(drop=True)
        
        # Clean metric names
        overall_data.iloc[:, 0] = overall_data.iloc[:, 0].str.strip()
        
        self.overall_metrics = overall_data
        return overall_data
    
    def analyze_user_distribution(self) -> Dict:
        """Analyze user distribution across cohorts"""
        print("👥 Analyzing user distribution...")
        
        # Get total users row
        total_users_row = self.overall_metrics[self.overall_metrics.iloc[:, 0] == 'Total Users']
        
        if not total_users_row.empty:
            user_data = {}
            for i, cohort in enumerate(self.cohorts):
                if i < len(total_users_row.columns) - 1:
                    value = total_users_row.iloc[0, i + 1]
                    if pd.notna(value) and value != '' and value > 0:
                        user_data[cohort] = float(value)
            
            return user_data
        return {}
    
    def analyze_engagement_metrics(self) -> Dict:
        """Analyze DAU and DTU metrics"""
        print("📊 Analyzing engagement metrics...")
        
        engagement_data = {}
        
        # Extract DAU and DTU
        dau_row = self.overall_metrics[self.overall_metrics.iloc[:, 0].str.contains('DAU', na=False)]
        dtu_row = self.overall_metrics[self.overall_metrics.iloc[:, 0].str.contains('DTU', na=False)]
        
        if not dau_row.empty:
            for i, cohort in enumerate(self.cohorts):
                if i < len(dau_row.columns) - 1:
                    value = dau_row.iloc[0, i + 1]
                    if pd.notna(value) and value != '' and value > 0:
                        engagement_data[f'{cohort}_DAU'] = float(value)
        
        if not dtu_row.empty:
            for i, cohort in enumerate(self.cohorts):
                if i < len(dtu_row.columns) - 1:
                    value = dtu_row.iloc[0, i + 1]
                    if pd.notna(value) and value != '' and value > 0:
                        engagement_data[f'{cohort}_DTU'] = float(value)
        
        return engagement_data
    
    def create_overall_visualizations(self):
        """Create visualizations for overall analysis"""
        print("📊 Creating overall analysis visualizations...")
        
        # 1. User Distribution Chart
        user_dist = self.analyze_user_distribution()
        if user_dist:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            cohorts = list(user_dist.keys())
            users = list(user_dist.values())
            
            bars = plt.bar(cohorts, users, color='skyblue', alpha=0.7)
            plt.title('Total Users by Cohort', fontsize=14, fontweight='bold')
            plt.xlabel('Cohorts')
            plt.ylabel('Number of Users')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, user_count in zip(bars, users):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(users)*0.01,
                        f'{user_count:,.0f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Engagement Metrics Comparison
        engagement_data = self.analyze_engagement_metrics()
        if engagement_data:
            dau_data = {k.replace('_DAU', ''): v for k, v in engagement_data.items() if 'DAU' in k}
            dtu_data = {k.replace('_DTU', ''): v for k, v in engagement_data.items() if 'DTU' in k}
            
            if dau_data and dtu_data:
                plt.subplot(2, 2, 2)
                x = np.arange(len(dau_data))
                width = 0.35
                
                dau_values = list(dau_data.values())
                dtu_values = list(dtu_data.values())
                
                bars1 = plt.bar(x - width/2, dau_values, width, label='DAU %', color='lightcoral', alpha=0.7)
                bars2 = plt.bar(x + width/2, dtu_values, width, label='DTU %', color='lightgreen', alpha=0.7)
                
                plt.title('DAU vs DTU by Cohort', fontsize=14, fontweight='bold')
                plt.xlabel('Cohorts')
                plt.ylabel('Percentage (%)')
                plt.xticks(x, list(dau_data.keys()), rotation=45, ha='right')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars1, dau_values):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
                
                for bar, value in zip(bars2, dtu_values):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. User Distribution Pie Chart
        if user_dist:
            plt.subplot(2, 2, 3)
            plt.pie(user_dist.values(), labels=user_dist.keys(), autopct='%1.1f%%', startangle=90)
            plt.title('User Distribution by Cohort', fontsize=14, fontweight='bold')
        
        # 4. Engagement Conversion Rate
        if engagement_data and dau_data and dtu_data:
            plt.subplot(2, 2, 4)
            conversion_rates = {}
            for cohort in dau_data.keys():
                if cohort in dtu_data:
                    dau = dau_data[cohort]
                    dtu = dtu_data[cohort]
                    conversion_rate = (dtu / dau * 100) if dau > 0 else 0
                    conversion_rates[cohort] = conversion_rate
            
            if conversion_rates:
                cohorts = list(conversion_rates.keys())
                rates = list(conversion_rates.values())
                
                bars = plt.bar(cohorts, rates, color='gold', alpha=0.7)
                plt.title('DAU to DTU Conversion Rate', fontsize=14, fontweight='bold')
                plt.xlabel('Cohorts')
                plt.ylabel('Conversion Rate (%)')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, rate in zip(bars, rates):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('overall_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Visualizations saved as 'overall_analysis.png'")
    
    def generate_overall_insights(self) -> Dict:
        """Generate insights from overall analysis"""
        print("💡 Generating overall insights...")
        
        insights = {}
        
        # User distribution insights
        user_dist = self.analyze_user_distribution()
        if user_dist:
            total_users = sum(user_dist.values())
            insights['total_users'] = total_users
            insights['largest_cohort'] = max(user_dist, key=user_dist.get)
            insights['smallest_cohort'] = min(user_dist, key=user_dist.get)
            insights['user_distribution'] = user_dist
        
        # Engagement insights
        engagement_data = self.analyze_engagement_metrics()
        if engagement_data:
            dau_data = {k.replace('_DAU', ''): v for k, v in engagement_data.items() if 'DAU' in k}
            dtu_data = {k.replace('_DTU', ''): v for k, v in engagement_data.items() if 'DTU' in k}
            
            if dau_data:
                insights['highest_dau'] = max(dau_data, key=dau_data.get)
                insights['lowest_dau'] = min(dau_data, key=dau_data.get)
                insights['dau_data'] = dau_data
            
            if dtu_data:
                insights['highest_dtu'] = max(dtu_data, key=dtu_data.get)
                insights['lowest_dtu'] = min(dtu_data, key=dtu_data.get)
                insights['dtu_data'] = dtu_data
        
        return insights
    
    def print_overall_summary(self):
        """Print a comprehensive summary of overall analysis"""
        print("\n" + "="*60)
        print("📊 OVERALL ANALYSIS SUMMARY")
        print("="*60)
        
        insights = self.generate_overall_insights()
        
        print(f"\n👥 USER DISTRIBUTION:")
        if 'total_users' in insights:
            print(f"   • Total Users: {insights['total_users']:,.0f}")
            print(f"   • Largest Cohort: {insights['largest_cohort']}")
            print(f"   • Smallest Cohort: {insights['smallest_cohort']}")
            
            # Show user distribution
            print(f"\n   📊 User Distribution:")
            for cohort, users in insights['user_distribution'].items():
                percentage = (users / insights['total_users']) * 100
                print(f"      - {cohort}: {users:,.0f} users ({percentage:.1f}%)")
        
        print(f"\n📈 ENGAGEMENT METRICS:")
        if 'dau_data' in insights:
            print(f"   • Highest DAU: {insights['highest_dau']} ({insights['dau_data'][insights['highest_dau']]:.1f}%)")
            print(f"   • Lowest DAU: {insights['lowest_dau']} ({insights['dau_data'][insights['lowest_dau']]:.1f}%)")
        
        if 'dtu_data' in insights:
            print(f"   • Highest DTU: {insights['highest_dtu']} ({insights['dtu_data'][insights['highest_dtu']]:.1f}%)")
            print(f"   • Lowest DTU: {insights['lowest_dtu']} ({insights['dtu_data'][insights['lowest_dtu']]:.1f}%)")
        
        print(f"\n🔍 KEY OBSERVATIONS:")
        if 'dau_data' in insights and 'dtu_data' in insights:
            # Find cohorts with both DAU and DTU data
            common_cohorts = set(insights['dau_data'].keys()) & set(insights['dtu_data'].keys())
            for cohort in common_cohorts:
                dau = insights['dau_data'][cohort]
                dtu = insights['dtu_data'][cohort]
                conversion_rate = (dtu / dau * 100) if dau > 0 else 0
                print(f"   • {cohort}: DAU {dau:.1f}% → DTU {dtu:.1f}% (Conversion: {conversion_rate:.1f}%)")

def main():
    """Main function to run overall analysis"""
    print("🚀 Starting Overall Analysis...")
    
    # Initialize analyzer
    analyzer = CohortAnalyzer('Cohort Wise Analysis Fam 2.0 - Sheet1.csv')
    
    # Load and clean data
    df = analyzer.load_and_clean_data()
    
    # Extract overall metrics
    overall_metrics = analyzer.extract_overall_metrics()
    
    # Create visualizations
    analyzer.create_overall_visualizations()
    
    # Print summary
    analyzer.print_overall_summary()
    
    print("\n✅ Overall analysis completed!")
    print("📊 Check 'overall_analysis.png' for visualizations")

if __name__ == "__main__":
    main()
