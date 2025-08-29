import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class Cohort18PlusAnalysis:
    def __init__(self, file_path: str):
        """Initialize the analyzer for 18+ cohort"""
        self.file_path = file_path
        self.df = None
        self.cohort_name = "18+"
        self.benchmark_cohort = "Combine All"
        self.cohort_data = {}
        self.benchmark_data = {}
        self.analysis_results = {}
        
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean the CSV data"""
        print("📊 Loading and cleaning data for 18+ Cohort Analysis...")
        
        # Load the CSV
        self.df = pd.read_csv(self.file_path)
        
        print(f"🎯 Analyzing '{self.cohort_name}' cohort vs '{self.benchmark_cohort}' benchmark")
        
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
        cohorts = self.df.columns[1:].tolist()
        for col in cohorts:
            if col in self.df.columns:
                # Convert to string first
                self.df[col] = self.df[col].astype(str)
                # Remove percentage signs, time units, and clean up
                self.df[col] = self.df[col].str.replace('%', '').str.replace('secs', '').str.replace('mins', '').str.replace('sec', '')
                # Convert to numeric, errors='coerce' will convert invalid values to NaN
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def extract_cohort_data(self):
        """Extract data for 18+ cohort and benchmark"""
        print("📈 Extracting 18+ cohort data...")
        
        cohort_data = {}
        benchmark_data = {}
        
        # Process each row that contains metrics
        for idx, row in self.df.iterrows():
            metric_name = str(row.iloc[0]).strip()
            
            # Skip empty or invalid metric names
            if pd.isna(metric_name) or metric_name == '' or metric_name == 'nan':
                continue
            
            # Skip section headers
            if metric_name in ['Overall', 'Spotlight', 'Home', 'DM Dashboard', 'Bubble']:
                continue
            
            # Extract 18+ cohort value
            if self.cohort_name in self.df.columns:
                cohort_value = row[self.cohort_name]
                if pd.notna(cohort_value) and cohort_value != '' and str(cohort_value) != 'nan':
                    try:
                        cohort_data[metric_name] = float(cohort_value)
                    except:
                        continue
            
            # Extract benchmark value
            if self.benchmark_cohort in self.df.columns:
                benchmark_value = row[self.benchmark_cohort]
                if pd.notna(benchmark_value) and benchmark_value != '' and str(benchmark_value) != 'nan':
                    try:
                        benchmark_data[metric_name] = float(benchmark_value)
                    except:
                        continue
        
        self.cohort_data = cohort_data
        self.benchmark_data = benchmark_data
        
        print(f"📊 Found {len(cohort_data)} metrics for 18+ cohort")
        return cohort_data, benchmark_data
    
    def analyze_overall_performance(self):
        """Analyze overall performance metrics"""
        print("📊 Analyzing 18+ Overall Performance...")
        
        overall_metrics = {
            'Total Users': 'User Base',
            'DAU (increase %)': 'Daily Active Users',
            'DTU': 'Daily Transaction Users', 
            'Median SPV (TS)': 'Session Value',
            'Avg Time Spent per session': 'Session Duration',
            'user scanning / total users coming to home (user wise)': 'Scanning Adoption',
            'avg dm session per day': 'DM Engagement',
            'users opening spotlight / total users coming to home': 'Spotlight Adoption'
        }
        
        overall_analysis = {}
        
        for metric, display_name in overall_metrics.items():
            if metric in self.cohort_data and metric in self.benchmark_data:
                cohort_val = self.cohort_data[metric]
                benchmark_val = self.benchmark_data[metric]
                
                if benchmark_val != 0:
                    performance_ratio = cohort_val / benchmark_val
                    performance_pct = ((cohort_val - benchmark_val) / benchmark_val) * 100
                else:
                    performance_ratio = float('inf') if cohort_val > 0 else 1
                    performance_pct = 0
                
                overall_analysis[display_name] = {
                    'cohort_value': cohort_val,
                    'benchmark_value': benchmark_val,
                    'performance_ratio': performance_ratio,
                    'performance_pct': performance_pct,
                    'metric_name': metric
                }
        
        self.analysis_results['overall'] = overall_analysis
        return overall_analysis
    
    def analyze_spotlight_performance(self):
        """Analyze Spotlight feature performance"""
        print("🔍 Analyzing 18+ Spotlight Performance...")
        
        spotlight_metrics = {
            'Daily absolute numbers for both swipe': 'Swipe Interactions',
            'Daily absolute numbers for both tap': 'Tap Interactions',
            'Avg. time from input entered → payment compose (search efficiency)': 'Search Efficiency',
            'Paste button shown ⇒ Paste button clicked': 'Paste Button CTR',
            '% of SS sessions where paste button was clicked (adoption)': 'Paste Adoption',
            '% of SS sessions where number button was clicked (adoption)': 'Number Button Adoption',
            'SS open ⇒ Recents clicks': 'Recents Usage',
            'Quick actions clicked vs In-app purchases clicked (ratio)': 'Quick Actions Ratio',
            '% of SS sessions with quick actions usage': 'Quick Actions Usage'
        }
        
        spotlight_analysis = {}
        
        for metric, display_name in spotlight_metrics.items():
            if metric in self.cohort_data and metric in self.benchmark_data:
                cohort_val = self.cohort_data[metric]
                benchmark_val = self.benchmark_data[metric]
                
                if benchmark_val != 0:
                    performance_ratio = cohort_val / benchmark_val
                    performance_pct = ((cohort_val - benchmark_val) / benchmark_val) * 100
                else:
                    performance_ratio = float('inf') if cohort_val > 0 else 1
                    performance_pct = 0
                
                spotlight_analysis[display_name] = {
                    'cohort_value': cohort_val,
                    'benchmark_value': benchmark_val,
                    'performance_ratio': performance_ratio,
                    'performance_pct': performance_pct,
                    'metric_name': metric
                }
        
        self.analysis_results['spotlight'] = spotlight_analysis
        return spotlight_analysis
    
    def analyze_dm_performance(self):
        """Analyze DM Dashboard performance"""
        print("💬 Analyzing 18+ DM Dashboard Performance...")
        
        dm_metrics = {
            'Avg. time spent per DM session (from DM open to last event)': 'Session Duration',
            '% of DM sessions with repeat opens in the same day (re-engagement within a day)': 'Re-engagement Rate',
            'Average clicks on pinned conversations per day per user': 'Pin Usage',
            '% of DM users using pins at least once (adoption)': 'Pin Adoption',
            'Average clicks on 3 dots per day per user': 'Menu Usage',
            '% of DM users clicking on 3 dots at least once (adoption)': 'Menu Adoption',
            '% of users using long press at least once (adoption) [more important]': 'Long Press Adoption',
            'Unique users clicks on send message button per day': 'Message Button Clicks',
            'Messages sent per session (to understand engagement depth)': 'Message Frequency',
            '% of UPI transactions initiated from DM vs Home screen': 'DM Payment Usage'
        }
        
        dm_analysis = {}
        
        for metric, display_name in dm_metrics.items():
            if metric in self.cohort_data and metric in self.benchmark_data:
                cohort_val = self.cohort_data[metric]
                benchmark_val = self.benchmark_data[metric]
                
                if benchmark_val != 0:
                    performance_ratio = cohort_val / benchmark_val
                    performance_pct = ((cohort_val - benchmark_val) / benchmark_val) * 100
                else:
                    performance_ratio = float('inf') if cohort_val > 0 else 1
                    performance_pct = 0
                
                dm_analysis[display_name] = {
                    'cohort_value': cohort_val,
                    'benchmark_value': benchmark_val,
                    'performance_ratio': performance_ratio,
                    'performance_pct': performance_pct,
                    'metric_name': metric
                }
        
        self.analysis_results['dm'] = dm_analysis
        return dm_analysis
    
    def analyze_home_screen_performance(self):
        """Analyze Home Screen performance"""
        print("🏠 Analyzing 18+ Home Screen Performance...")
        
        home_metrics = {
            'Full Screen Scanner / Home Screen Scanner': 'Scanner Usage Ratio'
        }
        
        home_analysis = {}
        
        for metric, display_name in home_metrics.items():
            if metric in self.cohort_data and metric in self.benchmark_data:
                cohort_val = self.cohort_data[metric]
                benchmark_val = self.benchmark_data[metric]
                
                if benchmark_val != 0:
                    performance_ratio = cohort_val / benchmark_val
                    performance_pct = ((cohort_val - benchmark_val) / benchmark_val) * 100
                else:
                    performance_ratio = float('inf') if cohort_val > 0 else 1
                    performance_pct = 0
                
                home_analysis[display_name] = {
                    'cohort_value': cohort_val,
                    'benchmark_value': benchmark_val,
                    'performance_ratio': performance_ratio,
                    'performance_pct': performance_pct,
                    'metric_name': metric
                }
        
        self.analysis_results['home'] = home_analysis
        return home_analysis
    
    def analyze_bubble_performance(self):
        """Analyze Bubble feature performance"""
        print("🫧 Analyzing 18+ Bubble Performance...")
        
        bubble_metrics = {
            'Recent Ticket Size': 'Average Transaction Size',
            'No. of recent payment per recent click': 'Payment Conversion',
            'recent bubble txn %': 'Transaction Rate',
            '% of users using bubbles / total exposed users': 'Bubble Adoption'
        }
        
        bubble_analysis = {}
        
        for metric, display_name in bubble_metrics.items():
            if metric in self.cohort_data and metric in self.benchmark_data:
                cohort_val = self.cohort_data[metric]
                benchmark_val = self.benchmark_data[metric]
                
                if benchmark_val != 0:
                    performance_ratio = cohort_val / benchmark_val
                    performance_pct = ((cohort_val - benchmark_val) / benchmark_val) * 100
                else:
                    performance_ratio = float('inf') if cohort_val > 0 else 1
                    performance_pct = 0
                
                bubble_analysis[display_name] = {
                    'cohort_value': cohort_val,
                    'benchmark_value': benchmark_val,
                    'performance_ratio': performance_ratio,
                    'performance_pct': performance_pct,
                    'metric_name': metric
                }
        
        self.analysis_results['bubble'] = bubble_analysis
        return bubble_analysis
    
    def analyze_payment_performance(self):
        """Analyze Payment-related performance"""
        print("💳 Analyzing 18+ Payment Performance...")
        
        payment_metrics = {
            'DTU': 'Daily Transaction Users',
            'time to pay per user per pay session': 'Payment Completion Time',
            'opening the spotlight → Payment compose': 'Spotlight to Payment',
            '% of UPI transactions initiated from DM vs Home screen': 'DM Payment Usage',
            'Recent Ticket Size': 'Average Transaction Size',
            'recent bubble txn %': 'Bubble Transaction Rate'
        }
        
        payment_analysis = {}
        
        for metric, display_name in payment_metrics.items():
            if metric in self.cohort_data and metric in self.benchmark_data:
                cohort_val = self.cohort_data[metric]
                benchmark_val = self.benchmark_data[metric]
                
                if benchmark_val != 0:
                    performance_ratio = cohort_val / benchmark_val
                    performance_pct = ((cohort_val - benchmark_val) / benchmark_val) * 100
                else:
                    performance_ratio = float('inf') if cohort_val > 0 else 1
                    performance_pct = 0
                
                payment_analysis[display_name] = {
                    'cohort_value': cohort_val,
                    'benchmark_value': benchmark_val,
                    'performance_ratio': performance_ratio,
                    'performance_pct': performance_pct,
                    'metric_name': metric
                }
        
        self.analysis_results['payment'] = payment_analysis
        return payment_analysis
    
    def create_overall_performance_viz(self):
        """Create overall performance visualization"""
        print("📊 Creating Overall Performance Visualization...")
        
        if 'overall' not in self.analysis_results:
            return
        
        overall_data = self.analysis_results['overall']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('18+ Cohort: Overall Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Performance vs Benchmark
        ax1 = axes[0, 0]
        metrics = list(overall_data.keys())
        performance_pcts = [data['performance_pct'] for data in overall_data.values()]
        colors = ['green' if pct > 0 else 'red' for pct in performance_pcts]
        
        bars = ax1.barh(metrics, performance_pcts, color=colors, alpha=0.7)
        ax1.set_title('Performance vs Benchmark (%)', fontweight='bold')
        ax1.set_xlabel('Performance Difference (%)')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, pct in zip(bars, performance_pcts):
            width = bar.get_width()
            ax1.text(width + (5 if width >= 0 else -5), bar.get_y() + bar.get_height()/2.,
                    f'{pct:+.1f}%', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
        
        # 2. Absolute Values Comparison
        ax2 = axes[0, 1]
        cohort_values = [data['cohort_value'] for data in overall_data.values()]
        benchmark_values = [data['benchmark_value'] for data in overall_data.values()]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, cohort_values, width, label='18+ Cohort', color='lightblue', alpha=0.7)
        bars2 = ax2.bar(x + width/2, benchmark_values, width, label='Benchmark', color='orange', alpha=0.7)
        
        ax2.set_title('Absolute Values: 18+ vs Benchmark', fontweight='bold')
        ax2.set_ylabel('Values')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance Ratio
        ax3 = axes[1, 0]
        ratios = [data['performance_ratio'] for data in overall_data.values()]
        colors = ['green' if ratio > 1 else 'red' for ratio in ratios]
        
        bars = ax3.bar(metrics, ratios, color=colors, alpha=0.7)
        ax3.set_title('Performance Ratio (18+ / Benchmark)', fontweight='bold')
        ax3.set_ylabel('Ratio')
        ax3.axhline(y=1, color='black', linestyle='-', alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        plt.setp(ax3.get_xticklabels(), ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{ratio:.2f}x', ha='center', va='bottom', fontsize=9)
        
        # 4. Key Metrics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = "18+ COHORT SUMMARY\n\n"
        
        # Top performers
        top_performers = sorted(overall_data.items(), key=lambda x: x[1]['performance_pct'], reverse=True)[:3]
        summary_text += "🏆 TOP PERFORMERS:\n"
        for metric, data in top_performers:
            summary_text += f"• {metric}: {data['performance_pct']:+.1f}%\n"
        
        summary_text += "\n"
        
        # Bottom performers
        bottom_performers = sorted(overall_data.items(), key=lambda x: x[1]['performance_pct'])[:3]
        summary_text += "⚠️ NEEDS IMPROVEMENT:\n"
        for metric, data in bottom_performers:
            summary_text += f"• {metric}: {data['performance_pct']:+.1f}%\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('01_overall_performance.png', dpi=300, bbox_inches='tight')
        print("📊 Overall performance visualization saved")
    
    def create_spotlight_analysis_viz(self):
        """Create Spotlight analysis visualization"""
        print("🔍 Creating Spotlight Analysis Visualization...")
        
        if 'spotlight' not in self.analysis_results:
            return
        
        spotlight_data = self.analysis_results['spotlight']
        
        if not spotlight_data:
            print("⚠️ No Spotlight data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('18+ Cohort: Spotlight Feature Analysis', fontsize=16, fontweight='bold')
        
        # 1. Feature Usage Comparison
        ax1 = axes[0, 0]
        metrics = list(spotlight_data.keys())
        cohort_values = [data['cohort_value'] for data in spotlight_data.values()]
        benchmark_values = [data['benchmark_value'] for data in spotlight_data.values()]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, cohort_values, width, label='18+ Cohort', color='lightcoral', alpha=0.7)
        bars2 = ax1.bar(x + width/2, benchmark_values, width, label='Benchmark', color='lightblue', alpha=0.7)
        
        ax1.set_title('Spotlight Metrics: 18+ vs Benchmark', fontweight='bold')
        ax1.set_ylabel('Values')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance Gap Analysis
        ax2 = axes[0, 1]
        performance_pcts = [data['performance_pct'] for data in spotlight_data.values()]
        colors = ['green' if pct > 0 else 'red' for pct in performance_pcts]
        
        bars = ax2.barh(metrics, performance_pcts, color=colors, alpha=0.7)
        ax2.set_title('Performance Gap (%)', fontweight='bold')
        ax2.set_xlabel('Difference from Benchmark (%)')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # 3. Efficiency Metrics
        ax3 = axes[1, 0]
        efficiency_metrics = {k: v for k, v in spotlight_data.items() 
                             if 'efficiency' in k.lower() or 'time' in k.lower()}
        
        if efficiency_metrics:
            eff_metrics = list(efficiency_metrics.keys())
            eff_cohort = [data['cohort_value'] for data in efficiency_metrics.values()]
            eff_benchmark = [data['benchmark_value'] for data in efficiency_metrics.values()]
            
            x = np.arange(len(eff_metrics))
            bars1 = ax3.bar(x - width/2, eff_cohort, width, label='18+ Cohort', color='gold', alpha=0.7)
            bars2 = ax3.bar(x + width/2, eff_benchmark, width, label='Benchmark', color='silver', alpha=0.7)
            
            ax3.set_title('Efficiency Metrics', fontweight='bold')
            ax3.set_ylabel('Time (seconds)')
            ax3.set_xticks(x)
            ax3.set_xticklabels(eff_metrics, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Efficiency\nMetrics Available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        
        # 4. Adoption Rates
        ax4 = axes[1, 1]
        adoption_metrics = {k: v for k, v in spotlight_data.items() 
                           if 'adoption' in k.lower() or '%' in k}
        
        if adoption_metrics:
            adopt_metrics = list(adoption_metrics.keys())
            adopt_cohort = [data['cohort_value'] for data in adoption_metrics.values()]
            adopt_benchmark = [data['benchmark_value'] for data in adoption_metrics.values()]
            
            x = np.arange(len(adopt_metrics))
            bars1 = ax4.bar(x - width/2, adopt_cohort, width, label='18+ Cohort', color='lightgreen', alpha=0.7)
            bars2 = ax4.bar(x + width/2, adopt_benchmark, width, label='Benchmark', color='lightpink', alpha=0.7)
            
            ax4.set_title('Adoption Rates (%)', fontweight='bold')
            ax4.set_ylabel('Adoption Rate (%)')
            ax4.set_xticks(x)
            ax4.set_xticklabels(adopt_metrics, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Adoption\nMetrics Available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.savefig('02_spotlight_analysis.png', dpi=300, bbox_inches='tight')
        print("🔍 Spotlight analysis visualization saved")
    
    def create_dm_analysis_viz(self):
        """Create DM Dashboard analysis visualization"""
        print("💬 Creating DM Analysis Visualization...")
        
        if 'dm' not in self.analysis_results:
            return
        
        dm_data = self.analysis_results['dm']
        
        if not dm_data:
            print("⚠️ No DM data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('18+ Cohort: DM Dashboard Analysis', fontsize=16, fontweight='bold')
        
        # 1. Engagement Metrics
        ax1 = axes[0, 0]
        engagement_metrics = {k: v for k, v in dm_data.items() 
                             if 'engagement' in k.lower() or 'session' in k.lower() or 'message' in k.lower()}
        
        if engagement_metrics:
            eng_metrics = list(engagement_metrics.keys())
            eng_cohort = [data['cohort_value'] for data in engagement_metrics.values()]
            eng_benchmark = [data['benchmark_value'] for data in engagement_metrics.values()]
            
            x = np.arange(len(eng_metrics))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, eng_cohort, width, label='18+ Cohort', color='lightblue', alpha=0.7)
            bars2 = ax1.bar(x + width/2, eng_benchmark, width, label='Benchmark', color='orange', alpha=0.7)
            
            ax1.set_title('Engagement Metrics', fontweight='bold')
            ax1.set_ylabel('Values')
            ax1.set_xticks(x)
            ax1.set_xticklabels(eng_metrics, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Feature Adoption
        ax2 = axes[0, 1]
        adoption_metrics = {k: v for k, v in dm_data.items() 
                           if 'adoption' in k.lower() or 'pin' in k.lower()}
        
        if adoption_metrics:
            adopt_metrics = list(adoption_metrics.keys())
            adopt_performance = [data['performance_pct'] for data in adoption_metrics.values()]
            colors = ['green' if pct > 0 else 'red' for pct in adopt_performance]
            
            bars = ax2.barh(adopt_metrics, adopt_performance, color=colors, alpha=0.7)
            ax2.set_title('Feature Adoption Performance (%)', fontweight='bold')
            ax2.set_xlabel('Performance vs Benchmark (%)')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)
        
        # 3. Usage Patterns
        ax3 = axes[1, 0]
        usage_metrics = {k: v for k, v in dm_data.items() 
                        if 'usage' in k.lower() or 'click' in k.lower()}
        
        if usage_metrics:
            usage_names = list(usage_metrics.keys())
            usage_ratios = [data['performance_ratio'] for data in usage_metrics.values()]
            colors = ['green' if ratio > 1 else 'red' for ratio in usage_ratios]
            
            bars = ax3.bar(usage_names, usage_ratios, color=colors, alpha=0.7)
            ax3.set_title('Usage Pattern Ratios', fontweight='bold')
            ax3.set_ylabel('Ratio (18+ / Benchmark)')
            ax3.axhline(y=1, color='black', linestyle='-', alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
            plt.setp(ax3.get_xticklabels(), ha='right')
            ax3.grid(True, alpha=0.3)
        
        # 4. DM Performance Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create DM summary
        summary_text = "18+ DM PERFORMANCE\n\n"
        
        # Best DM metrics
        best_dm = sorted(dm_data.items(), key=lambda x: x[1]['performance_pct'], reverse=True)[:3]
        summary_text += "🏆 STRENGTHS:\n"
        for metric, data in best_dm:
            summary_text += f"• {metric[:25]}...: {data['performance_pct']:+.1f}%\n"
        
        summary_text += "\n⚠️ OPPORTUNITIES:\n"
        worst_dm = sorted(dm_data.items(), key=lambda x: x[1]['performance_pct'])[:3]
        for metric, data in worst_dm:
            summary_text += f"• {metric[:25]}...: {data['performance_pct']:+.1f}%\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('03_dm_analysis.png', dpi=300, bbox_inches='tight')
        print("💬 DM analysis visualization saved")
    
    def create_payment_analysis_viz(self):
        """Create Payment analysis visualization"""
        print("💳 Creating Payment Analysis Visualization...")
        
        if 'payment' not in self.analysis_results:
            return
        
        payment_data = self.analysis_results['payment']
        
        if not payment_data:
            print("⚠️ No Payment data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('18+ Cohort: Payment Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Payment Metrics Overview
        ax1 = axes[0, 0]
        metrics = list(payment_data.keys())
        cohort_values = [data['cohort_value'] for data in payment_data.values()]
        benchmark_values = [data['benchmark_value'] for data in payment_data.values()]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, cohort_values, width, label='18+ Cohort', color='lightgreen', alpha=0.7)
        bars2 = ax1.bar(x + width/2, benchmark_values, width, label='Benchmark', color='lightcoral', alpha=0.7)
        
        ax1.set_title('Payment Metrics: 18+ vs Benchmark', fontweight='bold')
        ax1.set_ylabel('Values')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance Gaps
        ax2 = axes[0, 1]
        performance_pcts = [data['performance_pct'] for data in payment_data.values()]
        colors = ['green' if pct > 0 else 'red' for pct in performance_pcts]
        
        bars = ax2.barh(metrics, performance_pcts, color=colors, alpha=0.7)
        ax2.set_title('Payment Performance Gaps (%)', fontweight='bold')
        ax2.set_xlabel('Performance vs Benchmark (%)')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # 3. Transaction Analysis
        ax3 = axes[1, 0]
        transaction_metrics = {k: v for k, v in payment_data.items() 
                              if 'transaction' in k.lower() or 'dtu' in k.lower() or 'ticket' in k.lower()}
        
        if transaction_metrics:
            trans_names = list(transaction_metrics.keys())
            trans_cohort = [data['cohort_value'] for data in transaction_metrics.values()]
            trans_benchmark = [data['benchmark_value'] for data in transaction_metrics.values()]
            
            # Create a radar chart for transaction metrics
            angles = np.linspace(0, 2 * np.pi, len(trans_names), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            trans_cohort += trans_cohort[:1]
            trans_benchmark += trans_benchmark[:1]
            
            ax3 = plt.subplot(2, 2, 3, projection='polar')
            ax3.plot(angles, trans_cohort, 'o-', linewidth=2, label='18+ Cohort', color='blue')
            ax3.fill(angles, trans_cohort, alpha=0.25, color='blue')
            ax3.plot(angles, trans_benchmark, 'o-', linewidth=2, label='Benchmark', color='red')
            ax3.fill(angles, trans_benchmark, alpha=0.25, color='red')
            
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(trans_names)
            ax3.set_title('Transaction Metrics Radar', fontweight='bold', pad=20)
            ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 4. Payment Efficiency
        ax4 = axes[1, 1]
        efficiency_metrics = {k: v for k, v in payment_data.items() 
                             if 'time' in k.lower() or 'efficiency' in k.lower()}
        
        if efficiency_metrics:
            eff_names = list(efficiency_metrics.keys())
            eff_ratios = [data['performance_ratio'] for data in efficiency_metrics.values()]
            colors = ['red' if ratio > 1 else 'green' for ratio in eff_ratios]  # For time metrics, lower is better
            
            bars = ax4.bar(eff_names, eff_ratios, color=colors, alpha=0.7)
            ax4.set_title('Payment Efficiency Ratios', fontweight='bold')
            ax4.set_ylabel('Ratio (18+ / Benchmark)')
            ax4.axhline(y=1, color='black', linestyle='-', alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
            plt.setp(ax4.get_xticklabels(), ha='right')
            ax4.grid(True, alpha=0.3)
            
            # Add interpretation note
            ax4.text(0.02, 0.98, 'Lower is Better for Time Metrics', 
                    transform=ax4.transAxes, fontsize=8, 
                    verticalalignment='top', style='italic')
        
        plt.tight_layout()
        plt.savefig('04_payment_analysis.png', dpi=300, bbox_inches='tight')
        print("💳 Payment analysis visualization saved")
    
    def create_comprehensive_dashboard(self):
        """Create comprehensive dashboard"""
        print("📊 Creating Comprehensive Dashboard...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('18+ Cohort: Comprehensive Performance Dashboard', fontsize=18, fontweight='bold')
        
        # Collect all performance data
        all_performance = {}
        for section, data in self.analysis_results.items():
            for metric, perf_data in data.items():
                all_performance[f"{section}_{metric}"] = perf_data
        
        if not all_performance:
            print("⚠️ No performance data available for dashboard")
            return
        
        # 1. Overall Performance Score
        ax1 = axes[0, 0]
        performance_scores = [data['performance_pct'] for data in all_performance.values()]
        avg_performance = np.mean(performance_scores)
        positive_count = sum(1 for score in performance_scores if score > 0)
        total_count = len(performance_scores)
        
        # Create a gauge chart
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        colors = plt.cm.RdYlGn(np.linspace(0, 1, 100))
        for i in range(len(theta)-1):
            ax1.fill_between([theta[i], theta[i+1]], 0, 1, color=colors[i], alpha=0.7)
        
        # Add performance needle
        perf_angle = np.pi * (avg_performance + 100) / 200  # Map -100 to +100 to 0 to π
        ax1.arrow(0, 0, 0.8 * np.cos(perf_angle), 0.8 * np.sin(perf_angle), 
                 head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-0.2, 1.2)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f'Overall Performance\n{avg_performance:.1f}%', fontweight='bold')
        ax1.text(0, -0.1, f'{positive_count}/{total_count} metrics above benchmark', 
                ha='center', va='center', fontsize=10)
        
        # 2. Performance by Section
        ax2 = axes[0, 1]
        section_performance = {}
        for section, data in self.analysis_results.items():
            if data:
                section_scores = [perf_data['performance_pct'] for perf_data in data.values()]
                section_performance[section.title()] = np.mean(section_scores)
        
        if section_performance:
            sections = list(section_performance.keys())
            scores = list(section_performance.values())
            colors = ['green' if score > 0 else 'red' for score in scores]
            
            bars = ax2.barh(sections, scores, color=colors, alpha=0.7)
            ax2.set_title('Performance by Section', fontweight='bold')
            ax2.set_xlabel('Average Performance (%)')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                ax2.text(width + (2 if width >= 0 else -2), bar.get_y() + bar.get_height()/2.,
                        f'{score:.1f}%', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
        
        # 3. Top Strengths
        ax3 = axes[0, 2]
        ax3.axis('off')
        top_strengths = sorted(all_performance.items(), key=lambda x: x[1]['performance_pct'], reverse=True)[:5]
        
        strength_text = "🏆 TOP 5 STRENGTHS\n\n"
        for i, (metric, data) in enumerate(top_strengths, 1):
            clean_metric = metric.split('_', 1)[1] if '_' in metric else metric
            strength_text += f"{i}. {clean_metric[:30]}...\n   {data['performance_pct']:+.1f}%\n\n"
        
        ax3.text(0.1, 0.9, strength_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        # 4. Biggest Opportunities
        ax4 = axes[1, 0]
        ax4.axis('off')
        opportunities = sorted(all_performance.items(), key=lambda x: x[1]['performance_pct'])[:5]
        
        opp_text = "⚠️ TOP 5 OPPORTUNITIES\n\n"
        for i, (metric, data) in enumerate(opportunities, 1):
            clean_metric = metric.split('_', 1)[1] if '_' in metric else metric
            opp_text += f"{i}. {clean_metric[:30]}...\n   {data['performance_pct']:+.1f}%\n\n"
        
        ax4.text(0.1, 0.9, opp_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        
        # 5. Performance Distribution
        ax5 = axes[1, 1]
        performance_ranges = {
            'Excellent (>20%)': sum(1 for score in performance_scores if score > 20),
            'Good (0-20%)': sum(1 for score in performance_scores if 0 <= score <= 20),
            'Below (-20-0%)': sum(1 for score in performance_scores if -20 <= score < 0),
            'Poor (<-20%)': sum(1 for score in performance_scores if score < -20)
        }
        
        labels = list(performance_ranges.keys())
        sizes = list(performance_ranges.values())
        colors = ['darkgreen', 'lightgreen', 'orange', 'red']
        
        wedges, texts, autotexts = ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Performance Distribution', fontweight='bold')
        
        # 6. Metric Coverage
        ax6 = axes[1, 2]
        section_counts = {section.title(): len(data) for section, data in self.analysis_results.items() if data}
        
        if section_counts:
            sections = list(section_counts.keys())
            counts = list(section_counts.values())
            
            bars = ax6.bar(sections, counts, color='skyblue', alpha=0.7)
            ax6.set_title('Metrics Analyzed by Section', fontweight='bold')
            ax6.set_ylabel('Number of Metrics')
            ax6.tick_params(axis='x', rotation=45)
            plt.setp(ax6.get_xticklabels(), ha='right')
            ax6.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(count)}', ha='center', va='bottom', fontsize=9)
        
        # 7-9. Business Intelligence Insights
        ax7 = axes[2, 0]
        ax7.axis('off')
        
        bi_text = "📊 BUSINESS INTELLIGENCE\n\n"
        
        # Calculate key insights
        if 'overall' in self.analysis_results and 'User Base' in self.analysis_results['overall']:
            user_base = self.analysis_results['overall']['User Base']['cohort_value']
            bi_text += f"👥 User Base: {user_base:,.0f}\n\n"
        
        if avg_performance > 0:
            bi_text += f"✅ Overall trending positive\n   ({avg_performance:.1f}% avg performance)\n\n"
        else:
            bi_text += f"⚠️ Overall needs attention\n   ({avg_performance:.1f}% avg performance)\n\n"
        
        bi_text += f"📈 {positive_count}/{total_count} metrics outperforming\n\n"
        
        # Key recommendations
        if avg_performance < -10:
            bi_text += "🎯 PRIORITY: Major improvements needed"
        elif avg_performance < 0:
            bi_text += "🎯 FOCUS: Targeted optimizations"
        else:
            bi_text += "🎯 OPTIMIZE: Build on strengths"
        
        ax7.text(0.1, 0.9, bi_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # 8. Trend Analysis
        ax8 = axes[2, 1]
        # Create a mock trend visualization showing performance across sections
        if section_performance:
            sections = list(section_performance.keys())
            scores = list(section_performance.values())
            
            ax8.plot(sections, scores, 'o-', linewidth=2, markersize=8, color='blue')
            ax8.fill_between(sections, scores, alpha=0.3, color='blue')
            ax8.set_title('Performance Trend Across Sections', fontweight='bold')
            ax8.set_ylabel('Performance (%)')
            ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax8.tick_params(axis='x', rotation=45)
            plt.setp(ax8.get_xticklabels(), ha='right')
            ax8.grid(True, alpha=0.3)
        
        # 9. Action Items
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        action_text = "🎯 ACTION ITEMS\n\n"
        
        # Generate action items based on performance
        worst_section = min(section_performance.items(), key=lambda x: x[1]) if section_performance else None
        best_section = max(section_performance.items(), key=lambda x: x[1]) if section_performance else None
        
        if worst_section and worst_section[1] < -10:
            action_text += f"1. URGENT: Fix {worst_section[0]}\n   ({worst_section[1]:.1f}% performance)\n\n"
        
        if best_section and best_section[1] > 10:
            action_text += f"2. LEVERAGE: Scale {best_section[0]}\n   ({best_section[1]:.1f}% performance)\n\n"
        
        action_text += "3. MONITOR: Track weekly\n   performance changes\n\n"
        action_text += "4. OPTIMIZE: Focus on top\n   opportunity areas"
        
        ax9.text(0.1, 0.9, action_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('05_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        print("📊 Comprehensive dashboard saved")
    
    def generate_business_report(self):
        """Generate comprehensive business report"""
        print("📋 Generating Business Report...")
        
        report = []
        report.append("="*80)
        report.append("📊 18+ COHORT COMPREHENSIVE BUSINESS ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        # Executive Summary
        all_performance = {}
        for section, data in self.analysis_results.items():
            for metric, perf_data in data.items():
                all_performance[f"{section}_{metric}"] = perf_data
        
        if all_performance:
            performance_scores = [data['performance_pct'] for data in all_performance.values()]
            avg_performance = np.mean(performance_scores)
            positive_count = sum(1 for score in performance_scores if score > 0)
            total_count = len(performance_scores)
            
            report.append("🎯 EXECUTIVE SUMMARY")
            report.append("-" * 40)
            report.append(f"Overall Performance Score: {avg_performance:.1f}%")
            report.append(f"Metrics Outperforming Benchmark: {positive_count}/{total_count} ({(positive_count/total_count)*100:.1f}%)")
            
            if avg_performance > 10:
                report.append("Status: 🟢 STRONG PERFORMANCE")
            elif avg_performance > 0:
                report.append("Status: 🟡 MODERATE PERFORMANCE")
            elif avg_performance > -10:
                report.append("Status: 🟠 NEEDS ATTENTION")
            else:
                report.append("Status: 🔴 CRITICAL IMPROVEMENTS NEEDED")
            report.append("")
        
        # Section-wise Analysis
        for section_name, section_data in self.analysis_results.items():
            if not section_data:
                continue
                
            report.append(f"📊 {section_name.upper()} ANALYSIS")
            report.append("-" * 40)
            
            section_scores = [data['performance_pct'] for data in section_data.values()]
            section_avg = np.mean(section_scores)
            report.append(f"Section Performance: {section_avg:.1f}%")
            report.append("")
            
            # Top performers in section
            top_metrics = sorted(section_data.items(), key=lambda x: x[1]['performance_pct'], reverse=True)[:3]
            report.append("🏆 TOP PERFORMERS:")
            for metric, data in top_metrics:
                report.append(f"   • {metric}: {data['cohort_value']:.2f} vs {data['benchmark_value']:.2f} ({data['performance_pct']:+.1f}%)")
            report.append("")
            
            # Bottom performers in section
            bottom_metrics = sorted(section_data.items(), key=lambda x: x[1]['performance_pct'])[:3]
            report.append("⚠️ IMPROVEMENT OPPORTUNITIES:")
            for metric, data in bottom_metrics:
                report.append(f"   • {metric}: {data['cohort_value']:.2f} vs {data['benchmark_value']:.2f} ({data['performance_pct']:+.1f}%)")
            report.append("")
        
        # Strategic Recommendations
        report.append("💡 STRATEGIC RECOMMENDATIONS")
        report.append("-" * 40)
        
        # Generate recommendations based on performance
        section_performance = {}
        for section, data in self.analysis_results.items():
            if data:
                section_scores = [perf_data['performance_pct'] for perf_data in data.values()]
                section_performance[section] = np.mean(section_scores)
        
        if section_performance:
            worst_section = min(section_performance.items(), key=lambda x: x[1])
            best_section = max(section_performance.items(), key=lambda x: x[1])
            
            report.append(f"1. PRIORITY FOCUS: {worst_section[0].title()}")
            report.append(f"   - Current performance: {worst_section[1]:.1f}%")
            report.append(f"   - Requires immediate attention and resource allocation")
            report.append("")
            
            report.append(f"2. LEVERAGE STRENGTHS: {best_section[0].title()}")
            report.append(f"   - Current performance: {best_section[1]:.1f}%")
            report.append(f"   - Scale successful practices to other areas")
            report.append("")
            
            report.append("3. MONITORING & OPTIMIZATION:")
            report.append("   - Implement weekly performance tracking")
            report.append("   - Set up automated alerts for significant changes")
            report.append("   - Regular A/B testing for improvement opportunities")
            report.append("")
        
        # Business Impact Assessment
        report.append("💼 BUSINESS IMPACT ASSESSMENT")
        report.append("-" * 40)
        
        if 'overall' in self.analysis_results and 'User Base' in self.analysis_results['overall']:
            user_base = self.analysis_results['overall']['User Base']['cohort_value']
            report.append(f"18+ Cohort Size: {user_base:,.0f} users")
            
            if avg_performance < -20:
                report.append("Risk Level: HIGH - Significant user experience issues")
            elif avg_performance < 0:
                report.append("Risk Level: MEDIUM - Performance gaps affecting engagement")
            else:
                report.append("Risk Level: LOW - Stable performance with growth opportunities")
        
        report.append("")
        
        # Save report
        with open('18plus_business_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("📋 Business report saved to '18plus_business_report.txt'")
        
        # Print summary to console
        print("\n" + "\n".join(report[:20]))  # Print first 20 lines
        print("\n... (Full report saved to file)")

def main():
    """Main function to run 18+ cohort analysis"""
    print("🚀 Starting Comprehensive 18+ Cohort Analysis...")
    
    # Initialize analyzer
    analyzer = Cohort18PlusAnalysis('../Cohort Wise Analysis Fam 2.0 - Sheet1.csv')
    
    # Load and clean data
    df = analyzer.load_and_clean_data()
    
    # Extract cohort data
    cohort_data, benchmark_data = analyzer.extract_cohort_data()
    
    # Run all analyses
    print("\n📊 Running comprehensive analysis...")
    analyzer.analyze_overall_performance()
    analyzer.analyze_spotlight_performance()
    analyzer.analyze_dm_performance()
    analyzer.analyze_home_screen_performance()
    analyzer.analyze_bubble_performance()
    analyzer.analyze_payment_performance()
    
    # Create all visualizations
    print("\n📊 Creating visualizations...")
    analyzer.create_overall_performance_viz()
    analyzer.create_spotlight_analysis_viz()
    analyzer.create_dm_analysis_viz()
    analyzer.create_payment_analysis_viz()
    analyzer.create_comprehensive_dashboard()
    
    # Generate business report
    analyzer.generate_business_report()
    
    print("\n✅ 18+ Cohort Analysis Complete!")
    print("📁 All files saved in '18+' folder:")
    print("   📊 01_overall_performance.png")
    print("   🔍 02_spotlight_analysis.png")
    print("   💬 03_dm_analysis.png")
    print("   💳 04_payment_analysis.png")
    print("   📊 05_comprehensive_dashboard.png")
    print("   📋 18plus_business_report.txt")

if __name__ == "__main__":
    main()
