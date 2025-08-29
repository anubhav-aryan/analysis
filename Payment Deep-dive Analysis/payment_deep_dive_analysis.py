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

class PaymentDeepDiveAnalysis:
    def __init__(self, file_path: str):
        """Initialize payment deep-dive analysis"""
        self.file_path = file_path
        self.df = None
        self.benchmark_cohort = "Combine All"
        
        # Define cohort groups for analysis
        self.all_cohorts = []
        self.payment_metrics = []
        self.engagement_metrics = []
        self.efficiency_metrics = []
        
        self.kpi_data = {}
        self.payment_analysis = {}
        
    def load_and_clean_data(self):
        """Load and clean the CSV data"""
        print("💳 Loading payment deep-dive data...")
        
        # Load the CSV
        self.df = pd.read_csv(self.file_path)
        
        # Get all cohorts (exclude first column which contains metric names)
        self.all_cohorts = [col for col in self.df.columns[1:] if col.strip()]
        
        # Clean the data
        self._clean_data()
        
        return self.df
    
    def _clean_data(self):
        """Clean and standardize the data"""
        # Remove completely empty rows
        self.df = self.df.dropna(how='all')
        
        # Replace dashes with NaN
        self.df = self.df.replace('-', np.nan)
        
        # Clean percentage values and time values for all cohorts
        for col in self.all_cohorts:
            if col in self.df.columns:
                # Convert to string first
                self.df[col] = self.df[col].astype(str)
                # Remove percentage signs, time units, and clean up
                self.df[col] = self.df[col].str.replace('%', '').str.replace('secs', '').str.replace('mins', '').str.replace('sec', '')
                # Convert to numeric, errors='coerce' will convert invalid values to NaN
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def extract_payment_metrics(self):
        """Extract and categorize payment-related metrics"""
        print("💳 Extracting payment metrics...")
        
        # Initialize data storage
        self.kpi_data = {}
        
        # Define metric categories
        payment_keywords = ['pay', 'transaction', 'UPI', 'Ticket Size', 'compose', 'txn']
        engagement_keywords = ['Time Spent', 'session', 'Messages', 'repeat', 'opening']
        efficiency_keywords = ['time to pay', 'efficiency', 'search', 'input', 'clicks']
        
        # Process each row that contains metrics
        for idx, row in self.df.iterrows():
            metric_name = str(row.iloc[0]).strip()
            
            # Skip empty or invalid metric names
            if pd.isna(metric_name) or metric_name == '' or metric_name == 'nan':
                continue
            
            # Skip section headers
            if metric_name in ['Overall', 'Spotlight', 'Home', 'DM Dashboard', 'Bubble']:
                continue
            
            # Categorize metrics
            is_payment = any(keyword.lower() in metric_name.lower() for keyword in payment_keywords)
            is_engagement = any(keyword.lower() in metric_name.lower() for keyword in engagement_keywords)
            is_efficiency = any(keyword.lower() in metric_name.lower() for keyword in efficiency_keywords)
            
            if is_payment:
                self.payment_metrics.append(metric_name)
            elif is_engagement:
                self.engagement_metrics.append(metric_name)
            elif is_efficiency:
                self.efficiency_metrics.append(metric_name)
            
            # Store all metrics
            self.kpi_data[metric_name] = {}
            
            # Extract data for each cohort
            for cohort in self.all_cohorts:
                if cohort in self.df.columns:
                    cohort_value = row[cohort]
                    if pd.notna(cohort_value) and cohort_value != '' and str(cohort_value) != 'nan':
                        try:
                            self.kpi_data[metric_name][cohort] = float(cohort_value)
                        except:
                            continue
        
        print(f"💳 Found {len(self.payment_metrics)} payment metrics")
        print(f"💬 Found {len(self.engagement_metrics)} engagement metrics")
        print(f"⚡ Found {len(self.efficiency_metrics)} efficiency metrics")
        
        return self.kpi_data
    
    def analyze_time_to_pay_efficiency(self):
        """Analyze time-to-pay performance across cohorts"""
        print("⚡ Analyzing time-to-pay efficiency...")
        
        time_to_pay_analysis = {
            'cohort_performance': {},
            'efficiency_gaps': [],
            'optimization_opportunities': {}
        }
        
        # Find time-to-pay related metrics
        time_metrics = [m for m in self.kpi_data.keys() 
                       if 'time to pay' in m.lower() or 'efficiency' in m.lower()]
        
        for metric in time_metrics:
            metric_data = self.kpi_data[metric]
            benchmark_value = metric_data.get(self.benchmark_cohort, None)
            
            for cohort in self.all_cohorts:
                if cohort != self.benchmark_cohort and cohort in metric_data:
                    cohort_value = metric_data[cohort]
                    
                    if cohort not in time_to_pay_analysis['cohort_performance']:
                        time_to_pay_analysis['cohort_performance'][cohort] = {}
                    
                    # For time metrics, lower is better
                    if benchmark_value and benchmark_value > 0:
                        efficiency_ratio = cohort_value / benchmark_value
                        improvement_potential = ((benchmark_value - cohort_value) / benchmark_value) * 100
                    else:
                        efficiency_ratio = 1.0
                        improvement_potential = 0
                    
                    time_to_pay_analysis['cohort_performance'][cohort][metric] = {
                        'value': cohort_value,
                        'benchmark': benchmark_value,
                        'efficiency_ratio': efficiency_ratio,
                        'improvement_potential': improvement_potential
                    }
                    
                    # Track significant gaps
                    if abs(improvement_potential) > 50:
                        time_to_pay_analysis['efficiency_gaps'].append({
                            'cohort': cohort,
                            'metric': metric,
                            'gap': abs(improvement_potential),
                            'cohort_value': cohort_value,
                            'benchmark_value': benchmark_value,
                            'faster_slower': 'faster' if improvement_potential > 0 else 'slower'
                        })
        
        # Sort gaps by magnitude
        time_to_pay_analysis['efficiency_gaps'].sort(key=lambda x: x['gap'], reverse=True)
        
        # Calculate optimization opportunities
        for cohort in time_to_pay_analysis['cohort_performance']:
            cohort_metrics = time_to_pay_analysis['cohort_performance'][cohort]
            
            # Calculate average efficiency
            efficiency_ratios = [data['efficiency_ratio'] for data in cohort_metrics.values() if data['efficiency_ratio'] is not None]
            improvement_potentials = [data['improvement_potential'] for data in cohort_metrics.values()]
            
            if efficiency_ratios:
                time_to_pay_analysis['optimization_opportunities'][cohort] = {
                    'avg_efficiency_ratio': np.mean(efficiency_ratios),
                    'avg_improvement_potential': np.mean(improvement_potentials),
                    'efficiency_variance': np.var(efficiency_ratios),
                    'priority': 'HIGH' if np.mean(efficiency_ratios) > 1.5 else 'MEDIUM' if np.mean(efficiency_ratios) > 1.2 else 'LOW'
                }
        
        self.payment_analysis['time_to_pay'] = time_to_pay_analysis
        return time_to_pay_analysis
    
    def create_simple_dashboard(self):
        """Create simplified payment analysis dashboard"""
        print("💳 Creating Payment Analysis Dashboard...")
        
        time_data = self.payment_analysis['time_to_pay']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('💳 PAYMENT DEEP-DIVE ANALYSIS', fontsize=16, fontweight='bold')
        
        # 1. Top Efficiency Gaps
        ax1 = axes[0, 0]
        
        top_gaps = time_data['efficiency_gaps'][:8]
        
        if top_gaps:
            cohort_names = [gap['cohort'] for gap in top_gaps]
            gap_values = [gap['gap'] for gap in top_gaps]
            
            bars = ax1.barh(cohort_names, gap_values, color='lightcoral', alpha=0.8)
            ax1.set_title('Biggest Time-to-Pay Gaps', fontweight='bold')
            ax1.set_xlabel('Gap vs Benchmark (%)')
            ax1.grid(True, alpha=0.3)
        
        # 2. Priority Distribution
        ax2 = axes[0, 1]
        
        optimization_data = time_data['optimization_opportunities']
        
        if optimization_data:
            priorities = [data['priority'] for data in optimization_data.values()]
            priority_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for priority in priorities:
                priority_counts[priority] += 1
            
            priorities_list = list(priority_counts.keys())
            counts = list(priority_counts.values())
            colors = ['red', 'orange', 'green']
            
            bars = ax2.bar(priorities_list, counts, color=colors, alpha=0.8)
            ax2.set_title('Optimization Priority Distribution', fontweight='bold')
            ax2.set_ylabel('Number of Cohorts')
            ax2.grid(True, alpha=0.3)
        
        # 3. Metrics Coverage
        ax3 = axes[1, 0]
        
        categories = ['Payment', 'Engagement', 'Efficiency']
        counts = [len(self.payment_metrics), len(self.engagement_metrics), len(self.efficiency_metrics)]
        colors = ['gold', 'lightblue', 'lightgreen']
        
        bars = ax3.bar(categories, counts, color=colors, alpha=0.8)
        ax3.set_title('Analysis Coverage', fontweight='bold')
        ax3.set_ylabel('Number of Metrics')
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary Text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = "💳 PAYMENT ANALYSIS\nSUMMARY\n\n"
        
        if optimization_data:
            high_priority = [c for c, d in optimization_data.items() if d['priority'] == 'HIGH']
            summary_text += f"🚨 HIGH PRIORITY: {len(high_priority)}\n"
            for cohort in high_priority[:3]:
                summary_text += f"• {cohort}\n"
        
        if time_data['efficiency_gaps']:
            max_gap = max(time_data['efficiency_gaps'], key=lambda x: x['gap'])
            summary_text += f"\n🎯 BIGGEST GAP:\n{max_gap['cohort']}\n{max_gap['gap']:.0f}% slower"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('01_payment_analysis.png', dpi=300, bbox_inches='tight')
        print("💳 Payment analysis dashboard saved")
    
    def generate_simple_report(self):
        """Generate simplified payment analysis report"""
        print("�� Generating Payment Analysis Report...")
        
        report = []
        report.append("="*80)
        report.append("💳 PAYMENT DEEP-DIVE ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        # Executive Summary
        time_data = self.payment_analysis['time_to_pay']
        high_priority_count = len([c for c, d in time_data['optimization_opportunities'].items() if d['priority'] == 'HIGH'])
        
        report.append("🎯 EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"High Priority Issues: {high_priority_count} cohorts")
        
        if time_data['efficiency_gaps']:
            max_gap = max(time_data['efficiency_gaps'], key=lambda x: x['gap'])
            report.append(f"Biggest Gap: {max_gap['cohort']} ({max_gap['gap']:.1f}% slower)")
        
        report.append("")
        
        # Top Gaps
        report.append("🚨 TOP EFFICIENCY GAPS:")
        for i, gap in enumerate(time_data['efficiency_gaps'][:5], 1):
            report.append(f"{i}. {gap['cohort']}: {gap['gap']:.1f}% gap")
        
        report.append("")
        
        # Save report
        with open('payment_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("📋 Payment analysis report saved")
        return high_priority_count

def main():
    """Main function to run payment deep-dive analysis"""
    print("🚀 STARTING PAYMENT DEEP-DIVE ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = PaymentDeepDiveAnalysis('../Cohort Wise Analysis Fam 2.0 - Sheet1.csv')
    
    # Load and clean data
    df = analyzer.load_and_clean_data()
    
    # Extract payment metrics
    kpi_data = analyzer.extract_payment_metrics()
    
    # Run payment analyses
    time_to_pay_analysis = analyzer.analyze_time_to_pay_efficiency()
    
    # Create visualizations
    print(f"\n💳 Creating payment analysis visualizations...")
    analyzer.create_simple_dashboard()
    
    # Generate report
    high_priority_count = analyzer.generate_simple_report()
    
    print(f"\n✅ PAYMENT DEEP-DIVE ANALYSIS COMPLETED!")
    print(f"📁 Generated 1 PNG file + report")
    print(f"🚨 High Priority Issues: {high_priority_count} cohorts")

if __name__ == "__main__":
    main()
