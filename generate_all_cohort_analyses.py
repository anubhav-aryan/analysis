import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
import os
import shutil
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class UniversalCohortAnalysis:
    def __init__(self, file_path: str, cohort_name: str, folder_name: str):
        """Initialize the analyzer for any cohort"""
        self.file_path = file_path
        self.df = None
        self.cohort_name = cohort_name
        self.folder_name = folder_name
        self.benchmark_cohort = "Combine All"
        self.cohort_data = {}
        self.benchmark_data = {}
        self.analysis_results = {}
        
        # Color schemes for different cohorts
        self.color_schemes = {
            'Ultra Users': {'primary': 'gold', 'secondary': 'orange', 'accent': 'darkgoldenrod'},
            'Rep Set': {'primary': 'lightblue', 'secondary': 'blue', 'accent': 'navy'},
            'PPI': {'primary': 'lightgreen', 'secondary': 'green', 'accent': 'darkgreen'},
            'TPAP': {'primary': 'lightcoral', 'secondary': 'red', 'accent': 'darkred'},
            'Both': {'primary': 'mediumpurple', 'secondary': 'purple', 'accent': 'indigo'},
            'IOS ': {'primary': 'lightgray', 'secondary': 'gray', 'accent': 'black'},
            'Android': {'primary': 'lightsteelblue', 'secondary': 'steelblue', 'accent': 'darkslateblue'},
            'SLS ': {'primary': 'lightyellow', 'secondary': 'gold', 'accent': 'darkorange'},
            'DM': {'primary': 'lightpink', 'secondary': 'hotpink', 'accent': 'deeppink'},
            'Bubble': {'primary': 'lightcyan', 'secondary': 'cyan', 'accent': 'darkcyan'}
        }
        
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean the CSV data"""
        print(f"📊 Loading and cleaning data for {self.cohort_name} Cohort Analysis...")
        
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
        """Extract data for specific cohort and benchmark"""
        print(f"📈 Extracting {self.cohort_name} cohort data...")
        
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
            
            # Extract cohort value
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
        
        print(f"📊 Found {len(cohort_data)} metrics for {self.cohort_name} cohort")
        return cohort_data, benchmark_data
    
    def analyze_performance_section(self, section_name: str, metrics: Dict[str, str]):
        """Generic method to analyze any performance section"""
        print(f"📊 Analyzing {self.cohort_name} {section_name} Performance...")
        
        section_analysis = {}
        
        for metric, display_name in metrics.items():
            if metric in self.cohort_data and metric in self.benchmark_data:
                cohort_val = self.cohort_data[metric]
                benchmark_val = self.benchmark_data[metric]
                
                if benchmark_val != 0:
                    performance_ratio = cohort_val / benchmark_val
                    performance_pct = ((cohort_val - benchmark_val) / benchmark_val) * 100
                else:
                    performance_ratio = float('inf') if cohort_val > 0 else 1
                    performance_pct = 0
                
                section_analysis[display_name] = {
                    'cohort_value': cohort_val,
                    'benchmark_value': benchmark_val,
                    'performance_ratio': performance_ratio,
                    'performance_pct': performance_pct,
                    'metric_name': metric
                }
        
        return section_analysis
    
    def run_all_analyses(self):
        """Run all performance analyses"""
        
        # Define all metric sections
        sections = {
            'overall': {
                'Total Users': 'User Base',
                'DAU (increase %)': 'Daily Active Users',
                'DTU': 'Daily Transaction Users', 
                'Median SPV (TS)': 'Session Value',
                'Avg Time Spent per session': 'Session Duration',
                'user scanning / total users coming to home (user wise)': 'Scanning Adoption',
                'avg dm session per day': 'DM Engagement',
                'users opening spotlight / total users coming to home': 'Spotlight Adoption'
            },
            'spotlight': {
                'Daily absolute numbers for both swipe': 'Swipe Interactions',
                'Daily absolute numbers for both tap': 'Tap Interactions',
                'Avg. time from input entered → payment compose (search efficiency)': 'Search Efficiency',
                'Paste button shown ⇒ Paste button clicked': 'Paste Button CTR',
                '% of SS sessions where paste button was clicked (adoption)': 'Paste Adoption',
                '% of SS sessions where number button was clicked (adoption)': 'Number Button Adoption',
                'SS open ⇒ Recents clicks': 'Recents Usage',
                'Quick actions clicked vs In-app purchases clicked (ratio)': 'Quick Actions Ratio',
                '% of SS sessions with quick actions usage': 'Quick Actions Usage'
            },
            'dm': {
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
            },
            'home': {
                'Full Screen Scanner / Home Screen Scanner': 'Scanner Usage Ratio'
            },
            'bubble': {
                'Recent Ticket Size': 'Average Transaction Size',
                'No. of recent payment per recent click': 'Payment Conversion',
                'recent bubble txn %': 'Transaction Rate',
                '% of users using bubbles / total exposed users': 'Bubble Adoption'
            },
            'payment': {
                'DTU': 'Daily Transaction Users',
                'time to pay per user per pay session': 'Payment Completion Time',
                'opening the spotlight → Payment compose': 'Spotlight to Payment',
                '% of UPI transactions initiated from DM vs Home screen': 'DM Payment Usage',
                'Recent Ticket Size': 'Average Transaction Size',
                'recent bubble txn %': 'Bubble Transaction Rate'
            }
        }
        
        # Run all analyses
        for section_name, metrics in sections.items():
            self.analysis_results[section_name] = self.analyze_performance_section(section_name, metrics)
    
    def get_colors(self):
        """Get color scheme for this cohort"""
        return self.color_schemes.get(self.cohort_name, 
                                     {'primary': 'lightblue', 'secondary': 'blue', 'accent': 'darkblue'})
    
    def create_comprehensive_visualization(self):
        """Create comprehensive dashboard visualization"""
        print(f"📊 Creating {self.cohort_name} Comprehensive Dashboard...")
        
        colors = self.get_colors()
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'{self.cohort_name} Cohort: Comprehensive Performance Dashboard', fontsize=18, fontweight='bold')
        
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
        
        gauge_colors = plt.cm.RdYlGn(np.linspace(0, 1, 100))
        for i in range(len(theta)-1):
            ax1.fill_between([theta[i], theta[i+1]], 0, 1, color=gauge_colors[i], alpha=0.7)
        
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
            bar_colors = [colors['primary'] if score > 0 else 'lightcoral' for score in scores]
            
            bars = ax2.barh(sections, scores, color=bar_colors, alpha=0.7)
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
        
        strength_text = f"🏆 TOP 5 STRENGTHS\n{self.cohort_name.upper()}\n\n"
        for i, (metric, data) in enumerate(top_strengths, 1):
            clean_metric = metric.split('_', 1)[1] if '_' in metric else metric
            strength_text += f"{i}. {clean_metric[:30]}...\n   {data['performance_pct']:+.1f}%\n\n"
        
        ax3.text(0.1, 0.9, strength_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=colors['primary'], alpha=0.3))
        
        # 4. Biggest Opportunities
        ax4 = axes[1, 0]
        ax4.axis('off')
        opportunities = sorted(all_performance.items(), key=lambda x: x[1]['performance_pct'])[:5]
        
        opp_text = f"⚠️ TOP 5 OPPORTUNITIES\n{self.cohort_name.upper()}\n\n"
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
        pie_colors = ['darkgreen', 'lightgreen', 'orange', 'red']
        
        wedges, texts, autotexts = ax5.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Performance Distribution', fontweight='bold')
        
        # 6. Key Metrics Summary
        ax6 = axes[1, 2]
        if 'overall' in self.analysis_results and self.analysis_results['overall']:
            overall_data = self.analysis_results['overall']
            
            # Get top 3 overall metrics
            top_overall = sorted(overall_data.items(), key=lambda x: x[1]['performance_pct'], reverse=True)[:3]
            
            metrics = [item[0] for item in top_overall]
            cohort_values = [item[1]['cohort_value'] for item in top_overall]
            benchmark_values = [item[1]['benchmark_value'] for item in top_overall]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax6.bar(x - width/2, cohort_values, width, label=f'{self.cohort_name}', color=colors['primary'], alpha=0.7)
            bars2 = ax6.bar(x + width/2, benchmark_values, width, label='Benchmark', color='orange', alpha=0.7)
            
            ax6.set_title('Top 3 Overall Metrics', fontweight='bold')
            ax6.set_ylabel('Values')
            ax6.set_xticks(x)
            ax6.set_xticklabels(metrics, rotation=45, ha='right')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Business Intelligence
        ax7 = axes[2, 0]
        ax7.axis('off')
        
        bi_text = f"📊 {self.cohort_name.upper()}\nBUSINESS INTELLIGENCE\n\n"
        
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
                bbox=dict(boxstyle='round', facecolor=colors['secondary'], alpha=0.3))
        
        # 8. Section Performance Trend
        ax8 = axes[2, 1]
        if section_performance:
            sections = list(section_performance.keys())
            scores = list(section_performance.values())
            
            ax8.plot(sections, scores, 'o-', linewidth=2, markersize=8, color=colors['accent'])
            ax8.fill_between(sections, scores, alpha=0.3, color=colors['primary'])
            ax8.set_title('Performance Trend Across Sections', fontweight='bold')
            ax8.set_ylabel('Performance (%)')
            ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax8.tick_params(axis='x', rotation=45)
            plt.setp(ax8.get_xticklabels(), ha='right')
            ax8.grid(True, alpha=0.3)
        
        # 9. Risk Assessment
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        risk_text = f"🎯 {self.cohort_name.upper()}\nRISK ASSESSMENT\n\n"
        
        # Calculate risk level
        if avg_performance < -20:
            risk_level = "HIGH RISK"
            risk_color = "red"
            risk_text += "🔴 HIGH RISK\nSignificant issues\n\n"
        elif avg_performance < -10:
            risk_level = "MEDIUM RISK"
            risk_color = "orange"
            risk_text += "🟠 MEDIUM RISK\nNeeds attention\n\n"
        elif avg_performance < 0:
            risk_level = "LOW RISK"
            risk_color = "yellow"
            risk_text += "🟡 LOW RISK\nMinor improvements\n\n"
        else:
            risk_level = "PERFORMING WELL"
            risk_color = "green"
            risk_text += "🟢 PERFORMING WELL\nBuild on strengths\n\n"
        
        # Add specific recommendations
        worst_section = min(section_performance.items(), key=lambda x: x[1]) if section_performance else None
        if worst_section and worst_section[1] < -10:
            risk_text += f"PRIORITY:\nFix {worst_section[0]}\n({worst_section[1]:.1f}%)"
        
        ax9.text(0.1, 0.9, risk_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=risk_color, alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f'{self.folder_name}/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"📊 {self.cohort_name} comprehensive dashboard saved")
    
    def generate_business_report(self):
        """Generate comprehensive business report"""
        print(f"📋 Generating {self.cohort_name} Business Report...")
        
        report = []
        report.append("="*80)
        report.append(f"📊 {self.cohort_name.upper()} COHORT COMPREHENSIVE BUSINESS ANALYSIS REPORT")
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
            report.append(f"{self.cohort_name} Cohort Size: {user_base:,.0f} users")
            
            if avg_performance < -20:
                report.append("Risk Level: HIGH - Significant user experience issues")
            elif avg_performance < 0:
                report.append("Risk Level: MEDIUM - Performance gaps affecting engagement")
            else:
                report.append("Risk Level: LOW - Stable performance with growth opportunities")
        
        report.append("")
        
        # Save report
        safe_filename = self.cohort_name.replace(' ', '_').replace('/', '_').lower()
        with open(f'{self.folder_name}/{safe_filename}_business_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print(f"📋 {self.cohort_name} business report saved")
        
        return avg_performance, positive_count, total_count
    
    def create_summary_visualization(self):
        """Create a summary performance visualization"""
        print(f"📊 Creating {self.cohort_name} Summary Visualization...")
        
        colors = self.get_colors()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.cohort_name} Cohort: Performance Summary', fontsize=16, fontweight='bold')
        
        # 1. Overall Performance vs Benchmark
        ax1 = axes[0, 0]
        if 'overall' in self.analysis_results and self.analysis_results['overall']:
            overall_data = self.analysis_results['overall']
            metrics = list(overall_data.keys())
            performance_pcts = [data['performance_pct'] for data in overall_data.values()]
            bar_colors = [colors['primary'] if pct > 0 else 'lightcoral' for pct in performance_pcts]
            
            bars = ax1.barh(metrics, performance_pcts, color=bar_colors, alpha=0.7)
            ax1.set_title('Overall Performance vs Benchmark (%)', fontweight='bold')
            ax1.set_xlabel('Performance Difference (%)')
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, pct in zip(bars, performance_pcts):
                width = bar.get_width()
                ax1.text(width + (5 if width >= 0 else -5), bar.get_y() + bar.get_height()/2.,
                        f'{pct:+.1f}%', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
        
        # 2. Section Performance Summary
        ax2 = axes[0, 1]
        section_performance = {}
        for section, data in self.analysis_results.items():
            if data:
                section_scores = [perf_data['performance_pct'] for perf_data in data.values()]
                section_performance[section.title()] = np.mean(section_scores)
        
        if section_performance:
            sections = list(section_performance.keys())
            scores = list(section_performance.values())
            bar_colors = [colors['secondary'] if score > 0 else 'lightcoral' for score in scores]
            
            bars = ax2.bar(sections, scores, color=bar_colors, alpha=0.7)
            ax2.set_title('Performance by Section', fontweight='bold')
            ax2.set_ylabel('Average Performance (%)')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            plt.setp(ax2.get_xticklabels(), ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + (2 if height >= 0 else -4),
                        f'{score:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # 3. Top Strengths and Opportunities
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        # Collect all performance data for top/bottom analysis
        all_performance = {}
        for section, data in self.analysis_results.items():
            for metric, perf_data in data.items():
                all_performance[f"{section}_{metric}"] = perf_data
        
        if all_performance:
            top_strengths = sorted(all_performance.items(), key=lambda x: x[1]['performance_pct'], reverse=True)[:5]
            opportunities = sorted(all_performance.items(), key=lambda x: x[1]['performance_pct'])[:5]
            
            summary_text = f"{self.cohort_name.upper()} SUMMARY\n\n"
            summary_text += "🏆 TOP STRENGTHS:\n"
            for i, (metric, data) in enumerate(top_strengths, 1):
                clean_metric = metric.split('_', 1)[1] if '_' in metric else metric
                summary_text += f"{i}. {clean_metric[:25]}...\n   {data['performance_pct']:+.1f}%\n"
            
            summary_text += "\n⚠️ TOP OPPORTUNITIES:\n"
            for i, (metric, data) in enumerate(opportunities, 1):
                clean_metric = metric.split('_', 1)[1] if '_' in metric else metric
                summary_text += f"{i}. {clean_metric[:25]}...\n   {data['performance_pct']:+.1f}%\n"
            
            ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor=colors['primary'], alpha=0.3))
        
        # 4. Performance Score Gauge
        ax4 = axes[1, 1]
        if all_performance:
            performance_scores = [data['performance_pct'] for data in all_performance.values()]
            avg_performance = np.mean(performance_scores)
            positive_count = sum(1 for score in performance_scores if score > 0)
            total_count = len(performance_scores)
            
            # Create a simple performance indicator
            ax4.pie([positive_count, total_count - positive_count], 
                   labels=[f'Above Benchmark\n({positive_count})', f'Below Benchmark\n({total_count - positive_count})'],
                   colors=[colors['primary'], 'lightcoral'], autopct='%1.1f%%', startangle=90)
            ax4.set_title(f'Overall Score: {avg_performance:.1f}%', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.folder_name}/performance_summary.png', dpi=300, bbox_inches='tight')
        print(f"📊 {self.cohort_name} summary visualization saved")

def generate_cohort_analysis(cohort_name: str, folder_name: str):
    """Generate complete analysis for a specific cohort"""
    print(f"\n{'='*60}")
    print(f"🚀 STARTING {cohort_name.upper()} COHORT ANALYSIS")
    print(f"{'='*60}")
    
    # Initialize analyzer
    analyzer = UniversalCohortAnalysis('Cohort Wise Analysis Fam 2.0 - Sheet1.csv', cohort_name, folder_name)
    
    # Load and clean data
    df = analyzer.load_and_clean_data()
    
    # Extract cohort data
    cohort_data, benchmark_data = analyzer.extract_cohort_data()
    
    if not cohort_data:
        print(f"❌ No data found for {cohort_name} cohort")
        return None
    
    # Run all analyses
    print(f"\n📊 Running comprehensive analysis for {cohort_name}...")
    analyzer.run_all_analyses()
    
    # Create visualizations
    print(f"\n📊 Creating visualizations for {cohort_name}...")
    analyzer.create_summary_visualization()
    analyzer.create_comprehensive_visualization()
    
    # Generate business report
    avg_performance, positive_count, total_count = analyzer.generate_business_report()
    
    print(f"\n✅ {cohort_name} Cohort Analysis Complete!")
    print(f"📊 Overall Performance: {avg_performance:.1f}%")
    print(f"📈 Success Rate: {positive_count}/{total_count} metrics ({(positive_count/total_count)*100:.1f}%)")
    
    return {
        'cohort_name': cohort_name,
        'avg_performance': avg_performance,
        'success_rate': (positive_count/total_count)*100,
        'positive_count': positive_count,
        'total_count': total_count
    }

def main():
    """Main function to generate all cohort analyses"""
    print("🚀 STARTING COMPREHENSIVE COHORT ANALYSIS FOR ALL COHORTS")
    print("="*80)
    
    # Define all cohorts to analyze
    cohorts = [
        ('Ultra Users', 'Ultra Users'),
        ('Rep Set', 'Rep Set'),
        ('PPI', 'PPI'),
        ('TPAP', 'TPAP'),
        ('Both', 'Both'),
        ('IOS ', 'iOS'),  # Note: CSV has space after IOS
        ('Android', 'Android'),
        ('SLS ', 'SLS'),   # Note: CSV has space after SLS
        ('DM', 'DM'),
        ('Bubble', 'Bubble')
    ]
    
    results = []
    
    for cohort_name, folder_name in cohorts:
        try:
            result = generate_cohort_analysis(cohort_name, folder_name)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Error analyzing {cohort_name}: {str(e)}")
            continue
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY REPORT - ALL COHORTS")
    print(f"{'='*80}")
    
    if results:
        print("\n🏆 COHORT PERFORMANCE RANKING:")
        results.sort(key=lambda x: x['avg_performance'], reverse=True)
        
        for i, result in enumerate(results, 1):
            status = "🟢" if result['avg_performance'] > 10 else "🟡" if result['avg_performance'] > 0 else "🟠" if result['avg_performance'] > -10 else "🔴"
            print(f"{i:2d}. {status} {result['cohort_name']:15} | {result['avg_performance']:+6.1f}% | {result['success_rate']:5.1f}% success | {result['positive_count']:2d}/{result['total_count']:2d} metrics")
        
        # Calculate overall statistics
        avg_performance_all = np.mean([r['avg_performance'] for r in results])
        avg_success_rate = np.mean([r['success_rate'] for r in results])
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   • Average Performance Across All Cohorts: {avg_performance_all:.1f}%")
        print(f"   • Average Success Rate: {avg_success_rate:.1f}%")
        print(f"   • Total Cohorts Analyzed: {len(results)}")
        print(f"   • Cohorts Above Benchmark: {sum(1 for r in results if r['avg_performance'] > 0)}")
        print(f"   • Cohorts Needing Attention: {sum(1 for r in results if r['avg_performance'] < -10)}")
    
    print("\n✅ ALL COHORT ANALYSES COMPLETED!")
    print("📁 Check individual cohort folders for detailed reports and visualizations")

if __name__ == "__main__":
    main()
