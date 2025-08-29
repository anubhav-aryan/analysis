import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
from scipy import stats
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
        
    def load_and_clean_data(self) -> pd.DataFrame:
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
    
    def analyze_ticket_size_patterns(self):
        """Analyze ticket size vs engagement patterns"""
        print("💰 Analyzing ticket size patterns...")
        
        ticket_analysis = {
            'cohort_ticket_sizes': {},
            'size_engagement_relationship': {},
            'optimization_insights': {}
        }
        
        # Find ticket size metrics
        ticket_metrics = [m for m in self.kpi_data.keys() if 'Ticket Size' in m or 'ticket' in m.lower()]
        
        for metric in ticket_metrics:
            metric_data = self.kpi_data[metric]
            benchmark_value = metric_data.get(self.benchmark_cohort, None)
            
            for cohort in self.all_cohorts:
                if cohort != self.benchmark_cohort and cohort in metric_data:
                    ticket_size = metric_data[cohort]
                    
                    if cohort not in ticket_analysis['cohort_ticket_sizes']:
                        ticket_analysis['cohort_ticket_sizes'][cohort] = {}
                    
                    # Calculate vs benchmark
                    if benchmark_value and benchmark_value > 0:
                        vs_benchmark = ((ticket_size - benchmark_value) / benchmark_value) * 100
                    else:
                        vs_benchmark = 0
                    
                    ticket_analysis['cohort_ticket_sizes'][cohort][metric] = {
                        'value': ticket_size,
                        'vs_benchmark': vs_benchmark,
                        'tier': 'HIGH' if vs_benchmark > 20 else 'LOW' if vs_benchmark < -20 else 'MEDIUM'
                    }
        
        self.payment_analysis['ticket_size'] = ticket_analysis
        return ticket_analysis
    
    def create_time_to_pay_viz(self):
        """Create time-to-pay analysis visualization (PNG 1)"""
        print("⚡ Creating Time-to-Pay Analysis Visualization...")
        
        time_data = self.payment_analysis['time_to_pay']
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('⚡ TIME-TO-PAY EFFICIENCY ANALYSIS', fontsize=18, fontweight='bold')
        
        # 1. Top Efficiency Gaps
        ax1 = axes[0, 0]
        
        top_gaps = time_data['efficiency_gaps'][:8]
        
        if top_gaps:
            cohort_names = [gap['cohort'] for gap in top_gaps]
            gap_values = [gap['gap'] for gap in top_gaps]
            faster_slower = [gap['faster_slower'] for gap in top_gaps]
            
            # Color bars based on faster/slower
            colors = ['lightgreen' if fs == 'faster' else 'lightcoral' for fs in faster_slower]
            
            bars = ax1.barh(cohort_names, gap_values, color=colors, alpha=0.8)
            ax1.set_title('Biggest Time-to-Pay Gaps', fontweight='bold')
            ax1.set_xlabel('Gap vs Benchmark (%)')
            ax1.grid(True, alpha=0.3)
            
            # Add faster/slower labels
            for bar, fs, gap in zip(bars, faster_slower, gap_values):
                width = bar.get_width()
                ax1.text(width + gap*0.02, bar.get_y() + bar.get_height()/2.,
                        f'{fs}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # 2. Optimization Priority
        ax2 = axes[0, 1]
        
        optimization_data = time_data['optimization_opportunities']
        
        if optimization_data:
            cohorts = list(optimization_data.keys())
            priorities = [data['priority'] for data in optimization_data.values()]
            
            # Count by priority
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
            
            # Add count labels
            for bar, count in zip(bars, counts):
                if count > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(count)}', ha='center', va='bottom', fontsize=12)
        
        # 3. Payment Metrics Summary
        ax3 = axes[1, 0]
        
        # Show payment vs engagement vs efficiency metric counts
        categories = ['Payment', 'Engagement', 'Efficiency']
        counts = [len(self.payment_metrics), len(self.engagement_metrics), len(self.efficiency_metrics)]
        colors = ['gold', 'lightblue', 'lightgreen']
        
        bars = ax3.bar(categories, counts, color=colors, alpha=0.8)
        ax3.set_title('Metrics Categories Analyzed', fontweight='bold')
        ax3.set_ylabel('Number of Metrics')
        ax3.grid(True, alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=12)
        
        # 4. Optimization Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = "⚡ PAYMENT EFFICIENCY\nOPTIMIZATION SUMMARY\n\n"
        
        if optimization_data:
            # Find highest priority cohorts
            high_priority = [cohort for cohort, data in optimization_data.items() if data['priority'] == 'HIGH']
            medium_priority = [cohort for cohort, data in optimization_data.items() if data['priority'] == 'MEDIUM']
            
            summary_text += f"🚨 HIGH PRIORITY ({len(high_priority)}):\n"
            for cohort in high_priority[:3]:
                avg_ratio = optimization_data[cohort]['avg_efficiency_ratio']
                summary_text += f"• {cohort}: {avg_ratio:.2f}x slower\n"
            
            summary_text += f"\n⚠️ MEDIUM PRIORITY ({len(medium_priority)}):\n"
            for cohort in medium_priority[:3]:
                avg_ratio = optimization_data[cohort]['avg_efficiency_ratio']
                summary_text += f"• {cohort}: {avg_ratio:.2f}x slower\n"
        
        if time_data['efficiency_gaps']:
            max_gap = max(time_data['efficiency_gaps'], key=lambda x: x['gap'])
            summary_text += f"\n🏆 BIGGEST OPPORTUNITY:\n{max_gap['cohort']}\n"
            summary_text += f"{max_gap['gap']:.1f}% improvement potential"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('01_time_to_pay_analysis.png', dpi=300, bbox_inches='tight')
        print("⚡ Time-to-pay analysis visualization saved")
    
    def create_ticket_size_analysis_viz(self):
        """Create ticket size analysis visualization (PNG 2)"""
        print("💰 Creating Ticket Size Analysis Visualization...")
        
        ticket_data = self.payment_analysis['ticket_size']
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('💰 TICKET SIZE ANALYSIS', fontsize=18, fontweight='bold')
        
        # 1. Ticket Size Performance vs Benchmark
        ax1 = axes[0, 0]
        
        cohort_ticket_performance = {}
        for cohort, metrics in ticket_data['cohort_ticket_sizes'].items():
            avg_performance = np.mean([data['vs_benchmark'] for data in metrics.values()])
            cohort_ticket_performance[cohort] = avg_performance
        
        if cohort_ticket_performance:
            cohorts = list(cohort_ticket_performance.keys())
            performances = list(cohort_ticket_performance.values())
            
            # Color bars based on performance
            colors = ['darkgreen' if p > 20 else 'lightgreen' if p > 0 else 'lightcoral' if p > -20 else 'darkred' for p in performances]
            
            bars = ax1.bar(cohorts, performances, color=colors, alpha=0.8)
            ax1.set_title('Ticket Size Performance vs Benchmark', fontweight='bold')
            ax1.set_ylabel('Performance vs Benchmark (%)')
            ax1.tick_params(axis='x', rotation=45)
            plt.setp(ax1.get_xticklabels(), ha='right')
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, perf in zip(bars, performances):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + (2 if height >= 0 else -5),
                        f'{perf:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        # 2. Ticket Size Tiers Distribution
        ax2 = axes[0, 1]
        
        # Count cohorts by ticket size tier
        tier_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for cohort, metrics in ticket_data['cohort_ticket_sizes'].items():
            for metric, data in metrics.items():
                tier_counts[data['tier']] += 1
                break  # Only count each cohort once
        
        if sum(tier_counts.values()) > 0:
            tiers = list(tier_counts.keys())
            counts = list(tier_counts.values())
            colors = ['darkgreen', 'gold', 'lightcoral']
            
            bars = ax2.bar(tiers, counts, color=colors, alpha=0.8)
            ax2.set_title('Ticket Size Tier Distribution', fontweight='bold')
            ax2.set_ylabel('Number of Cohorts')
            ax2.grid(True, alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars, counts):
                if count > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(count)}', ha='center', va='bottom', fontsize=12)
        
        # 3. Top and Bottom Performers
        ax3 = axes[1, 0]
        
        if cohort_ticket_performance:
            # Sort by performance
            sorted_performance = sorted(cohort_ticket_performance.items(), key=lambda x: x[1], reverse=True)
            
            # Get top 5 and bottom 5
            top_performers = sorted_performance[:5]
            bottom_performers = sorted_performance[-5:]
            
            # Combine for visualization
            performers = top_performers + bottom_performers
            cohort_names = [item[0] for item in performers]
            performance_values = [item[1] for item in performers]
            
            # Color top vs bottom
            colors = ['darkgreen'] * len(top_performers) + ['lightcoral'] * len(bottom_performers)
            
            bars = ax3.barh(cohort_names, performance_values, color=colors, alpha=0.8)
            ax3.set_title('Top & Bottom Ticket Size Performers', fontweight='bold')
            ax3.set_xlabel('Performance vs Benchmark (%)')
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax3.grid(True, alpha=0.3)
            
            # Add performance values
            for bar, perf in zip(bars, performance_values):
                width = bar.get_width()
                ax3.text(width + (2 if width >= 0 else -2), bar.get_y() + bar.get_height()/2.,
                        f'{perf:+.1f}%', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
        
        # 4. Ticket Size Insights
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        insights_text = "💰 TICKET SIZE INSIGHTS\n\n"
        
        if cohort_ticket_performance:
            # Find best and worst performers
            best_performer = max(cohort_ticket_performance, key=cohort_ticket_performance.get)
            worst_performer = min(cohort_ticket_performance, key=cohort_ticket_performance.get)
            
            best_performance = cohort_ticket_performance[best_performer]
            worst_performance = cohort_ticket_performance[worst_performer]
            
            insights_text += f"🏆 TOP PERFORMER:\n{best_performer}\n{best_performance:+.1f}% vs benchmark\n\n"
            insights_text += f"⚠️ NEEDS ATTENTION:\n{worst_performer}\n{worst_performance:+.1f}% vs benchmark\n\n"
            
            # Calculate performance spread
            performance_spread = best_performance - worst_performance
            insights_text += f"📊 PERFORMANCE SPREAD:\n{performance_spread:.1f} percentage points\n\n"
            
            # Tier distribution insights
            high_tier = tier_counts.get('HIGH', 0)
            low_tier = tier_counts.get('LOW', 0)
            
            if high_tier > low_tier:
                insights_text += "✅ POSITIVE TREND:\nMore high-tier than low-tier\ncohorts"
            elif low_tier > high_tier:
                insights_text += "⚠️ OPTIMIZATION NEEDED:\nMore low-tier cohorts\nneed attention"
            else:
                insights_text += "📊 BALANCED DISTRIBUTION:\nEven spread across tiers"
        
        ax4.text(0.05, 0.95, insights_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('02_ticket_size_analysis.png', dpi=300, bbox_inches='tight')
        print("💰 Ticket size analysis visualization saved")
    
    def create_payment_dashboard(self):
        """Create comprehensive payment analysis dashboard (PNG 3)"""
        print("💳 Creating Payment Analysis Dashboard...")
        
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle('💳 COMPREHENSIVE PAYMENT ANALYSIS DASHBOARD', fontsize=18, fontweight='bold')
        
        # 1. Time-to-Pay Priority Matrix
        ax1 = axes[0, 0]
        
        time_data = self.payment_analysis['time_to_pay']
        optimization_data = time_data['optimization_opportunities']
        
        if optimization_data:
            priorities = [data['priority'] for data in optimization_data.values()]
            priority_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for priority in priorities:
                priority_counts[priority] += 1
            
            priorities_list = list(priority_counts.keys())
            counts = list(priority_counts.values())
            colors = ['red', 'orange', 'green']
            
            bars = ax1.bar(priorities_list, counts, color=colors, alpha=0.8)
            ax1.set_title('Time-to-Pay Priority Distribution', fontweight='bold')
            ax1.set_ylabel('Number of Cohorts')
            ax1.grid(True, alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars, counts):
                if count > 0:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(count)}', ha='center', va='bottom', fontsize=12)
        
        # 2. Ticket Size Tiers
        ax2 = axes[0, 1]
        
        ticket_data = self.payment_analysis['ticket_size']
        
        # Count cohorts by ticket size tier
        tier_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for cohort, metrics in ticket_data['cohort_ticket_sizes'].items():
            for metric, data in metrics.items():
                tier_counts[data['tier']] += 1
                break  # Only count each cohort once
        
        if sum(tier_counts.values()) > 0:
            tiers = list(tier_counts.keys())
            counts = list(tier_counts.values())
            colors = ['darkgreen', 'gold', 'lightcoral']
            
            wedges, texts, autotexts = ax2.pie(counts, labels=tiers, colors=colors, 
                                              autopct='%1.0f', startangle=90)
            ax2.set_title('Ticket Size Distribution', fontweight='bold')
        
        # 3. Metrics Coverage
        ax3 = axes[0, 2]
        
        categories = ['Payment', 'Engagement', 'Efficiency']
        counts = [len(self.payment_metrics), len(self.engagement_metrics), len(self.efficiency_metrics)]
        colors = ['gold', 'lightblue', 'lightgreen']
        
        bars = ax3.bar(categories, counts, color=colors, alpha=0.8)
        ax3.set_title('Analysis Coverage', fontweight='bold')
        ax3.set_ylabel('Number of Metrics')
        ax3.grid(True, alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=12)
        
        # 4. Top Efficiency Gaps
        ax4 = axes[1, 0]
        
        top_gaps = time_data['efficiency_gaps'][:6]
        
        if top_gaps:
            gap_cohorts = [gap['cohort'] for gap in top_gaps]
            gap_values = [gap['gap'] for gap in top_gaps]
            
            bars = ax4.barh(gap_cohorts, gap_values, color='lightcoral', alpha=0.8)
            ax4.set_title('Top Efficiency Gaps', fontweight='bold')
            ax4.set_xlabel('Gap vs Benchmark (%)')
            ax4.grid(True, alpha=0.3)
            
            # Add gap values
            for bar, gap in zip(bars, gap_values):
                width = bar.get_width()
                ax4.text(width + gap*0.02, bar.get_y() + bar.get_height()/2.,
                        f'{gap:.0f}%', ha='left', va='center', fontsize=9)
        
        # 5. Payment Performance Summary
        ax5 = axes[1, 1]
        ax5.axis('off')
        
        summary_text = "💳 PAYMENT PERFORMANCE\nSUMMARY\n\n"
        
        if optimization_data:
            high_priority = [c for c, d in optimization_data.items() if d['priority'] == 'HIGH']
            medium_priority = [c for c, d in optimization_data.items() if d['priority'] == 'MEDIUM']
            
            summary_text += f"🚨 HIGH PRIORITY: {len(high_priority)}\n"
            for cohort in high_priority[:3]:
                summary_text += f"• {cohort}\n"
            
            summary_text += f"\n⚠️ MEDIUM PRIORITY: {len(medium_priority)}\n"
            for cohort in medium_priority[:2]:
                summary_text += f"• {cohort}\n"
        
        if time_data['efficiency_gaps']:
            max_gap = max(time_data['efficiency_gaps'], key=lambda x: x['gap'])
            summary_text += f"\n🎯 BIGGEST GAP:\n{max_gap['cohort']}\n{max_gap['gap']:.0f}% slower"
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        
        # 6. Strategic Actions
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        strategy_text = "🎯 STRATEGIC ACTIONS\n\n"
        
        # Time-to-pay strategy
        if optimization_data:
            high_priority = [c for c, d in optimization_data.items() if d['priority'] == 'HIGH']
            if high_priority:
                strategy_text += "⚡ URGENT SPEED FIXES:\n"
                for cohort in high_priority[:2]:
                    strategy_text += f"• Optimize {cohort}\n"
                strategy_text += "\n"
        
        # Ticket size strategy
        ticket_data = self.payment_analysis['ticket_size']
        if 'cohort_ticket_sizes' in ticket_data:
            low_tier_cohorts = []
            for cohort, metrics in ticket_data['cohort_ticket_sizes'].items():
                for metric, data in metrics.items():
                    if data['tier'] == 'LOW':
                        low_tier_cohorts.append(cohort)
                    break
            
            if low_tier_cohorts:
                strategy_text += "💰 TICKET SIZE FOCUS:\n"
                for cohort in low_tier_cohorts[:2]:
                    strategy_text += f"• Boost {cohort} ticket size\n"
                strategy_text += "\n"
        
        strategy_text += "📊 MONITOR & TRACK:\n"
        strategy_text += "• Payment completion rates\n"
        strategy_text += "• Time-to-pay trends\n"
        strategy_text += "• Ticket size patterns"
        
        ax6.text(0.05, 0.95, strategy_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('03_payment_dashboard.png', dpi=300, bbox_inches='tight')
        print("💳 Payment analysis dashboard saved")
    
    def generate_payment_report(self):
        """Generate comprehensive payment analysis report"""
        print("📋 Generating Payment Deep-dive Report...")
        
        report = []
        report.append("="*100)
        report.append("💳 PAYMENT DEEP-DIVE ANALYSIS REPORT")
        report.append("="*100)
        report.append("")
        
        # Executive Summary
        report.append("🎯 EXECUTIVE SUMMARY")
        report.append("-" * 50)
        
        # Time-to-pay summary
        time_data = self.payment_analysis['time_to_pay']
        high_priority_count = len([c for c, d in time_data['optimization_opportunities'].items() if d['priority'] == 'HIGH'])
        
        report.append(f"⚡ PAYMENT EFFICIENCY:")
        report.append(f"   High Priority Issues: {high_priority_count} cohorts")
        
        if time_data['efficiency_gaps']:
            max_gap = max(time_data['efficiency_gaps'], key=lambda x: x['gap'])
            report.append(f"   Biggest Gap: {max_gap['cohort']} ({max_gap['gap']:.1f}% slower)")
        
        # Ticket size summary
        ticket_data = self.payment_analysis['ticket_size']
        if 'cohort_ticket_sizes' in ticket_data:
            total_cohorts = len(ticket_data['cohort_ticket_sizes'])
            report.append(f"\n💰 TICKET SIZE ANALYSIS:")
            report.append(f"   Cohorts Analyzed: {total_cohorts}")
        
        report.append("")
        
        # Detailed Time-to-Pay Analysis
        report.append("⚡ DETAILED TIME-TO-PAY ANALYSIS")
        report.append("-" * 50)
        
        report.append("🚨 TOP EFFICIENCY GAPS:")
        for i, gap in enumerate(time_data['efficiency_gaps'][:5], 1):
            report.append(f"{i}. {gap['cohort']}")
            report.append(f"   Gap: {gap['gap']:.1f}% ({gap['faster_slower']} than benchmark)")
            report.append(f"   Values: {gap['cohort_value']:.1f} vs {gap['benchmark_value']:.1f}")
        
        report.append("\n📊 OPTIMIZATION PRIORITIES:")
        for cohort, data in time_data['optimization_opportunities'].items():
            if data['priority'] in ['HIGH', 'MEDIUM']:
                report.append(f"   {data['priority']}: {cohort}")
                report.append(f"      Avg Efficiency Ratio: {data['avg_efficiency_ratio']:.2f}")
                report.append(f"      Improvement Potential: {data['avg_improvement_potential']:.1f}%")
        
        report.append("")
        
        # Ticket Size Analysis
        report.append("💰 TICKET SIZE ANALYSIS")
        report.append("-" * 50)
        
        if 'cohort_ticket_sizes' in ticket_data:
            # Calculate tier distribution
            tier_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            
            for cohort, metrics in ticket_data['cohort_ticket_sizes'].items():
                for metric, data in metrics.items():
                    tier_counts[data['tier']] += 1
                    break
            
            report.append("📊 Ticket Size Tier Distribution:")
            for tier, count in tier_counts.items():
                report.append(f"   {tier}: {count} cohorts")
        
        report.append("")
        
        # Strategic Recommendations
        report.append("💡 STRATEGIC RECOMMENDATIONS")
        report.append("-" * 50)
        
        report.append("1. TIME-TO-PAY OPTIMIZATION:")
        if high_priority_count > 0:
            report.append("   • URGENT: Address high-priority efficiency issues")
            high_priority_cohorts = [c for c, d in time_data['optimization_opportunities'].items() if d['priority'] == 'HIGH']
            report.append(f"   • Focus on: {', '.join(high_priority_cohorts[:3])}")
        else:
            report.append("   • Maintain current efficiency levels")
            report.append("   • Focus on incremental improvements")
        
        report.append("")
        report.append("2. TICKET SIZE STRATEGY:")
        if 'cohort_ticket_sizes' in ticket_data:
            low_tier_cohorts = []
            for cohort, metrics in ticket_data['cohort_ticket_sizes'].items():
                for metric, data in metrics.items():
                    if data['tier'] == 'LOW':
                        low_tier_cohorts.append(cohort)
                    break
            
            if low_tier_cohorts:
                report.append(f"   • BOOST: Low-tier cohorts need ticket size improvement")
                report.append(f"   • Target: {', '.join(low_tier_cohorts[:3])}")
            else:
                report.append("   • MAINTAIN: Good ticket size distribution")
        
        report.append("   • Monitor ticket size trends regularly")
        report.append("   • Balance ticket size with transaction frequency")
        
        report.append("")
        
        # Business Impact
        report.append("💼 BUSINESS IMPACT ASSESSMENT")
        report.append("-" * 50)
        
        # Calculate total improvement potential
        total_improvement_potential = 0
        improvement_count = 0
        
        for cohort, data in time_data['optimization_opportunities'].items():
            if data['priority'] in ['HIGH', 'MEDIUM']:
                total_improvement_potential += abs(data['avg_improvement_potential'])
                improvement_count += 1
        
        if improvement_count > 0:
            avg_improvement_potential = total_improvement_potential / improvement_count
            report.append(f"📈 Efficiency Improvement Potential:")
            report.append(f"   Average: {avg_improvement_potential:.1f}% across {improvement_count} cohorts")
            report.append(f"   High Priority Cohorts: {high_priority_count}")
        
        # Risk assessment
        risk_level = "HIGH" if high_priority_count > 3 else "MEDIUM" if high_priority_count > 1 else "LOW"
        report.append(f"\n⚠️ Risk Level: {risk_level}")
        
        if risk_level == "HIGH":
            report.append("   CRITICAL: Multiple cohorts have significant efficiency issues")
        elif risk_level == "MEDIUM":
            report.append("   MONITOR: Some cohorts need efficiency improvements")
        else:
            report.append("   STABLE: Payment efficiency generally good across cohorts")
        
        report.append("")
        
        # Save report
        with open('payment_deep_dive_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("📋 Payment deep-dive report saved")
        return high_priority_count, len(time_data['efficiency_gaps'])

def main():
    """Main function to run payment deep-dive analysis"""
    print("🚀 STARTING PAYMENT DEEP-DIVE ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = PaymentDeepDiveAnalysis('../Cohort Wise Analysis Fam 2.0 - Sheet1.csv')
    
    # Load and clean data
    df = analyzer.load_and_clean_data()
    
    # Extract payment metrics
    kpi_data = analyzer.extract_payment_metrics()
    
    # Run payment analyses
    time_to_pay_analysis = analyzer.analyze_time_to_pay_efficiency()
    ticket_size_analysis = analyzer.analyze_ticket_size_patterns()
    
    # Create visualizations
    print(f"\n💳 Creating payment analysis visualizations...")
    analyzer.create_time_to_pay_viz()              # PNG 1
    analyzer.create_ticket_size_analysis_viz()     # PNG 2
    analyzer.create_payment_dashboard()            # PNG 3
    
    # Generate report
    high_priority_count, total_gaps = analyzer.generate_payment_report()
    
    print(f"\n✅ PAYMENT DEEP-DIVE ANALYSIS COMPLETED!")
    print(f"📁 Generated 3 PNG files + comprehensive report")
    print(f"🚨 High Priority Issues: {high_priority_count} cohorts")
    print(f"⚡ Total Efficiency Gaps: {total_gaps}")

if __name__ == "__main__":
    main()