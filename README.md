# 📊 Comprehensive Cohort & KPI Analysis

## 🎯 Project Overview

This repository contains a comprehensive analysis of cohort-wise performance metrics for a digital payment/messaging application. The analysis includes detailed performance comparisons, KPI tier rankings, and strategic business insights across multiple user segments.

## 📁 Analysis Structure

### 🏠 Individual Cohort Analysis
Each cohort has its own detailed analysis folder containing:
- **5 PNG Visualizations**: Overall performance, Spotlight analysis, DM analysis, Payment analysis, Comprehensive dashboard
- **Business Report**: Detailed strategic insights and recommendations

**Cohorts Analyzed:**
- `18+` - Age group 18 and above
- `18-` - Age group under 18
- `Android` - Android platform users
- `iOS` - iOS platform users
- `Ultra Users` - Premium user segment
- `Rep Set` - Representative set users
- `PPI` - Prepaid instrument users
- `TPAP` - Third-party app provider users
- `Both` - Multi-feature users
- `SLS` - SLS feature users
- `DM` - Direct messaging users
- `Bubble` - Bubble feature users

### 🔄 Comparative Analysis
Head-to-head comparisons between related cohorts:
- **18+ vs 18- Analysis** - Age-based performance comparison
- **Android vs iOS Analysis** - Platform performance comparison
- **SLS vs DM vs Bubble Analysis** - Feature performance comparison
- **PPI vs TPAP vs Both Analysis** - Payment method comparison
- **Overall vs Ultra Users Analysis** - Premium vs average user comparison

### 📊 KPI Performance Tier Analysis
Comprehensive analysis of all KPIs across all cohorts:
- **Performance Heatmap** - Visual performance matrix
- **Champion Analysis** - Identification of top performers
- **Category Analysis** - KPI category breakdown
- **Tier Distribution** - Performance tier rankings
- **Strategic Dashboard** - Executive insights

## 🏆 Key Findings

### 🥇 Top Performers
- **"Both" Cohort**: 41.9% champion rate - Multi-feature users dominate
- **Rep Set**: 28.1% champion rate - Specialized excellence
- **18+**: 25.7% champion rate - Quality over quantity

### 🚨 Critical Insights
- **Platform Gap**: Android significantly outperforms iOS (+711% in user scanning)
- **Premium Problem**: Ultra Users underperform with only 3% champion rate
- **Feature Integration**: Multi-feature users ("Both") dramatically outperform single-feature users
- **Age Paradox**: 18+ excels while 18- has 0% champion rate despite larger size

### 📈 Biggest Performance Gaps
1. **User Scanning**: 740% gap (Android vs PPI)
2. **Spotlight Opening**: 675% gap (TPAP vs PPI)
3. **UPI from DM**: 255% gap (Android vs TPAP)
4. **DM Pins Adoption**: 252% gap (Rep Set vs TPAP)
5. **Bubble Transactions**: 231% gap (Rep Set vs iOS)

## 🛠️ Technical Implementation

### Python Scripts
- `comprehensive_kpi_analysis.py` - Main KPI tier analysis
- `generate_comparative_analyses.py` - Comparative analysis automation
- `generate_detailed_cohort_analyses.py` - Individual cohort analysis
- `overall_analysis.py` - Overall performance analysis
- `feature_analysis.py` - Feature-specific analysis

### Libraries Used
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical operations
- **matplotlib** - Data visualization
- **seaborn** - Enhanced visualizations

### Analysis Features
- **Performance Tier Classification**: Champion, Strong, Moderate, Weak, Critical
- **Benchmark Comparison**: All metrics compared against "Combine All" benchmark
- **Statistical Analysis**: Variance, correlation, and trend analysis
- **Business Intelligence**: Strategic recommendations and risk assessment

## 📋 Deliverables

### Visualizations (PNG Files)
Each analysis generates multiple high-resolution visualizations:
- Performance heatmaps
- Champion leaderboards
- Category breakdowns
- Tier distributions
- Strategic dashboards

### Reports (TXT Files)
Comprehensive business reports including:
- Executive summaries
- Performance rankings
- Strategic recommendations
- Risk assessments
- Action items

## 🎯 Business Impact

### Portfolio Health Score: 83.1/100
- **Excellence Rate**: 11.8% (Champion performance)
- **Risk Rate**: 16.9% (Critical performance - MEDIUM risk)

### Strategic Recommendations
1. **Scale "Both" Success** - Apply multi-feature practices across all cohorts
2. **Fix Platform Gaps** - Urgent Android-iOS optimization needed
3. **Optimize 18- Experience** - Huge user base with 0% champion rate
4. **Premium User Overhaul** - Ultra Users should be champions, not underperformers

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anubhav-aryan/analysis.git
   cd analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

3. **Run analysis**:
   ```bash
   # KPI Analysis
   cd "KPI Analysis"
   python3 comprehensive_kpi_analysis.py
   
   # Comparative Analysis
   python3 generate_comparative_analyses.py
   
   # Individual Cohort Analysis
   python3 generate_detailed_cohort_analyses.py
   ```

## 📊 Data Source

The analysis is based on the CSV file: `Cohort Wise Analysis Fam 2.0 - Sheet1.csv` containing:
- **49 KPIs** across multiple categories
- **13 cohorts** representing different user segments
- **Performance metrics** including engagement, payments, features, and user behavior

## 📈 Future Enhancements

- Real-time dashboard integration
- Automated alerting for performance drops
- Predictive analytics for cohort performance
- A/B testing framework integration
- Machine learning models for performance prediction

---

**📧 Contact**: For questions or collaboration, please reach out through GitHub issues or pull requests.

**📄 License**: This project is open source and available under the MIT License.
