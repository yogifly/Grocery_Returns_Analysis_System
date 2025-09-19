# Grocery Returns Analysis System

A comprehensivesolution for analyzing online grocery returns, identifying patterns in shelf life, seasonal/location vulnerabilities, and root causes of returns. The system helps inventory and procurement teams reduce wastage and improve freshness through data-driven insights.

## 🎯 Objectives

- Identify fruits/vegetables with high return rates and primary reasons for returns
- Detect seasonal and location-based patterns in product spoilage
- Estimate per-SKU shelf-life distribution for inventory decisions
- Provide actionable dashboard & reports for procurement, inventory, and delivery teams
- Suggest operational improvements (carriers, packaging, temperature control, reorder frequency)

## 🛠️ Tech Stack

- **Data Processing**: Python, Pandas, NumPy
- **Analytics**: Scikit-learn, SciPy, Lifelines (survival analysis)
- **NLP**: Custom classifier for return reason categorization
- **Visualization**: Plotly, Streamlit

## 📁 Project Structure

```bash
grocery-returns-analysis/
├── app.py                          # Main Streamlit dashboard
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── config/
│   └── settings.py                # Configuration settings
├── data/
│   ├── sample_data_generator.py   # Generate synthetic data
│   ├── orders.csv                 # Orders data 
│   ├── returns.csv                # Returns data 
│   └── products.csv               # Products data 
├── src/
│   ├── data_processor.py          # Data processing pipeline
│   ├── survival_analysis.py       # Shelf-life analysis
│   ├── predictive_models.py       # ML models for prediction
│   └── nlp_classifier.py          # Return reason classification
├── pages/
│   ├── 1_Data_Explorer.py         # Data exploration interface
│   
├── scripts/
│   └── generate_data.py           # Data generation script
├── utils/
│   └── report_generator.py        # Report generation utilities
└── models/                        # Trained models (generated)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
python scripts/generate_data.py
```

### 3. Launch Dashboard

```bash
streamlit run app.py
```

### 4. Access the Application

Open your browser and navigate to `http://localhost:8501`

## 📊 Features

### Main Dashboard
- **Executive Overview**: Key metrics, return rates, and trends
- **Deep Dive Analysis**: Product-specific and location-specific insights
- **Alerts & Recommendations**: Anomaly detection and actionable insights

### Data Explorer
- Raw data inspection and quality checks
- Statistical summaries and distributions
- Interactive filtering and exploration


## 📈 Key Metrics

- **Return Rate**: Percentage of orders returned
- **Stale Return Rate**: Percentage of returns due to spoilage
- **Time to Return**: Average days from delivery to return
- **Risk Score**: Composite score for inventory risk assessment
- **Shelf-Life Estimate**: Predicted product freshness duration

## 🎛️ Dashboard Filters

- Date range selection
- Product filtering (single or multiple)
- Location filtering (city-based)
- Carrier filtering
- Return reason filtering

## 📊 Visualizations

- Interactive Plotly charts and graphs
- Heatmaps for risk and return rate analysis
- Survival curves with confidence intervals
- Time-series plots with moving averages
- Statistical distribution plots
- Cohort analysis matrices

## 🔧 Configuration

Key settings can be modified in `config/settings.py`:

- Return rate alert thresholds
- Minimum sample sizes for analysis
- Color schemes for visualizations
- File paths and directories
- Analysis parameters


## 🎯 Business Impact

### Operational Improvements
- Identify problematic product-location combinations
- Optimize carrier selection based on performance data
- Improve packaging for temperature-sensitive products
- Adjust inventory levels based on shelf-life estimates

### Cost Savings
- Reduce return processing costs
- Minimize product waste and spoilage
- Optimize delivery routes and timing
- Improve customer satisfaction

### Data-Driven Decisions
- Evidence-based procurement strategies
- Seasonal adjustment of inventory levels
- Carrier contract negotiations with performance data
- Product quality improvement initiatives

## 🔮 Future Enhancements

- Integration with real-time inventory systems
- Weather data incorporation for spoilage prediction
- Computer vision for return photo analysis
- A/B testing framework for operational changes
- Real-time alerting system
- Mobile dashboard for field teams

## 📞 Support

For questions or issues:
1. Check the dashboard's built-in help sections
2. Review the code documentation
3. Examine the sample data structure
4. Test with different filter combinations


