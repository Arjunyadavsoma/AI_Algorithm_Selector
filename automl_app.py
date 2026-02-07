"""
Professional AutoML Dashboard
Modern Streamlit interface for automated machine learning
Author: AutoML Pro
Version: 3.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import time
from datetime import datetime
import automl_engine as aml

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AutoML Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown("""
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Headers */
    h1 {
        color: #1a202c;
        font-weight: 700;
        padding-bottom: 1rem;
        border-bottom: 4px solid #667eea;
    }
    
    h2 {
        color: #2d3748;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    h3 {
        color: #4a5568;
        font-weight: 500;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border-left-width: 5px;
    }
    
    /* Tables */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 12px 12px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        color: #4a5568;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #667eea;
        box-shadow: 0 -2px 8px rgba(102, 126, 234, 0.2);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 12px;
        font-weight: 600;
        color: #2d3748;
    }
    
    /* Success/Warning/Error messages */
    .element-container .stSuccess {
        background-color: #d1fae5;
        color: #065f46;
        border-left-color: #10b981;
    }
    
    .element-container .stWarning {
        background-color: #fef3c7;
        color: #92400e;
        border-left-color: #f59e0b;
    }
    
    .element-container .stError {
        background-color: #fee2e2;
        color: #991b1b;
        border-left-color: #ef4444;
    }
    
    .element-container .stInfo {
        background-color: #dbeafe;
        color: #1e40af;
        border-left-color: #3b82f6;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'df': None,
        'target_col': None,
        'automl': None,
        'results_df': None,
        'best_model': None,
        'dataset_info': None,
        'problem_type': None,
        'X_processed': None,
        'y_processed': None,
        'preprocessor': None,
        'label_encoder': None,
        'y_test': None,
        'y_pred': None,
        'training_complete': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def create_metric_card(label, value, delta=None):
    """Create styled metric card"""
    st.metric(label=label, value=value, delta=delta)

def format_large_number(num):
    """Format large numbers"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)

# =====================================================
# VISUALIZATION FUNCTIONS
# =====================================================

def plot_model_comparison(results_df, problem_type):
    """Create interactive model comparison chart"""
    metric_col = 'Accuracy' if problem_type == aml.ProblemType.CLASSIFICATION else 'R² Score'
    
    # Sort by metric
    results_sorted = results_df.sort_values(metric_col, ascending=True)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    colors = ['#ef4444' if 'Ensemble' in alg else '#667eea' for alg in results_sorted['Algorithm']]
    
    fig.add_trace(go.Bar(
        y=results_sorted['Algorithm'],
        x=results_sorted[metric_col],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=results_sorted[metric_col].apply(lambda x: f"{x:.4f}"),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>' + metric_col + ': %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'Model Performance Comparison ({metric_col})',
            font=dict(size=20, color='#1a202c', family='Inter')
        ),
        xaxis_title=metric_col,
        yaxis_title="Model",
        template='plotly_white',
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def plot_metrics_radar(results_df, problem_type):
    """Create radar chart for top 3 models"""
    if problem_type == aml.ProblemType.CLASSIFICATION:
        metric1, metric2 = 'Accuracy', 'F1 Score'
    else:
        metric1, metric2 = 'R² Score', 'MAE'
    
    top_3 = results_df.head(3)
    
    fig = go.Figure()
    
    categories = [metric1, metric2, 'Speed']
    
    for idx, row in top_3.iterrows():
        # Normalize speed (inverse of time)
        time_val = float(row['Training Time'].replace('s', ''))
        max_time = float(results_df['Training Time'].str.replace('s', '').astype(float).max())
        speed_score = 1 - (time_val / max_time)
        
        values = [
            row[metric1],
            row[metric2] if metric2 == 'F1 Score' else 1 - min(row[metric2], 1),
            speed_score
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=row['Algorithm']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title='Top 3 Models - Multi-Metric Comparison',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = [f"Class {i}" for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 14},
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance if available"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importances[indices],
            y=[feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in indices],
            orientation='h',
            marker=dict(
                color=importances[indices],
                colorscale='Viridis',
                line=dict(color='white', width=1)
            ),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importances',
            xaxis_title='Importance',
            yaxis_title='Feature',
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        return fig
    
    return None

def plot_prediction_distribution(y_true, y_pred, problem_type):
    """Plot prediction distribution"""
    if problem_type == aml.ProblemType.REGRESSION:
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Actual vs Predicted', 'Residuals'))
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(x=y_true, y=y_pred, mode='markers',
                       marker=dict(color='#667eea', opacity=0.6),
                       name='Predictions',
                       hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                       mode='lines', name='Perfect Prediction',
                       line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Residuals
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers',
                       marker=dict(color='#764ba2', opacity=0.6),
                       name='Residuals',
                       hovertemplate='Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'),
            row=1, col=2
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        fig.update_xaxes(title_text="Actual Value", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Value", row=1, col=1)
        fig.update_xaxes(title_text="Predicted Value", row=1, col=2)
        fig.update_yaxes(title_text="Residual", row=1, col=2)
        
        fig.update_layout(
            title='Prediction Analysis',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        
        return fig
    
    return None

# =====================================================
# MAIN APPLICATION
# =====================================================

def main():
    """Main application function"""
    
    # Header
    st.title("🤖 AutoML Pro")
    st.markdown("### Automated Machine Learning Made Simple")
    st.markdown("Upload your data, select target, and let AI find the best model automatically!")
    st.markdown("---")
    
    # =====================================================
    # SIDEBAR
    # =====================================================
    
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # File Upload
        uploaded_file = st.file_uploader(
            "📁 Upload Dataset (CSV)",
            type=['csv'],
            help="Upload your training data in CSV format"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
        
        st.markdown("---")
        
        # Target Selection
        if st.session_state.df is not None:
            target_col = st.selectbox(
                "🎯 Select Target Variable",
                options=st.session_state.df.columns.tolist(),
                help="Choose the column you want to predict"
            )
            st.session_state.target_col = target_col
            
            st.markdown("---")
            
            # Optimization Mode
            mode_options = {
                "⚡ Fast (Quick Results)": aml.OptimizationMode.FAST,
                "⚖️ Balanced (Recommended)": aml.OptimizationMode.BALANCED,
                "🎯 Precise (Best Accuracy)": aml.OptimizationMode.PRECISE
            }
            
            selected_mode = st.radio(
                "🔧 Training Mode",
                options=list(mode_options.keys()),
                index=1,
                help="Fast: 4-5 models, quick training\nBalanced: 6-7 models, good accuracy\nPrecise: 7-8 models, best results but slower"
            )
            
            optimization_mode = mode_options[selected_mode]
            
            st.markdown("---")
            
            # Advanced Settings
            with st.expander("🔬 Advanced Settings"):
                test_size = st.slider(
                    "Test Set Size (%)",
                    min_value=10,
                    max_value=40,
                    value=20,
                    step=5,
                    help="Percentage of data to use for testing"
                ) / 100.0
                
                show_advanced_viz = st.checkbox(
                    "Show Advanced Visualizations",
                    value=True,
                    help="Display additional charts and metrics"
                )
            
            st.markdown("---")
            
            # Action Button
            if st.button("🚀 Start AutoML Training", use_container_width=True):
                st.session_state.training_complete = False
                
                with st.spinner("🔄 Initializing AutoML..."):
                    # Create AutoML instance
                    automl = aml.AutoML(mode=optimization_mode)
                    st.session_state.automl = automl
                    
                    # Analyze dataset
                    dataset_info = automl.analyze(df, target_col)
                    st.session_state.dataset_info = dataset_info
                    st.session_state.problem_type = automl.problem_type
                    
                    # Show dataset info
                    st.sidebar.success(f"📊 Problem Type: **{automl.problem_type.value}**")
                    
                    # Preprocess data
                    X, y, preprocessor, label_encoder = automl.preprocessor.preprocess(
                        df, target_col, automl.problem_type
                    )
                    
                    st.session_state.X_processed = X
                    st.session_state.y_processed = y
                    st.session_state.preprocessor = preprocessor
                    st.session_state.label_encoder = label_encoder
                
                # Training progress in main area
                st.markdown("## 🏋️ Training Progress")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                progress_container = st.container()
                
                with progress_container:
                    col1, col2, col3 = st.columns(3)
                    metric_model = col1.empty()
                    metric_time = col2.empty()
                    metric_status = col3.empty()
                
                # Train models
                for update in automl.fit(df, target_col, test_size):
                    if update.get('type') == 'final':
                        # Store final results
                        st.session_state.results_df = update['results']
                        st.session_state.best_model = update['best_model']
                        st.session_state.y_test = update['y_test']
                        st.session_state.y_pred = update['y_pred']
                        st.session_state.training_complete = True
                        
                        progress_bar.progress(100)
                        status_text.success("✅ Training Complete!")
                        
                    else:
                        # Update progress
                        progress_pct = int((update['step'] / update['total']) * 100)
                        progress_bar.progress(progress_pct)
                        status_text.write(f"Training model {update['step']}/{update['total']}: **{update['model_name']}**")
                        
                        metric_model.metric("Current Model", update['model_name'])
                        metric_time.metric("Time", f"{update.get('time', 0):.2f}s")
                        metric_status.metric("Status", update['status'])
                        
                        time.sleep(0.1)  # Small delay for visual feedback
    
    # =====================================================
    # MAIN CONTENT
    # =====================================================
    
    if st.session_state.df is None:
        # Welcome Screen
        st.markdown("""
        ## 👋 Welcome to AutoML Pro!
        
        ### Get Started in 3 Easy Steps:
        
        1. **📁 Upload Your Data**
           - Click "Upload Dataset" in the sidebar
           - Supports CSV files
           - Works with any tabular data
        
        2. **🎯 Select Target Variable**
           - Choose the column you want to predict
           - AutoML will automatically detect if it's classification or regression
        
        3. **🚀 Start Training**
           - Click "Start AutoML Training"
           - Watch as multiple models compete
           - Get the best model automatically!
        
        ### ✨ Features:
        
        - 🤖 **Automatic Model Selection**: Tests 4-8 algorithms automatically
        - 📊 **Smart Preprocessing**: Handles missing values, encoding, scaling
        - 🏆 **Ensemble Learning**: Creates powerful model combinations
        - 📈 **Visual Analytics**: Interactive charts and comparisons
        - ⚡ **Hyperparameter Tuning**: Fine-tune models for best performance
        - 💾 **Export Models**: Download trained models for deployment
        
        ---
        
        ### 🎓 Supported Algorithms:
        
        **Classification:**
        - Logistic Regression
        - Decision Trees
        - Random Forests
        - Gradient Boosting
        - Support Vector Machines
        - K-Nearest Neighbors
        - Naive Bayes
        - Smart Ensemble
        
        **Regression:**
        - Linear Regression
        - Ridge / Lasso
        - Decision Trees
        - Random Forests
        - Gradient Boosting
        - Support Vector Regression
        - K-Nearest Neighbors
        - Smart Ensemble
        
        ---
        
        **Ready to begin?** Upload your dataset using the sidebar! 👉
        """)
        
        # Example datasets
        with st.expander("📚 Need Sample Data? Try These Examples"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **Classification Example**
                - Iris Dataset
                - Predict flower species
                - 150 rows, 4 features
                """)
            
            with col2:
                st.markdown("""
                **Regression Example**
                - House Prices
                - Predict sale price
                - 1000+ rows, 10+ features
                """)
            
            with col3:
                st.markdown("""
                **Binary Classification**
                - Titanic Survival
                - Predict survival
                - 891 rows, 11 features
                """)
    
    elif not st.session_state.training_complete:
        # Show dataset preview
        st.markdown("## 📊 Dataset Preview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Rows", format_large_number(len(st.session_state.df)))
        with col2:
            create_metric_card("Columns", len(st.session_state.df.columns))
        with col3:
            numeric_cols = len(st.session_state.df.select_dtypes(include=['number']).columns)
            create_metric_card("Numeric Features", numeric_cols)
        with col4:
            cat_cols = len(st.session_state.df.select_dtypes(include=['object']).columns)
            create_metric_card("Categorical Features", cat_cols)
        
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        # Dataset info
        if st.session_state.dataset_info:
            with st.expander("📋 Dataset Statistics", expanded=True):
                info = st.session_state.dataset_info
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Feature Types:**")
                    st.write(f"- Numeric: {info.feature_types.get('numeric', 0)}")
                    st.write(f"- Categorical: {info.feature_types.get('categorical', 0)}")
                    
                    if info.n_classes:
                        st.markdown(f"**Number of Classes:** {info.n_classes}")
                
                with col2:
                    if info.missing_values:
                        st.markdown("**Missing Values:**")
                        for col, count in list(info.missing_values.items())[:5]:
                            st.write(f"- {col}: {count}")
                    else:
                        st.success("✅ No missing values detected!")
        
        st.info("👆 Configure settings in the sidebar and click 'Start AutoML Training' to begin!")
    
    else:
        # Training complete - show results
        results_df = st.session_state.results_df
        problem_type = st.session_state.problem_type
        best_model = st.session_state.best_model
        
        # =====================================================
        # KEY METRICS
        # =====================================================
        
        st.markdown("## 🏆 Best Model Performance")
        
        best_result = results_df.iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🥇 Winner",
                best_result['Algorithm'],
                delta="Best Performance"
            )
        
        if problem_type == aml.ProblemType.CLASSIFICATION:
            with col2:
                st.metric("Accuracy", f"{best_result['Accuracy']:.4f}")
            with col3:
                st.metric("F1 Score", f"{best_result['F1 Score']:.4f}")
        else:
            with col2:
                st.metric("R² Score", f"{best_result['R² Score']:.4f}")
            with col3:
                st.metric("MAE", f"{best_result['MAE']:.4f}")
        
        with col4:
            st.metric("Training Time", best_result['Training Time'])
        
        st.markdown("---")
        
        # =====================================================
        # TABS
        # =====================================================
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Model Comparison",
            "📈 Visualizations",
            "🔬 Model Details",
            "⚡ Hyperparameter Tuning",
            "💾 Export & Deploy"
        ])
        
        # ==================== TAB 1: COMPARISON ====================
        with tab1:
            st.markdown("### All Model Results")
            
            # Results table
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Comparison chart
            st.plotly_chart(
                plot_model_comparison(results_df, problem_type),
                use_container_width=True
            )
            
            # Radar chart
            if len(results_df) >= 3:
                st.plotly_chart(
                    plot_metrics_radar(results_df, problem_type),
                    use_container_width=True
                )
        
        # ==================== TAB 2: VISUALIZATIONS ====================
        with tab2:
            st.markdown("### Model Performance Visualizations")
            
            if problem_type == aml.ProblemType.CLASSIFICATION and st.session_state.y_test is not None:
                # Confusion Matrix
                st.plotly_chart(
                    plot_confusion_matrix(
                        st.session_state.y_test,
                        st.session_state.y_pred,
                        labels=st.session_state.label_encoder.classes_ if st.session_state.label_encoder else None
                    ),
                    use_container_width=True
                )
                
                # Classification Report
                from sklearn.metrics import classification_report
                report = classification_report(
                    st.session_state.y_test,
                    st.session_state.y_pred,
                    target_names=st.session_state.label_encoder.classes_ if st.session_state.label_encoder else None,
                    output_dict=True
                )
                
                report_df = pd.DataFrame(report).transpose()
                st.markdown("#### Classification Report")
                st.dataframe(report_df.round(3), use_container_width=True)
            
            elif problem_type == aml.ProblemType.REGRESSION and st.session_state.y_test is not None:
                # Prediction plots
                fig = plot_prediction_distribution(
                    st.session_state.y_test,
                    st.session_state.y_pred,
                    problem_type
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
                
                mse = mean_squared_error(st.session_state.y_test, st.session_state.y_pred)
                rmse = np.sqrt(mse)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col2:
                    st.metric("MSE", f"{mse:.4f}")
                with col3:
                    try:
                        mape = mean_absolute_percentage_error(st.session_state.y_test, st.session_state.y_pred)
                        st.metric("MAPE", f"{mape:.2%}")
                    except:
                        pass
            
            # Feature Importance
            if hasattr(best_model, 'feature_importances_'):
                feature_names = st.session_state.automl.preprocessor.feature_names
                fig_importance = plot_feature_importance(best_model, feature_names)
                if fig_importance:
                    st.plotly_chart(fig_importance, use_container_width=True)
        
        # ==================== TAB 3: MODEL DETAILS ====================
        with tab3:
            st.markdown("### Best Model Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Type:**")
                st.code(type(best_model).__name__)
                
                st.markdown("**Parameters:**")
                params = best_model.get_params()
                st.json({k: str(v) for k, v in list(params.items())[:10]})
            
            with col2:
                st.markdown("**Model Summary:**")
                st.write(f"- Algorithm: {best_result['Algorithm']}")
                st.write(f"- Training Time: {best_result['Training Time']}")
                st.write(f"- Problem Type: {problem_type.value}")
                
                if hasattr(best_model, 'n_features_in_'):
                    st.write(f"- Features Used: {best_model.n_features_in_}")
        
        # ==================== TAB 4: TUNING ====================
        with tab4:
            st.markdown("### Hyperparameter Optimization")
            st.info("🎯 Fine-tune a specific model to potentially improve performance")
            
            # Model selection
            algo_list = [r['Algorithm'] for r in results_df.to_dict('records') if '🏆' not in r['Algorithm']]
            
            selected_algo = st.selectbox(
                "Select Algorithm to Tune",
                options=algo_list
            )
            
            cv_folds = st.slider(
                "Cross-Validation Folds",
                min_value=2,
                max_value=10,
                value=3,
                help="Number of folds for cross-validation"
            )
            
            if st.button("🎯 Start Hyperparameter Tuning"):
                with st.spinner(f"Tuning {selected_algo}..."):
                    try:
                        best_params, best_score, model_pkl = st.session_state.automl.tune(
                            st.session_state.X_processed,
                            st.session_state.y_processed,
                            selected_algo,
                            cv=cv_folds
                        )
                        
                        st.success(f"✅ Tuning Complete!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Best Parameters:**")
                            if best_params:
                                st.json(best_params)
                            else:
                                st.info("No parameters to tune for this model")
                        
                        with col2:
                            st.markdown("**Best Score:**")
                            st.metric("Cross-Val Score", f"{best_score:.4f}")
                        
                        # Download tuned model
                        st.download_button(
                            label="💾 Download Tuned Model",
                            data=model_pkl,
                            file_name=f"{selected_algo.lower().replace(' ', '_')}_tuned.pkl",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                    
                    except Exception as e:
                        st.error(f"❌ Tuning failed: {e}")
        
        # ==================== TAB 5: EXPORT ====================
        with tab5:
            st.markdown("### Export & Deployment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 💾 Download Models")
                
                # Download best model
                model_pkl = pickle.dumps(best_model)
                st.download_button(
                    label="📥 Download Best Model (.pkl)",
                    data=model_pkl,
                    file_name=f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
                
                # Download results
                results_csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=results_csv,
                    file_name=f"automl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Download preprocessor
                if st.session_state.preprocessor:
                    prep_pkl = pickle.dumps(st.session_state.preprocessor)
                    st.download_button(
                        label="📥 Download Preprocessor (.pkl)",
                        data=prep_pkl,
                        file_name="preprocessor.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
            
            with col2:
                st.markdown("#### 🚀 Deployment Guide")
                
                st.code("""
# Load and use your model

import pickle
import pandas as pd

# Load model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessor
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Prepare new data
new_data = pd.DataFrame({...})

# Preprocess
X_new = preprocessor.transform(new_data)

# Predict
predictions = model.predict(X_new)
                """, language="python")
                
                st.info("💡 Tip: Keep both the model and preprocessor together for production use!")

# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":
    main()