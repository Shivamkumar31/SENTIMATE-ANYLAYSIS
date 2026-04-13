"""
Sentiment Analysis Dashboard - Streamlit Web Interface
Analyze sentiment and emotions in text with beautiful visualizations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sentiment_utils import (
    SentimentEmotionAnalyzer,
    TextPreprocessor,
    calculate_statistics,
    batch_analyze_to_json
)
import json
from datetime import datetime
import time

# ============= PAGE CONFIGURATION =============

st.set_page_config(
    page_title="Sentiment & Emotion Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }
    </style>
""", unsafe_allow_html=True)

# ============= TITLE & INTRODUCTION =============

st.title("😊 Sentiment & Emotion Analysis Dashboard")
st.markdown("### Analyze emotions and sentiments in text using advanced NLP models")

# ============= LOAD MODELS (CACHED) =============

@st.cache_resource
def load_analyzer():
    """Load sentiment and emotion models (cached)"""
    st.info("📥 Loading models... (first time takes 30-60 seconds)")
    try:
        analyzer = SentimentEmotionAnalyzer()
        st.success("✅ Models loaded successfully!")
        return analyzer
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

analyzer = load_analyzer()

# ============= SIDEBAR CONFIGURATION =============

st.sidebar.header("⚙️ Settings & Options")

# Analysis mode selection
analysis_mode = st.sidebar.selectbox(
    "📊 Select Analysis Mode",
    ["Single Text", "Batch Upload", "Real-time Monitoring", "Compare Texts"]
)

# Text preprocessing options
st.sidebar.subheader("🔧 Text Processing")
clean_text = st.sidebar.checkbox("Clean text (remove URLs, mentions)", value=True)
normalize_text = st.sidebar.checkbox("Normalize repeated characters", value=True)

# Display options
st.sidebar.subheader("👁️ Display Options")
show_confidence = st.sidebar.checkbox("Show confidence scores", value=True)
show_detailed = st.sidebar.checkbox("Show detailed emotion breakdown", value=True)
show_processed = st.sidebar.checkbox("Show processed text", value=False)

# ============= VISUALIZATION FUNCTIONS =============

def create_sentiment_gauge(sentiment: str, confidence: float) -> go.Figure:
    """Create sentiment gauge chart"""
    colors = {"Positive": "#4CAF50", "Neutral": "#FFA500", "Negative": "#FF6B6B"}
    color = colors.get(sentiment, "#808080")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        title={'text': f"{sentiment}"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': "#FFE6E6"},
                {'range': [33, 66], 'color': "#FFF9E6"},
                {'range': [66, 100], 'color': "#E6F7E6"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def create_emotion_bar_chart(emotions_scores: dict) -> go.Figure:
    """Create emotion distribution bar chart"""
    emotion_names = [name.split()[-1] for name in emotions_scores.keys()]
    confidence_scores = list(emotions_scores.values())
    
    # Map emotions to colors
    colors_map = {
        'Sadness': '#4A90E2',
        'Fear': '#7B68EE',
        'Anger': '#DC143C',
        'Joy': '#FFD700',
        'Neutral': '#808080',
        'Surprise': '#FF69B4',
        'Disgust': '#228B22'
    }
    
    bar_colors = [colors_map.get(name, '#808080') for name in emotion_names]
    
    fig = go.Figure(data=[
        go.Bar(
            y=emotion_names,
            x=confidence_scores,
            orientation='h',
            marker=dict(color=bar_colors),
            text=[f"{v:.1%}" for v in confidence_scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Emotion Distribution",
        xaxis_title="Confidence",
        yaxis_title="Emotion",
        height=400,
        showlegend=False,
        margin=dict(l=100, r=10, t=50, b=10)
    )
    return fig

def create_sentiment_scores_chart(scores: dict) -> go.Figure:
    """Create sentiment scores radar/bar chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(scores.keys()),
            y=list(scores.values()),
            marker=dict(
                color=list(scores.values()),
                colorscale='RdYlGn',
                showscale=True,
                cmin=0,
                cmax=1
            ),
            text=[f"{v:.1%}" for v in scores.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Sentiment Score Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Confidence",
        height=400,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig

# ============= MAIN APP LOGIC =============

if analysis_mode == "Single Text":
    st.header("📝 Single Text Analysis")
    
    # Input area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste any text here...",
            height=150,
            key="single_text"
        )
    
    with col2:
        st.write("**Text Statistics:**")
        if text_input:
            st.metric("Characters", len(text_input))
            st.metric("Words", len(text_input.split()))
            st.metric("Sentences", len(text_input.split(".")))
    
    if text_input:
        # Analyze
        with st.spinner("🔍 Analyzing..."):
            result = analyzer.analyze(text_input)
        
        # Display processed text if requested
        if show_processed:
            with st.expander("📋 Processed Text"):
                st.text(result['processed_text'])
        
        # Results layout
        st.subheader("📊 Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("😊 Sentiment Analysis")
            
            sentiment_data = result['sentiment']
            
            # Gauge chart
            gauge_fig = create_sentiment_gauge(
                sentiment_data['label'],
                sentiment_data['confidence']
            )
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Sentiment metrics
            st.metric(
                "Sentiment",
                sentiment_data['label'],
                f"{sentiment_data['confidence']:.1%}" if show_confidence else ""
            )
            
            # Scores breakdown
            if show_detailed:
                st.write("**Confidence Scores:**")
                for sentiment, score in sentiment_data['scores'].items():
                    st.progress(score, text=f"{sentiment}: {score:.1%}")
        
        with col2:
            st.subheader("💭 Emotion Detection")
            
            emotion_data = result['emotion']
            
            # Emotion bar chart
            emotion_fig = create_emotion_bar_chart(emotion_data['all_emotions'])
            st.plotly_chart(emotion_fig, use_container_width=True)
            
            # Emotion metric
            st.metric(
                "Dominant Emotion",
                emotion_data['label'],
                f"{emotion_data['confidence']:.1%}" if show_confidence else ""
            )
        
        # Sentiment scores distribution
        st.subheader("🎯 Detailed Sentiment Scores")
        sentiment_fig = create_sentiment_scores_chart(sentiment_data['scores'])
        st.plotly_chart(sentiment_fig, use_container_width=True)
        
        # Export results
        st.subheader("💾 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            json_str = json.dumps(result, indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            csv_data = pd.DataFrame([{
                'Text': text_input,
                'Sentiment': sentiment_data['label'],
                'Sentiment_Confidence': sentiment_data['confidence'],
                'Emotion': emotion_data['label'],
                'Emotion_Confidence': emotion_data['confidence']
            }])
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data.to_csv(index=False),
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

elif analysis_mode == "Batch Upload":
    st.header("📤 Batch Text Analysis")
    
    # Input options
    input_type = st.radio("Choose input type:", ["Text List", "CSV File", "JSON File"])
    
    if input_type == "Text List":
        st.write("Enter multiple texts (one per line)")
        text_input = st.text_area(
            "Paste texts here:",
            placeholder="Line 1\nLine 2\nLine 3\n...",
            height=200
        )
        
        if text_input and st.button("🔍 Analyze All"):
            texts = [t.strip() for t in text_input.split('\n') if t.strip()]
            
            progress_bar = st.progress(0)
            results = []
            
            with st.spinner(f"📊 Analyzing {len(texts)} texts..."):
                for i, text in enumerate(texts):
                    result = analyzer.analyze(text)
                    results.append(result)
                    progress_bar.progress((i + 1) / len(texts))
            
            # Display results
            st.success(f"✅ Analyzed {len(texts)} texts successfully!")
            
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            stats = calculate_statistics(results)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Texts", stats['total_texts'])
            with col2:
                st.metric("Avg Sentiment Confidence", f"{stats['avg_sentiment_confidence']:.1%}")
            with col3:
                st.metric("Avg Emotion Confidence", f"{stats['avg_emotion_confidence']:.1%}")
            
            # Sentiment distribution
            st.subheader("📊 Results Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_df = pd.DataFrame(
                    list(stats['sentiment_distribution'].items()),
                    columns=['Sentiment', 'Count']
                )
                fig_sentiment = px.pie(
                    sentiment_df,
                    values='Count',
                    names='Sentiment',
                    color_discrete_map={
                        'Positive': '#4CAF50',
                        'Neutral': '#FFA500',
                        'Negative': '#FF6B6B'
                    }
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            with col2:
                emotion_df = pd.DataFrame(
                    list(stats['emotion_distribution'].items()),
                    columns=['Emotion', 'Count']
                ).sort_values('Count', ascending=False)
                fig_emotion = px.bar(
                    emotion_df,
                    x='Count',
                    y='Emotion',
                    orientation='h',
                    color='Count'
                )
                st.plotly_chart(fig_emotion, use_container_width=True)
            
            # Detailed results table
            st.subheader("📋 Detailed Results")
            
            table_data = []
            for result in results:
                table_data.append({
                    'Text': result['text'][:50] + "..." if len(result['text']) > 50 else result['text'],
                    'Sentiment': result['sentiment']['label'],
                    'Sentiment Confidence': f"{result['sentiment']['confidence']:.1%}",
                    'Emotion': result['emotion']['label'],
                    'Emotion Confidence': f"{result['emotion']['confidence']:.1%}"
                })
            
            df_results = pd.DataFrame(table_data)
            st.dataframe(df_results, use_container_width=True)
            
            # Export results
            st.subheader("💾 Export Batch Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                json_data = batch_analyze_to_json(results, pretty=True)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_data,
                    file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                csv_data = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

elif analysis_mode == "Real-time Monitoring":
    st.header("⏱️ Real-time Monitoring")
    
    st.write("Continuously analyze incoming texts (simulated)")
    
    if st.button("▶️ Start Monitoring"):
        st.info("Monitoring active for 30 seconds...")
        
        # Sample texts to simulate incoming data
        sample_texts = [
            "I love this! This is the best day ever!",
            "This is terrible and I'm very upset.",
            "The weather is neither good nor bad.",
            "I can't believe what just happened!",
            "I feel so sad and alone.",
            "That's disgusting and unacceptable!",
            "I'm thrilled with this amazing news!",
            "This product is absolutely wonderful!",
            "I'm deeply disappointed by this outcome.",
            "What an unexpected turn of events!"
        ]
        
        # Real-time plotting
        sentiment_placeholder = st.empty()
        emotion_placeholder = st.empty()
        timeline_placeholder = st.empty()
        
        timeline_data = []
        
        for i, text in enumerate(sample_texts):
            result = analyzer.analyze(text)
            
            timeline_data.append({
                'Time': i,
                'Sentiment': result['sentiment']['label'],
                'Emotion': result['emotion']['label'].split()[-1],
                'Sentiment_Conf': result['sentiment']['confidence'],
                'Emotion_Conf': result['emotion']['confidence']
            })
            
            # Update visualizations
            with sentiment_placeholder.container():
                st.write(f"**Latest Text**: {text}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Sentiment",
                        result['sentiment']['label'],
                        f"{result['sentiment']['confidence']:.1%}"
                    )
                with col2:
                    st.metric(
                        "Emotion",
                        result['emotion']['label'],
                        f"{result['emotion']['confidence']:.1%}"
                    )
            
            # Timeline chart
            df_timeline = pd.DataFrame(timeline_data)
            fig_timeline = px.line(
                df_timeline,
                x='Time',
                y='Sentiment_Conf',
                title="Sentiment Confidence Over Time",
                markers=True
            )
            timeline_placeholder.plotly_chart(fig_timeline, use_container_width=True)
            
            time.sleep(1)
        
        st.success("✅ Monitoring completed!")

elif analysis_mode == "Compare Texts":
    st.header("⚖️ Compare Multiple Texts")
    
    st.write("Compare sentiment and emotion across different texts")
    
    num_texts = st.number_input("Number of texts to compare:", 2, 5, 2)
    
    texts = []
    for i in range(num_texts):
        text = st.text_area(
            f"Text {i+1}:",
            placeholder=f"Enter text {i+1}...",
            height=100,
            key=f"compare_text_{i}"
        )
        if text:
            texts.append(text)
    
    if len(texts) == num_texts and st.button("🔍 Compare"):
        results = [analyzer.analyze(text) for text in texts]
        
        # Comparison table
        st.subheader("📊 Comparison Table")
        
        comparison_data = []
        for i, result in enumerate(results, 1):
            comparison_data.append({
                'Text #': i,
                'Sentiment': result['sentiment']['label'],
                'Sentiment Conf': f"{result['sentiment']['confidence']:.1%}",
                'Emotion': result['emotion']['label'],
                'Emotion Conf': f"{result['emotion']['confidence']:.1%}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Radar chart comparison
        st.subheader("🎯 Sentiment Scores Comparison")
        
        # Prepare data for comparison
        comparison_sentiments = []
        for i, result in enumerate(results, 1):
            for sentiment, score in result['sentiment']['scores'].items():
                comparison_sentiments.append({
                    'Text': f"Text {i}",
                    'Sentiment': sentiment,
                    'Score': score
                })
        
        df_sentiments = pd.DataFrame(comparison_sentiments)
        fig_comparison = px.bar(
            df_sentiments,
            x='Sentiment',
            y='Score',
            color='Text',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

# ============= FOOTER =============

st.markdown("---")

with st.expander("ℹ️ About This App"):
    st.markdown("""
    ### Sentiment & Emotion Analysis
    
    This dashboard uses advanced NLP models to analyze:
    
    **Sentiment Analysis:**
    - Positive: Favorable, happy, satisfied
    - Neutral: Objective, factual statements
    - Negative: Unfavorable, unhappy, dissatisfied
    
    **Emotion Detection:**
    - 😢 Sadness: Sorrowful, melancholic
    - 😨 Fear: Anxious, worried, afraid
    - 😠 Anger: Frustrated, upset, furious
    - 😊 Joy: Happy, delighted, cheerful
    - 😐 Neutral: No strong emotion
    - 😲 Surprise: Shocked, amazed
    - 🤢 Disgust: Repulsed, disgusted
    
    **Models Used:**
    - Sentiment: DistilBERT (fine-tuned on SST-2)
    - Emotion: RoBERTa (trained on emotion datasets)
    
    **Note:** Models are run locally for privacy. No data is sent to external servers.
    """)

with st.expander("🎓 How It Works"):
    st.markdown("""
    1. **Text Preprocessing**: Remove URLs, normalize text, handle special characters
    2. **Tokenization**: Convert text into tokens that the model can understand
    3. **Model Inference**: Pass tokens through trained neural networks
    4. **Post-processing**: Convert model outputs to human-readable results
    5. **Visualization**: Display results with confidence scores
    
    The models are pre-trained on large datasets and fine-tuned for sentiment/emotion tasks.
    """)
