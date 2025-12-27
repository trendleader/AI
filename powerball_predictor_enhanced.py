"""
Powerball Lottery Prediction Prototype - Enhanced Edition
Features:
- Real historical data from NY Open Data API
- CSV upload support
- Number exclusion filters
- Favorite numbers integration
- Ticket cost calculator
- Multiple prediction strategies
- OpenAI-powered analysis and predictions

DISCLAIMER: Lottery numbers are random - no algorithm can predict winning numbers.
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import random
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import requests
from io import StringIO
import json
import re
from openai import OpenAI

# Powerball rules
WHITE_BALLS = list(range(1, 70))  # 1-69
POWERBALL_RANGE = list(range(1, 27))  # 1-26
NUM_WHITE = 5

# Ticket pricing
TICKET_PRICE = 2.00
POWER_PLAY_PRICE = 1.00
DOUBLE_PLAY_PRICE = 1.00


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_real_data():
    """Fetch real Powerball data from NY Open Data API."""
    try:
        # NY Open Data Socrata API
        url = "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        
        # Parse the data - format is "Draw Date" and "Winning Numbers" (space-separated)
        processed_data = []
        
        for _, row in df.iterrows():
            try:
                date = row['Draw Date']
                numbers = str(row['Winning Numbers']).split()
                
                if len(numbers) >= 6:
                    processed_data.append({
                        'date': date,
                        'white_1': int(numbers[0]),
                        'white_2': int(numbers[1]),
                        'white_3': int(numbers[2]),
                        'white_4': int(numbers[3]),
                        'white_5': int(numbers[4]),
                        'powerball': int(numbers[5]),
                        'multiplier': row.get('Multiplier', None)
                    })
            except (ValueError, IndexError):
                continue
        
        result_df = pd.DataFrame(processed_data)
        result_df['date'] = pd.to_datetime(result_df['date'])
        result_df = result_df.sort_values('date', ascending=False).reset_index(drop=True)
        
        return result_df, True
        
    except Exception as e:
        st.warning(f"Could not fetch live data: {e}. Using sample data.")
        return generate_sample_data(), False


def generate_sample_data(n_drawings=500):
    """Generate sample historical data for demonstration."""
    np.random.seed(42)
    data = []
    base_date = datetime.now() - timedelta(days=n_drawings * 3.5)
    
    for i in range(n_drawings):
        white_balls = sorted(random.sample(range(1, 70), 5))
        powerball = random.randint(1, 26)
        draw_date = base_date + timedelta(days=i * 3.5)
        multiplier = random.choice([2, 3, 4, 5, 10])
        data.append({
            'date': draw_date,
            'white_1': white_balls[0],
            'white_2': white_balls[1],
            'white_3': white_balls[2],
            'white_4': white_balls[3],
            'white_5': white_balls[4],
            'powerball': powerball,
            'multiplier': multiplier
        })
    
    return pd.DataFrame(data)


def parse_uploaded_csv(uploaded_file):
    """Parse user-uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Try to detect column format
        if 'Winning Numbers' in df.columns:
            # NY format
            processed_data = []
            for _, row in df.iterrows():
                numbers = str(row['Winning Numbers']).split()
                if len(numbers) >= 6:
                    processed_data.append({
                        'date': row.get('Draw Date', ''),
                        'white_1': int(numbers[0]),
                        'white_2': int(numbers[1]),
                        'white_3': int(numbers[2]),
                        'white_4': int(numbers[3]),
                        'white_5': int(numbers[4]),
                        'powerball': int(numbers[5])
                    })
            return pd.DataFrame(processed_data)
        
        elif 'Num1' in df.columns or 'white_1' in df.columns:
            # Direct format
            return df
        
        else:
            st.error("Unrecognized CSV format. Please check the expected format below.")
            return None
            
    except Exception as e:
        st.error(f"Error parsing CSV: {e}")
        return None


def analyze_frequency(df, lookback=None):
    """Analyze frequency of all numbers."""
    if lookback:
        df = df.head(lookback)
    
    white_numbers = []
    for col in ['white_1', 'white_2', 'white_3', 'white_4', 'white_5']:
        white_numbers.extend(df[col].tolist())
    
    white_freq = Counter(white_numbers)
    powerball_freq = Counter(df['powerball'].tolist())
    
    return white_freq, powerball_freq


def get_hot_cold_numbers(freq, n=10, number_range=69):
    """Get hottest and coldest numbers."""
    full_freq = {i: freq.get(i, 0) for i in range(1, number_range + 1)}
    sorted_nums = sorted(full_freq.items(), key=lambda x: x[1], reverse=True)
    
    hot = [num for num, count in sorted_nums[:n]]
    cold = [num for num, count in sorted_nums[-n:]]
    
    return hot, cold


def get_overdue_numbers(df, number_range=69, is_powerball=False):
    """Find numbers that haven't appeared recently."""
    col_name = 'powerball' if is_powerball else None
    
    last_seen = {}
    
    for idx, row in df.iterrows():
        if is_powerball:
            num = row['powerball']
            if num not in last_seen:
                last_seen[num] = idx
        else:
            for col in ['white_1', 'white_2', 'white_3', 'white_4', 'white_5']:
                num = row[col]
                if num not in last_seen:
                    last_seen[num] = idx
    
    # Numbers never seen get max overdue
    max_range = 27 if is_powerball else 70
    for i in range(1, max_range):
        if i not in last_seen:
            last_seen[i] = len(df)
    
    # Sort by most overdue
    sorted_overdue = sorted(last_seen.items(), key=lambda x: x[1], reverse=True)
    return [num for num, _ in sorted_overdue]


def frequency_based_prediction(white_freq, pb_freq, strategy='hot', excluded_white=None, excluded_pb=None, favorite_white=None, favorite_pb=None):
    """Generate prediction based on frequency analysis."""
    excluded_white = excluded_white or []
    excluded_pb = excluded_pb or []
    favorite_white = favorite_white or []
    favorite_pb = favorite_pb or []
    
    # Get available numbers
    available_white = [n for n in WHITE_BALLS if n not in excluded_white]
    available_pb = [n for n in POWERBALL_RANGE if n not in excluded_pb]
    
    if strategy == 'hot':
        hot_white, _ = get_hot_cold_numbers(white_freq, n=25, number_range=69)
        hot_pb, _ = get_hot_cold_numbers(pb_freq, n=15, number_range=26)
        pool_white = [n for n in hot_white if n in available_white]
        pool_pb = [n for n in hot_pb if n in available_pb]
    elif strategy == 'cold':
        _, cold_white = get_hot_cold_numbers(white_freq, n=25, number_range=69)
        _, cold_pb = get_hot_cold_numbers(pb_freq, n=15, number_range=26)
        pool_white = [n for n in cold_white if n in available_white]
        pool_pb = [n for n in cold_pb if n in available_pb]
    else:  # mixed
        hot_white, cold_white = get_hot_cold_numbers(white_freq, n=20, number_range=69)
        hot_pb, cold_pb = get_hot_cold_numbers(pb_freq, n=10, number_range=26)
        combined_white = hot_white[:12] + cold_white[:12]
        combined_pb = hot_pb[:7] + cold_pb[:7]
        pool_white = [n for n in combined_white if n in available_white]
        pool_pb = [n for n in combined_pb if n in available_pb]
    
    # Start with favorites
    whites = [n for n in favorite_white if n in available_white][:5]
    
    # Fill remaining
    remaining_pool = [n for n in pool_white if n not in whites]
    if len(remaining_pool) < (5 - len(whites)):
        remaining_pool = [n for n in available_white if n not in whites]
    
    whites.extend(random.sample(remaining_pool, min(5 - len(whites), len(remaining_pool))))
    whites = sorted(whites[:5])
    
    # Powerball
    if favorite_pb and favorite_pb[0] in available_pb:
        pb = favorite_pb[0]
    elif pool_pb:
        pb = random.choice(pool_pb)
    else:
        pb = random.choice(available_pb)
    
    return whites, pb


def statistical_prediction(df, excluded_white=None, excluded_pb=None):
    """Use statistical patterns for prediction."""
    excluded_white = excluded_white or []
    excluded_pb = excluded_pb or []
    
    available_white = [n for n in WHITE_BALLS if n not in excluded_white]
    available_pb = [n for n in POWERBALL_RANGE if n not in excluded_pb]
    
    df_temp = df.copy()
    df_temp['sum'] = df_temp[['white_1', 'white_2', 'white_3', 'white_4', 'white_5']].sum(axis=1)
    avg_sum = df_temp['sum'].mean()
    std_sum = df_temp['sum'].std()
    
    target_sum = np.random.normal(avg_sum, std_sum / 2)
    target_sum = max(15, min(target_sum, 325))
    
    best_combo = None
    best_diff = float('inf')
    
    for _ in range(2000):
        if len(available_white) >= 5:
            whites = sorted(random.sample(available_white, 5))
            diff = abs(sum(whites) - target_sum)
            if diff < best_diff:
                best_diff = diff
                best_combo = whites
    
    pb = random.choice(available_pb) if available_pb else 1
    return best_combo or sorted(random.sample(available_white, 5)), pb


def delta_system_prediction(excluded_white=None, excluded_pb=None):
    """Use delta system for prediction."""
    excluded_white = excluded_white or []
    excluded_pb = excluded_pb or []
    
    available_white = [n for n in WHITE_BALLS if n not in excluded_white]
    available_pb = [n for n in POWERBALL_RANGE if n not in excluded_pb]
    
    # Generate using deltas
    for _ in range(100):
        first_num = random.randint(1, 20)
        if first_num in excluded_white:
            continue
            
        whites = [first_num]
        for _ in range(4):
            delta = random.randint(1, 15)
            next_num = whites[-1] + delta
            if next_num <= 69 and next_num not in excluded_white:
                whites.append(next_num)
        
        if len(whites) == 5:
            break
    
    # Fallback
    if len(whites) < 5:
        whites = sorted(random.sample(available_white, 5))
    else:
        whites = sorted(whites)
    
    pb = random.choice(available_pb) if available_pb else 1
    return whites, pb


def balanced_prediction(excluded_white=None, excluded_pb=None, favorite_white=None):
    """Generate balanced numbers across ranges."""
    excluded_white = excluded_white or []
    excluded_pb = excluded_pb or []
    favorite_white = favorite_white or []
    
    available_pb = [n for n in POWERBALL_RANGE if n not in excluded_pb]
    
    ranges = [(1, 14), (15, 28), (29, 42), (43, 56), (57, 69)]
    whites = []
    
    # Start with favorites in range
    for fav in favorite_white[:5]:
        if fav not in excluded_white:
            whites.append(fav)
    
    # Fill from ranges
    for low, high in ranges:
        if len(whites) >= 5:
            break
        available_in_range = [n for n in range(low, high + 1) 
                            if n not in excluded_white and n not in whites]
        if available_in_range:
            whites.append(random.choice(available_in_range))
    
    # Fallback fill
    available_white = [n for n in WHITE_BALLS if n not in excluded_white and n not in whites]
    while len(whites) < 5 and available_white:
        num = random.choice(available_white)
        whites.append(num)
        available_white.remove(num)
    
    whites = sorted(whites[:5])
    pb = random.choice(available_pb) if available_pb else 1
    return whites, pb


def quick_pick(excluded_white=None, excluded_pb=None):
    """True random quick pick."""
    excluded_white = excluded_white or []
    excluded_pb = excluded_pb or []
    
    available_white = [n for n in WHITE_BALLS if n not in excluded_white]
    available_pb = [n for n in POWERBALL_RANGE if n not in excluded_pb]
    
    whites = sorted(random.sample(available_white, min(5, len(available_white))))
    pb = random.choice(available_pb) if available_pb else 1
    return whites, pb


def calculate_ticket_cost(num_tickets, power_play=False, double_play=False):
    """Calculate total ticket cost."""
    base = num_tickets * TICKET_PRICE
    extras = 0
    if power_play:
        extras += num_tickets * POWER_PLAY_PRICE
    if double_play:
        extras += num_tickets * DOUBLE_PLAY_PRICE
    return base + extras


def get_openai_client(api_key):
    """Initialize OpenAI client with provided API key."""
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None


def ai_analyze_patterns(client, white_freq, pb_freq, df, model="gpt-4o-mini"):
    """Use OpenAI to analyze lottery patterns and provide insights."""
    
    # Prepare data summary
    hot_white, cold_white = get_hot_cold_numbers(white_freq, n=10, number_range=69)
    hot_pb, cold_pb = get_hot_cold_numbers(pb_freq, n=5, number_range=26)
    
    df_temp = df.head(50).copy()
    recent_draws = []
    for _, row in df_temp.iterrows():
        recent_draws.append(f"{row['white_1']}-{row['white_2']}-{row['white_3']}-{row['white_4']}-{row['white_5']} PB:{row['powerball']}")
    
    df_stats = df.copy()
    df_stats['sum'] = df_stats[['white_1', 'white_2', 'white_3', 'white_4', 'white_5']].sum(axis=1)
    avg_sum = df_stats['sum'].mean()
    
    prompt = f"""You are a lottery data analyst. Analyze this Powerball lottery data and provide interesting statistical observations. 
Remember: Lottery is random, but pattern analysis can be fun and educational.

DATA SUMMARY:
- Total drawings analyzed: {len(df)}
- Average sum of white balls: {avg_sum:.1f}

HOT WHITE BALLS (most frequent): {hot_white}
COLD WHITE BALLS (least frequent): {cold_white}
HOT POWERBALLS: {hot_pb}
COLD POWERBALLS: {cold_pb}

LAST 10 DRAWINGS:
{chr(10).join(recent_draws[:10])}

Provide:
1. 3-4 interesting pattern observations (with appropriate caveats about randomness)
2. Any notable streaks or anomalies
3. A "fun fact" about the data
4. Educational insight about lottery probability

Keep response concise and engaging. Use bullet points sparingly."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a friendly data analyst who explains lottery statistics in an engaging way while always reminding users that lottery is purely random chance."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI analysis: {str(e)}"


def ai_generate_predictions(client, white_freq, pb_freq, df, num_predictions=5, 
                           excluded_white=None, excluded_pb=None, 
                           favorite_white=None, favorite_pb=None,
                           user_prompt="", model="gpt-4o-mini"):
    """Use OpenAI to generate lottery number predictions with reasoning."""
    
    excluded_white = excluded_white or []
    excluded_pb = excluded_pb or []
    favorite_white = favorite_white or []
    favorite_pb = favorite_pb or []
    
    hot_white, cold_white = get_hot_cold_numbers(white_freq, n=15, number_range=69)
    hot_pb, cold_pb = get_hot_cold_numbers(pb_freq, n=8, number_range=26)
    
    df_stats = df.copy()
    df_stats['sum'] = df_stats[['white_1', 'white_2', 'white_3', 'white_4', 'white_5']].sum(axis=1)
    avg_sum = df_stats['sum'].mean()
    std_sum = df_stats['sum'].std()
    
    prompt = f"""Generate {num_predictions} Powerball lottery number combinations based on statistical analysis.

RULES:
- 5 white balls: numbers 1-69 (must be unique, sorted ascending)
- 1 Powerball: number 1-26

STATISTICAL CONTEXT:
- Hot white balls (most drawn): {hot_white[:10]}
- Cold white balls (least drawn): {cold_white[:10]}
- Hot Powerballs: {hot_pb[:5]}
- Cold Powerballs: {cold_pb[:5]}
- Typical sum range: {avg_sum - std_sum:.0f} to {avg_sum + std_sum:.0f} (avg: {avg_sum:.0f})

CONSTRAINTS:
- EXCLUDE these white balls: {excluded_white if excluded_white else 'None'}
- EXCLUDE these Powerballs: {excluded_pb if excluded_pb else 'None'}
- TRY TO INCLUDE favorites (white): {favorite_white if favorite_white else 'None'}
- TRY TO INCLUDE favorite Powerball: {favorite_pb if favorite_pb else 'None'}

{f"USER REQUEST: {user_prompt}" if user_prompt else ""}

Return ONLY valid JSON in this exact format:
{{
    "predictions": [
        {{"whites": [1, 2, 3, 4, 5], "powerball": 10, "reasoning": "Brief reason"}},
        ...
    ],
    "overall_strategy": "Brief explanation of your approach"
}}

Ensure all numbers respect the constraints. Be creative with your reasoning while acknowledging randomness."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a lottery number generator that creates statistically-informed combinations while acknowledging that lottery is random. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.8
        )
        
        content = response.choices[0].message.content
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            result = json.loads(json_match.group())
            
            # Validate predictions
            valid_predictions = []
            for pred in result.get('predictions', []):
                whites = pred.get('whites', [])
                pb = pred.get('powerball', 1)
                
                # Validate
                if (len(whites) == 5 and 
                    all(1 <= w <= 69 for w in whites) and
                    len(set(whites)) == 5 and
                    1 <= pb <= 26 and
                    not any(w in excluded_white for w in whites) and
                    pb not in excluded_pb):
                    valid_predictions.append({
                        'whites': sorted(whites),
                        'powerball': pb,
                        'reasoning': pred.get('reasoning', 'AI-generated combination')
                    })
            
            if valid_predictions:
                return {
                    'predictions': valid_predictions[:num_predictions],
                    'strategy': result.get('overall_strategy', 'AI-powered analysis')
                }
        
        # Fallback if parsing fails
        return None
        
    except Exception as e:
        st.error(f"Error generating AI predictions: {str(e)}")
        return None


def ai_lucky_numbers(client, user_info="", model="gpt-4o-mini"):
    """Generate personalized lucky numbers based on user input."""
    
    prompt = f"""Generate a single "lucky" Powerball combination based on this user info:
{user_info if user_info else "No specific info provided - create a balanced, interesting combination"}

Rules:
- 5 white balls (1-69, unique, sorted)
- 1 Powerball (1-26)

Return ONLY JSON:
{{"whites": [1, 2, 3, 4, 5], "powerball": 10, "meaning": "Brief fun explanation of why these numbers"}}

Be creative and fun with the meaning!"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a fun, creative lucky number generator. Create engaging explanations for number choices."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.9
        )
        
        content = response.choices[0].message.content
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            result = json.loads(json_match.group())
            whites = sorted(result.get('whites', []))
            if len(whites) == 5 and all(1 <= w <= 69 for w in whites):
                return result
        return None
    except Exception as e:
        return None


def ai_chat_assistant(client, user_message, df, white_freq, pb_freq, chat_history, model="gpt-4o-mini"):
    """Interactive chat about lottery statistics and strategies."""
    
    hot_white, cold_white = get_hot_cold_numbers(white_freq, n=10, number_range=69)
    hot_pb, cold_pb = get_hot_cold_numbers(pb_freq, n=5, number_range=26)
    
    df_stats = df.copy()
    df_stats['sum'] = df_stats[['white_1', 'white_2', 'white_3', 'white_4', 'white_5']].sum(axis=1)
    
    system_prompt = f"""You are a helpful Powerball lottery assistant with access to historical data.

CURRENT DATA CONTEXT:
- Total drawings: {len(df)}
- Hot white balls: {hot_white}
- Cold white balls: {cold_white}
- Hot Powerballs: {hot_pb}
- Cold Powerballs: {cold_pb}
- Average sum: {df_stats['sum'].mean():.1f}
- Sum std dev: {df_stats['sum'].std():.1f}

You can:
1. Answer questions about lottery statistics
2. Explain probability concepts
3. Discuss different picking strategies
4. Generate number suggestions when asked
5. Provide educational content about randomness

Always remind users that lottery is random chance. Be friendly and helpful.
If asked to generate numbers, provide them in a clear format."""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history
    for msg in chat_history[-10:]:  # Last 10 messages for context
        messages.append(msg)
    
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=600,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def display_lottery_ball(number, is_powerball=False, size=60):
    """Display a lottery ball with styling."""
    if is_powerball:
        bg = "linear-gradient(135deg, #ff4444, #cc0000)"
        color = "white"
    else:
        bg = "linear-gradient(135deg, #ffffff, #e8e8e8)"
        color = "#333"
    
    return f"""
    <div style='
        background: {bg}; 
        border-radius: 50%; 
        width: {size}px; 
        height: {size}px; 
        display: flex; 
        align-items: center; 
        justify-content: center;
        font-size: {int(size * 0.4)}px; 
        font-weight: bold; 
        color: {color};
        box-shadow: 3px 3px 8px rgba(0,0,0,0.3);
        margin: 5px auto;
        border: 2px solid {"#aa0000" if is_powerball else "#ccc"};
    '>{number}</div>
    """


# Main Streamlit App
def main():
    st.set_page_config(
        page_title="Powerball Predictor Pro",
        page_icon="🎱",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .big-number {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎱 Powerball Prediction Prototype - Enhanced Edition")
    st.caption("⚠️ DISCLAIMER: Lottery numbers are completely random. No algorithm can predict winning numbers. Play responsibly!")
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # OpenAI Configuration
    st.sidebar.subheader("🤖 OpenAI Integration")
    
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key:",
        type="password",
        help="Enter your OpenAI API key for AI-powered features",
        placeholder="sk-..."
    )
    
    openai_client = None
    if openai_api_key:
        if openai_api_key.startswith("sk-"):
            openai_client = get_openai_client(openai_api_key)
            if openai_client:
                st.sidebar.success("✅ API Key validated")
        else:
            st.sidebar.warning("⚠️ Invalid API key format")
    else:
        st.sidebar.info("💡 Add API key for AI features")
    
    openai_model = st.sidebar.selectbox(
        "AI Model:",
        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
        help="Select OpenAI model (gpt-4o-mini is fastest/cheapest)"
    )
    
    st.sidebar.markdown("---")
    
    # Data Source
    st.sidebar.subheader("📊 Data Source")
    data_source = st.sidebar.radio(
        "Select data source:",
        ["Live API (NY Open Data)", "Upload CSV", "Sample Data"]
    )
    
    if data_source == "Live API (NY Open Data)":
        with st.spinner("Fetching live data..."):
            df, is_live = fetch_real_data()
        if is_live:
            st.sidebar.success(f"✅ Loaded {len(df)} drawings from NY Open Data")
        else:
            st.sidebar.warning("⚠️ Using sample data")
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            df = parse_uploaded_csv(uploaded_file)
            if df is not None:
                st.sidebar.success(f"✅ Loaded {len(df)} drawings from CSV")
            else:
                df = generate_sample_data()
        else:
            st.sidebar.info("Upload a CSV or using sample data")
            df = generate_sample_data()
    else:
        df = generate_sample_data()
        st.sidebar.info("📋 Using sample data for demonstration")
    
    # Analysis Lookback
    st.sidebar.subheader("🔍 Analysis Period")
    lookback_options = {
        "All Time": None,
        "Last 50 Drawings": 50,
        "Last 100 Drawings": 100,
        "Last 200 Drawings": 200,
        "Last Year (~156)": 156
    }
    lookback_choice = st.sidebar.selectbox("Analyze data from:", list(lookback_options.keys()))
    lookback = lookback_options[lookback_choice]
    
    # Number Exclusions
    st.sidebar.subheader("🚫 Exclude Numbers")
    
    excluded_white = st.sidebar.multiselect(
        "Exclude White Balls (1-69):",
        options=WHITE_BALLS,
        default=[],
        help="Numbers you don't want in predictions"
    )
    
    excluded_pb = st.sidebar.multiselect(
        "Exclude Powerballs (1-26):",
        options=POWERBALL_RANGE,
        default=[],
        help="Powerball numbers you don't want"
    )
    
    # Favorite Numbers
    st.sidebar.subheader("⭐ Favorite Numbers")
    
    favorite_white = st.sidebar.multiselect(
        "Favorite White Balls (max 5):",
        options=[n for n in WHITE_BALLS if n not in excluded_white],
        default=[],
        max_selections=5,
        help="Numbers to prioritize in predictions"
    )
    
    favorite_pb = st.sidebar.multiselect(
        "Favorite Powerball (max 1):",
        options=[n for n in POWERBALL_RANGE if n not in excluded_pb],
        default=[],
        max_selections=1,
        help="Preferred Powerball number"
    )
    
    # Analyze frequency
    white_freq, pb_freq = analyze_frequency(df, lookback)
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Generate Predictions",
        "🤖 AI-Powered",
        "📊 Frequency Analysis",
        "🔥 Hot & Cold Numbers",
        "📈 Statistics",
        "💰 Cost Calculator"
    ])
    
    # TAB 1: Generate Predictions
    with tab1:
        st.header("Generate Your Numbers")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            strategy = st.selectbox(
                "Select Strategy",
                [
                    "🎲 Quick Pick (Random)",
                    "🔥 Hot Numbers (Most Frequent)",
                    "❄️ Cold Numbers (Least Frequent)",
                    "⚖️ Mixed Hot/Cold",
                    "📊 Statistical Pattern (Sum-Based)",
                    "📐 Delta System (Spacing)",
                    "🎯 Balanced Ranges"
                ]
            )
        
        with col2:
            num_predictions = st.slider("Number of predictions:", 1, 20, 5)
        
        with col3:
            st.write("")
            generate = st.button("🎰 Generate!", type="primary", use_container_width=True)
        
        # Show active filters
        if excluded_white or excluded_pb or favorite_white or favorite_pb:
            with st.expander("📋 Active Filters", expanded=False):
                filter_cols = st.columns(4)
                with filter_cols[0]:
                    if excluded_white:
                        st.write(f"**Excluded White:** {excluded_white}")
                with filter_cols[1]:
                    if excluded_pb:
                        st.write(f"**Excluded PB:** {excluded_pb}")
                with filter_cols[2]:
                    if favorite_white:
                        st.write(f"**Favorite White:** {favorite_white}")
                with filter_cols[3]:
                    if favorite_pb:
                        st.write(f"**Favorite PB:** {favorite_pb}")
        
        if generate:
            st.subheader("🎱 Your Predicted Numbers")
            
            predictions = []
            for i in range(num_predictions):
                if "Quick Pick" in strategy:
                    whites, pb = quick_pick(excluded_white, excluded_pb)
                elif "Hot Numbers" in strategy:
                    whites, pb = frequency_based_prediction(
                        white_freq, pb_freq, 'hot',
                        excluded_white, excluded_pb, favorite_white, favorite_pb
                    )
                elif "Cold Numbers" in strategy:
                    whites, pb = frequency_based_prediction(
                        white_freq, pb_freq, 'cold',
                        excluded_white, excluded_pb, favorite_white, favorite_pb
                    )
                elif "Mixed" in strategy:
                    whites, pb = frequency_based_prediction(
                        white_freq, pb_freq, 'mixed',
                        excluded_white, excluded_pb, favorite_white, favorite_pb
                    )
                elif "Statistical" in strategy:
                    whites, pb = statistical_prediction(df, excluded_white, excluded_pb)
                elif "Delta" in strategy:
                    whites, pb = delta_system_prediction(excluded_white, excluded_pb)
                else:  # Balanced
                    whites, pb = balanced_prediction(excluded_white, excluded_pb, favorite_white)
                
                predictions.append((whites, pb))
            
            # Display predictions
            for i, (whites, pb) in enumerate(predictions, 1):
                cols = st.columns([0.5, 1, 1, 1, 1, 1, 0.3, 1, 1.5])
                
                cols[0].markdown(f"**#{i}**")
                
                for j, num in enumerate(whites):
                    cols[j + 1].markdown(display_lottery_ball(num), unsafe_allow_html=True)
                
                cols[7].markdown(display_lottery_ball(pb, is_powerball=True), unsafe_allow_html=True)
                
                st.markdown("---")
            
            # Copy-friendly format
            st.subheader("📋 Copy-Friendly Format")
            copy_text = []
            for i, (whites, pb) in enumerate(predictions, 1):
                line = f"Ticket {i}: {' - '.join(map(str, whites))} | PB: {pb}"
                copy_text.append(line)
            
            st.code("\n".join(copy_text))
            
            # Quick cost estimate
            cost = calculate_ticket_cost(num_predictions)
            st.info(f"💵 Estimated cost for {num_predictions} tickets: **${cost:.2f}** (base price, no add-ons)")
    
    # TAB 2: AI-Powered Features
    with tab2:
        st.header("🤖 AI-Powered Features")
        
        if not openai_client:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use AI features.")
            st.info("""
            **How to get an API key:**
            1. Go to [platform.openai.com](https://platform.openai.com)
            2. Sign up or log in
            3. Navigate to API Keys section
            4. Create a new secret key
            5. Paste it in the sidebar
            
            **Note:** API usage incurs costs based on the model selected.
            """)
        else:
            ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs([
                "🔮 AI Predictions",
                "📊 AI Analysis", 
                "🍀 Lucky Numbers",
                "💬 Chat Assistant"
            ])
            
            # AI Predictions
            with ai_tab1:
                st.subheader("🔮 AI-Generated Predictions")
                st.caption("Let AI analyze patterns and generate number combinations")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    ai_user_prompt = st.text_area(
                        "Special instructions (optional):",
                        placeholder="e.g., 'Focus on numbers that haven't appeared recently' or 'Mix of hot and cold numbers'",
                        height=80
                    )
                
                with col2:
                    ai_num_predictions = st.slider("Number of predictions:", 1, 10, 5, key="ai_num")
                    ai_generate_btn = st.button("🤖 Generate AI Predictions", type="primary", use_container_width=True)
                
                if ai_generate_btn:
                    with st.spinner("🧠 AI is analyzing patterns..."):
                        result = ai_generate_predictions(
                            openai_client, white_freq, pb_freq, df,
                            num_predictions=ai_num_predictions,
                            excluded_white=excluded_white,
                            excluded_pb=excluded_pb,
                            favorite_white=favorite_white,
                            favorite_pb=favorite_pb,
                            user_prompt=ai_user_prompt,
                            model=openai_model
                        )
                    
                    if result:
                        st.success(f"**AI Strategy:** {result['strategy']}")
                        st.markdown("---")
                        
                        for i, pred in enumerate(result['predictions'], 1):
                            cols = st.columns([0.5, 1, 1, 1, 1, 1, 0.3, 1, 3])
                            
                            cols[0].markdown(f"**#{i}**")
                            
                            for j, num in enumerate(pred['whites']):
                                cols[j + 1].markdown(display_lottery_ball(num), unsafe_allow_html=True)
                            
                            cols[7].markdown(display_lottery_ball(pred['powerball'], is_powerball=True), unsafe_allow_html=True)
                            
                            cols[8].caption(f"💡 {pred['reasoning']}")
                            
                            st.markdown("---")
                        
                        # Copy format
                        st.subheader("📋 Copy-Friendly Format")
                        copy_text = []
                        for i, pred in enumerate(result['predictions'], 1):
                            line = f"Ticket {i}: {' - '.join(map(str, pred['whites']))} | PB: {pred['powerball']}"
                            copy_text.append(line)
                        st.code("\n".join(copy_text))
                    else:
                        st.error("Failed to generate AI predictions. Please try again.")
            
            # AI Analysis
            with ai_tab2:
                st.subheader("📊 AI Pattern Analysis")
                st.caption("Get AI-powered insights about lottery patterns")
                
                if st.button("🔍 Analyze Patterns", type="primary"):
                    with st.spinner("🧠 AI is analyzing the data..."):
                        analysis = ai_analyze_patterns(
                            openai_client, white_freq, pb_freq, df,
                            model=openai_model
                        )
                    
                    st.markdown("### 🎯 AI Insights")
                    st.markdown(analysis)
                    
                    st.info("💡 Remember: These patterns are historical observations. Lottery drawings are independent random events.")
            
            # Lucky Numbers
            with ai_tab3:
                st.subheader("🍀 Personalized Lucky Numbers")
                st.caption("Tell AI about yourself for personalized 'lucky' numbers")
                
                user_info = st.text_area(
                    "Tell me about yourself (optional):",
                    placeholder="e.g., 'My birthday is March 15, I love the number 7, my lucky color is blue, I was born in 1985...'",
                    height=100
                )
                
                if st.button("🍀 Generate My Lucky Numbers", type="primary"):
                    with st.spinner("✨ Finding your lucky numbers..."):
                        lucky = ai_lucky_numbers(openai_client, user_info, model=openai_model)
                    
                    if lucky:
                        st.success("🎉 Your Lucky Numbers!")
                        
                        cols = st.columns([1, 1, 1, 1, 1, 0.5, 1, 2])
                        
                        for j, num in enumerate(lucky['whites']):
                            cols[j].markdown(display_lottery_ball(num, size=70), unsafe_allow_html=True)
                        
                        cols[6].markdown(display_lottery_ball(lucky['powerball'], is_powerball=True, size=70), unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown(f"**✨ Why these numbers:** {lucky.get('meaning', 'AI-selected lucky combination')}")
                        
                        st.code(f"Lucky Pick: {' - '.join(map(str, lucky['whites']))} | PB: {lucky['powerball']}")
                    else:
                        st.error("Couldn't generate lucky numbers. Please try again.")
            
            # Chat Assistant
            with ai_tab4:
                st.subheader("💬 Lottery Chat Assistant")
                st.caption("Ask questions about lottery statistics, strategies, or get number suggestions")
                
                # Initialize chat history
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                # Display chat history
                chat_container = st.container()
                with chat_container:
                    for msg in st.session_state.chat_history:
                        if msg['role'] == 'user':
                            st.markdown(f"**🧑 You:** {msg['content']}")
                        else:
                            st.markdown(f"**🤖 Assistant:** {msg['content']}")
                        st.markdown("---")
                
                # Chat input
                col1, col2 = st.columns([5, 1])
                with col1:
                    user_message = st.text_input(
                        "Your message:",
                        placeholder="e.g., 'What are the odds of winning?' or 'Generate 3 balanced number sets'",
                        key="chat_input",
                        label_visibility="collapsed"
                    )
                with col2:
                    send_btn = st.button("Send", type="primary", use_container_width=True)
                
                if send_btn and user_message:
                    # Add user message to history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_message
                    })
                    
                    # Get AI response
                    with st.spinner("🤔 Thinking..."):
                        response = ai_chat_assistant(
                            openai_client, user_message, df, 
                            white_freq, pb_freq,
                            st.session_state.chat_history,
                            model=openai_model
                        )
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    st.rerun()
                
                # Clear chat button
                if st.session_state.chat_history:
                    if st.button("🗑️ Clear Chat"):
                        st.session_state.chat_history = []
                        st.rerun()
                
                # Suggested prompts
                with st.expander("💡 Suggested Questions"):
                    st.markdown("""
                    - "What are my odds of winning the jackpot?"
                    - "Explain the hot numbers strategy"
                    - "Generate 5 numbers using a balanced approach"
                    - "What's the most common sum of winning numbers?"
                    - "Should I pick my own numbers or use quick pick?"
                    - "What numbers haven't appeared recently?"
                    """)
    
    # TAB 3: Frequency Analysis (was TAB 2)
    with tab3:
        st.header("📊 Number Frequency Analysis")
        
        period_text = f"(Last {lookback} drawings)" if lookback else "(All time)"
        st.caption(period_text)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("White Ball Frequency")
            white_df = pd.DataFrame([
                {'Number': k, 'Frequency': v, 'Pct': v / sum(white_freq.values()) * 100}
                for k, v in sorted(white_freq.items())
            ])
            
            fig = px.bar(
                white_df, x='Number', y='Frequency',
                title='White Ball Frequency (1-69)',
                color='Frequency',
                color_continuous_scale='Blues',
                hover_data=['Pct']
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Powerball Frequency")
            pb_df = pd.DataFrame([
                {'Number': k, 'Frequency': v, 'Pct': v / sum(pb_freq.values()) * 100}
                for k, v in sorted(pb_freq.items())
            ])
            
            fig = px.bar(
                pb_df, x='Number', y='Frequency',
                title='Powerball Frequency (1-26)',
                color='Frequency',
                color_continuous_scale='Reds',
                hover_data=['Pct']
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        # Frequency tables
        with st.expander("📋 Detailed Frequency Tables"):
            tcol1, tcol2 = st.columns(2)
            with tcol1:
                st.write("**White Balls - Top 20**")
                top_white = white_df.nlargest(20, 'Frequency')
                st.dataframe(top_white, use_container_width=True, hide_index=True)
            with tcol2:
                st.write("**Powerball - All Numbers**")
                st.dataframe(pb_df.sort_values('Frequency', ascending=False), 
                           use_container_width=True, hide_index=True)
    
    # TAB 4: Hot & Cold Numbers
    with tab4:
        st.header("🔥 Hot & Cold Numbers")
        
        hot_white, cold_white = get_hot_cold_numbers(white_freq, n=10, number_range=69)
        hot_pb, cold_pb = get_hot_cold_numbers(pb_freq, n=5, number_range=26)
        overdue_white = get_overdue_numbers(df, number_range=69, is_powerball=False)[:10]
        overdue_pb = get_overdue_numbers(df, number_range=26, is_powerball=True)[:5]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔥 Hottest (Most Drawn)")
            
            st.write("**White Balls:**")
            hot_w_cols = st.columns(5)
            for i, num in enumerate(hot_white[:5]):
                hot_w_cols[i].metric(str(num), f"{white_freq[num]}x")
            hot_w_cols2 = st.columns(5)
            for i, num in enumerate(hot_white[5:10]):
                hot_w_cols2[i].metric(str(num), f"{white_freq[num]}x")
            
            st.write("**Powerball:**")
            hot_pb_cols = st.columns(5)
            for i, num in enumerate(hot_pb):
                hot_pb_cols[i].metric(str(num), f"{pb_freq[num]}x")
        
        with col2:
            st.subheader("❄️ Coldest (Least Drawn)")
            
            st.write("**White Balls:**")
            cold_w_cols = st.columns(5)
            for i, num in enumerate(cold_white[:5]):
                cold_w_cols[i].metric(str(num), f"{white_freq.get(num, 0)}x")
            cold_w_cols2 = st.columns(5)
            for i, num in enumerate(cold_white[5:10]):
                cold_w_cols2[i].metric(str(num), f"{white_freq.get(num, 0)}x")
            
            st.write("**Powerball:**")
            cold_pb_cols = st.columns(5)
            for i, num in enumerate(cold_pb):
                cold_pb_cols[i].metric(str(num), f"{pb_freq.get(num, 0)}x")
        
        with col3:
            st.subheader("⏰ Most Overdue")
            
            st.write("**White Balls:**")
            over_w_cols = st.columns(5)
            for i, num in enumerate(overdue_white[:5]):
                over_w_cols[i].markdown(f"**{num}**")
            over_w_cols2 = st.columns(5)
            for i, num in enumerate(overdue_white[5:10]):
                over_w_cols2[i].markdown(f"**{num}**")
            
            st.write("**Powerball:**")
            over_pb_cols = st.columns(5)
            for i, num in enumerate(overdue_pb):
                over_pb_cols[i].markdown(f"**{num}**")
    
    # TAB 5: Statistics
    with tab5:
        st.header("📈 Statistical Insights")
        
        # Calculate stats
        df_stats = df.copy()
        df_stats['sum'] = df_stats[['white_1', 'white_2', 'white_3', 'white_4', 'white_5']].sum(axis=1)
        df_stats['range'] = df_stats[['white_1', 'white_2', 'white_3', 'white_4', 'white_5']].max(axis=1) - \
                           df_stats[['white_1', 'white_2', 'white_3', 'white_4', 'white_5']].min(axis=1)
        
        # Key metrics
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        
        with mcol1:
            st.metric("Average Sum", f"{df_stats['sum'].mean():.1f}")
            st.caption(f"Std Dev: {df_stats['sum'].std():.1f}")
        with mcol2:
            st.metric("Median Sum", f"{df_stats['sum'].median():.0f}")
        with mcol3:
            st.metric("Sum Range", f"{df_stats['sum'].min():.0f} - {df_stats['sum'].max():.0f}")
        with mcol4:
            st.metric("Total Drawings", f"{len(df_stats):,}")
        
        # Sum distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df_stats, x='sum', nbins=40,
                title='Distribution of White Ball Sums',
                labels={'sum': 'Sum of 5 White Balls', 'count': 'Frequency'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.add_vline(x=df_stats['sum'].mean(), line_dash="dash", 
                         annotation_text=f"Mean: {df_stats['sum'].mean():.1f}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Odd/Even analysis
            def count_odds(row):
                return sum(1 for col in ['white_1', 'white_2', 'white_3', 'white_4', 'white_5'] 
                          if row[col] % 2 == 1)
            
            df_stats['odd_count'] = df_stats.apply(count_odds, axis=1)
            odd_dist = df_stats['odd_count'].value_counts().sort_index()
            
            fig = px.pie(
                values=odd_dist.values, 
                names=[f"{i} Odd / {5-i} Even" for i in odd_dist.index],
                title='Odd/Even Number Distribution',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional stats
        col3, col4 = st.columns(2)
        
        with col3:
            # High/Low distribution
            def count_high(row):
                return sum(1 for col in ['white_1', 'white_2', 'white_3', 'white_4', 'white_5'] 
                          if row[col] >= 35)
            
            df_stats['high_count'] = df_stats.apply(count_high, axis=1)
            high_dist = df_stats['high_count'].value_counts().sort_index()
            
            fig = px.bar(
                x=[f"{i} High / {5-i} Low" for i in high_dist.index],
                y=high_dist.values,
                title='High (35-69) vs Low (1-34) Distribution',
                labels={'x': 'Distribution', 'y': 'Count'},
                color_discrete_sequence=['#2ca02c']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Consecutive numbers
            def has_consecutive(row):
                nums = sorted([row[f'white_{i}'] for i in range(1, 6)])
                for i in range(len(nums) - 1):
                    if nums[i+1] - nums[i] == 1:
                        return True
                return False
            
            df_stats['has_consec'] = df_stats.apply(has_consecutive, axis=1)
            consec_pct = df_stats['has_consec'].mean() * 100
            
            fig = px.pie(
                values=[consec_pct, 100 - consec_pct],
                names=['Has Consecutive', 'No Consecutive'],
                title='Drawings with Consecutive Numbers',
                color_discrete_sequence=['#ff7f0e', '#d3d3d3']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 6: Cost Calculator
    with tab6:
        st.header("💰 Ticket Cost Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Calculate Your Investment")
            
            num_tickets = st.number_input("Number of Tickets:", min_value=1, max_value=100, value=5)
            power_play = st.checkbox("Add Power Play (+$1/ticket)", value=False, 
                                    help="Multiply non-jackpot prizes by 2x, 3x, 4x, 5x, or 10x")
            double_play = st.checkbox("Add Double Play (+$1/ticket)", value=False,
                                     help="Enter your numbers in a second drawing with set prizes")
            
            draws_per_week = st.selectbox("Drawings per week:", [1, 2, 3], index=1,
                                         help="Powerball draws Mon, Wed, Sat")
            weeks = st.slider("Number of weeks:", 1, 52, 1)
            
            total_tickets = num_tickets * draws_per_week * weeks
            total_cost = calculate_ticket_cost(total_tickets, power_play, double_play)
            
            st.markdown("---")
            
            st.metric("Total Tickets", f"{total_tickets:,}")
            st.metric("Total Cost", f"${total_cost:,.2f}")
            
            if weeks > 1:
                weekly = total_cost / weeks
                monthly = weekly * 4.33
                yearly = weekly * 52
                st.caption(f"Weekly: ${weekly:.2f} | Monthly: ~${monthly:.2f} | Yearly: ~${yearly:.2f}")
        
        with col2:
            st.subheader("💡 Prize Information")
            
            prize_data = pd.DataFrame({
                'Match': [
                    '5 + Powerball',
                    '5',
                    '4 + Powerball',
                    '4',
                    '3 + Powerball',
                    '3',
                    '2 + Powerball',
                    '1 + Powerball',
                    'Powerball only'
                ],
                'Prize': [
                    'JACKPOT',
                    '$1,000,000',
                    '$50,000',
                    '$100',
                    '$100',
                    '$7',
                    '$7',
                    '$4',
                    '$4'
                ],
                'Odds': [
                    '1 in 292,201,338',
                    '1 in 11,688,054',
                    '1 in 913,129',
                    '1 in 36,525',
                    '1 in 14,494',
                    '1 in 580',
                    '1 in 701',
                    '1 in 92',
                    '1 in 38'
                ]
            })
            
            st.dataframe(prize_data, use_container_width=True, hide_index=True)
            
            st.caption("*Power Play can multiply non-jackpot prizes. Match 5 with Power Play = $2,000,000")
            
            # Overall odds
            st.info("**Overall odds of winning any prize:** 1 in 24.9")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <strong>🎱 Powerball Predictor Pro</strong><br>
    Remember: The lottery is pure chance. Each drawing is independent.<br>
    Past results do NOT influence future outcomes. Play responsibly!<br>
    <em>Data source: NY Open Data API (data.ny.gov)</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
