"""
Multi-Game Lottery Prediction Prototype
Supports: Powerball, Pick 3, Pick 4, Cash 5 (Jersey Cash 5)

Features:
- Real historical data from APIs
- Multiple prediction strategies per game
- Number exclusion filters
- Favorite numbers integration
- Ticket cost calculator
- OpenAI-powered analysis and predictions
- Fireball/Power Play options

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

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# =============================================================================
# GAME CONFIGURATIONS
# =============================================================================

GAME_CONFIG = {
    "powerball": {
        "name": "Powerball",
        "icon": "🎱",
        "main_numbers": 5,
        "main_range": (1, 69),
        "bonus_name": "Powerball",
        "bonus_range": (1, 26),
        "ticket_price": 2.00,
        "extras": {
            "Power Play": 1.00,
            "Double Play": 1.00
        },
        "draws_per_week": 3,
        "draw_days": "Mon, Wed, Sat @ 10:59 PM",
        "color": "#c41e3a",
        "jackpot_odds": "1 in 292,201,338",
        "any_prize_odds": "1 in 24.9"
    },
    "megamillions": {
        "name": "Mega Millions",
        "icon": "💰",
        "main_numbers": 5,
        "main_range": (1, 70),
        "bonus_name": "Mega Ball",
        "bonus_range": (1, 24),
        "ticket_price": 5.00,
        "extras": {},  # Multiplier now included in base price
        "draws_per_week": 2,
        "draw_days": "Tue, Fri @ 11:00 PM",
        "color": "#ffd700",
        "jackpot_odds": "1 in 302,575,350",
        "any_prize_odds": "1 in 24"
    },
    "pick6": {
        "name": "Pick 6",
        "icon": "6️⃣",
        "main_numbers": 6,
        "main_range": (1, 46),
        "bonus_name": None,
        "bonus_range": None,
        "ticket_price": 2.00,
        "extras": {
            "Double Play": 1.00
        },
        "draws_per_week": 3,
        "draw_days": "Mon, Thu, Sat @ 10:57 PM",
        "color": "#9c27b0",
        "jackpot_odds": "1 in 13,983,816",
        "any_prize_odds": "1 in 44.7"
    },
    "cash5": {
        "name": "Jersey Cash 5",
        "icon": "5️⃣",
        "main_numbers": 5,
        "main_range": (1, 45),
        "bonus_name": "XTRA",
        "bonus_range": None,  # XTRA is a multiplier, not a number
        "ticket_price": 1.00,
        "extras": {
            "XTRA": 1.00
        },
        "draws_per_week": 7,
        "draw_days": "Daily @ 10:57 PM",
        "color": "#00c853",
        "jackpot_odds": "1 in 1,221,759",
        "any_prize_odds": "1 in 10.3"
    },
    "cash4life": {
        "name": "Cash4Life",
        "icon": "💵",
        "main_numbers": 5,
        "main_range": (1, 60),
        "bonus_name": "Cash Ball",
        "bonus_range": (1, 4),
        "ticket_price": 2.00,
        "extras": {
            "Doubler NJ": 1.00
        },
        "draws_per_week": 7,
        "draw_days": "Daily @ 9:00 PM",
        "color": "#2e7d32",
        "jackpot_odds": "1 in 21,846,048",
        "any_prize_odds": "1 in 8",
        "top_prize": "$1,000/Day for Life"
    },
    "pick3": {
        "name": "Pick 3",
        "icon": "3️⃣",
        "main_numbers": 3,
        "main_range": (0, 9),
        "bonus_name": None,
        "bonus_range": None,
        "ticket_price": 0.50,
        "extras": {
            "Fireball": 1.00
        },
        "draws_per_week": 14,  # Twice daily
        "draw_days": "Daily (Midday 12:59 PM & Evening 10:57 PM)",
        "color": "#ff6b35",
        "play_types": ["Straight", "Box", "Straight/Box", "Front Pair", "Back Pair"],
        "jackpot_odds": "1 in 1,000 (Straight)",
        "any_prize_odds": "Varies by play type"
    },
    "pick4": {
        "name": "Pick 4",
        "icon": "4️⃣",
        "main_numbers": 4,
        "main_range": (0, 9),
        "bonus_name": None,
        "bonus_range": None,
        "ticket_price": 0.50,
        "extras": {
            "Fireball": 1.00
        },
        "draws_per_week": 14,
        "draw_days": "Daily (Midday 12:59 PM & Evening 10:57 PM)",
        "color": "#1e88e5",
        "play_types": ["Straight", "Box", "Straight/Box", "Front Pair", "Back Pair"],
        "jackpot_odds": "1 in 10,000 (Straight)",
        "any_prize_odds": "Varies by play type"
    }
}

# =============================================================================
# DATA GENERATION & FETCHING
# =============================================================================

@st.cache_data(ttl=3600)
def fetch_powerball_data():
    """Fetch real Powerball data from NY Open Data API."""
    try:
        url = "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        processed_data = []
        
        for _, row in df.iterrows():
            try:
                date = row['Draw Date']
                numbers = str(row['Winning Numbers']).split()
                
                if len(numbers) >= 6:
                    processed_data.append({
                        'date': date,
                        'num_1': int(numbers[0]),
                        'num_2': int(numbers[1]),
                        'num_3': int(numbers[2]),
                        'num_4': int(numbers[3]),
                        'num_5': int(numbers[4]),
                        'bonus': int(numbers[5]),
                        'multiplier': row.get('Multiplier', None)
                    })
            except (ValueError, IndexError):
                continue
        
        result_df = pd.DataFrame(processed_data)
        result_df['date'] = pd.to_datetime(result_df['date'])
        result_df = result_df.sort_values('date', ascending=False).reset_index(drop=True)
        
        return result_df, True
        
    except Exception as e:
        return generate_sample_powerball_data(), False


def generate_sample_powerball_data(n_drawings=500):
    """Generate sample Powerball data."""
    np.random.seed(42)
    data = []
    base_date = datetime.now() - timedelta(days=n_drawings * 2.5)
    
    for i in range(n_drawings):
        numbers = sorted(random.sample(range(1, 70), 5))
        bonus = random.randint(1, 26)
        draw_date = base_date + timedelta(days=i * 2.5)
        data.append({
            'date': draw_date,
            'num_1': numbers[0],
            'num_2': numbers[1],
            'num_3': numbers[2],
            'num_4': numbers[3],
            'num_5': numbers[4],
            'bonus': bonus,
            'multiplier': random.choice([2, 3, 4, 5, 10])
        })
    
    return pd.DataFrame(data)


def generate_sample_pick3_data(n_drawings=500):
    """Generate sample Pick 3 data."""
    np.random.seed(43)
    data = []
    base_date = datetime.now() - timedelta(days=n_drawings * 0.5)
    
    for i in range(n_drawings):
        numbers = [random.randint(0, 9) for _ in range(3)]
        draw_date = base_date + timedelta(days=i * 0.5)
        data.append({
            'date': draw_date,
            'num_1': numbers[0],
            'num_2': numbers[1],
            'num_3': numbers[2],
            'draw_type': random.choice(['Midday', 'Evening'])
        })
    
    return pd.DataFrame(data)


def generate_sample_pick4_data(n_drawings=500):
    """Generate sample Pick 4 data."""
    np.random.seed(44)
    data = []
    base_date = datetime.now() - timedelta(days=n_drawings * 0.5)
    
    for i in range(n_drawings):
        numbers = [random.randint(0, 9) for _ in range(4)]
        draw_date = base_date + timedelta(days=i * 0.5)
        data.append({
            'date': draw_date,
            'num_1': numbers[0],
            'num_2': numbers[1],
            'num_3': numbers[2],
            'num_4': numbers[3],
            'draw_type': random.choice(['Midday', 'Evening'])
        })
    
    return pd.DataFrame(data)


def generate_sample_cash5_data(n_drawings=500):
    """Generate sample Cash 5 data."""
    np.random.seed(45)
    data = []
    base_date = datetime.now() - timedelta(days=n_drawings)
    
    for i in range(n_drawings):
        numbers = sorted(random.sample(range(1, 46), 5))
        draw_date = base_date + timedelta(days=i)
        data.append({
            'date': draw_date,
            'num_1': numbers[0],
            'num_2': numbers[1],
            'num_3': numbers[2],
            'num_4': numbers[3],
            'num_5': numbers[4],
            'xtra': random.choice([2, 3, 4, 5])
        })
    
    return pd.DataFrame(data)


@st.cache_data(ttl=3600)
def fetch_mega_millions_data():
    """Fetch real Mega Millions data from NY Open Data API."""
    try:
        url = "https://data.ny.gov/api/views/5xaw-6ayf/rows.csv?accessType=DOWNLOAD"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        processed_data = []
        
        for _, row in df.iterrows():
            try:
                date = row['Draw Date']
                numbers = str(row['Winning Numbers']).split()
                
                if len(numbers) >= 6:
                    processed_data.append({
                        'date': date,
                        'num_1': int(numbers[0]),
                        'num_2': int(numbers[1]),
                        'num_3': int(numbers[2]),
                        'num_4': int(numbers[3]),
                        'num_5': int(numbers[4]),
                        'bonus': int(numbers[5]),
                        'multiplier': row.get('Mega Ball', row.get('Multiplier', None))
                    })
            except (ValueError, IndexError):
                continue
        
        result_df = pd.DataFrame(processed_data)
        result_df['date'] = pd.to_datetime(result_df['date'])
        result_df = result_df.sort_values('date', ascending=False).reset_index(drop=True)
        
        return result_df, True
        
    except Exception as e:
        return generate_sample_mega_millions_data(), False


def generate_sample_mega_millions_data(n_drawings=500):
    """Generate sample Mega Millions data."""
    np.random.seed(46)
    data = []
    base_date = datetime.now() - timedelta(days=n_drawings * 3.5)
    
    for i in range(n_drawings):
        numbers = sorted(random.sample(range(1, 71), 5))
        bonus = random.randint(1, 24)
        draw_date = base_date + timedelta(days=i * 3.5)
        data.append({
            'date': draw_date,
            'num_1': numbers[0],
            'num_2': numbers[1],
            'num_3': numbers[2],
            'num_4': numbers[3],
            'num_5': numbers[4],
            'bonus': bonus,
            'multiplier': random.choice([2, 3, 4, 5, 10])
        })
    
    return pd.DataFrame(data)


def generate_sample_pick6_data(n_drawings=500):
    """Generate sample Pick 6 data."""
    np.random.seed(47)
    data = []
    base_date = datetime.now() - timedelta(days=n_drawings * 2.5)
    
    for i in range(n_drawings):
        numbers = sorted(random.sample(range(1, 47), 6))
        draw_date = base_date + timedelta(days=i * 2.5)
        data.append({
            'date': draw_date,
            'num_1': numbers[0],
            'num_2': numbers[1],
            'num_3': numbers[2],
            'num_4': numbers[3],
            'num_5': numbers[4],
            'num_6': numbers[5],
            'xtra': random.choice([1, 2, 3, 4, 5, 10])
        })
    
    return pd.DataFrame(data)


def generate_sample_cash4life_data(n_drawings=500):
    """Generate sample Cash4Life data."""
    np.random.seed(48)
    data = []
    base_date = datetime.now() - timedelta(days=n_drawings)
    
    for i in range(n_drawings):
        numbers = sorted(random.sample(range(1, 61), 5))
        bonus = random.randint(1, 4)
        draw_date = base_date + timedelta(days=i)
        data.append({
            'date': draw_date,
            'num_1': numbers[0],
            'num_2': numbers[1],
            'num_3': numbers[2],
            'num_4': numbers[3],
            'num_5': numbers[4],
            'bonus': bonus
        })
    
    return pd.DataFrame(data)


def load_game_data(game_type):
    """Load data for specified game type."""
    if game_type == "powerball":
        df, is_live = fetch_powerball_data()
        return df, is_live
    elif game_type == "megamillions":
        df, is_live = fetch_mega_millions_data()
        return df, is_live
    elif game_type == "pick3":
        return generate_sample_pick3_data(), False
    elif game_type == "pick4":
        return generate_sample_pick4_data(), False
    elif game_type == "cash5":
        return generate_sample_cash5_data(), False
    elif game_type == "pick6":
        return generate_sample_pick6_data(), False
    elif game_type == "cash4life":
        return generate_sample_cash4life_data(), False
    return None, False


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_frequency(df, game_type, lookback=None):
    """Analyze frequency of numbers for any game type."""
    if lookback:
        df = df.head(lookback)
    
    config = GAME_CONFIG[game_type]
    num_count = config["main_numbers"]
    
    all_numbers = []
    for i in range(1, num_count + 1):
        col = f'num_{i}'
        if col in df.columns:
            all_numbers.extend(df[col].tolist())
    
    main_freq = Counter(all_numbers)
    
    bonus_freq = None
    if config["bonus_name"] and 'bonus' in df.columns:
        bonus_freq = Counter(df['bonus'].tolist())
    
    return main_freq, bonus_freq


def get_hot_cold_numbers(freq, n=10, number_range=(1, 69)):
    """Get hottest and coldest numbers."""
    min_num, max_num = number_range
    full_freq = {i: freq.get(i, 0) for i in range(min_num, max_num + 1)}
    sorted_nums = sorted(full_freq.items(), key=lambda x: x[1], reverse=True)
    
    hot = [num for num, count in sorted_nums[:n]]
    cold = [num for num, count in sorted_nums[-n:]]
    
    return hot, cold


def get_digit_frequency(df, position, game_type):
    """Get frequency of digits at a specific position (for Pick games)."""
    col = f'num_{position}'
    if col in df.columns:
        return Counter(df[col].tolist())
    return Counter()


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def generate_powerball_prediction(strategy, main_freq, bonus_freq, excluded_main=None, 
                                  excluded_bonus=None, favorites_main=None, favorites_bonus=None):
    """Generate Powerball predictions."""
    excluded_main = excluded_main or []
    excluded_bonus = excluded_bonus or []
    favorites_main = favorites_main or []
    favorites_bonus = favorites_bonus or []
    
    available_main = [n for n in range(1, 70) if n not in excluded_main]
    available_bonus = [n for n in range(1, 27) if n not in excluded_bonus]
    
    if strategy == "hot":
        hot_main, _ = get_hot_cold_numbers(main_freq, n=25, number_range=(1, 69))
        hot_bonus, _ = get_hot_cold_numbers(bonus_freq, n=15, number_range=(1, 26))
        pool_main = [n for n in hot_main if n in available_main][:20]
        pool_bonus = [n for n in hot_bonus if n in available_bonus][:10]
    elif strategy == "cold":
        _, cold_main = get_hot_cold_numbers(main_freq, n=25, number_range=(1, 69))
        _, cold_bonus = get_hot_cold_numbers(bonus_freq, n=15, number_range=(1, 26))
        pool_main = [n for n in cold_main if n in available_main][:20]
        pool_bonus = [n for n in cold_bonus if n in available_bonus][:10]
    elif strategy == "balanced":
        # Pick from different ranges
        ranges = [(1, 14), (15, 28), (29, 42), (43, 56), (57, 69)]
        pool_main = []
        for low, high in ranges:
            range_nums = [n for n in range(low, high + 1) if n in available_main]
            if range_nums:
                pool_main.extend(random.sample(range_nums, min(4, len(range_nums))))
        pool_bonus = available_bonus
    else:  # random
        pool_main = available_main
        pool_bonus = available_bonus
    
    # Start with favorites
    main_nums = [n for n in favorites_main if n in available_main][:5]
    
    # Fill remaining
    remaining = [n for n in pool_main if n not in main_nums]
    if len(remaining) < (5 - len(main_nums)):
        remaining = [n for n in available_main if n not in main_nums]
    
    main_nums.extend(random.sample(remaining, 5 - len(main_nums)))
    main_nums = sorted(main_nums[:5])
    
    # Bonus number
    if favorites_bonus and favorites_bonus[0] in available_bonus:
        bonus = favorites_bonus[0]
    elif pool_bonus:
        bonus = random.choice(pool_bonus)
    else:
        bonus = random.choice(available_bonus)
    
    return main_nums, bonus


def generate_pick_prediction(game_type, strategy, position_freq, excluded=None, favorites=None):
    """Generate Pick 3 or Pick 4 predictions."""
    excluded = excluded or []
    favorites = favorites or []
    
    config = GAME_CONFIG[game_type]
    num_count = config["main_numbers"]
    
    numbers = []
    
    for pos in range(1, num_count + 1):
        freq = position_freq.get(pos, Counter())
        
        available = [n for n in range(0, 10) if n not in excluded]
        
        if strategy == "hot":
            hot, _ = get_hot_cold_numbers(freq, n=5, number_range=(0, 9))
            pool = [n for n in hot if n in available] or available
        elif strategy == "cold":
            _, cold = get_hot_cold_numbers(freq, n=5, number_range=(0, 9))
            pool = [n for n in cold if n in available] or available
        elif strategy == "mixed":
            hot, cold = get_hot_cold_numbers(freq, n=4, number_range=(0, 9))
            pool = [n for n in (hot[:2] + cold[:2]) if n in available] or available
        else:  # random
            pool = available
        
        # Use favorite if available for this position
        if len(favorites) > pos - 1 and favorites[pos - 1] in available:
            numbers.append(favorites[pos - 1])
        else:
            numbers.append(random.choice(pool))
    
    return numbers


def generate_cash5_prediction(strategy, main_freq, excluded=None, favorites=None):
    """Generate Cash 5 predictions."""
    excluded = excluded or []
    favorites = favorites or []
    
    available = [n for n in range(1, 46) if n not in excluded]
    
    if strategy == "hot":
        hot, _ = get_hot_cold_numbers(main_freq, n=20, number_range=(1, 45))
        pool = [n for n in hot if n in available][:15]
    elif strategy == "cold":
        _, cold = get_hot_cold_numbers(main_freq, n=20, number_range=(1, 45))
        pool = [n for n in cold if n in available][:15]
    elif strategy == "balanced":
        ranges = [(1, 9), (10, 18), (19, 27), (28, 36), (37, 45)]
        pool = []
        for low, high in ranges:
            range_nums = [n for n in range(low, high + 1) if n in available]
            if range_nums:
                pool.extend(random.sample(range_nums, min(3, len(range_nums))))
    else:
        pool = available
    
    # Start with favorites
    main_nums = [n for n in favorites if n in available][:5]
    
    # Fill remaining
    remaining = [n for n in pool if n not in main_nums]
    if len(remaining) < (5 - len(main_nums)):
        remaining = [n for n in available if n not in main_nums]
    
    main_nums.extend(random.sample(remaining, 5 - len(main_nums)))
    
    return sorted(main_nums[:5])


def generate_pick6_prediction(strategy, main_freq, excluded=None, favorites=None):
    """Generate Pick 6 predictions."""
    excluded = excluded or []
    favorites = favorites or []
    
    available = [n for n in range(1, 47) if n not in excluded]
    
    if strategy == "hot":
        hot, _ = get_hot_cold_numbers(main_freq, n=20, number_range=(1, 46))
        pool = [n for n in hot if n in available][:18]
    elif strategy == "cold":
        _, cold = get_hot_cold_numbers(main_freq, n=20, number_range=(1, 46))
        pool = [n for n in cold if n in available][:18]
    elif strategy == "balanced":
        ranges = [(1, 8), (9, 16), (17, 24), (25, 32), (33, 40), (41, 46)]
        pool = []
        for low, high in ranges:
            range_nums = [n for n in range(low, high + 1) if n in available]
            if range_nums:
                pool.extend(random.sample(range_nums, min(3, len(range_nums))))
    else:
        pool = available
    
    # Start with favorites
    main_nums = [n for n in favorites if n in available][:6]
    
    # Fill remaining
    remaining = [n for n in pool if n not in main_nums]
    if len(remaining) < (6 - len(main_nums)):
        remaining = [n for n in available if n not in main_nums]
    
    main_nums.extend(random.sample(remaining, 6 - len(main_nums)))
    
    return sorted(main_nums[:6])


def generate_cash4life_prediction(strategy, main_freq, bonus_freq, excluded_main=None, 
                                  excluded_bonus=None, favorites_main=None, favorites_bonus=None):
    """Generate Cash4Life predictions."""
    excluded_main = excluded_main or []
    excluded_bonus = excluded_bonus or []
    favorites_main = favorites_main or []
    favorites_bonus = favorites_bonus or []
    
    available_main = [n for n in range(1, 61) if n not in excluded_main]
    available_bonus = [n for n in range(1, 5) if n not in excluded_bonus]
    
    if strategy == "hot":
        hot, _ = get_hot_cold_numbers(main_freq, n=25, number_range=(1, 60))
        pool = [n for n in hot if n in available_main][:20]
    elif strategy == "cold":
        _, cold = get_hot_cold_numbers(main_freq, n=25, number_range=(1, 60))
        pool = [n for n in cold if n in available_main][:20]
    elif strategy == "balanced":
        ranges = [(1, 12), (13, 24), (25, 36), (37, 48), (49, 60)]
        pool = []
        for low, high in ranges:
            range_nums = [n for n in range(low, high + 1) if n in available_main]
            if range_nums:
                pool.extend(random.sample(range_nums, min(4, len(range_nums))))
    else:
        pool = available_main
    
    # Start with favorites
    main_nums = [n for n in favorites_main if n in available_main][:5]
    
    # Fill remaining
    remaining = [n for n in pool if n not in main_nums]
    if len(remaining) < (5 - len(main_nums)):
        remaining = [n for n in available_main if n not in main_nums]
    
    main_nums.extend(random.sample(remaining, 5 - len(main_nums)))
    main_nums = sorted(main_nums[:5])
    
    # Bonus (Cash Ball 1-4)
    if favorites_bonus and favorites_bonus[0] in available_bonus:
        bonus = favorites_bonus[0]
    else:
        bonus = random.choice(available_bonus)
    
    return main_nums, bonus


def generate_megamillions_prediction(strategy, main_freq, bonus_freq, excluded_main=None, 
                                     excluded_bonus=None, favorites_main=None, favorites_bonus=None):
    """Generate Mega Millions predictions."""
    excluded_main = excluded_main or []
    excluded_bonus = excluded_bonus or []
    favorites_main = favorites_main or []
    favorites_bonus = favorites_bonus or []
    
    available_main = [n for n in range(1, 71) if n not in excluded_main]
    available_bonus = [n for n in range(1, 25) if n not in excluded_bonus]
    
    if strategy == "hot":
        hot, _ = get_hot_cold_numbers(main_freq, n=25, number_range=(1, 70))
        pool = [n for n in hot if n in available_main][:20]
    elif strategy == "cold":
        _, cold = get_hot_cold_numbers(main_freq, n=25, number_range=(1, 70))
        pool = [n for n in cold if n in available_main][:20]
    elif strategy == "balanced":
        ranges = [(1, 14), (15, 28), (29, 42), (43, 56), (57, 70)]
        pool = []
        for low, high in ranges:
            range_nums = [n for n in range(low, high + 1) if n in available_main]
            if range_nums:
                pool.extend(random.sample(range_nums, min(4, len(range_nums))))
    else:
        pool = available_main
    
    # Start with favorites
    main_nums = [n for n in favorites_main if n in available_main][:5]
    
    # Fill remaining
    remaining = [n for n in pool if n not in main_nums]
    if len(remaining) < (5 - len(main_nums)):
        remaining = [n for n in available_main if n not in main_nums]
    
    main_nums.extend(random.sample(remaining, 5 - len(main_nums)))
    main_nums = sorted(main_nums[:5])
    
    # Mega Ball
    if favorites_bonus and favorites_bonus[0] in available_bonus:
        bonus = favorites_bonus[0]
    elif bonus_freq:
        hot_bonus, _ = get_hot_cold_numbers(bonus_freq, n=10, number_range=(1, 24))
        pool_bonus = [n for n in hot_bonus if n in available_bonus]
        bonus = random.choice(pool_bonus) if pool_bonus else random.choice(available_bonus)
    else:
        bonus = random.choice(available_bonus)
    
    return main_nums, bonus


# =============================================================================
# OPENAI FUNCTIONS
# =============================================================================

def get_openai_client(api_key):
    """Initialize OpenAI client."""
    if not OPENAI_AVAILABLE:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def ai_generate_predictions(client, game_type, df, main_freq, bonus_freq, num_predictions,
                           excluded_main, excluded_bonus, favorites_main, favorites_bonus,
                           user_prompt, model):
    """Use AI to generate lottery predictions."""
    config = GAME_CONFIG[game_type]
    
    hot_main, cold_main = get_hot_cold_numbers(main_freq, n=10, number_range=config["main_range"])
    
    if game_type in ["pick3", "pick4"]:
        game_rules = f"""
GAME: {config['name']}
- Pick {config['main_numbers']} digits from 0-9
- Digits CAN repeat
- Order matters for Straight plays"""
    else:
        game_rules = f"""
GAME: {config['name']}
- Pick {config['main_numbers']} numbers from {config['main_range'][0]}-{config['main_range'][1]}
- Numbers must be unique and sorted"""
        if config["bonus_name"]:
            game_rules += f"\n- Pick 1 {config['bonus_name']} from {config['bonus_range'][0]}-{config['bonus_range'][1]}"
    
    prompt = f"""{game_rules}

HOT NUMBERS: {hot_main}
COLD NUMBERS: {cold_main}
EXCLUDE: {excluded_main or 'None'}
FAVORITES: {favorites_main or 'None'}

{f"USER REQUEST: {user_prompt}" if user_prompt else ""}

Generate {num_predictions} predictions. Return ONLY JSON:
{{
    "predictions": [
        {{"numbers": [1, 2, 3], "bonus": 5, "reasoning": "Brief reason"}}
    ],
    "strategy": "Brief explanation"
}}

For Pick 3/4, "bonus" should be null. Numbers can repeat for Pick games."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You generate lottery numbers with statistical reasoning while acknowledging randomness. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.8
        )
        
        content = response.choices[0].message.content
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None


def ai_analyze_patterns(client, game_type, df, main_freq, model):
    """Use AI to analyze patterns."""
    config = GAME_CONFIG[game_type]
    hot, cold = get_hot_cold_numbers(main_freq, n=10, number_range=config["main_range"])
    
    prompt = f"""Analyze {config['name']} lottery data:
- Total drawings: {len(df)}
- Hot numbers: {hot}
- Cold numbers: {cold}

Provide:
1. 3 interesting pattern observations
2. Any notable streaks
3. A fun fact
4. Probability insight

Be concise and remind users lottery is random."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Friendly lottery data analyst. Always mention randomness."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# UI HELPER FUNCTIONS
# =============================================================================

def display_ball(number, is_bonus=False, game_type="powerball", size=55):
    """Display a lottery ball with game-specific styling."""
    config = GAME_CONFIG[game_type]
    
    if is_bonus:
        bg = f"linear-gradient(135deg, {config['color']}, {config['color']}dd)"
        color = "white"
        border = config['color']
    else:
        bg = "linear-gradient(135deg, #ffffff, #e8e8e8)"
        color = "#333"
        border = "#ccc"
    
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
        box-shadow: 2px 2px 6px rgba(0,0,0,0.25);
        margin: 5px auto;
        border: 2px solid {border};
    '>{number}</div>
    """


def display_pick_numbers(numbers, game_type, size=55):
    """Display Pick 3/4 style numbers."""
    config = GAME_CONFIG[game_type]
    html = "<div style='display: flex; gap: 10px; justify-content: center;'>"
    for num in numbers:
        html += f"""
        <div style='
            background: linear-gradient(135deg, {config["color"]}, {config["color"]}cc);
            border-radius: 10px;
            width: {size}px;
            height: {size}px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: {int(size * 0.5)}px;
            font-weight: bold;
            color: white;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        '>{num}</div>
        """
    html += "</div>"
    return html


def calculate_cost(game_type, num_tickets, extras_selected):
    """Calculate total ticket cost."""
    config = GAME_CONFIG[game_type]
    base = num_tickets * config["ticket_price"]
    
    extras_cost = 0
    for extra_name, extra_price in config.get("extras", {}).items():
        if extra_name in extras_selected:
            extras_cost += num_tickets * extra_price
    
    return base + extras_cost


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Multi-Game Lottery Predictor",
        page_icon="🎰",
        layout="wide"
    )
    
    st.title("🎰 Multi-Game Lottery Predictor")
    st.caption("⚠️ DISCLAIMER: Lottery is pure random chance. No algorithm can predict winning numbers. Play responsibly!")
    
    # ==========================================================================
    # SIDEBAR
    # ==========================================================================
    
    st.sidebar.header("🎮 Game Selection")
    
    game_type = st.sidebar.selectbox(
        "Select Game:",
        list(GAME_CONFIG.keys()),
        format_func=lambda x: f"{GAME_CONFIG[x]['icon']} {GAME_CONFIG[x]['name']}"
    )
    
    config = GAME_CONFIG[game_type]
    
    st.sidebar.markdown(f"""
    **{config['icon']} {config['name']}**
    - Numbers: {config['main_numbers']} from {config['main_range'][0]}-{config['main_range'][1]}
    - Draws: {config['draw_days']}
    - Base Price: ${config['ticket_price']:.2f}
    """)
    
    st.sidebar.markdown("---")
    
    # OpenAI Configuration
    st.sidebar.subheader("🤖 OpenAI Integration")
    
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key:",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key for AI features"
    )
    
    openai_client = None
    if openai_api_key and OPENAI_AVAILABLE:
        if openai_api_key.startswith("sk-"):
            openai_client = get_openai_client(openai_api_key)
            if openai_client:
                st.sidebar.success("✅ API Key active")
        else:
            st.sidebar.warning("⚠️ Invalid key format")
    elif not OPENAI_AVAILABLE:
        st.sidebar.info("💡 Install openai package for AI features")
    else:
        st.sidebar.info("💡 Add API key for AI features")
    
    openai_model = st.sidebar.selectbox(
        "AI Model:",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Number Exclusions
    st.sidebar.subheader("🚫 Exclude Numbers")
    
    min_num, max_num = config["main_range"]
    number_options = list(range(min_num, max_num + 1))
    
    excluded_main = st.sidebar.multiselect(
        f"Exclude Main Numbers ({min_num}-{max_num}):",
        options=number_options,
        default=[]
    )
    
    excluded_bonus = []
    if config["bonus_range"]:
        bonus_min, bonus_max = config["bonus_range"]
        excluded_bonus = st.sidebar.multiselect(
            f"Exclude {config['bonus_name']} ({bonus_min}-{bonus_max}):",
            options=list(range(bonus_min, bonus_max + 1)),
            default=[]
        )
    
    # Favorites
    st.sidebar.subheader("⭐ Favorite Numbers")
    
    favorites_main = st.sidebar.multiselect(
        f"Favorite Main Numbers (max {config['main_numbers']}):",
        options=[n for n in number_options if n not in excluded_main],
        default=[],
        max_selections=config["main_numbers"]
    )
    
    favorites_bonus = []
    if config["bonus_range"]:
        bonus_options = [n for n in range(config["bonus_range"][0], config["bonus_range"][1] + 1) 
                        if n not in excluded_bonus]
        favorites_bonus = st.sidebar.multiselect(
            f"Favorite {config['bonus_name']} (max 1):",
            options=bonus_options,
            default=[],
            max_selections=1
        )
    
    # ==========================================================================
    # LOAD DATA
    # ==========================================================================
    
    with st.spinner(f"Loading {config['name']} data..."):
        df, is_live = load_game_data(game_type)
    
    if is_live:
        st.success(f"✅ Loaded {len(df)} live drawings")
    else:
        st.info(f"📋 Using {len(df)} sample drawings for demonstration")
    
    # Analyze frequency
    main_freq, bonus_freq = analyze_frequency(df, game_type)
    
    # Position frequency for Pick games
    position_freq = {}
    if game_type in ["pick3", "pick4"]:
        for pos in range(1, config["main_numbers"] + 1):
            position_freq[pos] = get_digit_frequency(df, pos, game_type)
    
    # ==========================================================================
    # MAIN TABS
    # ==========================================================================
    
    if game_type in ["pick3", "pick4"]:
        tabs = st.tabs([
            "🎯 Generate Numbers",
            "🤖 AI Features",
            "📊 Digit Analysis",
            "🔥 Hot & Cold",
            "💰 Cost Calculator"
        ])
    else:
        tabs = st.tabs([
            "🎯 Generate Numbers",
            "🤖 AI Features",
            "📊 Frequency Analysis",
            "🔥 Hot & Cold",
            "📈 Statistics",
            "💰 Cost Calculator"
        ])
    
    # ==========================================================================
    # TAB: GENERATE NUMBERS
    # ==========================================================================
    
    with tabs[0]:
        st.header(f"{config['icon']} Generate {config['name']} Numbers")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            strategy = st.selectbox(
                "Strategy:",
                ["🎲 Random", "🔥 Hot Numbers", "❄️ Cold Numbers", "⚖️ Balanced/Mixed"],
                key="strategy"
            )
        
        with col2:
            num_predictions = st.slider("Predictions:", 1, 20, 5)
        
        with col3:
            st.write("")
            generate_btn = st.button("🎰 Generate!", type="primary", use_container_width=True)
        
        # Show active filters
        if excluded_main or favorites_main:
            with st.expander("📋 Active Filters"):
                if excluded_main:
                    st.write(f"**Excluded:** {excluded_main}")
                if favorites_main:
                    st.write(f"**Favorites:** {favorites_main}")
        
        if generate_btn:
            st.subheader("🎱 Your Numbers")
            
            strat_map = {
                "🎲 Random": "random",
                "🔥 Hot Numbers": "hot",
                "❄️ Cold Numbers": "cold",
                "⚖️ Balanced/Mixed": "balanced" if game_type not in ["pick3", "pick4"] else "mixed"
            }
            strat = strat_map.get(strategy, "random")
            
            predictions = []
            
            for i in range(num_predictions):
                if game_type == "powerball":
                    main, bonus = generate_powerball_prediction(
                        strat, main_freq, bonus_freq,
                        excluded_main, excluded_bonus, favorites_main, favorites_bonus
                    )
                    predictions.append({"main": main, "bonus": bonus})
                elif game_type == "megamillions":
                    main, bonus = generate_megamillions_prediction(
                        strat, main_freq, bonus_freq,
                        excluded_main, excluded_bonus, favorites_main, favorites_bonus
                    )
                    predictions.append({"main": main, "bonus": bonus})
                elif game_type == "cash4life":
                    main, bonus = generate_cash4life_prediction(
                        strat, main_freq, bonus_freq,
                        excluded_main, excluded_bonus, favorites_main, favorites_bonus
                    )
                    predictions.append({"main": main, "bonus": bonus})
                elif game_type == "pick6":
                    main = generate_pick6_prediction(
                        strat, main_freq, excluded_main, favorites_main
                    )
                    predictions.append({"main": main, "bonus": None})
                elif game_type in ["pick3", "pick4"]:
                    numbers = generate_pick_prediction(
                        game_type, strat, position_freq, excluded_main, favorites_main
                    )
                    predictions.append({"main": numbers, "bonus": None})
                elif game_type == "cash5":
                    main = generate_cash5_prediction(
                        strat, main_freq, excluded_main, favorites_main
                    )
                    predictions.append({"main": main, "bonus": None})
            
            # Display predictions
            for i, pred in enumerate(predictions, 1):
                if game_type in ["pick3", "pick4"]:
                    cols = st.columns([1, 4, 3])
                    cols[0].markdown(f"**#{i}**")
                    cols[1].markdown(display_pick_numbers(pred["main"], game_type), unsafe_allow_html=True)
                    cols[2].code(f"{''.join(map(str, pred['main']))}")
                else:
                    num_cols = len(pred["main"]) + (2 if pred["bonus"] else 1)
                    cols = st.columns([0.5] + [1] * len(pred["main"]) + ([0.3, 1] if pred["bonus"] else []) + [2])
                    
                    cols[0].markdown(f"**#{i}**")
                    for j, num in enumerate(pred["main"]):
                        cols[j + 1].markdown(display_ball(num, game_type=game_type), unsafe_allow_html=True)
                    
                    if pred["bonus"]:
                        cols[len(pred["main"]) + 2].markdown(
                            display_ball(pred["bonus"], is_bonus=True, game_type=game_type),
                            unsafe_allow_html=True
                        )
                
                st.markdown("---")
            
            # Copy-friendly format
            st.subheader("📋 Copy Format")
            copy_lines = []
            for i, pred in enumerate(predictions, 1):
                if game_type in ["pick3", "pick4"]:
                    copy_lines.append(f"#{i}: {''.join(map(str, pred['main']))}")
                else:
                    line = f"#{i}: {' - '.join(map(str, pred['main']))}"
                    if pred["bonus"]:
                        line += f" | {config['bonus_name']}: {pred['bonus']}"
                    copy_lines.append(line)
            
            st.code("\n".join(copy_lines))
            
            cost = calculate_cost(game_type, num_predictions, [])
            st.info(f"💵 Base cost: **${cost:.2f}** for {num_predictions} tickets")
    
    # ==========================================================================
    # TAB: AI FEATURES
    # ==========================================================================
    
    with tabs[1]:
        st.header("🤖 AI-Powered Features")
        
        if not openai_client:
            st.warning("⚠️ Enter your OpenAI API key in the sidebar to use AI features.")
            st.info("""
            **Get an API key:**
            1. Visit [platform.openai.com](https://platform.openai.com)
            2. Create account → API Keys → Create new key
            3. Paste in sidebar
            """)
        else:
            ai_tab1, ai_tab2 = st.tabs(["🔮 AI Predictions", "📊 AI Analysis"])
            
            with ai_tab1:
                st.subheader(f"🔮 AI {config['name']} Predictions")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    ai_prompt = st.text_area(
                        "Special instructions (optional):",
                        placeholder="e.g., 'Use more cold numbers' or 'Avoid sequences'",
                        height=80
                    )
                with col2:
                    ai_num = st.slider("Predictions:", 1, 10, 5, key="ai_num")
                    ai_btn = st.button("🤖 Generate", type="primary", use_container_width=True)
                
                if ai_btn:
                    with st.spinner("🧠 AI analyzing..."):
                        result = ai_generate_predictions(
                            openai_client, game_type, df, main_freq, bonus_freq,
                            ai_num, excluded_main, excluded_bonus,
                            favorites_main, favorites_bonus, ai_prompt, openai_model
                        )
                    
                    if result and "predictions" in result:
                        st.success(f"**Strategy:** {result.get('strategy', 'AI-generated')}")
                        
                        for i, pred in enumerate(result["predictions"], 1):
                            nums = pred.get("numbers", [])
                            bonus = pred.get("bonus")
                            reason = pred.get("reasoning", "")
                            
                            if game_type in ["pick3", "pick4"]:
                                cols = st.columns([1, 3, 4])
                                cols[0].markdown(f"**#{i}**")
                                cols[1].markdown(display_pick_numbers(nums[:config["main_numbers"]], game_type), unsafe_allow_html=True)
                                cols[2].caption(f"💡 {reason}")
                            else:
                                st.markdown(f"**#{i}:** {' - '.join(map(str, nums[:config['main_numbers']]))}" + 
                                           (f" | {config['bonus_name']}: {bonus}" if bonus else "") +
                                           f" — _{reason}_")
                            st.markdown("---")
                    else:
                        st.error("Failed to generate. Try again.")
            
            with ai_tab2:
                st.subheader("📊 AI Pattern Analysis")
                
                if st.button("🔍 Analyze", type="primary"):
                    with st.spinner("🧠 Analyzing..."):
                        analysis = ai_analyze_patterns(openai_client, game_type, df, main_freq, openai_model)
                    st.markdown(analysis)
    
    # ==========================================================================
    # TAB: FREQUENCY / DIGIT ANALYSIS
    # ==========================================================================
    
    with tabs[2]:
        if game_type in ["pick3", "pick4"]:
            st.header("📊 Digit Frequency by Position")
            
            cols = st.columns(config["main_numbers"])
            
            for pos in range(1, config["main_numbers"] + 1):
                with cols[pos - 1]:
                    st.subheader(f"Position {pos}")
                    freq = position_freq.get(pos, Counter())
                    
                    freq_df = pd.DataFrame([
                        {"Digit": k, "Count": v}
                        for k, v in sorted(freq.items())
                    ])
                    
                    fig = px.bar(freq_df, x="Digit", y="Count",
                                color="Count", color_continuous_scale="Blues")
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.header("📊 Number Frequency Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Main Numbers")
                freq_df = pd.DataFrame([
                    {"Number": k, "Frequency": v}
                    for k, v in sorted(main_freq.items())
                ])
                
                fig = px.bar(freq_df, x="Number", y="Frequency",
                            color="Frequency", color_continuous_scale="Blues")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            if bonus_freq:
                with col2:
                    st.subheader(config["bonus_name"])
                    bonus_df = pd.DataFrame([
                        {"Number": k, "Frequency": v}
                        for k, v in sorted(bonus_freq.items())
                    ])
                    
                    fig = px.bar(bonus_df, x="Number", y="Frequency",
                                color="Frequency", color_continuous_scale="Reds")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # TAB: HOT & COLD
    # ==========================================================================
    
    with tabs[3]:
        st.header("🔥 Hot & Cold Numbers")
        
        if game_type in ["pick3", "pick4"]:
            st.subheader("By Position")
            
            for pos in range(1, config["main_numbers"] + 1):
                freq = position_freq.get(pos, Counter())
                hot, cold = get_hot_cold_numbers(freq, n=3, number_range=(0, 9))
                
                cols = st.columns([1, 2, 2])
                cols[0].markdown(f"**Position {pos}**")
                cols[1].markdown(f"🔥 Hot: **{', '.join(map(str, hot))}**")
                cols[2].markdown(f"❄️ Cold: **{', '.join(map(str, cold))}**")
        else:
            hot_main, cold_main = get_hot_cold_numbers(main_freq, n=10, number_range=config["main_range"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔥 Hot (Most Frequent)")
                hot_cols = st.columns(5)
                for i, num in enumerate(hot_main[:5]):
                    hot_cols[i].metric(str(num), f"{main_freq[num]}x")
                if len(hot_main) > 5:
                    hot_cols2 = st.columns(5)
                    for i, num in enumerate(hot_main[5:10]):
                        hot_cols2[i].metric(str(num), f"{main_freq[num]}x")
            
            with col2:
                st.subheader("❄️ Cold (Least Frequent)")
                cold_cols = st.columns(5)
                for i, num in enumerate(cold_main[:5]):
                    cold_cols[i].metric(str(num), f"{main_freq.get(num, 0)}x")
                if len(cold_main) > 5:
                    cold_cols2 = st.columns(5)
                    for i, num in enumerate(cold_main[5:10]):
                        cold_cols2[i].metric(str(num), f"{main_freq.get(num, 0)}x")
            
            if bonus_freq:
                st.markdown("---")
                hot_bonus, cold_bonus = get_hot_cold_numbers(bonus_freq, n=5, number_range=config["bonus_range"])
                
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    st.markdown(f"**🔥 Hot {config['bonus_name']}:** {', '.join(map(str, hot_bonus))}")
                with bcol2:
                    st.markdown(f"**❄️ Cold {config['bonus_name']}:** {', '.join(map(str, cold_bonus))}")
    
    # ==========================================================================
    # TAB: STATISTICS (only for non-Pick games)
    # ==========================================================================
    
    if game_type not in ["pick3", "pick4"]:
        with tabs[4]:
            st.header("📈 Statistics")
            
            # Calculate sum stats
            num_cols = [f'num_{i}' for i in range(1, config["main_numbers"] + 1)]
            df_stats = df.copy()
            df_stats['sum'] = df_stats[num_cols].sum(axis=1)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Sum", f"{df_stats['sum'].mean():.1f}")
            col2.metric("Min Sum", f"{df_stats['sum'].min()}")
            col3.metric("Max Sum", f"{df_stats['sum'].max()}")
            
            fig = px.histogram(df_stats, x='sum', nbins=30,
                              title="Sum Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Odd/Even
            def count_odds(row):
                return sum(1 for col in num_cols if row[col] % 2 == 1)
            
            df_stats['odds'] = df_stats.apply(count_odds, axis=1)
            odd_dist = df_stats['odds'].value_counts().sort_index()
            
            fig = px.pie(values=odd_dist.values,
                        names=[f"{i} Odd / {config['main_numbers']-i} Even" for i in odd_dist.index],
                        title="Odd/Even Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================================================
    # TAB: COST CALCULATOR
    # ==========================================================================
    
    cost_tab_index = 4 if game_type in ["pick3", "pick4"] else 5
    
    with tabs[cost_tab_index]:
        st.header("💰 Ticket Cost Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Calculate Cost")
            
            num_tickets = st.number_input("Number of tickets:", 1, 100, 5)
            
            # Extras
            selected_extras = []
            if config.get("extras"):
                st.write("**Add-ons:**")
                for extra_name, extra_price in config["extras"].items():
                    if st.checkbox(f"{extra_name} (+${extra_price:.2f}/ticket)"):
                        selected_extras.append(extra_name)
            
            # Play type for Pick games
            if game_type in ["pick3", "pick4"]:
                play_type = st.selectbox("Play Type:", config.get("play_types", ["Straight"]))
                st.caption("Different play types have different odds and payouts")
            
            draws = st.slider("Number of draws:", 1, 14, 1)
            
            total_tickets = num_tickets * draws
            total_cost = calculate_cost(game_type, total_tickets, selected_extras)
            
            st.markdown("---")
            st.metric("Total Tickets", total_tickets)
            st.metric("Total Cost", f"${total_cost:.2f}")
        
        with col2:
            st.subheader("💡 Game Info")
            
            st.markdown(f"""
            **{config['icon']} {config['name']}**
            
            - **Base Price:** ${config['ticket_price']:.2f}
            - **Draws:** {config['draw_days']}
            - **Numbers:** Pick {config['main_numbers']} from {config['main_range'][0]}-{config['main_range'][1]}
            """)
            
            if config["bonus_name"] and config["bonus_range"]:
                st.markdown(f"- **{config['bonus_name']}:** Pick 1 from {config['bonus_range'][0]}-{config['bonus_range'][1]}")
            
            st.markdown("---")
            st.markdown("**Odds:**")
            st.markdown(f"- **Jackpot:** {config.get('jackpot_odds', 'N/A')}")
            st.markdown(f"- **Any Prize:** {config.get('any_prize_odds', 'N/A')}")
            
            # Game-specific prize info
            if game_type == "pick3":
                st.markdown("""
                **Pick 3 Prizes:**
                | Play Type | Odds | Prize |
                |-----------|------|-------|
                | Straight | 1:1,000 | $500 |
                | Box (3-way) | 1:333 | $160 |
                | Box (6-way) | 1:167 | $80 |
                | Front/Back Pair | 1:100 | $50 |
                """)
            elif game_type == "pick4":
                st.markdown("""
                **Pick 4 Prizes:**
                | Play Type | Odds | Prize |
                |-----------|------|-------|
                | Straight | 1:10,000 | $5,000 |
                | Box (4-way) | 1:2,500 | $1,200 |
                | Box (6-way) | 1:1,667 | $800 |
                | Box (12-way) | 1:833 | $400 |
                | Box (24-way) | 1:417 | $200 |
                """)
            elif game_type == "cash5":
                st.markdown("""
                **Jersey Cash 5 Prizes:**
                | Match | Odds | Prize |
                |-------|------|-------|
                | 5 of 5 | 1:1,221,759 | Jackpot |
                | 4 of 5 | 1:7,332 | $500 |
                | 3 of 5 | 1:170 | $15 |
                | 2 of 5 | 1:11 | Free Pick |
                """)
            elif game_type == "pick6":
                st.markdown("""
                **Pick 6 Prizes:**
                | Match | Odds | Prize |
                |-------|------|-------|
                | 6 of 6 | 1:13,983,816 | Jackpot |
                | 5 of 6 | 1:54,201 | ~$1,000+ |
                | 4 of 6 | 1:1,032 | ~$50+ |
                | 3 of 6 | 1:57 | $3 |
                """)
            elif game_type == "cash4life":
                st.markdown("""
                **Cash4Life Prizes:**
                | Match | Prize |
                |-------|-------|
                | 5 + Cash Ball | $1,000/Day for Life |
                | 5 | $1,000/Week for Life |
                | 4 + Cash Ball | $2,500 |
                | 4 | $500 |
                | 3 + Cash Ball | $100 |
                | 3 | $25 |
                """)
            elif game_type == "megamillions":
                st.markdown("""
                **Mega Millions Prizes:**
                | Match | Prize |
                |-------|-------|
                | 5 + Mega Ball | JACKPOT |
                | 5 | $1,000,000 |
                | 4 + Mega Ball | $10,000 |
                | 4 | $500 |
                | 3 + Mega Ball | $200 |
                | 3 | $10 |
                
                *Multiplier (2x-10x) included in $5 ticket price*
                """)
            elif game_type == "powerball":
                st.markdown("""
                **Powerball Prizes:**
                | Match | Prize |
                |-------|-------|
                | 5 + Powerball | JACKPOT |
                | 5 | $1,000,000 |
                | 4 + Powerball | $50,000 |
                | 4 | $100 |
                | 3 + Powerball | $100 |
                | 3 | $7 |
                """)
    
    # Footer
    st.markdown("---")
    st.caption(f"🎰 Multi-Game Lottery Predictor | {config['icon']} {config['name']} Mode | Remember: Lottery is random chance. Play responsibly!")


if __name__ == "__main__":
    main()
