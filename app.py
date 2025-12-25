from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.colors as mcolors

def load_data():
    BASE_DIR = Path(__file__).resolve().parent
    file_path = BASE_DIR / "clutch.xlsx"
    return pd.read_excel(file_path)

# Load data

df = load_data()
df['teamAbbrevs'] = df['teamAbbrevs'].apply(lambda x: x.split(',')[0].strip() if ',' in x else x)
df['headshot'] = 'https://assets.nhle.com/mugs/nhl/20242025/' + df['teamAbbrevs'] + '/' + df['playerId'].astype(str) + '.png'
df['logo'] = 'https://assets.nhle.com/logos/nhl/svg/' + df['teamAbbrevs'] + '_dark.svg'

tab1, tab2 = st.tabs(["Player Profile", "Full Rankings"])

with tab1:

    # Header with responsive columns
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("NHL Clutch Scorer Analysis")
        player_name = st.selectbox("Select a Player", sorted(df['Player'].unique()))

    player_data = df[df['Player'] == player_name].iloc[0]

    with col2:
        st.image(player_data['logo'], width = 500)  # Changed from width=500


    col_info, col_other = st.columns([2,1])  # Make col_info wider
    with col_info:
        with st.expander("ℹ️ About Clutch Score"):
            st.write("""
            **Clutch Score** weights goals scored in critical game situations:

            - **Tied games** (40%)  
            - **Down by 1** (40%)  
            - **Overtime** (20%)  

            A predictive model uses various underlying performance metrics (e.g. expected goals, high-danger scoring chances, shot attempts)
            to predict a player's clutch score. The model’s **expected clutch score** can then be compared 
            to a player’s **actual clutch score** to determine whether they are **exceeding or underperforming expectations**. Players exceeding predictions perform better under pressure than their stats suggest.
                    
            Actual clutch scores reflect performance of forwards from the 2024-2025 season through the current point 
            of the 2025-2026 season. Only players with 20+ total goals are displayed.
            """)



    # Main layout - use container_width for better mobile
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.image(player_data['headshot'])  # More responsive

    with col_right:
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted", f"{player_data['predicted_clutch_score_adjusted']:.2f}")
        
        with col2:
            st.metric("Current", f"{player_data['log_adjusted']:.2f}")
        
        with col3:
            residual_pct = ((player_data['log_adjusted'] - player_data['predicted_clutch_score_adjusted']) / 
                            player_data['predicted_clutch_score_adjusted'] * 100)
            st.metric("Difference", f"{residual_pct:.2f}%", delta=f"{residual_pct:.2f}%")
        
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        
        col_rank, col_tier = st.columns(2)
        
        with col_rank:
            rank = (df['log_adjusted'] > player_data['log_adjusted']).sum() + 1
            st.markdown(f"<h2 style='margin:0;'>{rank}</h2>", unsafe_allow_html=True)
            st.markdown("*Clutch Ranking*")
        
        def percentile_to_color(percentile):
            norm_value = percentile / 100
            cmap = mcolors.LinearSegmentedColormap.from_list("", ["#e74c3c", "#f39c12", "#2ecc71"])
            rgba = cmap(norm_value)
            return mcolors.rgb2hex(rgba)
        
        with col_tier:
            percentile = ((len(df) - rank) / len(df) * 100)
            color = percentile_to_color(percentile)
            st.markdown(f"<h2 style='color: {color}; font-weight: bold; margin:0;'>{round(percentile, 2)}%</h2>", unsafe_allow_html=True)
            st.markdown("*Percentile*")
        
        # Pie chart

    import plotly.graph_objects as go
    # Title with tight spacing
    st.markdown("<h3 style='margin-top:5px; margin-bottom:5px;'>Clutch Goal Breakdown</h3>", unsafe_allow_html=True)
    # Pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Down One', 'Tied', 'OT'],
        values=[player_data['goals_down_by_one'], player_data['goals_when_tied'], player_data['ot_goals']],
        hole=.4,
        marker=dict(colors=['#B12E38', '#276BB0', '#660089']),
        textfont=dict(size=15)              
    )])
    fig.update_layout(
    height=200,
    margin=dict(t=10, b=10, l=10, r=10),
    legend=dict(font=dict(size=14))      # Increase legend font size
    )

    col_left, col_right = st.columns([2.5, 1])  # left column smaller
    with col_left:
        st.plotly_chart(fig)
    with col_right:
        st.write("")  
        
    st.caption("Data retrieved from the NHL API and Natural Stat Trick") 
    

with tab2:
    st.title("NHL Clutch Rankings")

    df['interval'] = (
    "(" 
    + df['lower_bound_log'].round(2).astype(str)
    + ", "
    + df['upper_bound_log'].round(2).astype(str)
    + ")"
)


    
    # Prepare display dataframe
    display_df = df[['Player', 'teamAbbrevs', 'predicted_clutch_score_adjusted', 
                     'log_adjusted', 'interval', 'headshot', 'logo']].copy()
    
    
    display_df['residual_pct'] = ((display_df['log_adjusted'] - display_df['predicted_clutch_score_adjusted']) / 
                                   display_df['predicted_clutch_score_adjusted'] * 100).round(2)
    
    display_df = display_df.sort_values('log_adjusted', ascending=False).reset_index(drop=True)
    display_df.index += 1  # Start ranking at 1
    
    # Display with images (requires custom HTML or st.dataframe with column config)
    st.dataframe(
        display_df[['Player', 'teamAbbrevs', 'predicted_clutch_score_adjusted', 
                    'log_adjusted', 'residual_pct', 'interval']],
        column_config={
            "Player": "Player",
            "teamAbbrevs": "Team",
            "predicted_clutch_score_adjusted": "Predicted",
            "log_adjusted": "Actual",
            "residual_pct": st.column_config.NumberColumn(
                "Difference",
                format="%.2f%%"
            ),
            "interval": "Predicted Clutch Score Range"

        },
        hide_index=False
    )

