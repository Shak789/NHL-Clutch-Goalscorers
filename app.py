from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.colors as mcolors
import os
from datetime import datetime

st.markdown("""
<style>
@media only screen and (max-width: 768px) {
    .mobile-notice {
        display: block !important;
    }
}
@media only screen and (min-width: 769px) {
    .mobile-notice {
        display: none !important;
    }
}
</style>

<div class="mobile-notice">
    <p style="
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    ">
        📱 Rotate to landscape for best experience
    </p>
</div>
""", unsafe_allow_html=True)


def load_data():
    return pd.read_csv("clutch.csv")

def load_data_historical():
    return pd.read_csv("21_24_clutch.csv")


df = load_data()
df_historical = load_data_historical()

df['teamAbbrevs'] = df['teamAbbrevs'].apply(lambda x: x.split(',')[0].strip() if ',' in x else x)
df['headshot'] = 'https://assets.nhle.com/mugs/nhl/20252026/' + df['teamAbbrevs'] + '/' + df['playerId'].astype(str) + '.png'
df['logo'] = 'https://assets.nhle.com/logos/nhl/svg/' + df['teamAbbrevs'] + '_dark.svg'


def format_top_features(row):
        shap_cols = [c for c in row.index if c.startswith('shap_')]
        impacts = row[shap_cols].abs().sort_values(ascending=False).head(3)
        features = [col.replace('shap_', '').replace('_per_game', '') for col in impacts.index]
        return ', '.join(features)

df['key_factors'] = df.apply(format_top_features, axis=1)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Player Profile", "Full Rankings", "Trends", "Model Performance", "Methodology", "Insights"])

with tab1:

    # Header with responsive columns
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("NHL Clutch Scorer Analysis")
        player_name = st.selectbox("Select a Player", sorted(df['Player'].unique()))

    player_data = df[df['Player'] == player_name].iloc[0]

    with col2:
        st.image(player_data['logo'], width = 500)  # Changed from width=500


    col_info, col_other = st.columns([2.5,1])  # Make col_info wider
    with col_info:
        with st.expander("ℹ️ About Clutch Score"):
            st.write("""
            Clutch score measures how often players score per 60 minutes during high-pressure situations:

            - When the game is tied
            - When the team is down by 1 goal
                     
            Higher scores mean a player scores more frequently in close and tied games. The formula is shown below:

            ![Formula](https://raw.githubusercontent.com/shak789/NHL-Clutch-Goalscorers/main/formula.png)
                                             
            A statistical model uses various underlying performance metrics (scoring chances, assists, rebounds created, shooting %)
            to predict a player's clutch score. The model’s **expected clutch score** can then be compared 
            to a player’s **actual clutch score** to determine whether they are **exceeding or underperforming expectations**. Players exceeding predictions perform better under pressure than their stats suggest.         

            Actual clutch scores reflect performance of forwards from the 2024-2025 season through the current point 
            of the 2025-2026 season. Players must have minimum 150 TOI in clutch situations (i.e. when the game is tied or when the team is down by 1 goal) and score at least 10 goals per season.
                     
            For details on the model, see the *Methodology* tab above
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
            st.metric("Current", f"{player_data['actual_clutch_score_adjusted']:.2f}")
        
        with col3:
            residual_pct = ((player_data['actual_clutch_score_adjusted'] - player_data['predicted_clutch_score_adjusted']) / 
                            player_data['predicted_clutch_score_adjusted'] * 100)
            st.metric("Difference", f"{residual_pct:.2f}%", delta=f"{residual_pct:.2f}%")
        
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        
        col_rank, col_tier = st.columns(2)
        
        with col_rank:
            rank = (df['actual_clutch_score_adjusted'] > player_data['actual_clutch_score_adjusted']).sum() + 1
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

    col_left, col_right = st.columns(2)

    with col_left:

        player_seasons = pd.concat([df, df_historical], ignore_index=True)
        season_card = ['2021-2022', '2022-2023', '2023-2024', '2024-2026']
        player_seasons = player_seasons.loc[player_seasons['season'].isin(season_card)]
        player_seasons['rank'] = player_seasons.groupby('season')['actual_clutch_score_adjusted'].rank(ascending=False, method='min')
        player_seasons = player_seasons[player_seasons['Player'] == player_name].sort_values('season')

        
        st.markdown("<h5 style='margin-top:5px; margin-bottom:5px;'>Clutch Score by Season</h5>", unsafe_allow_html=True)
        fig_line = go.Figure(data=[go.Scatter(
            x=player_seasons['season'],
            y=player_seasons['actual_clutch_score_adjusted'],
            customdata=player_seasons[['rank']],
            mode='lines+markers',
            line=dict(color="#4A90E2", width=3),
            marker=dict(size=10, color="#FFFFFF"),
            hovertemplate='<b>%{x}</b><br>Clutch Score: %{y:.2f}<br>Rank: %{customdata[0]:.0f}<extra></extra>'
        )])
        fig_line.update_layout(
            height=250,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title='Season',
            yaxis_title='Clutch Score',
            font=dict(size=14)
        )
        st.plotly_chart(fig_line)

    with col_right:
        st.markdown("<h5 style='margin-top:5px; margin-bottom:5px;'>Impact of Metrics on Predicted Clutch Score</h5>", unsafe_allow_html=True)
        shap_cols = [c for c in player_data.index if c.startswith('shap_')]
        feature_impacts = player_data[shap_cols].abs().sort_values(ascending=False)
        feature_impacts.index = feature_impacts.index.str.replace('shap_', '').str.replace('_per_game', '')
        
        feature_names = {
            'iSCF_per_60': 'Scoring Chances',
            'ixG': 'Expected Goals',
            'iCF': 'Shot Attempts',
            'iFF': 'Unblocked Shot Attempts',
            'iHDCF': 'High-Danger Chances',
            'shots': 'Shots',
            'assists_per_60': 'Assists',
            'rebounds_created_per_60': 'Rebounds Created',
            'off_zone_starts': 'Offensive Zone Starts',
            'SH%': 'Shooting %'
        }
        feature_impacts.index = feature_impacts.index.map(lambda x: feature_names.get(x, x))
        
        fig_bar = px.bar(x=feature_impacts.values, y=feature_impacts.index, 
                        orientation='h')
       
        feature_impacts_pct = (feature_impacts / feature_impacts.sum()) * 100
        fig_bar = px.bar(x=feature_impacts_pct.values, y=feature_impacts_pct.index, 
                 orientation='h')
        fig_bar.update_traces(
            marker_color='#4A90E2',
            hovertemplate='<b>%{y}</b><br>Impact: %{x:.2f}%<extra></extra>'
        )
        fig_bar.update_layout(
            height=250,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
            xaxis_title="Relative Importance (%)",
            yaxis_title="Metric",
        )
        st.plotly_chart(fig_bar)
    
            
    st.caption("Data retrieved from the NHL API and Natural Stat Trick. Players must have minimum 150 TOI in clutch situations (i.e. when the game is tied or when the team is down by 1 goal) and score at least 10 goals per season.") 
    file_time = os.path.getmtime('clutch.csv')
    last_updated = datetime.fromtimestamp(file_time)
    st.caption(f"Data last updated: {last_updated.strftime('%Y-%m-%d')}. Dashboard refreshed daily at 9:00 a.m. EST.")

    

with tab2:
    
    player_seasons = pd.concat([df, df_historical], ignore_index=True)

    player_seasons['teamAbbrevs'] = player_seasons['teamAbbrevs'].apply(lambda x: x.split(',')[0].strip() if ',' in x else x)

    selected_season = st.selectbox(
        "Select Season",
        options = player_seasons['season'].unique(),
        index=0
    )

    display_df = player_seasons.copy()

    display_df = display_df[display_df['season'] == selected_season]

    display_df = display_df.sort_values('actual_clutch_score_adjusted', ascending=False).reset_index(drop=True)
    display_df.index += 1

    # Only show intervals/SHAP for current season
    show_advanced = (selected_season == '2024-2026')

    base_cols = ['Player', 'teamAbbrevs', 'actual_clutch_score_adjusted']

    if show_advanced:
        display_df['interval'] = (
            "(" + display_df['lower_bound_log'].round(2).astype(str) + ", " + 
            display_df['upper_bound_log'].round(2).astype(str) + ")"
        )


        display_df['residual_pct'] = (
            (display_df['actual_clutch_score_adjusted'] - display_df['predicted_clutch_score_adjusted']) / 
            display_df['predicted_clutch_score_adjusted'] * 100
        ).round(2)
    
        feature_names = {
            'shap_iSCF_per_60': 'Scoring Chances Impact',
            'shap_assists_per_60': 'Assists Impact',
            'shap_rebounds_created_per_60': 'Rebounds Created Impact',
            'shap_SH%': 'Shooting % Impact'
        }
        
        shap_cols = [c for c in display_df.columns if c.startswith('shap_')]
        shap_display_cols = {}
        
        for old_name, new_name in feature_names.items():
            if old_name in display_df.columns:
                display_df[f'{old_name}_pct'] = (
                    display_df[old_name].abs() / display_df[shap_cols].abs().sum(axis=1) * 100
                )
                shap_display_cols[f'{old_name}_pct'] = st.column_config.NumberColumn(
                    new_name, format="%.2f%%"
                )
        
        display_cols = ['Player', 'teamAbbrevs', 'predicted_clutch_score_adjusted', 'actual_clutch_score_adjusted', 'residual_pct',  'interval', 'Significantly_Clutch'] + [f'{k}_pct' for k in feature_names.keys()]
        
        column_config = {
            "Player": "Player",
            "teamAbbrevs": "Team",
            "predicted_clutch_score_adjusted": "Predicted Score",
            "actual_clutch_score_adjusted": "Actual Score",
            "residual_pct": st.column_config.NumberColumn("Difference", format="%.2f%%"),
            "interval": "Predicted Score Range",
            "Significantly_Clutch": "Inside/Outside Score Range",
            **shap_display_cols
        }
    else:
        display_cols = base_cols
        column_config = {
            "Player": "Player",
            "teamAbbrevs": "Team",
            "predicted_clutch_score_adjusted": "Predicted Score",
            "actual_clutch_score_adjusted": "Actual Score",
            "residual_pct": st.column_config.NumberColumn("Difference", format="%.2f%%")
        }
        st.info("Predictions and impact of metrics available for 2024-2026 season only")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        player_search = st.text_input("Search Player Name", "")
    with col2:
        all_teams = ['All'] + sorted(display_df['teamAbbrevs'].unique().tolist())
        selected_team = st.selectbox("Filter by Team", all_teams)
    

    # Apply filters
    filtered_df = display_df[display_cols].copy()
    if player_search:
        filtered_df = filtered_df[filtered_df['Player'].str.contains(player_search, case=False, na=False)]
    if selected_team != 'All':
        filtered_df = filtered_df[filtered_df['teamAbbrevs'] == selected_team]
    
    st.dataframe(filtered_df, column_config=column_config, hide_index=False)
    st.caption("Data retrieved from the NHL API and Natural Stat Trick. Players must have minimum 150 TOI in clutch situations (i.e. when the game is tied or when the team is down by 1 goal) and score at least 10 goals per season.") 


with tab3:

    player_seasons = pd.concat([df, df_historical], ignore_index=True)

    player_name = st.selectbox("Select a Player", sorted(player_seasons['Player'].unique()), key="player_trends_select")

    player_seasons['rank'] = player_seasons.groupby('season')['actual_clutch_score_adjusted'].rank(ascending=False, method='min')
    player_seasons = player_seasons[player_seasons['Player'] == player_name].sort_values('season')

    
    st.markdown("<h5 style='margin-top:5px; margin-bottom:5px;'>Clutch Score by Season</h5>", unsafe_allow_html=True)
    fig_line = go.Figure(data=[go.Scatter(
        x=player_seasons['season'],
        y=player_seasons['actual_clutch_score_adjusted'],
        customdata=player_seasons[['rank']],
        mode='lines+markers',
        line=dict(color="#4A90E2", width=3),
        marker=dict(size=10, color="#FFFFFF"),
        hovertemplate='<b>%{x}</b><br>Clutch Score: %{y:.2f}<br>Rank: %{customdata[0]:.0f}<extra></extra>'
    )])
    y_min = player_seasons['actual_clutch_score_adjusted'].min()
    y_max = player_seasons['actual_clutch_score_adjusted'].max()


    fig_line.update_layout(
        height=400,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title='Season',
        yaxis_title='Clutch Score',
        yaxis=dict(
            autorange=False,
            range=[y_min * 0.95, y_max * 1.05]
        ),
        font=dict(size=14)
    )

    st.plotly_chart(fig_line, key="player_trends_chart")
    st.caption("Data retrieved from the NHL API and Natural Stat Trick. Players must have minimum 150 TOI in clutch situations (i.e. when the game is tied or when the team is down by 1 goal) and score at least 10 goals per season.") 


with tab4:

    col1new, col2new = st.columns(2)

    with col1new:
        # Player name search
        player_search = st.text_input("Search Player Name", "", key="player_search_input")
    
    with col2new:
        # Team filter
        all_teams = ['All'] + sorted(display_df['teamAbbrevs'].unique().tolist())
        selected_team = st.selectbox("Filter by Team", all_teams, key="team_filter_selectbox")

    st.subheader("Model Performance: Actual vs. Predicted")

    filtered_df = df.copy()

    # Filter by player name (case-insensitive)
    if player_search:
        filtered_df = filtered_df[filtered_df['Player'].str.contains(player_search, case=False, na=False)]

    # Filter by team
    if selected_team != 'All':
        filtered_df = filtered_df[filtered_df['teamAbbrevs'] == selected_team]

    # Create interactive scatter plot
    fig = px.scatter(
        filtered_df,
        x='actual_clutch_score_adjusted',
        y='predicted_clutch_score_adjusted',
        hover_data=['Player', 'teamAbbrevs'],
        labels={
            'actual_clutch_score_adjusted': 'Actual Clutch Score',
            'predicted_clutch_score_adjusted': 'Predicted Clutch Score'
        },
        title='Actual vs. Predicted Clutch Performance'
    )

    # Add diagonal reference line
    fig.add_scatter(
        x=[df['actual_clutch_score_adjusted'].min(), 
        df['actual_clutch_score_adjusted'].max()],
        y=[df['actual_clutch_score_adjusted'].min(), 
        df['actual_clutch_score_adjusted'].max()],
        mode='lines',
        line=dict(color='red', width=2),
        name='Perfect Prediction',
        showlegend=False
    )

    # Style
    fig.update_traces(
    hovertemplate='<b>%{customdata[0]}</b><br>' +
                  'Team: %{customdata[1]}<br>' +
                  'Actual: %{x:.2f}<br>' +
                  'Predicted: %{y:.2f}<br>' +
                  '<extra></extra>'
    )
    fig.update_layout(
        height=600,
        hovermode='closest'
    )

    st.plotly_chart(fig)
    st.caption("Data retrieved from the NHL API and Natural Stat Trick. Players must have minimum 150 TOI in clutch situations (i.e. when the game is tied or when the team is down by 1 goal) and score at least 10 goals per season.") 


with tab5:
    with open("README.md", "r", encoding="utf-8") as f:
        st.markdown(f.read())

with tab6:
    with open("Insights.md", "r", encoding="utf-8") as f:
        st.markdown(f.read())