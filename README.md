This project applies machine learning to identify and predict NHL forwards who excel in “clutch” situations (close, tied, and overtime games). The goal is not only to measure  clutch performance but to model expected clutch scoring given a player’s underlying metrics.

The final model has been deployed to a [Streamlit Dashboard](https://shak789-nhl-clutch-goalscorers-app-dpjtq2.streamlit.app/) that is updated at 9:00 a.m. EST daily.

### 1. Data Sources
* NHL API: Player biographical information and traditional goal statistics
* Natural Stat Trick:
  - Goals scored in tied, one-goal deficit, and overtime situations
  - Advanced performance metrics (scoring chances, assists, ice time, rebounds)

### 2. Establishing a Definition of Clutch
I computed a "clutch score" for each player by weighting their goals in critical game situations:

- Goals per game when the team is down by one goal (45% weight)
- Goals per game when the game is tied (35% weight)
- Goals per game in overtime (20% weight)

Goals when the team is down by one goal received the highest weight because situations where the team is down by one goal are more high pressure than when the game is tied. Overtime goals received the lowest weight because they occur infrequently compared to other goals. They are also only scored during 3v3 play, which differs from regular 5v5.

### 3. Building a Classification Model
I first tried classification since it seemed logical to classify players as "clutch" and "non-clutch". I set thresholds for the clutch score and trained an XGBoost model on data from the 2020-2021 to the 2023-2024 NHL season. The model used various underlying performance metrics such as expected goals, high-danger scoring chances, shot attempts. While the model was successful in identifying elite players and those below average, it struggled with players who fell near the classification boundaries, where small differences in their stats made it difficult to confidently label them as clutch or non-clutch. 

### 4. Switching to a Regression Model
Linear regression was a more feasible approach since many of the features were strongly correlated with a clutch score. It would be easier to predict a player's clutch score rather than assigning the player an ambiguous label.  

However, there was multicollinearity among features, which would lead to instability in coefficients and make it difficult to interpret the impact of features on the clutch score. Therefore, a small subset of features were kept (scoring chances, assists, time on ice, rebounds created, offensive zone starts). 

I also refined the model by using Ridge regression and performed cross-validation to ensure there was less overfitting.

### 5. Dealing with Outliers
I used Cook's distance to identify influential points. The model underpredicted the clutch score of elite players because their feature stats set a "ceiling" for their clutch ability. The model also overestimated some elite players who had strong underlying metrics but did not perform well in clutch games. In addition, the model struggled with below-average players who scored clutch goals at a rate that did not match their advanced stats.  This prompted me to use a log transformation on the clutch score, which enabled the model to generate better predictions for elite players and reduced the number of influential points. 

After the transformation, the model still undervalued some players who performed better in close and tied situations than their metrics suggest. While influential points are often viewed negatively, they can show which players perform better under pressure than their stats suggest. 

### 6. Prediction Intervals
95% prediction intervals were generated for each player. If actual clutch scores fall outside the intervals, this indicates that clutch performance is significantly different from expectations. The intervals are generated using a bootstrap procedure with resampled residual noise, which ensures that the intervals reflect randomness in clutch performance.

### 7. Shap Values
SHAP values were calculated to explain which features most influenced each player's prediction.

### 8. Using the Model on a Final Test Set
The model was tested on player data from the 2024-2025 season to the current point of the 2025-2026 season. The R² of 70% indicates the model explains 70% of variance in clutch performance, which is strong given the inherent randomness in clutch situations. 

### Conclusion
Through this project, I hope that NHL fans can identify forwards who perform well in close game situations and use the regression model to determine if they are underperforming/overperforming expectations. The SHAP analysis should make the model less of a "black box" and enable users to gain more insight into playing styles that influence the predictions. For those more statistically inclined, the prediction intervals can show players who are truly "clutch". The influential points also identify genuinely clutch performers who exceed statistical expectations.

There are potential extensions for this model (e.g. including playoff data, goalie quality adjustments, venue effects). Third-period filtering would be ideal, as trailing with near the end of the game creates maximum pressure. Future versions could incorporate play-by-play timestamps. While the model has limitations, it provides a data-driven framework for evaluating clutch performance.
