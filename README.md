#### Overview

This project applies machine learning techniques to identify and predict NHL forwards who excel in “clutch” situations (close, tied, and overtime games). The goal is not only to measure clutch performance, but also model expected clutch scoring given a player’s underlying metrics and understand the reasoning behind the predictions.

The full code for the model can be viewed [here](https://github.com/Shak789/NHL-Clutch-Goalscorers/blob/main/Predicting%20NHL%20Clutch%20Goalscorers.pdf).

#### Data Sources
* NHL API: Player biographical information and traditional goal statistics
* Natural Stat Trick:
  - Goals scored in tied, one-goal deficit, and overtime situations
  - Advanced performance metrics (scoring chances, assists, ice time, rebounds, offensive zone starts)

#### Establishing a Definition of Clutch
A "clutch score" for each player is computed by weighting their goals in critical game situations:

- Goals per game when the team is down by one goal (45% weight)
- Goals per game when the game is tied (35% weight)
- Goals per game in overtime (20% weight)

Goals scored when the team is down by one goal received the highest weight because in comeback situations, there is more pressure to tie the game and avoid losing. Overtime goals received the lowest weight because they occur infrequently compared to other goals. They are also only scored during 3v3 play, which differs from regular 5v5.

#### Building a Classification Model
It seemed logical to classify players as "clutch" and "non-clutch". Thresholds were set for the clutch score and an XGBoost model was trained on data from the 2021-2022 to the 2023-2024 NHL season. The model used various underlying performance metrics such as expected goals, high-danger scoring chances, shot attempts. While the model was successful in identifying elite players and those below average, it struggled with players who fell near the classification boundaries, where small differences in their stats made it difficult to confidently label them as clutch or non-clutch. 

#### Switching to a Regression Model
Linear regression was a more feasible approach since many of the features were strongly correlated with a clutch score. It would be easier to predict a player's clutch score rather than assigning the player an ambiguous label.  

However, there was multicollinearity among features, which would lead to instability in coefficients and make it difficult to interpret the impact of features on the clutch score. Therefore, a small subset of features were kept (scoring chances, assists, time on ice, rebounds created, offensive zone starts). 

The model was refined using Ridge regression and cross-validation to ensure there was less overfitting.

#### Dealing with Outliers
Cook's distance helped identify influential points. The model underpredicted the clutch score of elite players because their feature stats set a "ceiling" for their clutch ability. The model also overestimated some elite players who had strong underlying metrics but did not perform well in clutch games. In addition, the model struggled with below-average players who scored clutch goals at a rate that did not match their advanced stats.  This resulted in a log transformation on the clutch score, which enabled the model to generate better predictions for elite players and reduced the number of influential points. 

After the transformation, the model still undervalued some players who performed better in close and tied situations than their metrics suggest. While influential points are often viewed negatively, they can show which players perform better under pressure than their stats suggest. 

#### Prediction Intervals
95% prediction intervals were generated for each player. If actual clutch scores fall outside the intervals, this indicates that clutch performance is significantly different from expectations. The intervals are generated using a bootstrap procedure with resampled residual noise, which ensures that the intervals reflect randomness in clutch performance.

#### SHAP Values
SHAP values were calculated to explain which features most influenced each player's prediction.

#### Using the Model on a Final Test Set
The model was tested on player data from the 2024-2025 season to the current point of the 2025-2026 season. The R² of 65% indicates the model explains 65% of variance in clutch performance, which is strong given the inherent randomness in clutch situations. 

#### Conclusion
Through this project, I hope that NHL fans can identify forwards who perform well in close game situations and use the regression model to determine if they are underperforming/overperforming expectations. The SHAP analysis should make the model less of a "black box" and enable users to gain more insight into playing styles that influence the predictions. For those more statistically inclined, the prediction intervals can show players who are truly "clutch". The influential points also identify genuinely clutch performers who exceed statistical expectations.

There are potential extensions for this model (e.g. including playoff data, goalie quality adjustments, venue effects). Third-period filtering would be ideal, as trailing with near the end of the game creates maximum pressure. Future versions could incorporate play-by-play timestamps. While the model has limitations, it provides a data-driven framework for evaluating clutch performance.
