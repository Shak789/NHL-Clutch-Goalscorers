#### Overview
This project applies machine learning techniques to identify and predict NHL forwards who excel in “clutch” situations (close, tied, and overtime games) during the regular season. The goal is not only to measure clutch performance, but also model expected clutch scoring given a player’s underlying metrics and understand the reasoning behind the predictions.

The full code for the model can be viewed [here](https://github.com/Shak789/NHL-Clutch-Goalscorers/blob/main/Predicting%20NHL%20Clutch%20Goalscorers.pdf).


#### Data Sources
* NHL API: Player biographical information and traditional goal statistics
* Natural Stat Trick:
  - Goals scored in tied, one-goal deficit, and overtime situations
  - Advanced performance metrics (scoring chances, assists, ice time, rebounds, offensive zone starts, shooting %)

#### Establishing a Definition of Clutch
A "clutch score" for each player is computed by weighting their goals in critical game situations:

- Goals per game when the team is down by one goal (45% weight)
- Goals per game when the game is tied (35% weight)
- Goals per game in overtime (20% weight)

While overtime goals have the highest Win Probability Added (WPA) of 100% and may be seen as the most important, they occur infrequenlty compared to other goals and all overtime periods are 5 minutes in the regular sesons. They are also scored during 3v3 play, which differs from regular 5v5. This means they received the lowest weight. Goals scored when the team is down by one goal received the highest weight because in comeback situations, there is more pressure to tie the game and avoid losing.  

#### Building a Classification Model
It seemed logical to classify players as "clutch" and "non-clutch". Thresholds were set for the clutch score and an XGBoost model was trained on data from the 2021-2022 to the 2023-2024 NHL season. The model used various underlying performance metrics such as expected goals, high-danger scoring chances, shot attempts. While the model was successful in identifying elite players and those below average, it struggled with players who fell near the classification boundaries, where small differences in their stats made it difficult to confidently label them as clutch or non-clutch. 

#### Switching to a Regression Model
Linear regression was a more feasible approach since many of the features were strongly correlated with a clutch score. It would be easier to predict a player's clutch score rather than assigning the player an ambiguous label. 

However, there was high multicollinearity among the features since they exhibited extreme Variance Inflation Factor (VIF) values of greater than 10. Therefore, ridge regression was used to limited the effect of multicollinearity on coefficients by shrinking correlated coefficients towards 0. This would improve their stability compared to standard OLS. Ridge regression was more appropriate than lasso regression, since it does not set coefficients to exactly 0 and preserves intepretability of the coefficients under multicollnearity. Ridge regression also reduced overfitting. After approximately 150 samples, both training and validation curves converge to an MSE of less than 2.

Time Series Cross-Validation was used to avoid leaking future information during traning due to the temporal nature of the data (2021-2024 seasons). The model showed good performance because it has a low MSE of approximately 1 and R² of 82% in training.

#### Dealing with Outliers
Cook's distance helped identify influential points. The model underpredicted the clutch score of elite players because their feature stats set a "ceiling" for their clutch ability. Some of these players  may have high leverage due to extreme. feature values (e.g. iSCF, SH%, assists). The model also overestimated some elite players who had strong underlying metrics but did not perform well in clutch games. Thus, a log transformation was applied on the clutch score, which enabled the model to generate better predictions for elite players and reduced the number of influential points. 

After the transformation, the model still undervalued some players who performed better in close and tied situations than their metrics suggest. While influential points are often viewed negatively, they can show which players perform better under pressure than their stats suggest. 

#### Prediction Intervals
95% prediction intervals were generated for each player. If actual clutch scores fall outside the intervals, this indicates that clutch performance is significantly different from expectations.
The intervals are generated using a bootstrap procedure with resampled residual noise, which ensures that the intervals reflect randomness in clutch performance. Approximately 5% of players fall outside range.

#### SHAP Values
SHAP values were calculated to explain which features most influenced each player's prediction. Due to the extremely high VIF values, multicollinearity may still be present even when using ridge regression. Therefore, SHAP is used with **feature_perturbation = "correlation_dependent"**. This accounts for correlations between the features when determining their contributions. Therefore, SHAP values will better reflect the true conditional contribution of each feature, rather than being distorted by multicollinearity.

#### Using the Model on a Final Test Set
The model was tested on player data from the 2024-2025 season to the current point of the 2025-2026 season. The R² of 77% indicates the model explains 77% of variance in clutch performance, which is strong given the inherent randomness in clutch situations. This is close to the R² of 82% in training, which suggests less overfitting.

### Temporal Stability
It is important to verify if clutch scoring truly exists. The year-over-year correlations (r = 0.57 for 2021-2022 vs 2022-2023, r = 0.54 for 2022-2023 vs 2023-2024, r = 0.52 for 2023-2024 vs 2024-2026) are all greater than 0.5, which shows clutch scoring is a repeatable skill rather than random variance.

#### Conclusion
Through this project, I hope that I developed a statistically sound goalscoring model. NHL fans, coaches and management can identify forwards who perform well in close game situations and use the regression model to determine if they are underperforming/overperforming expectations. The SHAP analysis should make the model less of a "black box" and enable users to gain more insight into playing styles that influence the predictions. For those more statistically inclined, the prediction intervals can show players who are truly "clutch". The influential points also identify genuinely clutch performers who exceed statistical expectations.

There are potential extensions for this model (e.g. including playoff data, goalie quality adjustments, venue effects). Third-period filtering would be ideal, as trailing with near the end of the game creates maximum pressure. Future versions could incorporate play-by-play timestamps. While the model has limitations, it provides a data-driven framework for evaluating clutch performance.
