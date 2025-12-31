#### Overview
This project applies machine learning techniques to identify and predict NHL forwards who excel in “clutch” situations (close, tied, and overtime games) during the regular season. The goal is not only to measure clutch performance, but also model expected clutch scoring given a player’s underlying metrics and understand the reasoning behind the predictions.

The full code for the model can be viewed [here](https://github.com/Shak789/NHL-Clutch-Goalscorers/blob/main/Predicting%20NHL%20Clutch%20Goalscorers.pdf).


#### Data Sources
* NHL API: Player biographical information and traditional goal statistics
* Natural Stat Trick:
  - Goals scored in tied and one-goal deficit situations
  - Advanced performance metrics (scoring chances per 60, assists per 60, rebounds created per 60, shooting %)

#### Clutch Score Definition
Clutch Score measures how often players score per 60 minutes during high-pressure situations:

- When the game is tied
- When the team is down by 1 goal
                     
Higher scores mean a player scores more frequently in close and tied games. The formula is shown below:

![Formula](https://raw.githubusercontent.com/shak789/NHL-Clutch-Goalscorers/main/formula.png)

The clutch score is normalized per 60 to show if players are efficiently scoring in critical situations, rather than providing a high score when the player is simply deployed more. 

A log transformation is applied to the score (as explained in the outliers section) and it is scaled for interpretability. The relative values matter more than the absolute numbers (e.g. a score of 30 is better than 25).                     

To avoid unreliable samples, players must have minimum 150 TOI in clutch situations (i.e. when the game is tied or when the team is down by 1 goal) and score at least 10 goals per season.

#### Building a Classification Model
It seemed logical to classify players as "clutch" and "non-clutch". Thresholds were set for the clutch score and an XGBoost model was trained on data from the 2021-2022 to the 2023-2024 NHL season. The model used various underlying performance metrics such as expected goals, high-danger scoring chances, shot attempts. While the model was successful in identifying elite players and those below average, it struggled with players who fell near the classification boundaries, where small differences in their stats made it difficult to confidently label them as clutch or non-clutch. 

#### Switching to a Regression Model
Linear regression is a more feasible approach since many of the features are moderately/strongly correlated with clutch score. It is easier to predict a player's clutch score rather than assigning the player an ambiguous label. The model is trained on data from 2021-2022 to 2023-2024 regular seasons. It is tested on data from 2024-2025 to the current point of the 2025-2026 seasons. The testing results are the prediction and residual values shown on the other pages of the dashboard.

#### Features for Model
The following features were used to predict a player's clutch score:

- iSCF (Scoring Chances)
- Shooting %
- Rebounds Created
- Assists

As with the clutch score, these features are normalized per 60 so that they can measure scoring efficiency, not volume.

![Formula](https://raw.githubusercontent.com/shak789/NHL-Clutch-Goalscorers/main/Feature_Formula.png)

#### Ridge Regression
There is high multicollinearity among the features since they exhibit extreme Variance Inflation Factor (VIF) values of greater than 10. Therefore, this project uses ridge regression to limit the effect of multicollinearity on coefficients by shrinking correlated coefficients towards 0. This improves the stability of coefficients compared to standard OLS. Ridge regression is more appropriate than lasso or elastic net regression, since it does not set coefficients to exactly 0 and preserves intepretability of the coefficients under multicollnearity. 

Ridge regression also reduces overfitting. After approximately 150 samples, both training and validation curves converge:

![Formula](https://raw.githubusercontent.com/shak789/NHL-Clutch-Goalscorers/main/Learning_Curves.png)

Time Series Cross-Validation is used to avoid leaking future information during traning due to the temporal nature of the data (2021-2024 seasons). The model showed decent performance with an R² of 0.63 after training. 

#### Dealing with Outliers
Cook's distance helps identify influential points. The model underpredicts the clutch score of elite players because their feature stats set a "ceiling" for their clutch ability. Some of these players may have high leverage due to extreme feature values (e.g. iSCF, SH%, assists). The model also overestimates some elite players who have strong underlying metrics but do not perform well in clutch games. Thus, a log transformation is applied to the clutch score, which enables the model to generate better predictions for elite players and reduces the number of influential points.

After the transformation, the model still undervalues some players who perform better in close and tied situations than their metrics suggest. While influential points are often viewed negatively, they can show which players perform better under pressure than their stats suggest.

#### Prediction Intervals
95% prediction intervals are generated for each player. If actual clutch scores fall outside the intervals, this indicates that clutch performance is significantly different from expectations.
The intervals are generated using a bootstrap procedure with resampled residual noise, which ensures that the intervals reflect randomness in clutch performance. Approximately 5% of players are outside range.

#### SHAP Values
SHAP values were calculated to explain which features most influenced each player's prediction. Due to the extremely high VIF values, multicollinearity may still be present even when using ridge regression. Therefore, SHAP is used with **feature_perturbation = "correlation_dependent"**. This accounts for correlations between the features when determining their contributions. Therefore, SHAP values will better reflect the true conditional contribution of each feature, rather than being distorted by multicollinearity. The SHAP plot below shows good distribution of features:

![Formula](https://raw.githubusercontent.com/shak789/NHL-Clutch-Goalscorers/main/SHAP_Plot.png)

#### Using the Model on a Final Test Set
The model was tested on player data from the 2024-2025 season to the current point of the 2025-2026 season. The R² of approximately 56% indicates the model explains 56% of variance in clutch performance. While this is lower than the 63% R^2 in training, it is reasonable since it is much harder to predict the scoring efficiency of players in high-pressure situations with inherent randomness.

### Temporal Stability
It is important to verify if clutch scoring truly exists. The year-over-year correlations (r = 0.437 for 2021-2022 vs 2022-2023, r = 0.370 for 2022-2023 vs 2023-2024, r = 0.375 for 2023-2024 vs 2024-2026) are all greater than 0.3. This shows that clutch scoring has some repeatability, but noise still exists in these situations. 

#### Conclusion
Through this project, I hope that I developed a statistically sound goalscoring model. NHL fans, coaches and management can identify forwards who perform well in close game situations and use the regression model to determine if they are underperforming/overperforming expectations. The SHAP analysis should make the model less of a "black box" and enable users to gain more insight into playing styles that influence the predictions. For those more statistically inclined, the prediction intervals can show players who are truly "clutch". The influential points also identify genuinely clutch performers who exceed statistical expectations.

There are potential extensions for this model (e.g. including playoff data, goalie quality adjustments, venue effects). Third-period filtering would be ideal, as trailing with near the end of the game creates maximum pressure. Future versions could incorporate play-by-play timestamps. While the model has limitations, it provides a data-driven framework for evaluating clutch performance.
