# Predicting Clutch Goalscorers in the NHL

In this project, I explored various machine learning techniques to determine the best performing NHL forwards in close and tied games (often referred to as "clutch" moments).

### 1. Data Sources
I extracted data from the NHL API and Natural Stat Trick. The NHL API offer a good foundation for player bios and common goal statistics. Natural Stat Trick provides goals scored by players in close and tied games, as well as advanced stats that can be used as features for a machine learning model.

### 2. Establishing a Definition of Clutch
I computed a "clutch score" for each player by weighting their goals in critical game situations:

- Goals per game when the game is tied (40% weight)
- Goals per game when the team is down by one goal (40% weight)
- Goals per game in overtime (20% weight)

### 3. Choosing features
Various underlying performance metrics (e.g. expected goals, high-danger scoring chances, shot attempts) were used to predict a player's clutch score. 

### 4. Building a Classification Model
I attempted to classify players as "clutch" and "non-clutch" by setting thresholds for the clutch score, which was trained on data from the 2020-2021 to the 2023-2024 NHL season. While the model was successful in identifying elite players and those below average, it struggled with players who fell near the classification boundaries, where small differences in their stats made it difficult to confidently label them as clutch or non-clutch.

### 5. Switching to a Regression Model
Linear regression was a more feasible approach since many of the features were strongly correlated with a clutch score. It would, therefore, be easier to predict a player's clutch score rather than assigning the player an ambiguous label.  

I refined the model by using Ridge regression and performed cross-validation to ensure there was no overfitting.

### 6. Dealing with Outliers
I used Cook's Distance to identify influential points. The model underpredicted the clutch score of elite players because their feature stats set a "ceiling" for their clutch ability. The model also overestimated some elite players who had strong underlying metrics but did not perform well in clutch games. In addition, the model struggled with below-average players who scored clutch goals at a rate that did not match their advanced stats.  

This prompted me to use a log transformation, which enabled the model to generate better predictions for elite players and reduced the number of influential points. 

The model still undervalued some players who performed better in close and tied situations than their metrics suggest. On the other hand, some players were overvalued because metrics that may not fully reflect their clutch performance.  While influential points are often viewed negatively, they can show which players perform better under pressure than their stats suggest

### 7. Prediction Intervals
95% prediction intervals were generated for each player. If the actual clutch scores were inside the intervals, this would show the player performance in close and tied situations was statistically significant.

### 8. Using the Model on a Final Test Set
The model was tested on player data from the 2024-2025 season to the current point of the 2025-2026 season. The R^2 was 70% which was close to the training R^2 of 79%. 
