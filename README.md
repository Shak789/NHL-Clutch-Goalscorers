# Predicting Clutch Goalscorers in the NHL using Machine Learning Techniques

In this project, I explored various machine learning techniques to determine the best performing NHL forwards in close and tied games (often referred to as "clutch" moments). The process involved several key steps:

### 1. Identifying Correct Sources of Data
I needed to scrape data from the NHL API and Natural Stat Trick. The NHL API offers a good foundation for player bios and common goal statistics. However, Natural Stat Trick provides statistics on goals scored by players in close and tied games, as well as advanced stats that can be used as features for a machine learning model.

### 2. Data Cleaning
I merged data from the NHL API and Natural Stat Trick, then ensured the data was accurate and filtered appropriately.

### 3. Establishing a Definition of Clutch
I computed a "clutch score" for each player by weighting their goals in critical game situations:

- Goals when the game is tied (30% weight)
- Goals when the team is down by one goal (30% weight)
- Goals when the team is up by one goal (20% weight)
- Goals in overtime (20% weight)

### 4. Choosing features
Regardless of the model used, Natural Stat Trick provided a good set of advanced analytics to predict a player's clutch score. These included shots, expected goals, scoring chances, Corsi and Fenwick.

### 5. Building a Classification Model
I attempted to classify players as "clutch" and "non-clutch" by setting thresholds for the clutch score. I used metrics such as expected goals, scoring chances, and other advanced statistics as features. The model was trained on data from the 2020-2021 to 2022-2023 NHL seasons. While the model was successful in identifying elite players and those below average, it struggled with players who fell near the classification boundaries, where small differences in their stats made it difficult to confidently label them as clutch or non-clutch.

### 6. Switching to a Regression Model
I realized that linear regression was a more feasible approach since many of the features were strongly correlated with a clutch score. It would, therefore, be easier to predict a player's clutch score rather than assigning the player an ambiguous label.  

I refined the model by using Ridge regression and performed cross-validation to ensure there was no overfitting.

### 7. Dealing with Outliers
I used Cook's Distance to identify influential points. I discovered that the model underpredicted the clutch score of elite players because their feature stats set a "ceiling" for their clutch ability. The model also overestimated some elite players who had strong underlying metrics but did not perform well in clutch games. In addition, the model struggled with below-average players who scored clutch goals at a rate that did not match their advanced stats.  

This prompted me to use a log transformation, which enabled the model to generate better predictions for elite players and reduced the number of influential points. 

The model still undervalued some players who performed better in close and tied situations than their metrics suggest. On the other hand, some players were overvalued because metrics that may not fully reflect their clutch performance.  While influential points are often viewed negatively, they can provide valuable insights into players who perform well in high-pressure situations, even if they aren’t considered elite based on traditional metrics.

Finally, some below-average players become influential since the log transformation tends to amplify the difference between smaller actual and predicted values.

### 8. Using the Model on a Final Test Set
After I was satisfied with the model, I used it to predict the clutch score of players based on their statistics from the start of the 2023-2024 season to the current point of the 2024-2025 season.  
In the coming weeks, I plan to deploy the model and connect it to a Power BI dashboard, which will provide real-time updates of a player's current clutch score and their predicted clutch score.
