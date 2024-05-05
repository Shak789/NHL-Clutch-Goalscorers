# NHL-Clutch-Goalscorers

## Abstract
Traditional National Hockey League (NHL) statistics simply focus on the amount of goals scored by players, which overlooks their ability to perform in critical situations, such as close or tied games. This research aims to identify NHL players who excel in scoring goals during pivotal moments and establish a classification system to discern their performance in critical game scenarios. Using data retrieved from the NHL API, such as Shot Attempts Percentage (SAT%) and Unblocked Shot Attempts Percentage (USAT%) observed during close and tied game situations, alongside game-winning goals, a ranking system was constructed to evaluate NHL players based on their performance in critical game contexts. In addition, a random forest binary classification model was developed to categorize players based on their performance in critical situations. As a result of the high negative imbalance in the dataset, various metrics such as precision and recall, as well as class weightings, were used to assess the accuracy of the model. While the model was reliable in correctly identifying top NHL goalscorers during pivotal game situations, further research is needed to determine the model's predictive accuracy in classifying players.

## Methods
All player data was retrieved from the NHL API. The statistics considered were:
- **Shot Attempts (SAT) % in Close or Tied Games:** SAT% measures the shot differentials of players. A higher SAT% implies that the player is controlling the puck for longer periods of time and generating more shot attempts than opponents, which can contribute to increased goalscoring opportunities.

- **Unblocked Shot Attempts (USAT) % in Close or Tied Games:** USAT% is a similar metric to SAT% but excludes blocked shots. U

- **Game-Winning Goals:** A game-winning goal is the goal that was scored by a player to put their team ahead and win the game.

The averages of the percentiles for close and tied situations, as well as game-winning goals, were caluclated to determine a percentile ranking for a player’s clutch goalscoring. The weighted average of these stasitics was taken with a slighlty higher weighting assigned to game-winning goals because SAT% and USAT% which may provide inaccuracies due to the quality of shots. Full player rankings can be seen in the Power BI vizulaization in this repository.




