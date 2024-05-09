# Analysis of NHL Goalscoring in Critical Situations

<a href="https://www.researchgate.net/publication/380347690_Analysis_of_NHL_Goalscoring_in_Critical_Situations">Link to full paper</a>

## Abstract
Traditional National Hockey League (NHL) statistics simply focus on the number of goals scored by players, overlooking their performance in critical situations, such as close or tied games. This research aims to identify NHL players who excel in scoring goals during pivotal moments and establish a classification system to discern their performance in critical game scenarios. Using data retrieved from the NHL API, such as Shot Attempts Percentage (SAT%) and Unblocked Shot Attempts Percentage (USAT%) observed during close and tied game situations, alongside game-winning goals, a ranking system was constructed to evaluate NHL players based on their ability to score goals in critical game contexts. In addition, a random forest binary classification model was developed to categorize players based on their metrics in critical situations. As a result of the high negative imbalance in the dataset, various methods such as precision and recall, as well as class weightings, were employed to assess the accuracy of the model. While the model was reliable in correctly identifying top NHL goalscorers during pivotal game situations, further research is needed to determine its predictive accuracy in classifying players.

## Ranking Top Performers in Critical Situations
The statistics considered were:
- **Shot Attempts (SAT) % in Close or Tied Games:** SAT% measures the sum of shots on goal, missed shots, and blocked shots over shots against, missed shots against and blocked shots against. 

- **Unblocked Shot Attempts (USAT) % in Close or Tied Games:** USAT% is a similar metric to SAT% but excludes blocked shots.

- **Game-Winning Goals:** A game-winning goal is the goal that was scored by a player to put their team ahead and win the game.

The weighted averages of the percentiles for close and tied situations, as well as game-winning goals, were caluclated to determine a percentile ranking for a player’s clutch goalscoring. A slightly higher weighting was assigned to game-winning goals because SAT% and USAT% may provide inaccuracies due to the quality of shots. Full player rankings can be seen in the Power BI visualization in this repository.

## Classification Model
A random forest model was selected to categorize players based on their performance in close and tied game scenarios. From the analysis of the distributions of USAT% and SAT% in close and tied game situations, as well as game-winning goals, a player was be classified as a clutch goalscorer by meeting the following criteria:
a)	At least one SAT% and USAT% metric in close and tied game situations that exceeds 55%.
b)	Scored more game-winning goals than 80% of other players in the dataset.

The binary classification presented a problem of negative class imbalance because over 80% of players in the dataset were classified as non-clutch goalscorers. Thus, various metrics were used to assess the accuracy of the model.

## Key Findings from Classification Model
The binary classification model was very effective in correctly identifying clutch NHL goalscorers in the dataset, as shown by its high recall value and the model’s performance after changing class weightings to favour the majority group.
However, it is difficult to determine the likelihood of the model correctly classifying an NHL player as a clutch goalscorer because the negative class imbalance increases the precision. 

One potential solution is to analyze data from multiple NHL seasons. While this may not eliminate class imbalance, it can increase training on the minority class and lead to better predictive performance. 





