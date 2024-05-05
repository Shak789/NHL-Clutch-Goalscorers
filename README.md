# NHL-Clutch-Goalscorers

## Abstract
Traditional National Hockey League (NHL) statistics simply focus on the amount of goals scored by players, which overlooks their ability to perform in critical situations, such as close or tied games. This research aims to identify NHL players who excel in scoring goals during pivotal moments and establish a classification system to discern their performance in critical game scenarios. Using data retrieved from the NHL API, such as Shot Attempts Percentage (SAT%) and Unblocked Shot Attempts Percentage (USAT%) observed during close and tie game situations, alongside game-winning goals, a ranking system was constructed to evaluate NHL players based on their performance in critical game contexts. In addition, a random forest binary classification model was developed to categorize players based on their performance in critical situations. As a result of the high negative imbalance in the dataset, various metrics such as precision and recall, as well as class weightings, were used to assess the accuracy of the model. While the model was reliable in correctly identifying top NHL goalscorers during pivotal game situations, further research is needed to determine the model's predictive accuracy in classifying players.

## Metrics Used ot Evaluate Performance in Critical Situations
- **Shot Attempts (SAT) % in Close or Tied Games:** SAT% measures the shot differentials of players. A higher SAT% implies that the player is controlling the puck for longer periods of time and generating more shot attempts than opponents, which can contribute to increased goalscoring opportunities.

- **Unblocked Shot Attempts (USAT) % in Close or Tied Games:** USAT% is a similar metric to SAT% but excludes blocked shots. USAT% evaluates a player’s ability to direct shots towards the opposing team’s goaltender and create goalscoring chances.

- **Game-Winning Goals:** A game-winning goal is the goal that was scored by a player to put their team ahead and win the game.  Game-winning goals demonstrate which players can score goals in crucial moments of games.

## Potential Shortcomings with Metrics
Other statistics such as goals scored in the third period as well as game-tying goals would have been beneficial to analyze. However, these metrics are not available on the NHL API and are not provided by any other sources. In addition, SAT% and USAT% do not account for the quality of the shots and can overstate a player's offensive contribution if the shots generated lack sufficient scoring threat. 

## Developing a Ranking System
The averages of the percentiles for close and tied situations, as well as game-winning goals, were computed to determine a percentile ranking for a player’s clutch goalscoring. The following weightings were assigned to each of the percentile rankings:
a)	Close Percentile: 30%
b)	Tied Percentile: 30%
c)	Game-Winning Goals Percentile: 40%
A higher percentile was assigned to the game-winning goals because This statistic tracks actual goals scored in critical situations, unlike SAT% and USAT% which may provide inaccuracies due to the quality of shots.

