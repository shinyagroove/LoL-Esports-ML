# Predict which role a player played given their post-game data

## Overview
This data science project investigates ingame action intensity within professional League of Legends esports matches in 2022. The dataset used for this analysis, which is available [here](https://oracleselixir.com/tools/downloads), was originally compiled from detailed match statistics. The exploratory data analysis on this dataset can be found [here](https://shinyagroove.github.io/LoL-Esports-Action-Analysis/).

---

## Framing the Problem

The problem we want to predict is `Predict which role a player played given their post-game data`. This is a classification problem, and multiclass classification. Thre response variable is `position`. The metric used to evaluate the models is accuracy. Since this problem uses post-game data to predict the role of player, we can use any column within this dataset.

---


## Baseline Model


The baseline model is a simple decision tree classifier. 

Features used in the model are following:

`position`: Indicating which role the player has played
`champion`: Name of champion the player has played
`kills`: Number of kills
`deaths`: Number of deaths
`assists`: Number of assists
`dpm`: Average damage to champions per minute
`damageshare`: Average share of team’s total damage to champions
`damagetakenperminute`: 
`damagemitigatedperminute`: 
`wpm`: Average wards placed per minute
`wcpm`: Average wards cleared per minute
`vspm`: Vision score per minute
`earnedgoldshare`: Percentage of gold player earned against the team
`minionkills`: Number of minions killed
`monsterkills`: Number of monsters killed


`position` and `champion` are nominal data, the rest are all quantitative data.

One hot encoding was performed on `champion` column, we did not change `position` since it is our response variable.

`Model Performance`:
training set accuracy: 0.9904171761828614
test set accuracy: 0.951368638886276


---


## Final Model

Features added:
`fightparticipationrate`: (kill+assist) / (total kills + total deaths across all players in the same team)
The reason why we choose to feature engineer fight participation rate is because we believe top role in general are less likely to have high participation rate because the main objective in early game(dragons) only spawn near bottom lane, and it is usually where the team fight would happen. It is pretty far and takes decent amount of time to get to from top lane, so we thought this would be a good estimation of team fight participation rate that could help extinguish the difference between top role and mid role or bot role for example.


`dmg_dmgtaken_ratio`: damage / damge taken per minute
The reason why we choose to add this feature is because we believe this can help extinguish bot role from other roles. To have a good dmg_dmgtaken_ratio, it requires the player to do damage while taking less damage, this is usually easier to accomplish on champions with longer attack range. Typically, bot role ad carry champions tend to be the best at dealing damage at range.



**Note: Because the way fight participation rate is calculated, it requires the input data to have a strict structure that starting from the very first row, every 10 rows must correspond to the same match, and first five rows must contain all the players from red or blue team, and same for the second five rows. We cannot put this feature engineering step into the pipeline because it has to be done before we split training set and test set. The model is fit on dataset with added columns, so we will have to calculate and add these new features first so we can predict using the model.**


We have tested 4 different models: Decision Tree, Random Forest, KNN, Logistic Regression

Model 1 Decision Tree: best parameters: {'dt__max_depth': 50}, training set accuracy: 0.990087950333929, test set accuracy: 0.9527796068102719

Model 2 Random Forest: best parameters: {'rfc__max_depth': 162}, training set accuracy: 1.0, test set accuracy: 0.9608691562411814

Model 3 KNN: best parameters: {'knn__n_neighbors': 22}, training set accuracy: 0.9637733985514063, test set accuracy: 0.9607280594487818

Model 4 Logistic Regression: best parameters: {'logreg__C': 1.0}, training set accuracy: 0.9677241087385947, test set accuracy: 0.9659016085034333


Our final choice of model is: Random Forest
We used GridSearchCV to search for the optimal max_depth for random forest and found the best performance around 162.

`Accuracy improvement`:
**training set 0.00958282381, test set 0.00950051735**

Although the improvement in accuracy is marginal compared to our baseline model, it is still a slight improvement.


---


## Fairness Analysis


X: Frequently Played Champions
Y: Less Frequently Played Champions

We will define frequently played champion as champion that is played more than the median number of games played by each champion, less frequently played champion will be less than or equal to median.

`Null Hypothesis`:
The model's precision for frequently played champions is the same as the precision for less frequently played champions. Any observed differences in precision are due to random chance.

`Alternative Hypothesis`:
The model's precision for frequently played champions is significantly different from the precision for less frequently played champions. The differences in precision are not due to random chance, indicating a potential bias in the model.

`Test Statistics`:
We used the difference between the precision of frequently played champions and less frequently played champion as our test statistics.

We will use p-value cutoff 0.05, and the p-value we got is 0.0. Because p-value we got is 0 and is below the cutoff, we will reject the null hypothesis. Therefore, we suspect that our model does worse for champions that are less frequently played.




---