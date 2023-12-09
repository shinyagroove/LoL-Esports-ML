Student: Qilong Zou, Haotian Zhu

Course: DSC 80

Project 5(Final Project)

---
# League Of Legends Position Prediction Model

## Introduction
This data science project attempts to create a ML model to predict which role a player played **given the post-game data** using data from professional League of Legends esports matches in 2022. The dataset used for this analysis, which is available [here](https://oracleselixir.com/tools/downloads), was originally compiled from detailed match statistics. The exploratory data analysis on this dataset can be found [here](https://shinyagroove.github.io/LoL-Esports-Action-Analysis/).

---

## Framing the Problem

The problem we want to predict is "**Which role a player played given their post-game data**". Since the  response variable is `position`, which is a categorical value with multiple classes(i.e. `top`,`mid`, `sup`, `jg`, `bot`), we will use **Classification with multi-class** for this problem. As for the metric, because our data is balanced(there will never be a team with 5 ADC),  accuracy is totally OK to handle the situation. 

***Important Note***: This problem uses post-game data to predict the role of player, we can use any column within this dataset.

---


## Baseline Model

### Step 1: Data Cleaning
1. Drop any row that have missing value in any of the column we are using.
2. Drop all the rows containing team data because we are analyzing position of each player. (each match has 12 rows, 10 rows corresponding to each player and 2 rows that contain team statistics)

 
> Features used in the model are following:

- `position`: Indicating which role the player has played. (**Nominal**)
- `champion`: Name of champion the player has played. (**Nominal**)
- `kills`: Number of kills. (quantitative)
- `deaths`: Number of deaths. (quantitative)
- `assists`: Number of assists. (quantitative)
- `dpm`: Average damage to champions per minute. (quantitative)
- `damageshare`: Average share of team’s total damage to champions. (quantitative)
- `damagetakenperminute`: Damage taken per minute. (quantitative)
- `damagemitigatedperminute`: Damage mitigated per minute. (quantitative)
- `wpm`: Average wards placed per minute. (quantitative)
- `wcpm`: Average wards cleared per minute. (quantitative)
- `vspm`: Vision score per minute. (quantitative)
- `earnedgoldshare`: Percentage of gold player earned against the team. (quantitative)
- `minionkills`: Number of minions killed. (quantitative)
- `monsterkills`: Number of monsters killed. (quantitative)

***Note***:`position` and `champion` are nominal data, the rest are all quantitative data.

### Step 2: Missingness assessment

The following data frame shows the missingness of each our columns.


|                          |   missingness |\n|:-------------------------|--------------:|\n| league                   |             0 |\n| position                 |             0 |\n| champion                 |             0 |\n| kills                    |             0 |\n| deaths                   |             0 |\n| assists                  |             0 |\n| dpm                      |            10 |\n| damageshare              |            10 |\n| damagetakenperminute     |            10 |\n| damagemitigatedperminute |         18190 |\n| wpm                      |            10 |\n| wcpm                     |            10 |\n| vspm                     |            10 |\n| earnedgoldshare          |             0 |\n| minionkills              |            10 |\n| monsterkills             |            10 |

Same as we discovered in previous project, missingness of data is strongly correlated to the country where the match is held.
Since we cannot safely impute these data without loss of generality for match held in China or World Series, we will go ahead and drop any column we plan to use that has missing values.





### Step 3: Pick and Apply Baseline Model
The baseline model is a **simple decision tree classifier** with `depth=50`, which we will use to predict which role a player played using post-match data(multi-class), position is the choice of our response variable as it is the direct measurement of which role the player played in the match.

> We found that Total Variance Distance by group of `champion` column is huge, as we can see below, some champion can be considered as **toplane champion**, some being **support champion**, etc
<iframe src="assets/championDistr.html" width=1040 height=720 frameBorder=0></iframe>

Hence, one hot encoding was performed on `champion` column for the sake of a better performance of our model. Then, `Decision Tree classifier(depth=50)` was performed.


### Result
training set accuracy: 0.9904171761828614
test set accuracy: 0.951368638886276

A test set accuracy of 95% is pretty high! However, there is still space of improvements. Some columns have a plenty of outliers which will disturb our prediction, under such circumstances, we'd better use Robust or such transformers for a more accurate result. The following graph is the within group distribution of `dpm` column.

<iframe src="assets/dpm.html" width=1040 height=720 frameBorder=0></iframe>


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