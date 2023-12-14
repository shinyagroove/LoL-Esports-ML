Student: Qilong Zou, Haotian Zhu

Course: DSC 80

Project 5(Final Project)

---
# League Of Legends Position Prediction Model

## Introduction
This data science project attempts to create a ML model to predict which role a player played given the post-game data using data from professional League of Legends esports matches in 2022. The dataset used for this analysis, which is available [here](https://oracleselixir.com/tools/downloads), was originally compiled from detailed match statistics. The exploratory data analysis on this dataset can be found [here](https://shinyagroove.github.io/LoL-Esports-Action-Analysis/) and [here](https://zhtdbb1.github.io/League-of-Legends-Aggressiveness-Analysis-2017-2018/).

---

## Framing the Problem

In our prediction model, we aim to determine "**the role a player played given their post-game data**". This is a case of **multiclass classification**. The response variable `position` is a direct measurement of the role of player which is a categorical value with multiple classes(i.e. `top`,`mid`, `sup`, `jg`, `bot`). Multiclass classification is appropriate here because each player can only occupy one of these distinct roles. For evaluating our model, we will use accuracy as our primary metric. This choice is made because our dataset is inherently balanced in the context of team compositions – a team can not have multiple players in the same role. Therefore, the likelihood of class imbalance affecting our accuracy metric is low. Since our problem involves predicting a player's role based on post-game data, we would have access to the entire dataset of that match after the match has concluded, allowing us to utilize all available data for this model.


### Missingness assessment

The following data frame shows the missingness of each our columns.

|                          |   missingness |
|:-------------------------|--------------:|
| league                   |             0 |
| position                 |             0 |
| champion                 |             0 |
| kills                    |             0 |
| deaths                   |             0 |
| assists                  |             0 |
| dpm                      |            10 |
| damageshare              |            10 |
| damagetakenperminute     |            10 |
| damagemitigatedperminute |         18190 |
| wpm                      |            10 |
| wcpm                     |            10 |
| vspm                     |            10 |
| earnedgoldshare          |             0 |
| minionkills              |            10 |
| monsterkills             |            10 |

Same as we discovered in previous project, missingness of data is strongly correlated to the country(assumption about DGP) where the match is held.
Since we cannot safely impute these data without loss of generality for match held in China or World Series, we will go ahead and drop any column we plan to use that has missing values.

### Data Cleaning
1. Drop any row that have missing value in any of the column we are using.
2. Drop all the rows containing team data because we are analyzing position of each player. (each match has 12 rows, 10 rows corresponding to each player and 2 rows that contain team statistics)


---


## Baseline Model

> Features we will be including in our model:

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

***Note***: For additional details on how features are distributed across various positions, please refer to the [Appendix](#Appendix) section.

Our model is a decision tree classifier with a specified depth of 50. This model was chosen for its simplicity and interpretability, particularly useful as a baseline.

We performed the necessary one hot encoding on the `champion` column in order to build our baseline model.



### Baseline Model Result

**Training set accuracy**: 0.9904171761828614  
**Test set accuracy**: 0.951368638886276  

The performance as indicated by the training set accuracy and the test set accuracy, suggests that our model is performing quite well. It has a high traning set accuracy, and certain amount of drop in performance from training to test data is normal and expected. The high training accuracy does raise a concern about overfitting. However, the fact that the test accuracy remains high is reassuring.


---


## Final Model


### Features
New Features added:
- `fightparticipationrate`: Calculated by (kill+assist) / (total kills + total assists across all players in the same team)

- `dmg_dmgtaken_ratio`: damage / damge taken per minute

- `RobustScaler` on all columns (Only for KNN and Logistic Regression)

**Note** We need this since KNN and Logistic Regression models are sensitive to the scale of data.

Explanation:

- The reason why we choose to feature engineer fight participation rate is because we believe top role in general are less likely to have high participation rate because the main objective in early game(dragons) only spawn near bottom lane, and it is usually where the team fight would happen. It is pretty far and takes decent amount of time to get to from top lane, so we thought this would be a good estimation of team fight participation rate that could help extinguish the difference between top role and mid role or bot role for example.

- The reason why we choose to add `dmg_dmgtaken_ratio` is because we believe this can help extinguish bot role from other roles. To have a good dmg_dmgtaken_ratio, it requires the player to do damage while taking less damage, this is usually easier to accomplish on champions with longer attack range. Typically, bot role ad carry champions tend to be the best at dealing damage at range.

***Note***: Because the way fight participation rate is calculated, it requires the input data to have a strict structure that starting from the very first row, every 10 rows must correspond to the same match, and first five rows must contain all the players from red or blue team, and same for the second five rows. We cannot put this feature engineering step into the pipeline because it has to be done before we split training set and test set. The model is fit on dataset with added columns, so we will have to calculate and add these new features first so we can predict using the model.


Resulting Dataframe:

| position   | champion   |   kills |   deaths |   assists |     dpm |   damageshare |   damagetakenperminute |   damagemitigatedperminute |    wpm |   wcpm |   vspm |   earnedgoldshare |   minionkills |   monsterkills |   fightparticipationrate |   dmg_dmgtaken_ratio |
|:-----------|:-----------|--------:|---------:|----------:|--------:|--------------:|-----------------------:|---------------------------:|-------:|-------:|-------:|------------------:|--------------:|---------------:|-------------------------:|---------------------:|
| top        | Renekton   |       2 |        3 |         2 | 552.294 |     0.278784  |               1072.4   |                    777.793 | 0.2802 | 0.2102 | 0.9107 |          0.253859 |           220 |             11 |                 0.142857 |             0.515008 |
| jng        | Xin Zhao   |       2 |        5 |         6 | 412.084 |     0.208009  |                944.273 |                    650.158 | 0.2102 | 0.6305 | 1.6813 |          0.19022  |            33 |            115 |                 0.285714 |             0.436403 |
| mid        | LeBlanc    |       2 |        2 |         3 | 499.405 |     0.252086  |                581.646 |                    227.776 | 0.6655 | 0.2452 | 1.0158 |          0.210665 |           177 |             16 |                 0.178571 |             0.858605 |
| bot        | Samira     |       2 |        4 |         2 | 389.002 |     0.196358  |                463.853 |                    218.879 | 0.4203 | 0.2102 | 0.8757 |          0.242201 |           208 |             18 |                 0.142857 |             0.838632 |
| sup        | Leona      |       1 |        5 |         6 | 128.301 |     0.0647631 |                475.026 |                    490.123 | 1.0158 | 0.4904 | 2.4168 |          0.103054 |            42 |              0 |                 0.25     |             0.270093 |



### Models selection and performance

We have tested 4 different models: 
1. [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree) 
2. [Random Forest](https://en.wikipedia.org/wiki/Random_forest)
3. [KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
4. [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)

The optimal hyperparameters found by `GridSearchCV` for all four models:

1. Decision Tree: 
- best parameters: `max_depth= 50`
- training set accuracy: 0.990087950333929
- test set accuracy: 0.9527796068102719

2. Random Forest: 
- best parameters: `max_depth = 162`
- training set accuracy: 1.0
- test set accuracy: 0.9608691562411814

3. KNN: 
- best parameters: `n_neighbors= 22`
- training set accuracy: 0.9637733985514063
- test set accuracy: 0.9607280594487818

4. Logistic Regression: 
- best parameters: `C = 1.0`
- training set accuracy: 0.9677241087385947
- test set accuracy: 0.9659016085034333


Our final choice of model is: **Random Forest**

> Accuracy improvement:
training set: 0.00958282381
test set: 0.00950051735

Overall, the performance improvements on both the training set and test set is about `1%`. Although numerically small, but they are significant because the baseline performances were already high.


### Visualization of the model's performance

<iframe src="assets/confusion.html" width=1040 height=720 frameBorder=0></iframe>

Based on the information provided in the confusion matrix, it's clear that the model significantly underperformed in correctly predicting the top player, this notable deficiency points to a need for further refinement.


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

We will use p-value cutoff 0.05

<iframe src="assets/permutation.html" width=1040 height=720 frameBorder=0></iframe>


The p-value we got is 0.0. Because p-value we got is 0 and is below the cutoff, we will reject the null hypothesis. Therefore, we suspect that our model does worse for champions that are less frequently played.



---


## <a name="Appendix"></a> Distribution-of-Features

<iframe src="assets/championDistr.html" width=1040 height=720 frameBorder=0></iframe>
<iframe src="assets/damagemitigatedperminute.html" width=1040 height=720 frameBorder=0></iframe>
<iframe src="assets/damageshare.html" width=1040 height=720 frameBorder=0></iframe>
<iframe src="assets/damagetakenperminute.html" width=1040 height=720 frameBorder=0></iframe>
<iframe src="assets/deaths.html" width=1040 height=720 frameBorder=0></iframe>
<iframe src="assets/dpm.html" width=1040 height=720 frameBorder=0></iframe>
<iframe src="assets/kills.html" width=1040 height=720 frameBorder=0></iframe>
<iframe src="assets/vspm.html" width=1040 height=720 frameBorder=0></iframe>
<iframe src="assets/wcpm.html" width=1040 height=720 frameBorder=0></iframe>
<iframe src="assets/wpm.html" width=1040 height=720 frameBorder=0></iframe>


---

