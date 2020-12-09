# Project Introduction

Through looking at a simple data set containing the scores of each NFL game, the betting line involved, and a few other explanatory details about the game being played, I decided to set out to answer questions I had about the accuracy of NFL betting lines. 

I tried predicting many different things due to the various amounts of features I needed to create for a pretty basic dataset, so ultimately I tried predicting a few things based on what model I wanted to use. Among the variables I tried predicting were the actual score difference, the margin the favorite to win beat the spread by, and a binary result of whether the home team can be expected to win or not.

## Importing and Cleaning the Data Section


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import regex as re
plt.style.use('ggplot')
```


```python
games = pd.read_csv('/Users/scottgill/Desktop/CurrentCourseAssignments/Stat426/Project/Data/spreadspoke_scores.csv')
teams = pd.read_csv('/Users/scottgill/Desktop/CurrentCourseAssignments/Stat426/Project/Data/nfl_teams.csv')
stadiums = pd.read_csv('/Users/scottgill/Desktop/CurrentCourseAssignments/Stat426/Project/Data/nfl_stadiums.csv', encoding = "ISO-8859-1")

```


```python
#Removing games from 2020 that haven't happened yet
games.isnull().sum()
games = games[games['score_home'].notna()]
print(games.isnull().sum())
print(games.shape)
print(games.dtypes)
```

    schedule_date              0
    schedule_season            0
    schedule_week              0
    schedule_playoff           0
    team_home                  0
    score_home                 0
    score_away                 0
    team_away                  0
    team_favorite_id        2479
    spread_favorite         2479
    over_under_line         2489
    stadium                    0
    stadium_neutral            0
    weather_temperature      825
    weather_wind_mph         825
    weather_humidity        4409
    weather_detail         10122
    dtype: int64
    (12797, 17)
    schedule_date           object
    schedule_season          int64
    schedule_week           object
    schedule_playoff          bool
    team_home               object
    score_home             float64
    score_away             float64
    team_away               object
    team_favorite_id        object
    spread_favorite        float64
    over_under_line         object
    stadium                 object
    stadium_neutral           bool
    weather_temperature    float64
    weather_wind_mph       float64
    weather_humidity        object
    weather_detail          object
    dtype: object



```python
games2000 = games[(games['schedule_season'] > 1999) & (games['schedule_season'] < 2020) ] #grabbing games only later than 2000, but before 2020 started (with no fans)
```


```python
games2000.schedule_playoff.value_counts() #checking to see how many playoff games we have in the dataset
```




    False    5104
    True      220
    Name: schedule_playoff, dtype: int64




```python
#Dropping unnecessary columns from stadiums df
stadiums = stadiums.drop(['stadium_address', 'stadium_weather_station_code', 'STATION', 'NAME', 'LATITUDE', 'LONGITUDE'], axis=1)
```


```python
games2000.rename(columns={'stadium': 'stadium_name'}, inplace=True) #changing column name so we can merge easier

#Joining both dataframes together
df = pd.merge(games2000, stadiums, on = 'stadium_name', how = 'left')
```

    /Applications/anaconda3/lib/python3.8/site-packages/pandas/core/frame.py:4125: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      return super().rename(



```python
#Dropping more unnecessary columns
df = df.drop(['stadium_close', 'weather_humidity'], axis=1) #too many NA's in weather humidity
```


```python
#Changing datatype for schedule date column
df['schedule_date'] = pd.to_datetime(df['schedule_date'])
```


```python
print(df.dtypes)
df
```

    schedule_date           datetime64[ns]
    schedule_season                  int64
    schedule_week                   object
    schedule_playoff                  bool
    team_home                       object
    score_home                     float64
    score_away                     float64
    team_away                       object
    team_favorite_id                object
    spread_favorite                float64
    over_under_line                 object
    stadium_name                    object
    stadium_neutral                   bool
    weather_temperature            float64
    weather_wind_mph               float64
    weather_detail                  object
    stadium_location                object
    stadium_open                   float64
    stadium_type                    object
    stadium_weather_type            object
    stadium_capacity                object
    stadium_surface                 object
    ELEVATION                      float64
    dtype: object





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>schedule_date</th>
      <th>schedule_season</th>
      <th>schedule_week</th>
      <th>schedule_playoff</th>
      <th>team_home</th>
      <th>score_home</th>
      <th>score_away</th>
      <th>team_away</th>
      <th>team_favorite_id</th>
      <th>spread_favorite</th>
      <th>...</th>
      <th>weather_temperature</th>
      <th>weather_wind_mph</th>
      <th>weather_detail</th>
      <th>stadium_location</th>
      <th>stadium_open</th>
      <th>stadium_type</th>
      <th>stadium_weather_type</th>
      <th>stadium_capacity</th>
      <th>stadium_surface</th>
      <th>ELEVATION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>False</td>
      <td>Atlanta Falcons</td>
      <td>36.0</td>
      <td>28.0</td>
      <td>San Francisco 49ers</td>
      <td>ATL</td>
      <td>-6.5</td>
      <td>...</td>
      <td>72.0</td>
      <td>0.0</td>
      <td>DOME</td>
      <td>Atlanta, GA</td>
      <td>1992.0</td>
      <td>indoor</td>
      <td>dome</td>
      <td>71,250</td>
      <td>FieldTurf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>False</td>
      <td>Buffalo Bills</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>Tennessee Titans</td>
      <td>BUF</td>
      <td>-1.0</td>
      <td>...</td>
      <td>70.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>Orchard Park, NY</td>
      <td>1973.0</td>
      <td>outdoor</td>
      <td>cold</td>
      <td>73,967</td>
      <td>FieldTurf</td>
      <td>178.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>False</td>
      <td>Cleveland Browns</td>
      <td>7.0</td>
      <td>27.0</td>
      <td>Jacksonville Jaguars</td>
      <td>JAX</td>
      <td>-10.5</td>
      <td>...</td>
      <td>75.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>Cleveland, OH</td>
      <td>1999.0</td>
      <td>outdoor</td>
      <td>cold</td>
      <td>68,000</td>
      <td>Grass</td>
      <td>238.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>False</td>
      <td>Dallas Cowboys</td>
      <td>14.0</td>
      <td>41.0</td>
      <td>Philadelphia Eagles</td>
      <td>DAL</td>
      <td>-6.0</td>
      <td>...</td>
      <td>95.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>Irving, TX</td>
      <td>1971.0</td>
      <td>outdoor</td>
      <td>moderate</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>163.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>False</td>
      <td>Green Bay Packers</td>
      <td>16.0</td>
      <td>20.0</td>
      <td>New York Jets</td>
      <td>GB</td>
      <td>-2.5</td>
      <td>...</td>
      <td>69.0</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>Green Bay, WI</td>
      <td>1957.0</td>
      <td>outdoor</td>
      <td>cold</td>
      <td>80,735</td>
      <td>Grass</td>
      <td>209.4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5319</th>
      <td>2020-01-12</td>
      <td>2019</td>
      <td>Division</td>
      <td>True</td>
      <td>Green Bay Packers</td>
      <td>28.0</td>
      <td>23.0</td>
      <td>Seattle Seahawks</td>
      <td>GB</td>
      <td>-4.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Green Bay, WI</td>
      <td>1957.0</td>
      <td>outdoor</td>
      <td>cold</td>
      <td>80,735</td>
      <td>Grass</td>
      <td>209.4</td>
    </tr>
    <tr>
      <th>5320</th>
      <td>2020-01-12</td>
      <td>2019</td>
      <td>Division</td>
      <td>True</td>
      <td>Kansas City Chiefs</td>
      <td>51.0</td>
      <td>31.0</td>
      <td>Houston Texans</td>
      <td>KC</td>
      <td>-10.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Kansas City, MO</td>
      <td>1972.0</td>
      <td>outdoor</td>
      <td>cold</td>
      <td>76,416</td>
      <td>Grass</td>
      <td>264.9</td>
    </tr>
    <tr>
      <th>5321</th>
      <td>2020-01-19</td>
      <td>2019</td>
      <td>Conference</td>
      <td>True</td>
      <td>Kansas City Chiefs</td>
      <td>35.0</td>
      <td>24.0</td>
      <td>Tennessee Titans</td>
      <td>KC</td>
      <td>-7.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Kansas City, MO</td>
      <td>1972.0</td>
      <td>outdoor</td>
      <td>cold</td>
      <td>76,416</td>
      <td>Grass</td>
      <td>264.9</td>
    </tr>
    <tr>
      <th>5322</th>
      <td>2020-01-19</td>
      <td>2019</td>
      <td>Conference</td>
      <td>True</td>
      <td>San Francisco 49ers</td>
      <td>37.0</td>
      <td>20.0</td>
      <td>Green Bay Packers</td>
      <td>SF</td>
      <td>-8.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Santa Clara, CA</td>
      <td>2014.0</td>
      <td>outdoor</td>
      <td>moderate</td>
      <td>68,500</td>
      <td>Grass</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>5323</th>
      <td>2020-02-02</td>
      <td>2019</td>
      <td>Superbowl</td>
      <td>True</td>
      <td>Kansas City Chiefs</td>
      <td>31.0</td>
      <td>20.0</td>
      <td>San Francisco 49ers</td>
      <td>KC</td>
      <td>-1.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Miami Gardens, FL</td>
      <td>1987.0</td>
      <td>outdoor</td>
      <td>warm</td>
      <td>65,326</td>
      <td>Grass</td>
      <td>8.8</td>
    </tr>
  </tbody>
</table>
<p>5324 rows × 23 columns</p>
</div>



## Cleaning the Data/Dealing with the many existing nulls


```python
df.isna().sum()
```




    schedule_date              0
    schedule_season            0
    schedule_week              0
    schedule_playoff           0
    team_home                  0
    score_home                 0
    score_away                 0
    team_away                  0
    team_favorite_id           0
    spread_favorite            0
    over_under_line            0
    stadium_name               0
    stadium_neutral            0
    weather_temperature      321
    weather_wind_mph         321
    weather_detail          3859
    stadium_location         178
    stadium_open             189
    stadium_type             180
    stadium_weather_type     180
    stadium_capacity        1145
    stadium_surface         1145
    ELEVATION               1069
    dtype: int64




```python
df["schedule_playoff"] = df["schedule_playoff"].astype(int)
```


```python
#Changing weeks that are strings to integers for later model predictions
df.loc[(df.schedule_week == '18'), 'schedule_week'] = '17'
df.loc[(df.schedule_week == 'Wildcard') | (df.schedule_week == 'WildCard'), 'schedule_week'] = '18'
df.loc[(df.schedule_week == 'Division'), 'schedule_week'] = '19'
df.loc[(df.schedule_week == 'Conference'), 'schedule_week'] = '20'
df.loc[(df.schedule_week == 'Superbowl') | (df.schedule_week == 'SuperBowl'), 'schedule_week'] = '21'
df['schedule_week'] = df.schedule_week.astype(int)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>schedule_date</th>
      <th>schedule_season</th>
      <th>schedule_week</th>
      <th>schedule_playoff</th>
      <th>team_home</th>
      <th>score_home</th>
      <th>score_away</th>
      <th>team_away</th>
      <th>team_favorite_id</th>
      <th>spread_favorite</th>
      <th>...</th>
      <th>weather_temperature</th>
      <th>weather_wind_mph</th>
      <th>weather_detail</th>
      <th>stadium_location</th>
      <th>stadium_open</th>
      <th>stadium_type</th>
      <th>stadium_weather_type</th>
      <th>stadium_capacity</th>
      <th>stadium_surface</th>
      <th>ELEVATION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>Atlanta Falcons</td>
      <td>36.0</td>
      <td>28.0</td>
      <td>San Francisco 49ers</td>
      <td>ATL</td>
      <td>-6.5</td>
      <td>...</td>
      <td>72.0</td>
      <td>0.0</td>
      <td>DOME</td>
      <td>Atlanta, GA</td>
      <td>1992.0</td>
      <td>indoor</td>
      <td>dome</td>
      <td>71,250</td>
      <td>FieldTurf</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>Buffalo Bills</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>Tennessee Titans</td>
      <td>BUF</td>
      <td>-1.0</td>
      <td>...</td>
      <td>70.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>Orchard Park, NY</td>
      <td>1973.0</td>
      <td>outdoor</td>
      <td>cold</td>
      <td>73,967</td>
      <td>FieldTurf</td>
      <td>178.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>Cleveland Browns</td>
      <td>7.0</td>
      <td>27.0</td>
      <td>Jacksonville Jaguars</td>
      <td>JAX</td>
      <td>-10.5</td>
      <td>...</td>
      <td>75.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>Cleveland, OH</td>
      <td>1999.0</td>
      <td>outdoor</td>
      <td>cold</td>
      <td>68,000</td>
      <td>Grass</td>
      <td>238.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>Dallas Cowboys</td>
      <td>14.0</td>
      <td>41.0</td>
      <td>Philadelphia Eagles</td>
      <td>DAL</td>
      <td>-6.0</td>
      <td>...</td>
      <td>95.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>Irving, TX</td>
      <td>1971.0</td>
      <td>outdoor</td>
      <td>moderate</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>163.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>Green Bay Packers</td>
      <td>16.0</td>
      <td>20.0</td>
      <td>New York Jets</td>
      <td>GB</td>
      <td>-2.5</td>
      <td>...</td>
      <td>69.0</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>Green Bay, WI</td>
      <td>1957.0</td>
      <td>outdoor</td>
      <td>cold</td>
      <td>80,735</td>
      <td>Grass</td>
      <td>209.4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5319</th>
      <td>2020-01-12</td>
      <td>2019</td>
      <td>19</td>
      <td>1</td>
      <td>Green Bay Packers</td>
      <td>28.0</td>
      <td>23.0</td>
      <td>Seattle Seahawks</td>
      <td>GB</td>
      <td>-4.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Green Bay, WI</td>
      <td>1957.0</td>
      <td>outdoor</td>
      <td>cold</td>
      <td>80,735</td>
      <td>Grass</td>
      <td>209.4</td>
    </tr>
    <tr>
      <th>5320</th>
      <td>2020-01-12</td>
      <td>2019</td>
      <td>19</td>
      <td>1</td>
      <td>Kansas City Chiefs</td>
      <td>51.0</td>
      <td>31.0</td>
      <td>Houston Texans</td>
      <td>KC</td>
      <td>-10.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Kansas City, MO</td>
      <td>1972.0</td>
      <td>outdoor</td>
      <td>cold</td>
      <td>76,416</td>
      <td>Grass</td>
      <td>264.9</td>
    </tr>
    <tr>
      <th>5321</th>
      <td>2020-01-19</td>
      <td>2019</td>
      <td>20</td>
      <td>1</td>
      <td>Kansas City Chiefs</td>
      <td>35.0</td>
      <td>24.0</td>
      <td>Tennessee Titans</td>
      <td>KC</td>
      <td>-7.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Kansas City, MO</td>
      <td>1972.0</td>
      <td>outdoor</td>
      <td>cold</td>
      <td>76,416</td>
      <td>Grass</td>
      <td>264.9</td>
    </tr>
    <tr>
      <th>5322</th>
      <td>2020-01-19</td>
      <td>2019</td>
      <td>20</td>
      <td>1</td>
      <td>San Francisco 49ers</td>
      <td>37.0</td>
      <td>20.0</td>
      <td>Green Bay Packers</td>
      <td>SF</td>
      <td>-8.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Santa Clara, CA</td>
      <td>2014.0</td>
      <td>outdoor</td>
      <td>moderate</td>
      <td>68,500</td>
      <td>Grass</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>5323</th>
      <td>2020-02-02</td>
      <td>2019</td>
      <td>21</td>
      <td>1</td>
      <td>Kansas City Chiefs</td>
      <td>31.0</td>
      <td>20.0</td>
      <td>San Francisco 49ers</td>
      <td>KC</td>
      <td>-1.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Miami Gardens, FL</td>
      <td>1987.0</td>
      <td>outdoor</td>
      <td>warm</td>
      <td>65,326</td>
      <td>Grass</td>
      <td>8.8</td>
    </tr>
  </tbody>
</table>
<p>5324 rows × 23 columns</p>
</div>




```python
#Cleaning Stadium location

df[df['stadium_location'].isnull()].team_home.value_counts() #Seeing which Teams have NA's for stadium location
df.loc[df.team_home == "Washington Redskins", "stadium_location"] = 'Landover, MD'
df.loc[df.team_home == "Oakland Raiders", "stadium_location"] = 'Oakland, CA'
df.loc[df.team_home == "Tampa Bay Buccaneers", "stadium_location"] = 'Tampa, FL'

#Jaguars are messy because they've played many games in London as the "Home" team
df['stadium_location'] = df['stadium_location'].fillna('Jacksonville, FL')

df.stadium_location.value_counts()


```




    East Rutherford, NJ    328
    Foxborough, MA         184
    Philadelphia, PA       174
    Pittsburgh, PA         173
    Green Bay, WI          172
    Indianapolis, IN       172
    Seattle, WA            171
    Denver, CO             169
    Kansas City, MO        167
    Baltimore, MD          167
    Charlotte, NC          166
    Atlanta, GA            166
    Oakland, CA            166
    Glendale, AZ           165
    Tampa, FL              165
    Minneapolis, MN        164
    Nashville, TN          163
    Cincinnati, OH         163
    Landover, MD           162
    Miami Gardens, FL      162
    New Orleans, LA        162
    Cleveland, OH          159
    Chicago, IL            157
    Jacksonville, FL       155
    Orchard Park, NY       153
    Houston, TX            152
    Detroit, MI            147
    San Diego, CA          140
    St. Louis, MO          130
    San Francisco, CA      115
    Arlington, TX           93
    Irving, TX              73
    Los Angeles, CA         53
    Santa Clara, CA         51
    London, UK              22
    Pontiac, MI             16
    Champaign, IL           10
    Toronto, Canada          6
    Baton Rouge, LA          4
    San Antonio, TX          3
    Mexico City, Mexico      2
    Miami, FL                1
    Tempe, AZ                1
    Name: stadium_location, dtype: int64




```python
#Cleaning the stadium open column
df[df['stadium_open'].isnull()].team_home.value_counts() #Redskins 1997, Jags 1995, Bills 1973, Saints 1975, Raiders 1966

df.loc[df.team_home == "Washington Redskins", "stadium_open"] = 1997
df.loc[df.team_home == "Jacksonville Jaguars", "stadium_open"] = 1995
df.loc[df.team_home == "Buffalo Bills", "stadium_open"] = 1973
df.loc[df.team_home == "New Orleans Saints", "stadium_open"] = 1975
df.loc[df.team_home == "Oakland Raiders", "stadium_open"] = 1966
df['stadium_open'] = df['stadium_open'].fillna(df['stadium_open'].mean()) #filling the last 3 with just the column mean

```


```python
#stadium_capacity cleaning - uncomment the first line once and run
df['stadium_capacity'] = df['stadium_capacity'].str.replace(',', '').astype(float)
df['stadium_capacity'] = df['stadium_capacity'].fillna(df.groupby('team_home')['stadium_capacity'].transform('mean'))
df['stadium_capacity'] = df['stadium_capacity'].fillna(65488) #this accounts for the Washington Football Team, which had no attendance record in this dataset. A reliable source stated that this was their average game attendance in the 2000s
```


```python
#Stadium_type cleaning

df[df['stadium_type'].isnull()].team_home.value_counts()
df[df['stadium_type'].isnull()]
df.stadium_type.value_counts()

#Cleaning the Redskins
df.loc[df.team_home == "Washington Redskins", "stadium_type"] = 'outdoor'

#Rest of the NA's were played outdoor, including the two Super Bowls
df['stadium_type'] = df['stadium_type'].fillna('outdoor')

```


```python
#Cleaning the Temperature column
df['weather_temperature'] = df['weather_temperature'].fillna(df.groupby('team_home')['weather_temperature'].transform('mean'))
```


```python
#Cleaning the Wind column
df['weather_wind_mph'] = df['weather_wind_mph'].fillna(df.groupby('team_home')['weather_wind_mph'].transform('mean'))
```


```python
#Cleaning Elevation column
meters = 0.3048

print(df[df['ELEVATION'].isnull()].team_home.value_counts()) #Looks like all Saints, Falcons, Redskins, Lions, Colts, and Rams games are missing elevation
df.loc[df.stadium_location == "New Orleans, LA", "ELEVATION"] = 3*meters #Only could find elevation in feet, dataset is in meters, data according to ArcGIS
df.loc[df.stadium_location == "Atlanta, GA", "ELEVATION"] = 997*meters
df.loc[df.stadium_location == "Landover, MD", "ELEVATION"] = 197*meters
df.loc[df.team_home == "Detroit Lions", "ELEVATION"] = 604*meters
df.loc[df.stadium_location == "Minneapolis, MN", "ELEVATION"] = 853*meters
df.loc[df.stadium_location == "St. Louis, MO", "ELEVATION"] = 466*meters
df.loc[df.stadium_location == "London, UK", "ELEVATION"] = 36*meters
df.loc[df.stadium_location == "Indianapolis, IN", "ELEVATION"] = 709*meters
df['ELEVATION'] = df['ELEVATION'].fillna(df['ELEVATION'].median()) #filling the last few missing with the column median

```

    New Orleans Saints      168
    Atlanta Falcons         166
    Washington Redskins     162
    Detroit Lions           160
    Minnesota Vikings       147
    St. Louis Rams          132
    Indianapolis Colts       71
    Jacksonville Jaguars     20
    Chicago Bears            10
    Buffalo Bills             7
    Oakland Raiders           5
    Los Angeles Rams          4
    Miami Dolphins            3
    Tampa Bay Buccaneers      3
    San Francisco 49ers       2
    Los Angeles Chargers      2
    Pittsburgh Steelers       1
    New England Patriots      1
    Cleveland Browns          1
    Cincinnati Bengals        1
    Philadelphia Eagles       1
    Arizona Cardinals         1
    Kansas City Chiefs        1
    Name: team_home, dtype: int64



```python
#stadium_weather_type cleaning
#Using ffill to take surrounding observations and fill NA's - this should work since it will take values based on time of when it will be warm/cold around it
df['stadium_weather_type'] = df['stadium_weather_type'].fillna(method = 'ffill') #hope is that data proportion will be preserved
df.stadium_weather_type.value_counts() #proportions appear to be preserved
```




    cold        2038
    dome        1387
    moderate    1186
    warm         713
    Name: stadium_weather_type, dtype: int64




```python
#Stadium surface cleaning - not done yet

df[df['stadium_surface'].isnull()].team_home.value_counts() #Too many teams to try and correct data for all of them
print(df.stadium_surface.value_counts())

df.stadium_surface = df.stadium_surface.fillna(method = 'ffill') #Using ffill again to preserve proportion
df.stadium_surface.value_counts() #proportion appears preserved again
```

    Grass        2505
    FieldTurf    1674
    Name: stadium_surface, dtype: int64





    Grass        3348
    FieldTurf    1976
    Name: stadium_surface, dtype: int64




```python
#Cleaning Weather Detail column - Assuming observations with no data available had no conditions since it wasn't accounted for
df.weather_detail = df.weather_detail.fillna('Normal')

print(df.weather_detail.value_counts())

#Combining data to create less levels
df.loc[df.weather_detail == "Rain | Fog", "weather_detail"] = 'Rain'
df.loc[df.weather_detail == "DOME (Open Roof)", "weather_detail"] = 'Normal'
df.loc[df.weather_detail == "Snow | Fog", "weather_detail"] = 'Snow'
df.loc[df.weather_detail == "Snow | Freezing Rain", "weather_detail"] = 'Snow'

#Dome should also be normal weather since there's no condition
df.loc[df.weather_detail == "DOME", "weather_detail"] = 'Normal'

df.weather_detail.value_counts()
```

    Normal                  3859
    DOME                    1233
    Rain                     106
    DOME (Open Roof)          56
    Fog                       28
    Rain | Fog                22
    Snow                      14
    Snow | Fog                 5
    Snow | Freezing Rain       1
    Name: weather_detail, dtype: int64





    Normal    5148
    Rain       128
    Fog         28
    Snow        20
    Name: weather_detail, dtype: int64




```python
df.isna().sum()
```




    schedule_date           0
    schedule_season         0
    schedule_week           0
    schedule_playoff        0
    team_home               0
    score_home              0
    score_away              0
    team_away               0
    team_favorite_id        0
    spread_favorite         0
    over_under_line         0
    stadium_name            0
    stadium_neutral         0
    weather_temperature     0
    weather_wind_mph        0
    weather_detail          0
    stadium_location        0
    stadium_open            0
    stadium_type            0
    stadium_weather_type    0
    stadium_capacity        0
    stadium_surface         0
    ELEVATION               0
    dtype: int64



## Adding New/Necessary Columns


```python
df['score_difference'] = abs(df['score_home'] - df['score_away']) #calculating a margin for each game
```


```python
#Converting the Over/Under score to a numeric score
df["over_under_line"] = pd.to_numeric(df["over_under_line"])
```


```python
#Creating a column to see how close the over/prediction is to actual
df['over_under_accuracy'] = (df['score_home'] + df['score_away']) - df['over_under_line']
```


```python
#Creating columns for nicknames of home/away teams to make divisional games make more sense
df["home_nickname"] = df["team_home"].str.split().str[-1]

df["away_nickname"] = df["team_away"].str.split().str[-1]
```


```python
#Create dictionaries for division of teams
Divisions = {
    'NFCW': ['Cardinals', '49ers', 'Rams', 'Seahawks'],
    'NFCN': ['Bears', 'Packers', 'Lions', 'Vikings'],
    'NFCE': ['Cowboys', 'Eagles', 'Redskins', 'Giants'],
    'NFCS': ['Panthers', 'Saints', 'Buccaneers', 'Falcons'],
    'AFCW': ['Chiefs', 'Chargers', 'Broncos', 'Raiders'],
    'AFCN': ['Browns', 'Bengals', 'Steelers', 'Ravens'],
    'AFCE': ['Jets', 'Patriots', 'Dolphins', 'Bills'],
    'AFCS': ['Titans', 'Texans', 'Colts', 'Jaguars']
}
divisions = pd.DataFrame(Divisions)
divisions

div_dict = {
    'Cardinals': 'NFCW', '49ers': 'NFCW', 'Rams': 'NFCW', 'Seahawks': 'NFCW', 
    'Bears': 'NFCN', 'Packers': 'NFCN', 'Lions': 'NFCN', 'Vikings': 'NFCN', 
    'Cowboys': 'NFCE', 'Eagles': 'NFCE', 'Redskins': 'NFCE', 'Giants': 'NFCE', 
    'Panthers': 'NFCS','Saints': 'NFCS',  'Buccaneers': 'NFCS', 'Falcons': 'NFCS', 
    'Chiefs': 'AFCW', 'Chargers': 'AFCW', 'Broncos': 'AFCW', 'Raiders': 'AFCW', 
    'Browns': 'AFCN', 'Bengals': 'AFCN', 'Steelers': 'AFCN', 'Ravens': 'AFCN', 
    'Jets': 'AFCE', 'Patriots': 'AFCE', 'Dolphins': 'AFCE', 'Bills': 'AFCE', 
    'Titans': 'AFCS','Texans': 'AFCS',  'Colts': 'AFCS', 'Jaguars': 'AFCS', 
}
div_dict
```




    {'Cardinals': 'NFCW',
     '49ers': 'NFCW',
     'Rams': 'NFCW',
     'Seahawks': 'NFCW',
     'Bears': 'NFCN',
     'Packers': 'NFCN',
     'Lions': 'NFCN',
     'Vikings': 'NFCN',
     'Cowboys': 'NFCE',
     'Eagles': 'NFCE',
     'Redskins': 'NFCE',
     'Giants': 'NFCE',
     'Panthers': 'NFCS',
     'Saints': 'NFCS',
     'Buccaneers': 'NFCS',
     'Falcons': 'NFCS',
     'Chiefs': 'AFCW',
     'Chargers': 'AFCW',
     'Broncos': 'AFCW',
     'Raiders': 'AFCW',
     'Browns': 'AFCN',
     'Bengals': 'AFCN',
     'Steelers': 'AFCN',
     'Ravens': 'AFCN',
     'Jets': 'AFCE',
     'Patriots': 'AFCE',
     'Dolphins': 'AFCE',
     'Bills': 'AFCE',
     'Titans': 'AFCS',
     'Texans': 'AFCS',
     'Colts': 'AFCS',
     'Jaguars': 'AFCS'}




```python
df["home_division"] = df["home_nickname"].map(div_dict)
df["away_division"] = df["away_nickname"].map(div_dict)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>schedule_date</th>
      <th>schedule_season</th>
      <th>schedule_week</th>
      <th>schedule_playoff</th>
      <th>team_home</th>
      <th>score_home</th>
      <th>score_away</th>
      <th>team_away</th>
      <th>team_favorite_id</th>
      <th>spread_favorite</th>
      <th>...</th>
      <th>stadium_weather_type</th>
      <th>stadium_capacity</th>
      <th>stadium_surface</th>
      <th>ELEVATION</th>
      <th>score_difference</th>
      <th>over_under_accuracy</th>
      <th>home_nickname</th>
      <th>away_nickname</th>
      <th>home_division</th>
      <th>away_division</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>Atlanta Falcons</td>
      <td>36.0</td>
      <td>28.0</td>
      <td>San Francisco 49ers</td>
      <td>ATL</td>
      <td>-6.5</td>
      <td>...</td>
      <td>dome</td>
      <td>71250.0</td>
      <td>FieldTurf</td>
      <td>303.8856</td>
      <td>8.0</td>
      <td>17.5</td>
      <td>Falcons</td>
      <td>49ers</td>
      <td>NFCS</td>
      <td>NFCW</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>Buffalo Bills</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>Tennessee Titans</td>
      <td>BUF</td>
      <td>-1.0</td>
      <td>...</td>
      <td>cold</td>
      <td>73967.0</td>
      <td>FieldTurf</td>
      <td>178.0000</td>
      <td>3.0</td>
      <td>-11.0</td>
      <td>Bills</td>
      <td>Titans</td>
      <td>AFCE</td>
      <td>AFCS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>Cleveland Browns</td>
      <td>7.0</td>
      <td>27.0</td>
      <td>Jacksonville Jaguars</td>
      <td>JAX</td>
      <td>-10.5</td>
      <td>...</td>
      <td>cold</td>
      <td>68000.0</td>
      <td>Grass</td>
      <td>238.0000</td>
      <td>20.0</td>
      <td>-4.5</td>
      <td>Browns</td>
      <td>Jaguars</td>
      <td>AFCN</td>
      <td>AFCS</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>Dallas Cowboys</td>
      <td>14.0</td>
      <td>41.0</td>
      <td>Philadelphia Eagles</td>
      <td>DAL</td>
      <td>-6.0</td>
      <td>...</td>
      <td>moderate</td>
      <td>80000.0</td>
      <td>Grass</td>
      <td>163.4000</td>
      <td>27.0</td>
      <td>15.5</td>
      <td>Cowboys</td>
      <td>Eagles</td>
      <td>NFCE</td>
      <td>NFCE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>Green Bay Packers</td>
      <td>16.0</td>
      <td>20.0</td>
      <td>New York Jets</td>
      <td>GB</td>
      <td>-2.5</td>
      <td>...</td>
      <td>cold</td>
      <td>80735.0</td>
      <td>Grass</td>
      <td>209.4000</td>
      <td>4.0</td>
      <td>-8.0</td>
      <td>Packers</td>
      <td>Jets</td>
      <td>NFCN</td>
      <td>AFCE</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5319</th>
      <td>2020-01-12</td>
      <td>2019</td>
      <td>19</td>
      <td>1</td>
      <td>Green Bay Packers</td>
      <td>28.0</td>
      <td>23.0</td>
      <td>Seattle Seahawks</td>
      <td>GB</td>
      <td>-4.5</td>
      <td>...</td>
      <td>cold</td>
      <td>80735.0</td>
      <td>Grass</td>
      <td>209.4000</td>
      <td>5.0</td>
      <td>5.5</td>
      <td>Packers</td>
      <td>Seahawks</td>
      <td>NFCN</td>
      <td>NFCW</td>
    </tr>
    <tr>
      <th>5320</th>
      <td>2020-01-12</td>
      <td>2019</td>
      <td>19</td>
      <td>1</td>
      <td>Kansas City Chiefs</td>
      <td>51.0</td>
      <td>31.0</td>
      <td>Houston Texans</td>
      <td>KC</td>
      <td>-10.0</td>
      <td>...</td>
      <td>cold</td>
      <td>76416.0</td>
      <td>Grass</td>
      <td>264.9000</td>
      <td>20.0</td>
      <td>31.5</td>
      <td>Chiefs</td>
      <td>Texans</td>
      <td>AFCW</td>
      <td>AFCS</td>
    </tr>
    <tr>
      <th>5321</th>
      <td>2020-01-19</td>
      <td>2019</td>
      <td>20</td>
      <td>1</td>
      <td>Kansas City Chiefs</td>
      <td>35.0</td>
      <td>24.0</td>
      <td>Tennessee Titans</td>
      <td>KC</td>
      <td>-7.0</td>
      <td>...</td>
      <td>cold</td>
      <td>76416.0</td>
      <td>Grass</td>
      <td>264.9000</td>
      <td>11.0</td>
      <td>8.0</td>
      <td>Chiefs</td>
      <td>Titans</td>
      <td>AFCW</td>
      <td>AFCS</td>
    </tr>
    <tr>
      <th>5322</th>
      <td>2020-01-19</td>
      <td>2019</td>
      <td>20</td>
      <td>1</td>
      <td>San Francisco 49ers</td>
      <td>37.0</td>
      <td>20.0</td>
      <td>Green Bay Packers</td>
      <td>SF</td>
      <td>-8.0</td>
      <td>...</td>
      <td>moderate</td>
      <td>68500.0</td>
      <td>Grass</td>
      <td>2.4000</td>
      <td>17.0</td>
      <td>10.5</td>
      <td>49ers</td>
      <td>Packers</td>
      <td>NFCW</td>
      <td>NFCN</td>
    </tr>
    <tr>
      <th>5323</th>
      <td>2020-02-02</td>
      <td>2019</td>
      <td>21</td>
      <td>1</td>
      <td>Kansas City Chiefs</td>
      <td>31.0</td>
      <td>20.0</td>
      <td>San Francisco 49ers</td>
      <td>KC</td>
      <td>-1.5</td>
      <td>...</td>
      <td>warm</td>
      <td>65326.0</td>
      <td>Grass</td>
      <td>8.8000</td>
      <td>11.0</td>
      <td>-2.0</td>
      <td>Chiefs</td>
      <td>49ers</td>
      <td>AFCW</td>
      <td>NFCW</td>
    </tr>
  </tbody>
</table>
<p>5324 rows × 29 columns</p>
</div>




```python
#Creating a function that identifies whether the participating teams are in the same division
def divisional(row):
    if row['home_division'] == row['away_division']:
        val = 'True'
    else:
        val = 'False'
    return val

df['division_game'] = df.apply(divisional, axis=1)


df.loc[df.division_game == 'False', 'division_game'] = 0
df.loc[df.division_game == 'True', 'division_game'] = 1
```


```python
# Creating a resulting Winner column identifying which team wins the game
df['Winner'] = df.loc[(df.score_home > df.score_away), 'team_home']
df['Winner'] = df['Winner'].fillna(df.team_away)

```


```python
#Creating a binary column to determine whether home team won or not
df['result'] = (df.score_home > df.score_away).astype(int)
```


```python
#Creating another list for team ID to make working with team_favorite_id easier
team_abb = [('ATL','Falcons'),
            ('BUF', 'Bills'),
            ('JAX','Jaguars'),
            ('DAL', 'Cowboys'),
            ('GB', 'Packers'),
            ('IND','Colts'),
            ('MIA', 'Dolphins'),
            ('MIN','Vikings'),
            ('TB', 'Buccaneers'),
            ('NYG','Giants'),
            ('OAK', 'Raiders'),
            ('BAL','Ravens'),
            ('WAS', 'Redskins'),
            ('LAR', 'Rams'),
            ('ARI', 'Cardinals'),
            ('CIN', 'Bengals'),
            ('DEN', 'Broncos'),
            ('PHI', 'Eagles'),
            ('LAC','Chargers'),
            ('CAR','Panthers'),
            ('TEN','Titans'),
            ('NYJ', 'Jets'),
            ('CHI', 'Bears'),
            ('PIT', 'Steelers'),
            ('KC', 'Chiefs'),
            ('NE','Patriots'),
            ('SEA','Seahawks'),
            ('NO', 'Saints'),
            ('DET', 'Lions'),
            ('SF', '49ers'),
            ('CLE', 'Browns'),
            ('HOU', 'Texans')]
```


```python
# Changing team abbreviations to be the actual team nickname
for team in team_abb:
    df.team_favorite_id.replace(team[0], team[1],inplace=True)
    
```


```python
# creating binary home favorite and away favorite columns
df['home_favorite'] = np.where(df['home_nickname']==df['team_favorite_id'], 1, 0)
df['away_favorite'] = np.where(df['away_nickname']==df['team_favorite_id'], 1, 0)
df.home_favorite.fillna(0, inplace=True)
df.away_favorite.fillna(0, inplace=True)

```


```python
# Determines whether or not favorite was the home team, and how much the favorite won by
def Fav_MoV (row):
    # Determines whether the favorite was the home team
    if re.search(row['team_favorite_id'], row['team_home'], re.IGNORECASE):
        return row['score_home']-row['score_away']
    else:
        return row['score_away']-row['score_home']
    

df['fav_MoV'] = df.apply(lambda row: Fav_MoV(row), axis=1)

```


```python
#Function to determine how much the favorite beat the spread by
def Fav_Spread_MoV (row):
    return row['spread_favorite'] + row['fav_MoV']  

df['fav_spread_MoV'] = df.apply(lambda row: Fav_Spread_MoV(row),axis=1)
```


```python
df.columns
```




    Index(['schedule_date', 'schedule_season', 'schedule_week', 'schedule_playoff',
           'team_home', 'score_home', 'score_away', 'team_away',
           'team_favorite_id', 'spread_favorite', 'over_under_line',
           'stadium_name', 'stadium_neutral', 'weather_temperature',
           'weather_wind_mph', 'weather_detail', 'stadium_location',
           'stadium_open', 'stadium_type', 'stadium_weather_type',
           'stadium_capacity', 'stadium_surface', 'ELEVATION', 'score_difference',
           'over_under_accuracy', 'home_nickname', 'away_nickname',
           'home_division', 'away_division', 'division_game', 'Winner', 'result',
           'home_favorite', 'away_favorite', 'fav_MoV', 'fav_spread_MoV'],
          dtype='object')




```python
#Rearranging column names
df = df[['schedule_date', 'schedule_season', 'schedule_week', 'schedule_playoff', 'division_game', 
       'team_home', 'home_nickname', 'score_home', 'score_away', 'team_away', 'away_nickname',  'Winner',
       'stadium_name', 'stadium_neutral', 'weather_temperature',
       'weather_wind_mph', 'weather_detail', 'stadium_location', 'stadium_open', 'stadium_type',
       'stadium_weather_type', 'stadium_capacity', 'stadium_surface', 'ELEVATION',
        'team_favorite_id', 'spread_favorite', 'score_difference', 'over_under_line', 'over_under_accuracy',
         'fav_MoV', 'fav_spread_MoV', 'home_favorite', 'away_favorite', 'result',]]
```

## EDA Section - Figuring out which variable would be best to try to predict


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>schedule_date</th>
      <th>schedule_season</th>
      <th>schedule_week</th>
      <th>schedule_playoff</th>
      <th>division_game</th>
      <th>team_home</th>
      <th>home_nickname</th>
      <th>score_home</th>
      <th>score_away</th>
      <th>team_away</th>
      <th>...</th>
      <th>team_favorite_id</th>
      <th>spread_favorite</th>
      <th>score_difference</th>
      <th>over_under_line</th>
      <th>over_under_accuracy</th>
      <th>fav_MoV</th>
      <th>fav_spread_MoV</th>
      <th>home_favorite</th>
      <th>away_favorite</th>
      <th>result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Atlanta Falcons</td>
      <td>Falcons</td>
      <td>36.0</td>
      <td>28.0</td>
      <td>San Francisco 49ers</td>
      <td>...</td>
      <td>Falcons</td>
      <td>-6.5</td>
      <td>8.0</td>
      <td>46.5</td>
      <td>17.5</td>
      <td>8.0</td>
      <td>1.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Buffalo Bills</td>
      <td>Bills</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>Tennessee Titans</td>
      <td>...</td>
      <td>Bills</td>
      <td>-1.0</td>
      <td>3.0</td>
      <td>40.0</td>
      <td>-11.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Cleveland Browns</td>
      <td>Browns</td>
      <td>7.0</td>
      <td>27.0</td>
      <td>Jacksonville Jaguars</td>
      <td>...</td>
      <td>Jaguars</td>
      <td>-10.5</td>
      <td>20.0</td>
      <td>38.5</td>
      <td>-4.5</td>
      <td>20.0</td>
      <td>9.5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Dallas Cowboys</td>
      <td>Cowboys</td>
      <td>14.0</td>
      <td>41.0</td>
      <td>Philadelphia Eagles</td>
      <td>...</td>
      <td>Cowboys</td>
      <td>-6.0</td>
      <td>27.0</td>
      <td>39.5</td>
      <td>15.5</td>
      <td>-27.0</td>
      <td>-33.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000-09-03</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Green Bay Packers</td>
      <td>Packers</td>
      <td>16.0</td>
      <td>20.0</td>
      <td>New York Jets</td>
      <td>...</td>
      <td>Packers</td>
      <td>-2.5</td>
      <td>4.0</td>
      <td>44.0</td>
      <td>-8.0</td>
      <td>-4.0</td>
      <td>-6.5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5319</th>
      <td>2020-01-12</td>
      <td>2019</td>
      <td>19</td>
      <td>1</td>
      <td>0</td>
      <td>Green Bay Packers</td>
      <td>Packers</td>
      <td>28.0</td>
      <td>23.0</td>
      <td>Seattle Seahawks</td>
      <td>...</td>
      <td>Packers</td>
      <td>-4.5</td>
      <td>5.0</td>
      <td>45.5</td>
      <td>5.5</td>
      <td>5.0</td>
      <td>0.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5320</th>
      <td>2020-01-12</td>
      <td>2019</td>
      <td>19</td>
      <td>1</td>
      <td>0</td>
      <td>Kansas City Chiefs</td>
      <td>Chiefs</td>
      <td>51.0</td>
      <td>31.0</td>
      <td>Houston Texans</td>
      <td>...</td>
      <td>Chiefs</td>
      <td>-10.0</td>
      <td>20.0</td>
      <td>50.5</td>
      <td>31.5</td>
      <td>20.0</td>
      <td>10.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5321</th>
      <td>2020-01-19</td>
      <td>2019</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>Kansas City Chiefs</td>
      <td>Chiefs</td>
      <td>35.0</td>
      <td>24.0</td>
      <td>Tennessee Titans</td>
      <td>...</td>
      <td>Chiefs</td>
      <td>-7.0</td>
      <td>11.0</td>
      <td>51.0</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>4.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5322</th>
      <td>2020-01-19</td>
      <td>2019</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>San Francisco 49ers</td>
      <td>49ers</td>
      <td>37.0</td>
      <td>20.0</td>
      <td>Green Bay Packers</td>
      <td>...</td>
      <td>49ers</td>
      <td>-8.0</td>
      <td>17.0</td>
      <td>46.5</td>
      <td>10.5</td>
      <td>17.0</td>
      <td>9.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5323</th>
      <td>2020-02-02</td>
      <td>2019</td>
      <td>21</td>
      <td>1</td>
      <td>0</td>
      <td>Kansas City Chiefs</td>
      <td>Chiefs</td>
      <td>31.0</td>
      <td>20.0</td>
      <td>San Francisco 49ers</td>
      <td>...</td>
      <td>Chiefs</td>
      <td>-1.5</td>
      <td>11.0</td>
      <td>53.0</td>
      <td>-2.0</td>
      <td>11.0</td>
      <td>9.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5324 rows × 34 columns</p>
</div>




```python
#Plotting distributions for a few different variables
plt.hist(df.score_difference)
plt.figtext(0.65,0.5, df.score_difference.describe().loc[['mean','std']].to_string())
plt.title('Score Margin')
plt.show()
plt.hist(df.spread_favorite)
plt.figtext(0.35,0.5, df.spread_favorite.describe().loc[['mean','std']].to_string())
plt.title('Predicted Spread')
plt.show()
plt.hist(df.over_under_line)
plt.figtext(0.65,0.5, df.over_under_line.describe().loc[['mean','std']].to_string())
plt.title('Over/Under Score')
plt.show()
plt.hist(df.over_under_accuracy)
plt.figtext(0.65,0.5, df.over_under_accuracy.describe().loc[['mean','std']].to_string())
plt.title('Predicted Over/Under Accuracy')
plt.show()
plt.hist(df.fav_MoV)
plt.figtext(0.65,0.5, df.fav_MoV.describe().loc[['mean','std']].to_string())
plt.title('Favorite Margin of Victory')
plt.show()
plt.hist(df.fav_spread_MoV)
plt.figtext(0.65,0.5, df.fav_spread_MoV.describe().loc[['mean','std']].to_string())
plt.title('Favorite Spread Margin of Victory')
plt.show()
```


![png](output_47_0.png)



![png](output_47_1.png)



![png](output_47_2.png)



![png](output_47_3.png)



![png](output_47_4.png)



![png](output_47_5.png)



```python
df.describe().transpose() #Running summary statistics for games

#Biggest observation - on average, the favorite wins by 5. However the spread MoV average is right on the money at -.06
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>schedule_season</th>
      <td>5324.0</td>
      <td>2009.527047</td>
      <td>5.754236</td>
      <td>2000.0000</td>
      <td>2005.0</td>
      <td>2010.0</td>
      <td>2015.000000</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>schedule_week</th>
      <td>5324.0</td>
      <td>9.511833</td>
      <td>5.271909</td>
      <td>1.0000</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>14.000000</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>schedule_playoff</th>
      <td>5324.0</td>
      <td>0.041322</td>
      <td>0.199053</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>score_home</th>
      <td>5324.0</td>
      <td>23.159842</td>
      <td>10.382276</td>
      <td>0.0000</td>
      <td>16.0</td>
      <td>23.0</td>
      <td>30.000000</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>score_away</th>
      <td>5324.0</td>
      <td>20.710368</td>
      <td>10.088146</td>
      <td>0.0000</td>
      <td>13.0</td>
      <td>20.0</td>
      <td>27.000000</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>weather_temperature</th>
      <td>5324.0</td>
      <td>60.289000</td>
      <td>15.354138</td>
      <td>-6.0000</td>
      <td>50.0</td>
      <td>63.0</td>
      <td>72.000000</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>weather_wind_mph</th>
      <td>5324.0</td>
      <td>6.380589</td>
      <td>5.324449</td>
      <td>0.0000</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>10.000000</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>stadium_open</th>
      <td>5324.0</td>
      <td>1989.093967</td>
      <td>19.077342</td>
      <td>1909.0000</td>
      <td>1975.0</td>
      <td>1997.0</td>
      <td>2002.000000</td>
      <td>2017.0</td>
    </tr>
    <tr>
      <th>stadium_capacity</th>
      <td>5324.0</td>
      <td>69969.871740</td>
      <td>7302.570992</td>
      <td>27000.0000</td>
      <td>65500.0</td>
      <td>68756.0</td>
      <td>76117.331288</td>
      <td>93605.0</td>
    </tr>
    <tr>
      <th>ELEVATION</th>
      <td>5324.0</td>
      <td>178.415612</td>
      <td>283.340807</td>
      <td>0.9144</td>
      <td>8.8</td>
      <td>145.4</td>
      <td>216.103200</td>
      <td>1611.2</td>
    </tr>
    <tr>
      <th>spread_favorite</th>
      <td>5324.0</td>
      <td>-5.397164</td>
      <td>3.423281</td>
      <td>-26.5000</td>
      <td>-7.0</td>
      <td>-4.5</td>
      <td>-3.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>score_difference</th>
      <td>5324.0</td>
      <td>11.692900</td>
      <td>9.281484</td>
      <td>0.0000</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>17.000000</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>over_under_line</th>
      <td>5324.0</td>
      <td>43.259673</td>
      <td>4.946316</td>
      <td>30.0000</td>
      <td>40.0</td>
      <td>43.5</td>
      <td>46.500000</td>
      <td>63.5</td>
    </tr>
    <tr>
      <th>over_under_accuracy</th>
      <td>5324.0</td>
      <td>0.610537</td>
      <td>13.498027</td>
      <td>-39.5000</td>
      <td>-9.0</td>
      <td>0.0</td>
      <td>9.500000</td>
      <td>68.5</td>
    </tr>
    <tr>
      <th>fav_MoV</th>
      <td>5324.0</td>
      <td>5.333020</td>
      <td>13.944501</td>
      <td>-45.0000</td>
      <td>-3.0</td>
      <td>4.0</td>
      <td>14.000000</td>
      <td>59.0</td>
    </tr>
    <tr>
      <th>fav_spread_MoV</th>
      <td>5324.0</td>
      <td>-0.064144</td>
      <td>13.475267</td>
      <td>-52.0000</td>
      <td>-8.5</td>
      <td>-0.5</td>
      <td>8.125000</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>home_favorite</th>
      <td>5324.0</td>
      <td>0.648385</td>
      <td>0.477519</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>away_favorite</th>
      <td>5324.0</td>
      <td>0.322878</td>
      <td>0.467620</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>result</th>
      <td>5324.0</td>
      <td>0.570060</td>
      <td>0.495114</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



EDA of Playoff Games


```python
playoffs = df.loc[df['schedule_playoff'] == True]
playoffs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>schedule_date</th>
      <th>schedule_season</th>
      <th>schedule_week</th>
      <th>schedule_playoff</th>
      <th>division_game</th>
      <th>team_home</th>
      <th>home_nickname</th>
      <th>score_home</th>
      <th>score_away</th>
      <th>team_away</th>
      <th>...</th>
      <th>team_favorite_id</th>
      <th>spread_favorite</th>
      <th>score_difference</th>
      <th>over_under_line</th>
      <th>over_under_accuracy</th>
      <th>fav_MoV</th>
      <th>fav_spread_MoV</th>
      <th>home_favorite</th>
      <th>away_favorite</th>
      <th>result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>248</th>
      <td>2000-12-30</td>
      <td>2000</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>Miami Dolphins</td>
      <td>Dolphins</td>
      <td>23.0</td>
      <td>17.0</td>
      <td>Indianapolis Colts</td>
      <td>...</td>
      <td>Colts</td>
      <td>-1.5</td>
      <td>6.0</td>
      <td>42.0</td>
      <td>-2.0</td>
      <td>-6.0</td>
      <td>-7.5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>249</th>
      <td>2000-12-30</td>
      <td>2000</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>New Orleans Saints</td>
      <td>Saints</td>
      <td>31.0</td>
      <td>28.0</td>
      <td>St. Louis Rams</td>
      <td>...</td>
      <td>Rams</td>
      <td>-5.5</td>
      <td>3.0</td>
      <td>55.0</td>
      <td>4.0</td>
      <td>-3.0</td>
      <td>-8.5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>250</th>
      <td>2000-12-31</td>
      <td>2000</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>Baltimore Ravens</td>
      <td>Ravens</td>
      <td>21.0</td>
      <td>3.0</td>
      <td>Denver Broncos</td>
      <td>...</td>
      <td>Ravens</td>
      <td>-3.5</td>
      <td>18.0</td>
      <td>41.0</td>
      <td>-17.0</td>
      <td>18.0</td>
      <td>14.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>251</th>
      <td>2000-12-31</td>
      <td>2000</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>Philadelphia Eagles</td>
      <td>Eagles</td>
      <td>21.0</td>
      <td>3.0</td>
      <td>Tampa Bay Buccaneers</td>
      <td>...</td>
      <td>Buccaneers</td>
      <td>-3.0</td>
      <td>18.0</td>
      <td>34.0</td>
      <td>-10.0</td>
      <td>-18.0</td>
      <td>-21.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>252</th>
      <td>2001-01-06</td>
      <td>2000</td>
      <td>19</td>
      <td>1</td>
      <td>0</td>
      <td>Minnesota Vikings</td>
      <td>Vikings</td>
      <td>34.0</td>
      <td>16.0</td>
      <td>New Orleans Saints</td>
      <td>...</td>
      <td>Vikings</td>
      <td>-8.0</td>
      <td>18.0</td>
      <td>49.5</td>
      <td>0.5</td>
      <td>18.0</td>
      <td>10.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5319</th>
      <td>2020-01-12</td>
      <td>2019</td>
      <td>19</td>
      <td>1</td>
      <td>0</td>
      <td>Green Bay Packers</td>
      <td>Packers</td>
      <td>28.0</td>
      <td>23.0</td>
      <td>Seattle Seahawks</td>
      <td>...</td>
      <td>Packers</td>
      <td>-4.5</td>
      <td>5.0</td>
      <td>45.5</td>
      <td>5.5</td>
      <td>5.0</td>
      <td>0.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5320</th>
      <td>2020-01-12</td>
      <td>2019</td>
      <td>19</td>
      <td>1</td>
      <td>0</td>
      <td>Kansas City Chiefs</td>
      <td>Chiefs</td>
      <td>51.0</td>
      <td>31.0</td>
      <td>Houston Texans</td>
      <td>...</td>
      <td>Chiefs</td>
      <td>-10.0</td>
      <td>20.0</td>
      <td>50.5</td>
      <td>31.5</td>
      <td>20.0</td>
      <td>10.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5321</th>
      <td>2020-01-19</td>
      <td>2019</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>Kansas City Chiefs</td>
      <td>Chiefs</td>
      <td>35.0</td>
      <td>24.0</td>
      <td>Tennessee Titans</td>
      <td>...</td>
      <td>Chiefs</td>
      <td>-7.0</td>
      <td>11.0</td>
      <td>51.0</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>4.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5322</th>
      <td>2020-01-19</td>
      <td>2019</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>San Francisco 49ers</td>
      <td>49ers</td>
      <td>37.0</td>
      <td>20.0</td>
      <td>Green Bay Packers</td>
      <td>...</td>
      <td>49ers</td>
      <td>-8.0</td>
      <td>17.0</td>
      <td>46.5</td>
      <td>10.5</td>
      <td>17.0</td>
      <td>9.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5323</th>
      <td>2020-02-02</td>
      <td>2019</td>
      <td>21</td>
      <td>1</td>
      <td>0</td>
      <td>Kansas City Chiefs</td>
      <td>Chiefs</td>
      <td>31.0</td>
      <td>20.0</td>
      <td>San Francisco 49ers</td>
      <td>...</td>
      <td>Chiefs</td>
      <td>-1.5</td>
      <td>11.0</td>
      <td>53.0</td>
      <td>-2.0</td>
      <td>11.0</td>
      <td>9.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>220 rows × 34 columns</p>
</div>




```python
plt.hist(playoffs.score_difference)
plt.figtext(0.65,0.5, playoffs.score_difference.describe().loc[['mean','std']].to_string())
plt.title('Playoff Score Margin')
plt.show()
plt.hist(playoffs.spread_favorite)
plt.figtext(0.35,0.5, playoffs.spread_favorite.describe().loc[['mean','std']].to_string())
plt.title('Playoff Predicted Spread')
plt.show()
plt.hist(playoffs.over_under_line)
plt.figtext(0.65,0.5, playoffs.over_under_line.describe().loc[['mean','std']].to_string())
plt.title('Playoff Over/Under Score')
plt.show()
plt.hist(playoffs.over_under_accuracy)
plt.figtext(0.65,0.5, playoffs.over_under_accuracy.describe().loc[['mean','std']].to_string())
plt.title('Playoff Predicted Over/Under Accuracy')
plt.show()
plt.hist(playoffs.fav_MoV)
plt.figtext(0.65,0.5, df.fav_MoV.describe().loc[['mean','std']].to_string())
plt.title('Playoff Favorite Margin of Victory')
plt.show()
plt.hist(playoffs.fav_spread_MoV)
plt.figtext(0.65,0.5, df.fav_spread_MoV.describe().loc[['mean','std']].to_string())
plt.title('Playoff Favorite Spread Margin of Victory')
plt.show()
```


![png](output_51_0.png)



![png](output_51_1.png)



![png](output_51_2.png)



![png](output_51_3.png)



![png](output_51_4.png)



![png](output_51_5.png)



```python
playoffs.describe().transpose()
#Summary statistic means and std's look incredibly similar to regular season, no significant differences other than less scoring
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>schedule_season</th>
      <td>220.0</td>
      <td>2009.500000</td>
      <td>5.779431</td>
      <td>2000.0000</td>
      <td>2004.750000</td>
      <td>2009.500000</td>
      <td>2014.250000</td>
      <td>2019.000000</td>
    </tr>
    <tr>
      <th>schedule_week</th>
      <td>220.0</td>
      <td>19.000000</td>
      <td>0.955637</td>
      <td>18.0000</td>
      <td>18.000000</td>
      <td>19.000000</td>
      <td>20.000000</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>schedule_playoff</th>
      <td>220.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.0000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>score_home</th>
      <td>220.0</td>
      <td>25.145455</td>
      <td>10.636232</td>
      <td>0.0000</td>
      <td>17.000000</td>
      <td>24.000000</td>
      <td>31.000000</td>
      <td>51.000000</td>
    </tr>
    <tr>
      <th>score_away</th>
      <td>220.0</td>
      <td>20.695455</td>
      <td>9.527800</td>
      <td>0.0000</td>
      <td>14.000000</td>
      <td>20.000000</td>
      <td>27.000000</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>weather_temperature</th>
      <td>220.0</td>
      <td>56.265997</td>
      <td>13.931600</td>
      <td>-6.0000</td>
      <td>50.463855</td>
      <td>53.811688</td>
      <td>72.000000</td>
      <td>78.151316</td>
    </tr>
    <tr>
      <th>weather_wind_mph</th>
      <td>220.0</td>
      <td>6.142345</td>
      <td>4.307688</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>7.513072</td>
      <td>8.739854</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>stadium_open</th>
      <td>220.0</td>
      <td>1989.387218</td>
      <td>19.459598</td>
      <td>1926.0000</td>
      <td>1975.000000</td>
      <td>1998.500000</td>
      <td>2002.000000</td>
      <td>2017.000000</td>
    </tr>
    <tr>
      <th>stadium_capacity</th>
      <td>220.0</td>
      <td>70253.552372</td>
      <td>6502.286616</td>
      <td>53250.0000</td>
      <td>65515.000000</td>
      <td>68949.500000</td>
      <td>76125.000000</td>
      <td>93605.000000</td>
    </tr>
    <tr>
      <th>ELEVATION</th>
      <td>220.0</td>
      <td>188.241655</td>
      <td>318.134830</td>
      <td>0.9144</td>
      <td>12.250000</td>
      <td>142.036800</td>
      <td>217.552400</td>
      <td>1611.200000</td>
    </tr>
    <tr>
      <th>spread_favorite</th>
      <td>220.0</td>
      <td>-5.534091</td>
      <td>3.006837</td>
      <td>-16.0000</td>
      <td>-7.500000</td>
      <td>-4.750000</td>
      <td>-3.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>score_difference</th>
      <td>220.0</td>
      <td>11.768182</td>
      <td>8.740077</td>
      <td>1.0000</td>
      <td>4.000000</td>
      <td>10.000000</td>
      <td>17.000000</td>
      <td>41.000000</td>
    </tr>
    <tr>
      <th>over_under_line</th>
      <td>220.0</td>
      <td>44.800000</td>
      <td>6.001978</td>
      <td>31.0000</td>
      <td>41.000000</td>
      <td>45.000000</td>
      <td>48.500000</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>over_under_accuracy</th>
      <td>220.0</td>
      <td>1.040909</td>
      <td>13.752481</td>
      <td>-39.5000</td>
      <td>-8.125000</td>
      <td>-0.250000</td>
      <td>7.125000</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>fav_MoV</th>
      <td>220.0</td>
      <td>4.268182</td>
      <td>14.043178</td>
      <td>-41.0000</td>
      <td>-5.000000</td>
      <td>4.000000</td>
      <td>14.000000</td>
      <td>41.000000</td>
    </tr>
    <tr>
      <th>fav_spread_MoV</th>
      <td>220.0</td>
      <td>-1.265909</td>
      <td>13.819977</td>
      <td>-42.0000</td>
      <td>-10.000000</td>
      <td>-1.000000</td>
      <td>8.000000</td>
      <td>35.000000</td>
    </tr>
    <tr>
      <th>home_favorite</th>
      <td>220.0</td>
      <td>0.822727</td>
      <td>0.382770</td>
      <td>0.0000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>away_favorite</th>
      <td>220.0</td>
      <td>0.150000</td>
      <td>0.357886</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>result</th>
      <td>220.0</td>
      <td>0.622727</td>
      <td>0.485809</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
home_win = "{:.2f}".format((sum((df.result == 1) & (df.stadium_neutral == 0)) / len(df)) * 100)
away_win = "{:.2f}".format((sum((df.result == 0) & (df.stadium_neutral == 0)) / len(df)) * 100)
under_line = "{:.2f}".format((sum((df.score_home + df.score_away) < df.over_under_line) / len(df)) * 100)
over_line = "{:.2f}".format((sum((df.score_home + df.score_away) > df.over_under_line) / len(df)) * 100)

#Summary Statistics to consider before gambling in the first place
print("Number of Games: " + str(len(df)))
print("Home Straight Up Win Percentage: " + home_win + "%")
print("Away Straight Up Win Percentage: " + away_win + "%")
print("Under Percentage: " + under_line + "%")
print("Over Percentage: " + over_line + "%")
```

    Number of Games: 5324
    Home Straight Up Win Percentage: 56.56%
    Away Straight Up Win Percentage: 42.26%
    Under Percentage: 49.81%
    Over Percentage: 48.46%


## Machine Learning Section

### Classification Methods - predicting result of a game in whether the home team wins or not


```python
df.columns
```




    Index(['schedule_date', 'schedule_season', 'schedule_week', 'schedule_playoff',
           'division_game', 'team_home', 'home_nickname', 'score_home',
           'score_away', 'team_away', 'away_nickname', 'Winner', 'stadium_name',
           'stadium_neutral', 'weather_temperature', 'weather_wind_mph',
           'weather_detail', 'stadium_location', 'stadium_open', 'stadium_type',
           'stadium_weather_type', 'stadium_capacity', 'stadium_surface',
           'ELEVATION', 'team_favorite_id', 'spread_favorite', 'score_difference',
           'over_under_line', 'over_under_accuracy', 'fav_MoV', 'fav_spread_MoV',
           'home_favorite', 'away_favorite', 'result'],
          dtype='object')




```python
from sklearn.model_selection import train_test_split, cross_val_score
y = df['result']
X = df[['schedule_season', 'schedule_week', 'schedule_playoff', 'division_game',
        'weather_temperature', 'weather_wind_mph', 'spread_favorite', 'score_difference',
       'over_under_line', 'over_under_accuracy', 'fav_spread_MoV', 'home_favorite', 'away_favorite']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=785)
```


```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(min_samples_leaf=3)
tree.fit(X_train,y_train)
yhattree = tree.predict(X_test)
y_prob_tree = tree.predict_proba(X_test)[:,1]

print("Decision Tree Accuracy: ",accuracy_score(y_test, yhattree))
print("Decision Tree F1: ", f1_score(y_test, yhattree))
print("Decision Tree AUC: ", roc_auc_score(y_test, yhattree))
```

    Decision Tree Accuracy:  0.9912390488110138
    Decision Tree F1:  0.9923664122137406
    Decision Tree AUC:  0.9912309368191721



```python
#Random Forest
rf = RandomForestClassifier(n_jobs=-1)
parameters = {
    'n_estimators': [50, 250, 500],
    'max_depth': [5, 10, 50, 100, None]
}

cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(X_train, y_train)

rf = RandomForestClassifier(max_depth=100, n_estimators=500, n_jobs=-1)
rf.fit(X_train, y_train)
yhatrf = (rf.predict(X_test))
y_prob_rf = rf.predict_proba(X_test)[:,1]

print("Random Forest Accuracy: ", accuracy_score(y_test, yhatrf))
print("Random Forest F1: ", f1_score(y_test, yhatrf))
print("Random Forest AUC: ", roc_auc_score(y_test, y_prob_rf))
```

    Random Forest Accuracy:  0.9912390488110138
    Random Forest F1:  0.9923747276688453
    Random Forest AUC:  0.9996788094322697



```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import RFE

#Using an Least Discriminate Analysis to try and narrow down my model
original = LDA()

# choose the best features for my model
rfe = RFE(original, 5)
rfe = rfe.fit(X, y)

# features
print(X.columns)
print(rfe.support_)
print(rfe.ranking_)

```

    Index(['schedule_season', 'schedule_week', 'schedule_playoff', 'division_game',
           'weather_temperature', 'weather_wind_mph', 'spread_favorite',
           'score_difference', 'over_under_line', 'over_under_accuracy',
           'fav_spread_MoV', 'home_favorite', 'away_favorite'],
          dtype='object')
    [False False False  True False False  True False False False  True  True
      True]
    [6 4 3 1 7 9 1 2 5 8 1 1 1]


    /Applications/anaconda3/lib/python3.8/site-packages/sklearn/utils/validation.py:68: FutureWarning: Pass n_features_to_select=5 as keyword args. From version 0.25 passing these as positional arguments will result in an error
      warnings.warn("Pass {} as keyword args. From version 0.25 "



```python
y = df['result']
new_X = df[['division_game', 'spread_favorite', 'fav_spread_MoV', 'home_favorite', 'away_favorite']]
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size = .3, random_state=785)
```


```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(min_samples_leaf=3)
tree.fit(X_train,y_train)
yhattree = tree.predict(X_test)
y_prob_tree = tree.predict_proba(X_test)[:,1]

print("Decision Tree Accuracy: ",accuracy_score(y_test, yhattree))
print("Decision Tree F1: ", f1_score(y_test, yhattree))
print("Decision Tree AUC: ", roc_auc_score(y_test, yhattree))
```

    Decision Tree Accuracy:  0.9949937421777222
    Decision Tree F1:  0.9956379498364231
    Decision Tree AUC:  0.9950708061002178


After narrowing down factors and ultimately settling upon those last 5, I concluded those were the best five in determining whether or not the result of the game would be the home team winning.

### Linear Regression/KNN Attempts - predicting score difference

I tried doing linear regression, except it didn't work well since so many variables were based off of each other. KNN regression model seemed to perform much better.


```python
sns.pairplot(df, 
             x_vars=['weather_temperature', 'spread_favorite', 'over_under_line'], 
             y_vars='score_difference', size=7, kind='reg')
```

    /Applications/anaconda3/lib/python3.8/site-packages/seaborn/axisgrid.py:2071: UserWarning: The `size` parameter has been renamed to `height`; please update your code.
      warnings.warn(msg, UserWarning)





    <seaborn.axisgrid.PairGrid at 0x7f9f938ebcd0>




![png](output_65_2.png)



```python
from sklearn.model_selection import train_test_split, cross_val_score
y = df['score_difference']
X = df[['spread_favorite', 'over_under_line', 'schedule_week', 'schedule_playoff', 'division_game', 'fav_spread_MoV']] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=785)
```


```python
lm = LinearRegression()
lm.fit(X_train, y_train)
yhat_test = lm.predict(X_test)
mse_lm_tt = mean_squared_error(y_test, yhat_test)
print(mse_lm_tt)
print(r2_score(y_test, yhat_test))
```

    68.50817231495157
    0.24315533614559737



```python
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)
yhat_knn_test = knn.predict(X_test)
mse_knn_tt = mean_squared_error(y_test, yhat_knn_test)
print(mse_knn_tt)
print(r2_score(y_test, yhat_knn_test))
```

    1.5561118064246975
    0.9828088405038893



```python
lm = LinearRegression()
scores_lm = cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')
mse_lm_cv = -1*scores_lm.mean()
print(mse_lm_cv)
```

    66.59114763593605



```python
knn = KNeighborsRegressor(n_neighbors=3)
scores_knn = cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error')
mse_knn_cv = -1*scores_knn.mean()
print(mse_knn_cv)
```

    1.249993926341808



```python
print('LinReg Train/Test: {0:.4f}'.format(mse_lm_tt))
print('KNN Train/Test: {0:.4f}'.format(mse_knn_tt))
print('LinReg 10-fold CV: {0:.4f}'.format(mse_lm_cv))
print('KNN 10-fold CV: {0:.4f}'.format(mse_knn_cv))

```

    LinReg Train/Test: 68.5082
    KNN Train/Test: 1.5561
    LinReg 10-fold CV: 66.5911
    KNN 10-fold CV: 1.2500



```python
ols = LinearRegression()
model = ols.fit(X_train,y_train)


#Showing the MSE for this model
print("MSE (Training) : {:.3f}".format(mean_squared_error(y_train,ols.predict(X_train))))

#One way to check the fit (and evidence of overfitting) is to see 
#how it preforms in sample vs out of sample
print("OLS accuracy on training set: {:.3f}".format(ols.score(X_train, y_train)))


```

    MSE (Training) : 65.403
    OLS accuracy on training set: 0.223

