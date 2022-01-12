import pandas as pd
import pybaseball as pb
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

'''
I'm so sick of losing fantasy baseball year in and year out. Hopefully a data-
driven approach (and a league of managers who won't look much deeper than Yahoo's
top-level stats)
'''
# Pull pybaseball data (min 10 PAs) from 2021 season
batters = pd.DataFrame(pb.batting_stats(2021,qual=10))

# Calculate fantasy points based on stats, verify it matches Yahoo
batters["Points"] = (batters["R"]*2 +
                    batters["H"]*1 +
                    batters["2B"]*1 +
                    batters["3B"]*2 +
                    batters["HR"]*3 +
                    batters["RBI"]*2 +
                    batters["SB"]*1 +
                    batters["CS"]*-1 +
                    batters["BB"]*1 +
                    batters["IBB"]*1 +
                    batters["HBP"]*1 +
                    batters["SO"]*-0.5)
# Create CSV
batters.to_csv("batterpoints.csv")

# Calculate the stats that most highly correlate with Fantasy Points, drop NaNs
corrs = batters.corr().Points
corrs.dropna(inplace=True)
corrs = corrs.sort_values(ascending=False)
top_pos = corrs.head(30)
bot_pos = corrs.tail(10)
variables = list(top_pos.head(30).index) + list(bot_pos.head(10).index)

# Some of these are mostly NaNs, so we'll drop those columns
to_remove = ["Points",
             "SB-X (pi)",
            "wSB/C (pi)",
            "KN% (pi)",
            "KN%",
            "XX% (pi)",
            "KN% (sc)",
            "CS% (pi)",
            "vSB (pi)",
            "SB% (pi)"]

reg_variables = []
for par in variables:
    if par not in to_remove:
        reg_variables.append(par)

# Time for regression
y = batters["Points"]
x = batters[reg_variables]
model = sm.OLS(y,x)
residuals = model.fit()

print(residuals.summary())

# Looks good, but let's drop the non-significant (p>0.05, at least to start) variables

