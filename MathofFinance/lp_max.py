import pulp as p
from pulp import *
import pandas as pd

df = pd.DataFrame()
my_list = [0,2]
for scenario in my_list:

    Lp_prob = p.LpProblem("Problem", p.LpMaximize)

    x1 = p.LpVariable("x1", lowBound=0, upBound=1)
    x2 = p.LpVariable("x2", lowBound=0, upBound=1)
    x3 = p.LpVariable("x3", lowBound=0, upBound=1)
    x4 = p.LpVariable("x4", lowBound=0, upBound=1)
    x5 = p.LpVariable("x5", lowBound=0, upBound=1)
    x6 = p.LpVariable("x6", lowBound=0, upBound=1)
    x7 = p.LpVariable("x7", lowBound=0, upBound=1)
    x8 = p.LpVariable("x8", lowBound=0, upBound=1)
    x9 = p.LpVariable("x9", lowBound=0, upBound=1)
    x10 = p.LpVariable("x10", lowBound=0, upBound=1)

    Lp_prob += 4*x1 + 4*x2 + 3*x3 + 4.3*x4 + x5 + 1.5*x6 + 2.5*x7 + 0.3*x8 + x9 + 2*x10

    Lp_prob += 2*x1 + 3*x2 + 1.5*x3 + 2.2*x4 + 0.5*x5 + 1.5*x6 + 2.5*x7 + 0.1*x8 + 0.6*x9 + x10 <= 5
    Lp_prob += x1 + x2 + x3 + x4 <= 1
    # two case scenarios 0 or 2
    Lp_prob += x2 + x4 + x6 + x7 == scenario
    ########################################
    Lp_prob += x5 + x6 + x7 <= 1
    Lp_prob += x8 + x9 + x10 <= 1

    Lp_prob.solve()

    #print("Status:", LpStatus[Lp_prob.status])

    for v in Lp_prob.variables():
        #printing solutions since they have to be non-zero vars
        if v.varValue>0:
            npv = str(v.varValue)
            #print(solution)
            df.loc[str(v.name), scenario] = npv

    df.loc["npv",scenario] = p.value(Lp_prob.objective)

#compare the Dataframe columns 0 & 2 to get the MAX npv
if df.loc["npv",0] > df.loc["npv",2]:
    print(df[0].dropna())
elif df.loc["npv",2] > df.loc["npv",0]:
    print(df[2].dropna())