import streamlit as st
import pandas as pd
import numpy as np
from pulp import *


def budget_allocate(df, date_range):
    if df is not None:
        df = df[df['Cost'] > 0]
        df = df.groupby(['Channel']) \
               .agg({'Cost': np.sum, 'Revenue': np.sum}) \
               .reset_index().sort_values(['Channel']) \
               .reset_index(drop=True)
        df['ROAS'] = df['Revenue'] / df['Cost']

        prob = LpProblem('Maximise Revenue Problem', LpMaximize)

        coef = np.array(df['ROAS'])
        df = df.reset_index()
        Mapping = df[['index', 'Channel']]
        no_variable = df.shape[0]

        base = df[['index', 'Cost']]
        base = base.fillna(0)

        base['+20'] = base['Cost'] + (base['Cost'] * 0.2)
        base['-20'] = base['Cost'] - (base['Cost'] * 0.2)

        for x in range(1, no_variable + 1):
            globals()["x" + str(x)] = LpVariable("x" + str(x),
                                                 base[base['index'] == x - 1]['-20'].item(),
                                                 base[base['index'] == x - 1]['+20'].item())

        s = ""
        for x in range(1, no_variable + 1):
            s += f"x{x}+"

        s = s.rstrip('+')
        prob += eval(s) == base['Cost'].sum()

        r = ""
        for x in range(1, no_variable + 1):
            r += f"coef.item({x - 1})*x{x}+"

        r = r.rstrip('+')
        prob += eval(r)

        status = prob.solve()

        dc = pd.DataFrame()
        for v in prob.variables():
            dc = dc.append({'Name': v.name, 'value': v.varValue}, ignore_index=True)

        dc['index'] = dc['Name'].str.split('x', expand=True)[1]
        dc['index'] = dc['index'].astype(int)
        dc['index'] = dc['index'] - 1

        final_pred = df.merge(dc)
        final_pred = Mapping.merge(final_pred)
        final_pred = final_pred[['Channel', 'Cost', 'Revenue', 'ROAS', 'value']]
        final_pred.columns = ['Channel', 'Spend Before', 'Revenue', 'ROAS', 'Optimal Spend']

        budget_change = []
        for spend1, spend2 in final_pred[['Spend Before', 'Optimal Spend']].values:
            if spend1 - spend2 > 0:
                budget_change.append('Reduce budget')
            elif spend1 - spend2 < 0:
                budget_change.append('Increase budget')
            else:
                budget_change.append('Remain')
        final_pred['Suggestion'] = budget_change

        status_cols = st.columns((1, 1, 1))
        with status_cols[0]:
            st.subheader(f'Date Range: {date_range}')
        with status_cols[1]:
            st.subheader(f'LP Status: {LpStatus[status]}')

        hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
        """
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.table(final_pred.style.format({'Spend Before': '{:.2f}',
                                          'Revenue': '{:.2f}',
                                          'ROAS': '{:.2f}',
                                          'Optimal Spend': '{:.2f}'}))
