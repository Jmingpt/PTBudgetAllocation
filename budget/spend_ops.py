import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def secant_method(f, x1, x2, tolerance=1e-6):
    while abs(f(x1)) > tolerance:
        fx1 = f(x1)
        fx2 = f(x2)
        xtemp = x1
        x1 = x1 - (x1 - x2) * fx1 / (fx1 - fx2)
        x2 = xtemp
    return x1


def derivative(f, a, method='central', h=0.0001):
    if method == 'central':
        return (f(a + h) - f(a - h))/(2*h)
    elif method == 'forward':
        return (f(a + h) - f(a))/h
    elif method == 'backward':
        return (f(a) - f(a - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


def modelPlot(x, x2, y1, y2, min_max_point):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x,
                             y=y1,
                             mode='markers',
                             name='Cost vs. ROAS'))

    fig.add_trace(go.Scatter(x=x2,
                             y=y2,
                             mode='lines',
                             name='Trend Line'))

    fig.add_vline(x=min_max_point,
                  line_color='red')

    fig.update_layout(title='Spending Optimisation',
                      width=800,
                      height=700,
                      xaxis_title='Spending',
                      yaxis_title='ROAS')

    return fig


def spend_optimisation(df):
    if df is not None:
        df_plot = df.groupby('YearWeek') \
                    .agg({'Cost': np.sum, 'Revenue': np.sum}) \
                    .reset_index()
        df_plot = df_plot[df_plot['Cost'] > 0]
        df_plot = df_plot.sort_values('Cost', ascending=True)
        df_plot['ROAS'] = df_plot['Revenue']/df_plot['Cost']
        x = df_plot['Cost'].values
        x_plot = np.linspace(min(x), max(x), 1000)
        y = df_plot['ROAS'].values

        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(x.reshape(-1, 1))
        x_ploy = poly.fit_transform(x_plot.reshape(-1, 1))
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(poly_features, y)
        y_predicted = poly_reg_model.predict(x_ploy)

        if poly_reg_model.coef_[1] < 0:
            min_max_value = max(y_predicted)
            if min_max_value > y_predicted[0] and min_max_value > y_predicted[-1]:
                positioning_df = pd.DataFrame(data=x_plot, columns=['x'])
                positioning_df['y'] = y_predicted
                min_max_point = positioning_df[positioning_df['y'] == min_max_value]['x'].values[0]
            else:
                min_max_point = None
        else:
            min_max_value = min(y_predicted)
            if min_max_value < y_predicted[0] and min_max_value < y_predicted[-1]:
                positioning_df = pd.DataFrame(data=x_plot, columns=['x'])
                positioning_df['y'] = y_predicted
                min_max_point = positioning_df[positioning_df['y'] == min_max_value]['x'].values[0]
            else:
                min_max_point = None

        fig = modelPlot(x, x_plot, y, y_predicted, min_max_point)
        st.plotly_chart(fig, use_container_width=True)
