import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


def modelplot(df, date_range):
    config = {'displayModeBar': False}
    df_plot = df.sort_values('coef', ascending=False)
    x = df_plot['params'].values
    y = [round(i, 2) for i in df_plot['coef'].values]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x,
                         y=y,
                         text=y,
                         textposition='outside'))

    fig.update_layout(title=f"MMM Model [{date_range}]",
                      height=500,
                      yaxis_title='Importance Score',
                      yaxis_range=[min(y) - abs(max(y)) / 5, max(y) + abs(max(y)) / 5])

    st.plotly_chart(fig, use_container_width=True, config=config)


def predictmodel(X, y, y_pred):
    config = {'displayModeBar': False}
    x = [yw[0:4]+'_'+yw[4:6] for yw in X]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x,
                             y=y,
                             mode='lines',
                             name='Actual'))

    fig.add_trace(go.Scatter(x=x,
                             y=y_pred,
                             mode='lines',
                             name='Predicted'))

    fig.update_layout(title='Prediction',
                      height=400,
                      yaxis_title='Revenue')

    st.plotly_chart(fig, use_container_width=True, config=config)


def mmm_model(mmm_df, date_range):
    if mmm_df is not None and date_range is not None:
        X = mmm_df.drop('Revenue', axis=1)
        y = mmm_df['Revenue']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        score = r2_score(y, y_pred)

        y_pred = model.predict(X)

        coef = []
        for i, j in zip(model.feature_importances_, X.columns):
            coef.append([i, j])
        plot_df = pd.DataFrame(coef, columns=['coef', 'params'])
        st.subheader(f"R\u00b2: {score:.4f}")

        modelplot(plot_df, date_range)
        predictmodel(X.index, y, y_pred)
