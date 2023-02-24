import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO


def dfmapping(df_ga, df_fb):
    cols_seq = ['Date', 'Channel', 'Cost', 'Revenue']
    df_fb['Channel'] = 'Facebook'
    df_fb = df_fb[cols_seq]
    df_ga = df_ga[cols_seq]
    df = pd.concat([df_ga, df_fb], ignore_index=True)

    date_range = '{} - {}'.format(df['Date'].min().strftime('%Y/%m/%d'), df['Date'].max().strftime('%Y/%m/%d'))

    df['isoyear'] = df['Date'].apply(lambda x: x.isocalendar()[0])
    df['isoweek'] = df['Date'].apply(lambda x: x.isocalendar()[1])
    df['YearWeek'] = df.apply(lambda x: f'{x.isoyear}{x.isoweek:02d}', axis=1)
    df = df.drop(['Date', 'isoyear', 'isoweek'], axis=1)

    pivot_df = pd.pivot_table(df, values='Cost', index=['YearWeek'], columns=['Channel'], aggfunc=np.sum)
    pivot_df = pivot_df.reset_index().sort_values('YearWeek', ascending=True).reset_index(drop=True)
    pivot_df = pivot_df.fillna(0)

    revenue_df = df.groupby('YearWeek') \
                   .agg({'Revenue': np.sum}) \
                   .reset_index() \
                   .sort_values('YearWeek', ascending=True) \
                   .reset_index(drop=True)

    mmm = pd.merge(pivot_df, revenue_df, on='YearWeek', how='left')
    mmm_df = mmm.groupby(['YearWeek']).agg(np.sum)

    total_rows = mmm_df.shape[0]
    dropped_cols = []
    for col in mmm_df.columns:
        na_rows = mmm_df[mmm_df[col] == 0].shape[0]
        missing_rate = na_rows/total_rows
        missing_criteria = 0.45
        if missing_rate > missing_criteria:
            dropped_cols.append(col)
            mmm_df.drop(col, axis=1, inplace=True)

    st.info(f'**Dropped columns**: {dropped_cols}. **Reason**: More than {missing_criteria * 100}% missing value.')
    return df, mmm_df, date_range


def upload_files(uploaded_files):
    ga_required_cols = ['Date', 'Channel', 'Cost', 'Revenue']
    fb_required_cols = ['Date', 'Cost', 'Revenue']
    if len(uploaded_files) == 2:
        uploadfile_error = 0
        for f in uploaded_files:
            if 'ga' in str(f.name).lower() or 'google' in str(f.name).lower():
                bytes_data = f.read()
                s = str(bytes_data, 'utf-8')
                data = StringIO(s)
                df_ga = pd.read_csv(data)

                try:
                    df_ga = df_ga[ga_required_cols]
                    df_ga['Date'] = pd.to_datetime(df_ga['Date'])
                except:
                    uploadfile_error += 1
                    st.error(f'Please ensure the columns in **Google Ads** file are followed the sequence: **{ga_required_cols}**')

            elif 'fb' in str(f.name).lower() or 'facebook' in str(f.name).lower():
                bytes_data = f.read()
                s = str(bytes_data, 'utf-8')
                data = StringIO(s)
                df_fb = pd.read_csv(data)

                try:
                    df_fb = df_fb[fb_required_cols]
                    df_fb['Date'] = pd.to_datetime(df_fb['Date'])
                except:
                    uploadfile_error += 1
                    st.error(f'Please ensure the columns in **Google Ads** file are followed the sequence: **{fb_required_cols}**')

        if uploadfile_error < 1:
            df, mmm_df, date_range = dfmapping(df_ga, df_fb)

            return df, mmm_df, date_range

    elif len(uploaded_files) < 2 or len(uploaded_files) > 2:
        for f in uploaded_files:
            if 'ga' in str(f.name).lower() or 'google' in str(f.name).lower():
                st.warning('Missing **Facebook** Ads file. Please upload the remaining file.', icon='⚠️')
            elif 'fb' in str(f.name).lower() or 'facebook' in str(f.name).lower():
                st.warning('Missing **Google Ads** file. Please upload the remaining file.', icon='⚠️')

        return None, None, None
