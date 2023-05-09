import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns 
import streamlit as st
from PIL import Image

## Read xlsx sheets
dfs = {} 

for mes in ['noviembre', 'diciembre', 'enero', 'febrero']:
    dfs[mes] = pd.read_excel('Test Cohorts.xlsx', sheet_name=mes)

def preprocess_data(df):

    #Extract the type of plan from Invoice column and save it into a new column
    df["Plan"] = df["Invoice"].str.extract(r'([P]\d)')

    #Extract the duration of plan from Invoice column and save it into a new column
    df["Duration"] = df["Invoice"].fillna("").str.split("-").str[2]
    df["Duration"] = pd.to_numeric(df["Duration"], errors='coerce')
    df["Duration"].fillna(0, inplace=True)
    df["Duration"] = df["Duration"].astype(int)

    #Remove columns with nulls
    df.drop(['Local VAT', 'Create Date', 'Local Original Amount','EXTRAE', 'extrae'], axis=1, inplace=True, errors='ignore')
    
    #Keep only transactions with status paid
    df = df.loc[df['Status'] == 'PAID']
    
    return df

for mes in ['noviembre', 'diciembre', 'enero', 'febrero']:
    dfs[mes] = preprocess_data(dfs[mes])

months = ['noviembre', 'diciembre', 'enero', 'febrero']
df_monthly_sales = {}
for month in months:
    df_monthly_sales[month] = dfs[month]['Amount Usd'].sum()

total_sales = sum(df_monthly_sales.values())

def plot_transactions_per_month():
    # Get the number of plans split by month
    number_plan1 = [ dfs[month]['Plan'].value_counts()[0] for month in months ]
    number_plan2 = [ dfs[month]['Plan'].value_counts()[1] for month in months ]
    colors_plan1 = ['green']
    colors_plan2 = [(0.5, 0, 0)]

    X_axis = np.arange(len(months))

    fig, ax = plt.subplots()
    ax.bar(X_axis - 0.2, number_plan1, 0.4, label = 'Plan 1', color=colors_plan1)
    ax.bar(X_axis + 0.2, number_plan2, 0.4, label = 'Plan 2', color=colors_plan2)

    ax.set_xticks(X_axis)
    ax.set_xticklabels(months)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of transactions")
    ax.set_title("Number of transactions per month")
    ax.legend()
    
    # Add labels to bars
    for i, v in enumerate(number_plan1):
        plt.text(i - 0.2, v + 0.05, str(v), color='black', fontweight='bold', ha='center')
    
    for i, v in enumerate(number_plan2):
        plt.text(i + 0.2, v + 0.05, str(v), color='black', fontweight='bold', ha='center')

    return fig
    


# Let's put all the data in the same data frame and keep only month, customer ID and plan

# Create an empty dataframe
df_simplified = pd.DataFrame()

# Add the relevant data, month by month
for i, mes in enumerate(months):
    
    month_df  = dfs[mes]
    month_df['order_month'] = i # Use numerical indexes for the months
                                # November is 0, december is 1, january is 2, february is 3
    
    df_simplified = pd.concat([df_simplified, month_df])

df_simplified = df_simplified[ ['order_month', 'Invoice', 'Duration', 'User account' ] ].drop_duplicates()

df_extended = df_simplified

for index, transaction in df_simplified.iterrows():
    if transaction['Duration']>1:
        # Add placeholder transaction
        to_concat = []
        for i in range(1,transaction['Duration']):
            if transaction['order_month']+i < len(months):
                to_concat.append(pd.DataFrame(                     
                    { 'order_month' : [transaction['order_month']+i], \
                      'Invoice' : [transaction['Invoice']], \
                      'Duration' : [1], \
                      'User account': [transaction['User account']] }))
        if to_concat:
            df_extended = pd.concat([df_extended] + to_concat, ignore_index=True)

df_extended.drop(['Duration', 'Invoice'], axis=1, inplace=True, errors='ignore')
from operator import attrgetter

# Assign a cohort to each transaction.
# A transaction cohort is the month of the oldest transaction of the user
df_extended['cohort'] = df_extended.groupby('User account')['order_month'].transform('min')


df_cohort = df_extended.groupby(['cohort', 'order_month']) \
              .agg(n_customers=('User account', 'nunique')) \
              .reset_index(drop=False)

df_cohort['period_number'] = \
    (df_cohort.order_month - df_cohort.cohort) # number of periods between the cohort month and the month of the purchase.

cohort_pivot = df_cohort.pivot_table(index = 'cohort',
                                     columns = 'period_number',
                                     values = 'n_customers')

for i, month in enumerate(months):
    if not i in cohort_pivot.index:
        cohort_pivot.loc[i] = np.nan
        cohort_pivot[0][i] = 0
cohort_pivot.sort_index(inplace=True)

cohort_size = cohort_pivot.iloc[:,0]
retention_matrix = cohort_pivot.divide(cohort_size, axis = 0)

amount_per_country = [ dfs[month].groupby('Country')['Amount Usd'].sum().tolist() for month in months ]

X_axis = np.arange(len(months))

country_name = ['BR','PA','MX','PE','CO','CL']

def amount_by_country():
    fig3, ax = plt.subplots()
    for country_index in range(0,6):
        amounts = [ row[country_index] for row in amount_per_country ]
        bars = ax.bar(X_axis+0.1*country_index-0.25, amounts, 0.1, label = country_name[country_index])
  
        
    ax.set_xticks(X_axis)
    ax.set_xticklabels(months)
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount (USD)")
    ax.set_title("Amount of transactions (USD) per month, split by country")
    ax.legend()
    
    return fig3



# crear figura y ejes
def retention_users():
    with sns.axes_style("white"):
        fig2, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True, gridspec_kw={'width_ratios': [1, 11]})
    
    # retention matrix
        sns.heatmap(retention_matrix, 
                mask=retention_matrix.isnull(), 
                annot=True, 
                fmt='.0%', 
                cmap='RdYlGn', 
                ax=ax[1])
    ax[1].set_title('Monthly Cohorts: User Retention', fontsize=16)
    ax[1].set(xlabel='# of periods',
              ylabel='month')

    # cohort size
    cohort_size_df = pd.DataFrame(cohort_size).rename(columns={0: 'cohort_size'})
    white_cmap = mcolors.ListedColormap(['white'])
    sns.heatmap(cohort_size_df, 
                annot=True, 
                cbar=False, 
                fmt='g', 
                cmap=white_cmap, 
                ax=ax[0])

    fig2.tight_layout()
    return fig2

def retention_userint():
    with sns.axes_style("white"):
        fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True, gridspec_kw={'width_ratios': [1, 11]})
    
    # retention matrix
        sns.heatmap(cohort_pivot, 
                mask=cohort_pivot.isnull(), 
                annot=True, 
                fmt='g', 
                cmap='RdYlGn', 
                ax=ax[1])
    ax[1].set_title('Monthly Cohorts: User Retention', fontsize=16)
    ax[1].set(xlabel='# of periods',
              ylabel='month')

    # cohort size
    cohort_size_df = pd.DataFrame(cohort_size).rename(columns={0: 'cohort_size'})
    white_cmap = mcolors.ListedColormap(['white'])
    sns.heatmap(cohort_size_df, 
                annot=True, 
                cbar=False, 
                fmt='g', 
                cmap=white_cmap, 
                ax=ax[0])

    fig.tight_layout()
    return fig

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
image = Image.open('logo-olaclick-blanco 2.png')
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Análisis de Ventas')

st.sidebar.image(image, caption='¡Más de 200 000 menús creados!', use_column_width=True)

st.sidebar.subheader('Parameters')
time_hist_color = st.sidebar.selectbox('Country', (country_name)) 

st.sidebar.subheader('User Retention Parameter')
donut_theta = st.sidebar.selectbox('Select Data', ('g', 'g%'))

st.sidebar.subheader('Line chart Parameters')
plot_data = st.sidebar.multiselect('Select data', ['transactions', 'amounts'], ['amounts', 'transactions'])
plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created with ❤️ by Sebastián Godoy.
''')


# Row A
st.markdown('### Metrics')
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales USD", "62.060,86")
col2.metric("Retention Rate", "64,25%")
col3.metric("Total Users", "1508")

# Row B

c1, c2 = st.columns((5,5))
with c1:
    st.markdown('### Transactions Per Month By Plan')
    fig = plot_transactions_per_month()
    st.pyplot(fig, use_container_width=True)
with c2:
    st.markdown('### Amounts By Countries')
    fig3 = amount_by_country()
    st.pyplot(fig3, use_container_width=True)

# Row C
st.markdown('### Monthly Cohorts: User Retention')
fig4 = retention_userint()
st.pyplot(fig4, use_container_width=True)

# Row D
st.markdown('### Monthly Cohorts: User Retention %')
fig2 = retention_users()
st.pyplot(fig2, use_container_width=True)