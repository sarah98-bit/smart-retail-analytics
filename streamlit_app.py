# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(page_title="Smart Retail Analytics", layout="wide")
st.title("🛍️ Smart Retail Analytics: Segment and Recommend")

# === Upload Data ===
st.sidebar.header("Upload CSV Files")
customer_file = st.sidebar.file_uploader("Upload Customer Dataset", type=["csv"])
transaction_file = st.sidebar.file_uploader("Upload Transactions Dataset", type=["csv"])

if customer_file:
    df = pd.read_csv(customer_file)
    st.subheader("Customer Data Preview")
    st.dataframe(df.head())

    # Encode & Scale
    if 'Gender' in df.columns:
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]
    X_scaled = StandardScaler().fit_transform(X)

    # Elbow + KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    st.subheader("Elbow Method")
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title("Elbow Method")
    ax.set_xlabel("# of Clusters")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)

    k = st.slider("Select number of clusters", 2, 10, 5)
    model = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = model.fit_predict(X_scaled)

    st.subheader("Customer Segmentation")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', ax=ax2, palette='tab10')
    st.pyplot(fig2)

    st.write("Segmented Data")
    st.dataframe(df)

if transaction_file:
    st.subheader("Transaction Basket Data Preview")
    trans_df = pd.read_csv(transaction_file, header=None)
    transactions = trans_df[0].apply(lambda x: x.split(",")).tolist()

    # Apriori
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_array, columns=te.columns_)

    min_support = st.slider("Minimum Support", 0.1, 1.0, 0.3)
    frequent_itemsets = apriori(df_trans, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules = rules.sort_values(by='lift', ascending=False)

    st.subheader("Association Rules")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
