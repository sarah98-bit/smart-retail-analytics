# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import numpy as np

st.set_page_config(page_title="Smart Retail Analytics", layout="wide")
st.title("🛍️ Smart Retail Analytics: Segment and Recommend")

# === Upload Data ===
st.sidebar.header("Upload CSV Files")
customer_file = st.sidebar.file_uploader("Upload Customer Dataset", type=["csv"])
transaction_file = st.sidebar.file_uploader("Upload Transactions Dataset", type=["csv"])

# === Customer Segmentation Section ===
if customer_file:
    try:
        df = pd.read_csv(customer_file)
        st.subheader("📋 Customer Data Preview")
        st.dataframe(df.head())
        
        # Display basic info about the dataset
        st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Data preprocessing
        df_processed = df.copy()
        
        # Encode categorical variables
        if 'Gender' in df_processed.columns:
            le_gender = LabelEncoder()
            df_processed['Gender'] = le_gender.fit_transform(df_processed['Gender'])
            st.write("✅ Gender column encoded")

        # Define features for clustering
        available_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']
        features = [f for f in available_features if f in df_processed.columns]
        
        if len(features) < 2:
            st.error("❌ Need at least 2 numeric features for clustering. Please check your data columns.")
        else:
            st.write(f"**Features used for clustering:** {', '.join(features)}")
            
            X = df_processed[features]
            
            # Handle missing values
            if X.isnull().sum().sum() > 0:
                st.warning("⚠️ Missing values detected. Filling with median values.")
                X = X.fillna(X.median())
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Elbow Method for optimal clusters
            st.subheader("📈 Elbow Method for Optimal Clusters")
            wcss = []
            K_range = range(1, 11)
            
            with st.spinner("Computing WCSS for different cluster numbers..."):
                for i in K_range:
                    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
                    kmeans.fit(X_scaled)
                    wcss.append(kmeans.inertia_)

            # Plot elbow curve
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(K_range, wcss, marker='o', linewidth=2, markersize=8)
            ax.set_title("Elbow Method For Optimal k", fontsize=16)
            ax.set_xlabel("Number of Clusters (k)", fontsize=12)
            ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=12)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Cluster selection
            col1, col2 = st.columns([1, 2])
            with col1:
                k = st.slider("Select number of clusters", 2, 10, 5)
                
            with col2:
                st.info(f"💡 **Tip:** Look for the 'elbow' in the curve above. The optimal number of clusters is typically where the WCSS starts to decrease more slowly.")

            # Apply K-Means clustering
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            df_processed['Cluster'] = model.fit_predict(X_scaled)

            # Visualization
            st.subheader("🎯 Customer Segmentation Results")
            
            if 'Annual Income (k$)' in df_processed.columns and 'Spending Score (1-100)' in df_processed.columns:
                fig2, ax2 = plt.subplots(figsize=(12, 8))
                scatter = sns.scatterplot(
                    data=df_processed, 
                    x='Annual Income (k$)', 
                    y='Spending Score (1-100)', 
                    hue='Cluster', 
                    ax=ax2, 
                    palette='tab10',
                    s=100,
                    alpha=0.7
                )
                ax2.set_title("Customer Segments", fontsize=16)
                ax2.set_xlabel("Annual Income (k$)", fontsize=12)
                ax2.set_ylabel("Spending Score (1-100)", fontsize=12)
                
                # Add cluster centers
                centers_scaled = scaler.inverse_transform(model.cluster_centers_)
                if len(features) >= 2:
                    income_idx = features.index('Annual Income (k$)') if 'Annual Income (k$)' in features else 0
                    spending_idx = features.index('Spending Score (1-100)') if 'Spending Score (1-100)' in features else 1
                    ax2.scatter(centers_scaled[:, income_idx], centers_scaled[:, spending_idx], 
                              c='red', marker='x', s=200, linewidths=3, label='Centroids')
                ax2.legend()
                st.pyplot(fig2)
            else:
                st.info("📊 Income and Spending Score columns not found. Showing cluster distribution instead.")
                
                # Show cluster distribution
                cluster_counts = df_processed['Cluster'].value_counts().sort_index()
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                cluster_counts.plot(kind='bar', ax=ax3, color='skyblue')
                ax3.set_title("Customer Distribution Across Clusters")
                ax3.set_xlabel("Cluster")
                ax3.set_ylabel("Number of Customers")
                plt.xticks(rotation=0)
                st.pyplot(fig3)

            # Cluster Analysis
            st.subheader("📊 Cluster Analysis")
            
            # Calculate cluster statistics
            cluster_stats = df_processed.groupby('Cluster')[features].agg(['mean', 'count']).round(2)
            
            for cluster_id in range(k):
                with st.expander(f"🎯 Cluster {cluster_id} Details"):
                    cluster_data = df_processed[df_processed['Cluster'] == cluster_id]
                    st.write(f"**Size:** {len(cluster_data)} customers ({len(cluster_data)/len(df_processed)*100:.1f}%)")
                    
                    # Show statistics for each feature
                    for feature in features:
                        mean_val = cluster_data[feature].mean()
                        st.write(f"**{feature}:** {mean_val:.2f} (avg)")

            # Display segmented data
            st.subheader("📋 Segmented Customer Data")
            st.dataframe(df_processed)
            
            # Download option
            csv = df_processed.to_csv(index=False)
            st.download_button(
                label="📥 Download Segmented Data as CSV",
                data=csv,
                file_name='segmented_customers.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"❌ Error processing customer data: {str(e)}")
        st.write("**Please check:**")
        st.write("- File format is correct (CSV)")
        st.write("- Required columns exist (Age, Annual Income, Spending Score)")
        st.write("- Data contains numeric values")

# === Market Basket Analysis Section ===
if transaction_file:
    st.subheader("🛒 Market Basket Analysis")
    
    try:
        df_raw = pd.read_csv(transaction_file)
        st.write("**Raw Transaction Data Preview:**")
        st.dataframe(df_raw.head())
        st.write(f"**Dataset Shape:** {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

        # Transaction preprocessing
        transactions = []
        
        if {"TransactionID", "ProductID"}.issubset(df_raw.columns):
            st.info("📋 Detected structured format (TransactionID, ProductID)")
            # Convert structured format to basket format
            basket_df = df_raw.groupby("TransactionID")["ProductID"].apply(lambda x: list(x.astype(str))).reset_index()
            transactions = basket_df["ProductID"].tolist()
            
        elif df_raw.shape[1] == 1:
            st.info("📋 Detected single column format (comma-separated items)")
            # Single column with comma-separated items
            transactions = df_raw.iloc[:, 0].apply(
                lambda x: [item.strip() for item in str(x).split(",") if item.strip()] if pd.notna(x) else []
            ).tolist()
            
        else:
            st.info("📋 Detected multi-column basket format")
            # Multiple columns, each representing an item
            transactions = df_raw.apply(
                lambda row: [str(item).strip() for item in row if pd.notna(item) and str(item).strip() != ''], 
                axis=1
            ).tolist()

        # Remove empty transactions
        transactions = [t for t in transactions if len(t) > 0]
        
        if not transactions:
            st.error("❌ No valid transactions found. Please check your data format.")
        else:
            # Show transaction statistics
            st.write(f"📊 **Total Valid Transactions:** {len(transactions)}")
            st.write(f"📊 **Average Items per Transaction:** {np.mean([len(t) for t in transactions]):.2f}")
            st.write(f"📊 **Max Items in a Transaction:** {max(len(t) for t in transactions)}")
            
            # Show sample transactions
            st.write("**Sample Transactions:**")
            for i, transaction in enumerate(transactions[:5]):
                st.write(f"Transaction {i+1}: {transaction}")

            # Transaction Encoder
            te = TransactionEncoder()
            te_array = te.fit(transactions).transform(transactions)
            df_trans = pd.DataFrame(te_array, columns=te.columns_)

            # Show item frequency analysis
            st.subheader("📈 Item Frequency Analysis")
            item_frequency = df_trans.sum().sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 15 Most Frequent Items:**")
                top_items = item_frequency.head(15)
                st.dataframe(top_items.to_frame('Frequency'))
                
            with col2:
                # Plot item frequency
                fig_freq, ax_freq = plt.subplots(figsize=(10, 6))
                top_items.plot(kind='bar', ax=ax_freq, color='lightcoral')
                ax_freq.set_title("Top 15 Most Frequent Items")
                ax_freq.set_xlabel("Items")
                ax_freq.set_ylabel("Frequency")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig_freq)

            # Apriori Algorithm Settings
            st.subheader("⚙️ Apriori Algorithm Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                min_support = st.slider(
                    "Minimum Support", 
                    0.001, 1.0, 0.02, step=0.001,
                    help="Minimum support threshold. Lower values find more patterns but may include noise."
                )
                
            with col2:
                st.write(f"**Items must appear in at least:**")
                st.write(f"• {min_support*100:.1f}% of transactions")
                st.write(f"• {int(min_support * len(transactions))} transactions")

            # Run Apriori Algorithm
            with st.spinner("🔍 Finding frequent itemsets..."):
                frequent_itemsets = apriori(df_trans, min_support=min_support, use_colnames=True)

            if frequent_itemsets.empty:
                st.warning(f"⚠️ No frequent itemsets found with minimum support of {min_support:.3f}")
                
                st.info("💡 **Suggestions:**")
                st.write("- Try lowering the minimum support threshold")
                st.write("- Check if your data has enough co-occurring items")
                st.write("- Verify your transaction format is correct")
                
                # Auto-try with lower support
                lower_supports = [0.01, 0.005, 0.001]
                for lower_support in lower_supports:
                    if min_support > lower_support:
                        st.write(f"🔄 Automatically trying with lower support ({lower_support})...")
                        frequent_itemsets = apriori(df_trans, min_support=lower_support, use_colnames=True)
                        if not frequent_itemsets.empty:
                            st.success(f"✅ Found patterns with support {lower_support}!")
                            break

            if not frequent_itemsets.empty:
                st.success(f"✅ Found {len(frequent_itemsets)} frequent itemsets!")
                
                # Display frequent itemsets
                st.subheader("📋 Frequent Itemsets")
                
                # Add itemset size column
                frequent_itemsets['itemset_size'] = frequent_itemsets['itemsets'].apply(len)
                
                # Show itemsets by size
                itemset_sizes = frequent_itemsets['itemset_size'].value_counts().sort_index()
                st.write("**Itemsets by Size:**")
                for size, count in itemset_sizes.items():
                    st.write(f"• Size {size}: {count} itemsets")
                
                # Display top frequent itemsets
                display_itemsets = frequent_itemsets.copy()
                display_itemsets['itemsets'] = display_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
                display_itemsets = display_itemsets.sort_values('support', ascending=False)
                
                st.dataframe(display_itemsets[['itemsets', 'support', 'itemset_size']].head(20))

                # Association Rules Generation
                st.subheader("🔗 Association Rules Generation")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_threshold = st.slider(
                        "Minimum Lift Threshold", 
                        0.1, 5.0, 1.0, step=0.1,
                        help="Minimum lift value. Values > 1 indicate positive correlation."
                    )
                    
                with col2:
                    metric_choice = st.selectbox(
                        "Metric for Rules",
                        ["lift", "confidence", "support"],
                        help="Metric to use for generating association rules"
                    )

                # Generate association rules
                with st.spinner("⚡ Generating association rules..."):
                    rules = association_rules(frequent_itemsets, metric=metric_choice, min_threshold=min_threshold)
                
                if rules.empty:
                    st.warning(f"❌ No association rules found with {metric_choice} >= {min_threshold}")
                    st.info("💡 Try lowering the threshold or changing the metric")
                else:
                    # Sort rules by lift (or chosen metric)
                    rules = rules.sort_values(by='lift', ascending=False)
                    
                    st.success(f"✅ Found {len(rules)} association rules!")
                    
                    # Rules summary
                    st.subheader("📊 Rules Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Rules", len(rules))
                    with col2:
                        st.metric("Avg Confidence", f"{rules['confidence'].mean():.3f}")
                    with col3:
                        st.metric("Avg Lift", f"{rules['lift'].mean():.3f}")

                    # Display rules with better formatting
                    st.subheader("📋 Association Rules")
                    
                    display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
                    display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    display_rules = display_rules.round(3)
                    
                    # Add filters
                    st.write("**Filter Rules:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, step=0.1)
                    with col2:
                        min_lift_filter = st.slider("Minimum Lift", 0.1, 5.0, 1.0, step=0.1)
                    
                    # Apply filters
                    filtered_rules = display_rules[
                        (display_rules['confidence'] >= min_confidence) & 
                        (display_rules['lift'] >= min_lift_filter)
                    ]
                    
                    st.write(f"**Showing {len(filtered_rules)} rules (filtered from {len(display_rules)})**")
                    st.dataframe(filtered_rules)
                    
                    # Top Rules Section
                    if len(filtered_rules) > 0:
                        st.subheader("🔥 Top 10 Rules by Lift")
                        top_rules = filtered_rules.head(10)
                        
                        for i, (idx, rule) in enumerate(top_rules.iterrows(), 1):
                            with st.expander(f"Rule {i}: {rule['antecedents']} → {rule['consequents']} (Lift: {rule['lift']})"):
                                st.write(f"**If a customer buys:** {rule['antecedents']}")
                                st.write(f"**Then they will likely buy:** {rule['consequents']}")
                                st.write(f"**Support:** {rule['support']:.3f} ({rule['support']*len(transactions):.0f} transactions)")
                                st.write(f"**Confidence:** {rule['confidence']:.3f} ({rule['confidence']*100:.1f}% chance)")
                                st.write(f"**Lift:** {rule['lift']:.3f} ({'Positive' if rule['lift'] > 1 else 'Negative'} correlation)")
                    
                    # Download options
                    st.subheader("📥 Download Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        rules_csv = filtered_rules.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Rules as CSV",
                            data=rules_csv,
                            file_name='association_rules.csv',
                            mime='text/csv'
                        )
                    
                    with col2:
                        itemsets_csv = display_itemsets.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Frequent Itemsets as CSV",
                            data=itemsets_csv,
                            file_name='frequent_itemsets.csv',
                            mime='text/csv'
                        )

            else:
                st.error("❌ Unable to find any frequent patterns in your data")
                st.write("**Possible issues:**")
                st.write("- Dataset too small (need more transactions)")
                st.write("- Items don't frequently appear together")
                st.write("- Minimum support threshold too high")
                st.write("- Data preprocessing issues")
                
                # Show data diagnostics
                st.subheader("🔍 Data Diagnostics")
                st.write(f"• Total unique items: {len(te.columns_)}")
                st.write(f"• Most frequent item appears in: {item_frequency.max()}/{len(transactions)} transactions ({item_frequency.max()/len(transactions)*100:.1f}%)")
                st.write(f"• Least frequent item appears in: {item_frequency.min()}/{len(transactions)} transactions ({item_frequency.min()/len(transactions)*100:.1f}%)")

    except Exception as e:
        st.error(f"❌ Error processing transaction data: {str(e)}")
        st.write("**Please check:**")
        st.write("- File format is CSV")
        st.write("- For structured format: columns named 'TransactionID' and 'ProductID'")
        st.write("- For basket format: comma-separated items or items in separate columns")
        st.write("- Data contains valid product/item names")

