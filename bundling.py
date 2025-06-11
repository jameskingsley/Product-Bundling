import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy.sparse import csr_matrix

st.title("Interactive Product Bundling Explorer")

#Upload dataset
uploaded_file = st.file_uploader("Upload a retail/sales dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    #Column selection
    st.subheader("Select Columns")
    col1 = st.selectbox("Transaction/Invoice column", df.columns)
    col2 = st.selectbox("Product/Item column", df.columns)

    #Frequency Filter
    st.subheader("Filter Products by Frequency")
    min_occurrence = st.slider("Minimum times a product must appear", 1, 100, 5)
    product_counts = df[col2].value_counts()
    top_products = product_counts[product_counts >= min_occurrence].index.tolist()
    df_filtered = df[df[col2].isin(top_products)]

    # Diagnostics
    st.write(f"Products after filtering: **{len(top_products)}**")
    st.write(f"Transactions: **{df[col1].nunique()}**")

    # Step 4: Generate Bundles
    if st.button("Generate Bundles"):
        st.info("Generating bundles... Please wait.")

        # Group products by transaction
        transactions = df_filtered.groupby(col1)[col2].apply(lambda x: [str(item) for item in x]).tolist()

        # One-hot encode transactions using sparse matrix
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        sparse_matrix = csr_matrix(te_ary)
        df_encoded_sparse = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=te.columns_)

        # Support slider
        st.subheader("Set Minimum Support")
        min_support = st.slider("Minimum Support", 0.001, 0.1, 0.01, 0.001)

        # Frequent itemsets
        frequent_itemsets = apriori(df_encoded_sparse, min_support=min_support, use_colnames=True)

        if frequent_itemsets.empty:
            st.error("🚫 No frequent itemsets found. Try lowering the support or product frequency threshold.")
        else:
            # Association rules
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

            if rules.empty:
                st.warning("⚠️ No association rules found. Try adjusting thresholds.")
            else:
                # Filter rules
                st.subheader("Filter Bundles by Lift")
                min_lift = st.slider("Minimum Lift", 0.0, 30.0, 5.0, 0.1)

                filtered = rules[rules['lift'] >= min_lift]
                st.write(f"🔎 Found **{len(filtered)}** bundles with support ≥ {min_support:.3f} and lift ≥ {min_lift:.2f}")

                if filtered.empty:
                    st.warning("No bundles meet the criteria. Try lowering thresholds.")
                else:
                    for idx, row in filtered.iterrows():
                        bundle = row['antecedents'] | row['consequents']
                        st.markdown(f"### 🧺 Bundle: {', '.join(bundle)}")
                        st.markdown(f"- **Support:** {row['support']:.4f}")
                        st.markdown(f"- **Confidence:** {row['confidence']:.4f}")
                        st.markdown(f"- **Lift:** {row['lift']:.4f}")
                        st.markdown("---")

                st.info("💡 Use these insights for combo deals, promotions, or cross-selling strategies.")

else:
    st.info("📂 Upload a dataset to begin.")
