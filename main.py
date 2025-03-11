import os
import csv
import json
import datetime
from io import BytesIO
import streamlit as st
import pandas as pd
import pdfplumber
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
import plotly.express as px

# ✅ Set Streamlit Page Layout
st.set_page_config(
    page_title="Transaction Processor Agent", page_icon="📄", layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(BASE_DIR, "assets", "images", "yavar-logo.png")

# ✅ Sidebar: Logo, API Key, AI Model Selection
st.image(image_path, width=100)

# ✅ App Title (Fixed at the Top)
st.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <h1 style="font-size: 32px; font-weight: bold; color: #004AAD;">Transaction Processor Agent</h1>
        <p style="font-size: 14px; font-weight: 400; color: #666;">Extract & categorize transactions with AI</p>
    </div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        padding: 10px;
        background-color: #FFDADA;
        color: #8B0000;
        font-size: 15px;
        font-weight: 600;
        border-radius: 8px;
        border-left: 6px solid #D32F2F;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);">
        ⚠️ Please mask any <b>Personally Identifiable Information (PII)</b> before uploading the PDF.
    </div>
""",
    unsafe_allow_html=True,
)
# ✅ Push Copyright Notice to the Bottom
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #F8F9FA;
            padding: 10px;
            text-align: center;
            font-size: 14px;
            font-weight: bold;
            color: #555;
            border-top: 2px solid #1976D2;
        }
    </style>
    <div class="footer">© 2025 Copyright Yavar TechWorks Pte Ltd., All rights reserved.</div>
""",
    unsafe_allow_html=True,
)

# ✅ Sidebar: Logo, API Key, AI Model Selection
st.image(image_path, width=100)
# ✅ AI Model Selection in Sidebar (Defined ONCE)
st.sidebar.header("Model Selection")
ai_model = st.sidebar.radio(
    "Choose AI Model", ["DeepSeek", "Gemini"], horizontal=True, key="ai_model_select"
)

api_key = st.sidebar.text_input("Enter API Key 🔑 ", type="password")

# ✅ Tabs for Processing Modes
tab1, tab2, tab3 = st.tabs(
    ["📄 Single Document Processing", "📂 Bulk Processing", "Analytics Dashboard"]
)


def extract_text_from_pdf(pdf_file):
    """Extract text from a valid, non-corrupt PDF file."""
    try:
        pdf_data = pdf_file.read()
        pdf_stream = BytesIO(pdf_data)  # ✅ Create a new BytesIO object each time

        text_pages = []
        with pdfplumber.open(pdf_stream) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    text_pages.append((page_num, text.strip()))
        return text_pages

    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return []


def save_qa_log(entry):
    log_file = "qa_history_log.csv"  # ✅ Define log file path

    # ✅ Check if file exists, create with headers if not
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # ✅ Write headers only if the file is newly created
        if not file_exists:
            writer.writerow(["Date", "Document", "Question", "Answer"])

        # ✅ Append new entry
        writer.writerow(entry)


def save_feedback(feedback_entry, filename="feedback_log.csv"):
    """Saves feedback data to a CSV file."""
    headers = [
        "Date",
        "Document",
        "Description",
        "Corrected Vendor",
        "Original Vendor",
        "Deposits_Credits",
        "Withdrawals_Debits",
        "Comments",
    ]

    # ✅ Open CSV in append mode
    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # ✅ Write headers only if file is empty
        if file.tell() == 0:
            writer.writerow(headers)

        # ✅ Write feedback data
        writer.writerow(feedback_entry)


# ✅ Load Vendor List
def load_vendor_list(vendor_file):
    if vendor_file is not None:
        if vendor_file.name.endswith(".csv"):
            return pd.read_csv(vendor_file)["Payee"].tolist()
        else:
            return pd.read_excel(vendor_file)["Payee"].tolist()
    return []


# ✅ Process Transactions with AI Model
def process_and_categorize(text, vendor_list, api_key, ai_model):
    """Processes transactions and categorizes them in one API call."""
    prompt = f"""
    Extract structured transactions from the bank statement and match them to vendors.
 
    **STRICT RULES:**
    - Use vendor names **ONLY** from this list:
      {json.dumps(vendor_list, indent=2)}
    - Do **NOT** assume vendors. If no match is found, return **"Unknown"**.
    - Do **NOT** modify transaction descriptions.
    - Return **pure JSON output** ONLY. No explanations, no additional text.
 
    **Statement Text:**
    {text}
 
    **Output Format (ONLY JSON)**
    ```json
    [
        {{"Date": "MM/DD/YYYY", "Description": "transaction details", "Deposits_Credits": number, "Withdrawals_Debits": number, "Vendor Name": "matched vendor"}},
        {{"Date": "MM/DD/YYYY", "Description": "another transaction", "Deposits_Credits": number, "Withdrawals_Debits": number, "Vendor Name": "matched vendor"}}
    ]
    ```
 
    **Example:**
    ```json
    [
        {{
            "Date": "11/01/2023",
            "Description": "Overdraft Fee for a Transaction Posted on 10/31 $143.00 Dell",
            "Deposits_Credits": 0,
            "Withdrawals_Debits": 35.00,
            "Vendor Name": "Overdraft Fee"
        }},
        {{
            "Date": "11/01/2023",
            "Description": "ATM Cash Deposit on 11/01 1530 Heitman St Fort Myers FL",
            "Deposits_Credits": 600.00,
            "Withdrawals_Debits": 0,
            "Vendor Name": "ATM"
        }}
    ]
    ```
    """

    if ai_model == "DeepSeek":
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )

        if response.status_code == 200:
            try:
                json_data = response.json()["choices"][0]["message"]["content"]
                return json.loads(json_data.strip("```json").strip("```"))
            except Exception as e:
                st.error(f"❌ Error processing DeepSeek response: {e}")
                return []
        else:
            st.error(f"❌ API call failed: {response.status_code} - {response.text}")
            return []

    elif ai_model == "Gemini":
        gemini = ChatGoogleGenerativeAI(
            model="gemini-2.0-pro-exp-02-05", google_api_key=api_key, temperature=0
        )

        try:
            response = gemini.invoke(prompt)
            # Debugging: Show Raw API Response
            st.text_area("🔍 Gemini Raw Response", response.content, height=200)
            if not response or not response.content.strip():
                st.error("❌ Gemini API returned an empty response.")
                return []

            transactions = json.loads(response.content.strip("```json").strip("```"))

            for tx in transactions:
                tx["Deposits_Credits"] = tx.get("Deposits_Credits", 0) or 0
                tx["Withdrawals_Debits"] = tx.get("Withdrawals_Debits", 0) or 0

            return transactions

        except Exception as e:
            st.error(f"❌ Unexpected Gemini API error: {str(e)}")
            return []


# ✅ Single Document Processing
with tab1:
    st.subheader("📄 Single Document Processing")
    pdf_file = st.file_uploader(
        "📄 Upload a Transaction Statement (PDF)", type=["pdf"], key="single_pdf"
    )
    vendor_file = st.file_uploader(
        "📂 Upload a Vendor List (CSV or Excel)",
        type=["csv", "xls", "xlsx"],
        key="single_vendor",
    )
    process_button = st.button("Process Document", key="single_process")

    if process_button and pdf_file and vendor_file:
        vendor_list = load_vendor_list(vendor_file)
        text_pages = extract_text_from_pdf(pdf_file)

        all_transactions = []
        progress_bar = st.progress(0)  # Progress indicator

        for i, (page_num, page_text) in enumerate(text_pages):
            st.info(f"📄 Processing Page {page_num} of {len(text_pages)}...")
            transactions = process_and_categorize(
                page_text, vendor_list, api_key, ai_model
            )
            all_transactions.extend(transactions)
            progress_bar.progress((i + 1) / len(text_pages))

        if all_transactions:
            st.session_state.transactions = pd.DataFrame(all_transactions)
            st.success("✅ Transactions extracted & categorized successfully!")

    # ✅ Show feedback & download ONLY in Single Document Processing tab
    if "transactions" in st.session_state and not st.session_state.transactions.empty:
        df = st.session_state.transactions
        st.markdown(
            "<h4 style='color: #1976D2; font-weight: bold;'>📋 Processed Transactions</h4>",
            unsafe_allow_html=True,
        )
        st.dataframe(df, use_container_width=True)

        st.markdown(
            "<h5 style='color: #444;'>📝 Provide Feedback</h5>", unsafe_allow_html=True
        )

        # ✅ Step 1: Select a transaction to correct
        selected_desc = st.selectbox(
            "Select a Transaction to Correct",
            df["Description"].unique(),
            key="single_desc",
            help="Choose a transaction from the list",
        )

        if not df[df["Description"] == selected_desc].empty:
            filtered_row = df[df["Description"] == selected_desc].iloc[0]

            # ✅ Vendor Selection (Dropdown + Freeflow Text)
            existing_vendors = (
                df["Vendor Name"].dropna().unique().tolist()
            )  # Get unique vendor names
            selected_vendor_dropdown = st.selectbox(
                "Correct Vendor (Choose from list)",
                existing_vendors,
                index=(
                    existing_vendors.index(filtered_row["Vendor Name"])
                    if filtered_row["Vendor Name"] in existing_vendors
                    else 0
                ),
                key="single_vendor_dropdown",
                help="Select the correct vendor name",
            )
            new_vendor_text = st.text_input(
                "Or Enter a New Vendor Name",
                "",
                key="single_vendor_text",
                help="Type a new vendor name if it's not in the list",
            )

            # ✅ Determine final vendor choice
            correct_vendor = (
                new_vendor_text.strip()
                if new_vendor_text.strip()
                else selected_vendor_dropdown
            )

            correct_deposits = st.number_input(
                "Deposits_Credits",
                value=float(filtered_row["Deposits_Credits"]),
                step=0.01,
                key="single_deposit_fb",
                help="Update deposit amount if incorrect",
            )
            correct_withdrawals = st.number_input(
                "Withdrawals_Debits",
                value=float(filtered_row["Withdrawals_Debits"]),
                step=0.01,
                key="single_withdraw_fb",
                help="Update withdrawal amount if incorrect",
            )
            comments = st.text_area(
                "Additional Comments (Optional)",
                key="single_comments_fb",
                help="Provide any additional feedback",
            )

            col1, col2 = st.columns([1, 1.2])
            with col1:
                submit_feedback = st.button(
                    "✅ Submit Feedback",
                    key="single_feedback_btn",
                    help="Submit your feedback",
                )
            with col2:
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download CSV",
                    csv_data,
                    "transactions.csv",
                    "text/csv",
                    help="Download the updated transactions",
                )

            if submit_feedback:
                # ✅ Capture feedback details
                feedback_entry = [
                    [
                        datetime.date.today(),
                        selected_desc,
                        correct_vendor,
                        filtered_row["Vendor Name"],  # Old vendor name
                        correct_deposits,
                        correct_withdrawals,
                        comments,
                    ]
                ]

                save_feedback(feedback_entry)  # ✅ Save to feedback log

                # ✅ Update transaction DataFrame
                df.loc[
                    df["Description"] == selected_desc,
                    ["Vendor Name", "Deposits_Credits", "Withdrawals_Debits"],
                ] = [
                    correct_vendor,
                    float(correct_deposits),
                    float(correct_withdrawals),
                ]

                st.session_state.transactions = df  # ✅ Update session state

                # ✅ Generate updated CSV
                csv_data = df.to_csv(index=False).encode("utf-8")

                st.success("✅ Feedback submitted successfully!")

                # ✅ Show updated download button
                # st.download_button("⬇ Download Updated Transactions CSV", csv_data, "transactions.csv", "text/csv", help="Download the updated transactions")


# ✅ Bulk Processing (Supports Folder Upload & Separate CSV for Each PDF)
with tab2:
    st.subheader("📂 Bulk Transaction Processing")

    # ✅ File Uploads (Folder Selection & PDF Upload)
    uploaded_folder = st.file_uploader(
        "📂 Select PDF Files from a Folder",
        type=["pdf"],
        accept_multiple_files=True,
        key="bulk_folder",
    )
    vendor_file = st.file_uploader(
        "📂 Upload a Vendor List (CSV or Excel)",
        type=["csv", "xls", "xlsx"],
        key="bulk_vendor",
    )

    process_bulk_button = st.button("🚀 Process Bulk Documents", key="bulk_process")

    if process_bulk_button and uploaded_folder and vendor_file:
        vendor_list = load_vendor_list(vendor_file)
        progress_bar = st.progress(0)

        total_pages = sum(
            len(extract_text_from_pdf(BytesIO(pdf_file.getvalue())))
            for pdf_file in uploaded_folder
        )
        processed_pages = 0

        session_bulk_csvs = {}  # ✅ Store separate DataFrames per PDF

        for pdf_idx, pdf_file in enumerate(uploaded_folder):
            pdf_data = pdf_file.getvalue()  # ✅ Read once into memory
            text_pages = extract_text_from_pdf(
                BytesIO(pdf_data)
            )  # ✅ Use BytesIO(pdf_data)

            if not text_pages:
                st.error(f"❌ Skipping file {pdf_file.name}: Unable to read content.")
                continue  # ✅ Skip unreadable PDFs

            file_name = pdf_file.name  # ✅ Store filename
            transactions_per_pdf = []  # ✅ Store transactions for this PDF only

            for i, (page_num, page_text) in enumerate(text_pages):
                st.info(
                    f"📄 Processing File {pdf_idx + 1}/{len(uploaded_folder)} - Page {page_num}/{len(text_pages)}"
                )
                transactions = process_and_categorize(
                    page_text, vendor_list, api_key, ai_model
                )

                # ✅ Add filename to each transaction
                for txn in transactions:
                    txn["Document"] = file_name

                transactions_per_pdf.extend(transactions)
                processed_pages += 1
                progress_bar.progress(processed_pages / total_pages)

            if transactions_per_pdf:
                session_bulk_csvs[file_name] = pd.DataFrame(
                    transactions_per_pdf
                )  # ✅ Save per file

        # ✅ Store in session state
        st.session_state.bulk_csvs = session_bulk_csvs
        st.success("✅ Bulk Transactions Processed! Each PDF has its own CSV.")

    # ✅ Show feedback & download per document
    if "bulk_csvs" in st.session_state and st.session_state.bulk_csvs:
        selected_doc = st.selectbox(
            "📂 Select a Document for Feedback",
            list(st.session_state.bulk_csvs.keys()),
            key="bulk_doc",
        )
        df_selected = st.session_state.bulk_csvs[selected_doc]

        st.markdown(
            f"<h4 style='color: #1976D2; font-weight: bold;'>📋 Processed Transactions - {selected_doc}</h4>",
            unsafe_allow_html=True,
        )
        st.dataframe(df_selected, use_container_width=True)

        st.markdown(
            "<h5 style='color: #444;'>📝 Provide Feedback</h5>", unsafe_allow_html=True
        )

        selected_desc = st.selectbox(
            "Select a Transaction to Correct",
            df_selected["Description"].unique(),
            key="bulk_desc",
        )

        if not df_selected[df_selected["Description"] == selected_desc].empty:
            filtered_row = df_selected[
                df_selected["Description"] == selected_desc
            ].iloc[0]

            # ✅ Vendor Selection (Dropdown + Freeflow Text)
            existing_vendors = df_selected["Vendor Name"].dropna().unique().tolist()
            selected_vendor_dropdown = st.selectbox(
                "Correct Vendor (Choose from list)",
                existing_vendors,
                index=(
                    existing_vendors.index(filtered_row["Vendor Name"])
                    if filtered_row["Vendor Name"] in existing_vendors
                    else 0
                ),
                key="bulk_vendor_dropdown",
            )
            new_vendor_text = st.text_input(
                "Or Enter a New Vendor Name", "", key="bulk_vendor_text"
            )

            correct_vendor = (
                new_vendor_text.strip()
                if new_vendor_text.strip()
                else selected_vendor_dropdown
            )
            correct_deposits = st.number_input(
                "Deposits_Credits",
                value=float(filtered_row["Deposits_Credits"]),
                step=0.01,
                key="bulk_deposit_fb",
            )
            correct_withdrawals = st.number_input(
                "Withdrawals_Debits",
                value=float(filtered_row["Withdrawals_Debits"]),
                step=0.01,
                key="bulk_withdraw_fb",
            )
            comments = st.text_area(
                "Additional Comments (Optional)", key="bulk_comments_fb"
            )

            col1, col2 = st.columns([1, 1.2])
            with col1:
                submit_feedback = st.button(
                    "✅ Submit Feedback", key="bulk_feedback_btn"
                )
            with col2:
                csv_data = df_selected.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"⬇ Download {selected_doc} CSV",
                    csv_data,
                    f"{selected_doc}.csv",
                    "text/csv",
                )

            if submit_feedback:
                feedback_entry = [
                    [
                        datetime.date.today(),
                        selected_doc,
                        selected_desc,
                        correct_vendor,
                        filtered_row["Vendor Name"],
                        correct_deposits,
                        correct_withdrawals,
                        comments,
                    ]
                ]

                save_feedback(feedback_entry)  # ✅ Save to feedback log

                df_selected.loc[
                    df_selected["Description"] == selected_desc,
                    ["Vendor Name", "Deposits_Credits", "Withdrawals_Debits"],
                ] = [
                    correct_vendor,
                    float(correct_deposits),
                    float(correct_withdrawals),
                ]

                st.session_state.bulk_csvs[selected_doc] = (
                    df_selected  # ✅ Update session state
                )

                # ✅ Generate updated CSV
                csv_data = df_selected.to_csv(index=False).encode("utf-8")

                st.success("✅ Feedback submitted successfully!")

                # ✅ Show updated download button
                st.download_button(
                    f"⬇ Download Updated {selected_doc} CSV",
                    csv_data,
                    f"{selected_doc}.csv",
                    "text/csv",
                )

# ✅ Ensure session state for Q&A history
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# ✅ Ensure session state for storing bulk CSVs
if "bulk_csvs" not in st.session_state:
    st.session_state.bulk_csvs = {}

# ✅ Analytics Dashboard (Supports Multiple CSVs)
with tab3:
    st.markdown(
        "<h2 style='color:#004AAD;'>📊 AI Analytics Dashboard</h2>",
        unsafe_allow_html=True,
    )

    # ✅ Gather available CSVs
    available_csvs = {}

    if "transactions" in st.session_state and not st.session_state.transactions.empty:
        available_csvs["Single Document"] = st.session_state.transactions

    if "bulk_csvs" in st.session_state and st.session_state.bulk_csvs:
        available_csvs.update(st.session_state.bulk_csvs)  # ✅ Add bulk CSVs

    # ✅ No CSVs available
    if not available_csvs:
        st.warning("⚠ No transactions available. Process a document first.")
    else:
        # ✅ Select CSV for analysis
        selected_csv = st.selectbox(
            "📂 Select CSV for Analysis",
            list(available_csvs.keys()),
            key="analytics_csv",
        )
        df = available_csvs[selected_csv]

        # ✅ Ensure valid dates before conversion
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
        df = df.dropna(subset=["Date"])  # ✅ Remove invalid dates

        # ✅ Transactions Over Time (Month-wise)
        st.markdown(
            f"<h4 style='color:#1976D2;'>📊 Transactions Over Time (Monthly) - {selected_csv}</h4>",
            unsafe_allow_html=True,
        )
        df_grouped = (
            df.groupby(df["Date"].dt.to_period("M"))[
                ["Deposits_Credits", "Withdrawals_Debits"]
            ]
            .sum()
            .reset_index()
        )
        df_grouped["Date"] = df_grouped["Date"].astype(str)
        fig = px.line(
            df_grouped,
            x="Date",
            y=["Deposits_Credits", "Withdrawals_Debits"],
            title="Transactions Over Time (Monthly)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ✅ Transactions Over Time (Day-wise)
        st.markdown(
            f"<h4 style='color:#1976D2;'>📆 Transactions Over Time (Daily) - {selected_csv}</h4>",
            unsafe_allow_html=True,
        )
        df_grouped_day = (
            df.groupby(df["Date"])[["Deposits_Credits", "Withdrawals_Debits"]]
            .sum()
            .reset_index()
        )
        fig_day = px.line(
            df_grouped_day,
            x="Date",
            y=["Deposits_Credits", "Withdrawals_Debits"],
            title="Transactions Over Time (Daily)",
        )
        st.plotly_chart(fig_day, use_container_width=True)

        # ✅ Vendor-Based Summary
        st.markdown(
            "<h4 style='color:#1976D2;'>📌 Vendor-Based Summary</h4>",
            unsafe_allow_html=True,
        )
        vendor_summary = (
            df.groupby("Vendor Name")[["Deposits_Credits", "Withdrawals_Debits"]]
            .sum()
            .reset_index()
        )
        fig_bar = px.bar(
            vendor_summary,
            x="Vendor Name",
            y=["Deposits_Credits", "Withdrawals_Debits"],
            title="Top Vendors by Transactions",
            barmode="group",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ✅ AI Chatbot (Below Graphs) - Fixed Alignment
        st.markdown(
            """
            <style>
            .chat-container {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                margin-top: 20px;
                text-align: center;
            }
            .chat-header {
                font-size: 18px;
                font-weight: bold;
                color: #004AAD;
                text-align: center;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown(
            '<p class="chat-header">💬 Ask About Transactions</p>',
            unsafe_allow_html=True,
        )

        query = st.text_area(
            "🔍 Enter your question...", key="query_analytics", height=100
        )

        col1, col2, col3 = st.columns([4, 2, 2])  # ✅ Adjusted button alignment
        with col2:
            ask_button = st.button(" Get Insights", key="query_btn")

        # st.markdown('</div>', unsafe_allow_html=True)  # ✅ Close chat container

        # ✅ AI Processing & Response
        if ask_button and query.strip():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)

            context = f"Analyze the following transaction data:\n{df.to_json(orient='records', indent=2)}"
            full_prompt = f"""
            {context}
            User Question: {query}
 
            **Instructions for Response Formatting:**
            - **Provide clear, structured financial insights.**
            - **Summarize key findings concisely.**
            - **Use structured tables for numerical analysis where possible.**
            - **If needed, return JSON Table format for structured display.**
            """

            try:
                if ai_model == "DeepSeek":
                    payload = {
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": full_prompt}],
                        "temperature": 0,
                    }
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    }
                    response = requests.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers=headers,
                        json=payload,
                    )

                    if response.status_code == 200:
                        response_text = response.json()["choices"][0]["message"][
                            "content"
                        ]
                    else:
                        st.error(f"❌ DeepSeek API error: {response.status_code}")
                        response_text = "Error fetching response from DeepSeek."

                elif ai_model == "Gemini":
                    gemini = ChatGoogleGenerativeAI(
                        model="gemini-2.0-pro-exp-02-05",
                        google_api_key=api_key,
                        temperature=0,
                    )
                    response = gemini.invoke(full_prompt)
                    response_text = (
                        response.content if response else "No response received."
                    )

                # ✅ Detect if response contains JSON table
                if response_text.strip().startswith(
                    "{"
                ) or response_text.strip().startswith("["):
                    try:
                        table_data = json.loads(response_text)  # ✅ Parse JSON
                        df_response = pd.DataFrame(
                            table_data
                        )  # ✅ Convert to DataFrame
                        st.markdown("#### Table")
                        st.dataframe(df_response, use_container_width=True)
                    except:
                        st.markdown(
                            f"**📝 Summary:** {response_text}"
                        )  # ✅ Fallback to text response
                else:
                    st.markdown(
                        f"**📝 Summary:** {response_text}"
                    )  # ✅ Direct text response

                # ✅ Store Q&A in session state & log it
                if "qa_history" not in st.session_state:
                    st.session_state.qa_history = []

                entry = {
                    "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Document": selected_csv,
                    "Question": query,
                    "Answer": response_text,
                }
                st.session_state.qa_history.append(entry)

                # ✅ Save Q&A log (without download option)
                save_qa_log(entry)

            except Exception as e:
                st.error(f"❌ Error generating insights: {e}")

            st.markdown("</div>", unsafe_allow_html=True)  # ✅ Close chat container

        # ✅ Q&A History (Expandable)
        with st.expander("📜 View Q&A History"):
            if "qa_history" in st.session_state and st.session_state.qa_history:
                qa_df = pd.DataFrame(st.session_state.qa_history)
                st.dataframe(qa_df, use_container_width=True)
            else:
                st.info(
                    "📝 No Q&A history yet. Ask a question to start logging interactions!"
                )
