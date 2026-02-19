import os
import pandas as pd
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

# -------------------------------
# Setup OpenAI API Key
# -------------------------------
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=os.environ["OPENAI_API_KEY"]
)

# -------------------------------
# System Instruction Prompt
# -------------------------------
system_prompt = """
You are an agentic AI system designed to detect and prevent payment fraud in online transactions.
Your orchestration involves multiple specialized agents working collaboratively.
Each agent must perform its role independently, share signals with others, and escalate findings to the Case Management Agent.
Goals:
- Detect fraudulent transactions with high accuracy
- Minimize false positives
- Provide compliance-ready case records
- Continuously learn from resolved cases
Workflow:
1. Transaction Monitoring Agent scans transactions for anomalies
2. Behavioral Analysis Agent compares against historical user patterns
3. Device Fingerprinting Agent validates device trustworthiness
4. Identity Verification Agent confirms user legitimacy
5. Case Management Agent consolidates findings and escalates high-risk cases
"""


# -------------------------------
# Define Agents
# -------------------------------
transaction_monitoring_agent = Agent(
    role="Transaction Monitoring Agent",
    goal="Scan transactions for anomalies and flag suspicious activity.",
    backstory="Expert in transaction analysis, anomaly detection, and fraud signals.",
    llm=llm,
    verbose=True
)

behavioral_analysis_agent = Agent(
    role="Behavioral Analysis Agent",
    goal="Compare current transaction activity against historical user behavior.",
    backstory="Specialist in behavioral analytics and pattern recognition.",
    llm=llm,
    verbose=True
)

device_fingerprinting_agent = Agent(
    role="Device Fingerprinting Agent",
    goal="Validate device trustworthiness and detect anomalies.",
    backstory="Expert in device fingerprinting, IP analysis, and geolocation checks.",
    llm=llm,
    verbose=True
)

identity_verification_agent = Agent(
    role="Identity Verification Agent",
    goal="Confirm legitimacy of the user using identity verification signals.",
    backstory="Handles MFA, biometrics, and KYC validation.",
    llm=llm,
    verbose=True
)

case_management_agent = Agent(
    role="Case Management Agent",
    goal="Consolidate findings, escalate high-risk cases, and generate compliance-ready reports.",
    backstory="Responsible for case aggregation, escalation, and reporting.",
    llm=llm,
    verbose=True
)

# -------------------------------
# Define Tasks
# -------------------------------
transaction_task = Task(
    description="Monitor incoming transactions and flag anomalies.",
    agent=transaction_monitoring_agent,
    expected_output="A list of suspicious transactions with preliminary risk scores"
)

behavior_task = Task(
    description="Analyze flagged transactions against historical user behavior.",
    agent=behavioral_analysis_agent,
    expected_output="Behavioral risk score and deviation signals"
)

device_task = Task(
    description="Validate device fingerprint and detect mismatches.",
    agent=device_fingerprinting_agent,
    expected_output="Device trust score and anomaly report"
)

identity_task = Task(
    description="Verify user identity using MFA, biometrics, and KYC.",
    agent=identity_verification_agent,
    expected_output="Identity verification result (legitimate or suspicious)"
)

case_task = Task(
    description="Aggregate all agent outputs, escalate high-risk cases, and generate reports.",
    agent=case_management_agent,
    expected_output="Final fraud case record and escalation decision"
)

# -------------------------------
# Orchestrate Crew
# -------------------------------
fraud_detection_crew = Crew(
    agents=[
        transaction_monitoring_agent,
        behavioral_analysis_agent,
        device_fingerprinting_agent,
        identity_verification_agent,
        case_management_agent
    ],
    tasks=[transaction_task, behavior_task, device_task, identity_task, case_task],
    verbose=True
)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🛡️ Fraud Detection System")
st.write("Upload a transaction dataset (CSV) to run fraud detection using multi-agent orchestration.")

uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head(10))

    if st.button("Run Fraud Detection"):
        sample_transactions = df.head(10).to_dict(orient="records")

        result = fraud_detection_crew.kickoff(inputs={
            "system_prompt": system_prompt,#"Detect fraud in online transactions using multi-agent orchestration.",
            "transactions": sample_transactions
        })

        st.success("Fraud Detection Completed!")

        st.json(result)
