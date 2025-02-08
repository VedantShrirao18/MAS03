import pandas as pd
import random
import together
from fastapi import Query
from classification import classify_issue_ml

# Load Data
CSV_FILE = "/home/vedant/Downloads/ai_agent/data/your_data.csv"
jira_data = pd.read_csv(CSV_FILE)

TOGETHER_API_KEY = "your_together_api_key"
together.api_key = TOGETHER_API_KEY
client = together.Together()

# 🔹 Query Handling
def query_agent(query: str):
    if "add issue" in query.lower():
        return add_issue_to_csv(query.replace("add issue", "").strip())
    return ai_pipeline(query)
def get_llm_response(prompt):
    """Call LLM to generate a response."""
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content.strip()
    return response

def retrieval_agent(query):
    """Retrieve relevant data from Jira CSV based on LLM decision."""
    prompt = f"Does this query require Access to JIRA CSV file ?\nQuery: {query}\nAnswer only 'YES' or 'NO'."
    decision = get_llm_response(prompt)

    return decision

def reasoning_agent(query,issues):
    """Process and reason about the retrieved data."""
    if issues == "YES" :
        return query_llm_about_csv(query)
    return get_llm_response(query)  # If LLM-only, proceed with LLM reasoning

def action_agent(query, insights):
    """Generate a final user response."""
    prompt = f"User Query: {query}\nResponse: {insights}\nProvide a clear and accurate response."
    return get_llm_response(prompt)

def ai_pipeline(query):
    """Main AI Pipeline using ReAct (Reasoning + Acting) Agents."""
    retrieved_data = retrieval_agent(query)
    insights =  reasoning_agent(query ,retrieved_data)
    return action_agent(query, insights)

# def get_together_response(prompt):
#     response = client.chat.completions.create(
#         model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
#         messages=[{"role": "user", "content": prompt}],
#     ).choices[0].message.content.strip()
#     return response

# 🔹 Add Issue to CSV (ML Classification)

#-------------------------------------------------------------------------------

def query_llm_about_csv(user_query):
    csv_content = jira_data.to_string()
    prompt = f"Based on the following dataset, answer the user's query: {user_query}\n\n{csv_content}"
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content.strip()
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error querying LLM about CSV data")

#-------------------------------------------------------------------------------
def add_issue_to_csv(query):
    issue_id, classification = classify_issue_ml(query)
    new_issue = {
        "Issue ID": issue_id,
        "Summary": query,
        "Description": f"Fix issue related to {query.lower()}",
        "Status": random.choice(["To Do", "In Progress", "Done"]),
        "Priority": random.choice(["Low", "Medium", "High", "Critical"]),
        "Assignee": random.choice(["Anna Lee", "James Brown", "Jane Smith", "John Doe"]),
        "Reporter": random.choice(["Anna Lee", "James Brown", "Jane Smith", "John Doe"]),
        "Created Date": "2025-02-07",
        "Due Date": "2025-02-20",
        "Classification": classification
    }

    df_new = pd.DataFrame([new_issue])
    df_new.to_csv(CSV_FILE, mode="a", header=False, index=False)
    return {"message": "Issue added successfully", "data": new_issue}
