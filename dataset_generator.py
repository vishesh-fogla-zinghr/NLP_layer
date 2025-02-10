import pandas as pd
import random

# Define sample patterns for each category
transactional_patterns = [
    "Apply for {leave_type} leave on {date}",
    "Request reimbursement for {expense}",
    "Update my {account_detail} in the system",
    "Change my {personal_detail}",
    "Submit my claim for {reason}",
    "File a request for {request_type}",
    "Schedule a meeting with HR",
    "Apply for {benefit} benefits",
    "Request a work-from-home day on {date}"
]

informational_patterns = [
    "What is the policy for {policy_type}?",
    "How many {leave_type} days do I have left?",
    "Explain the process for {process}",
    "Can I {action} under company policy?",
    "What are the company benefits for {benefit}?",
    "How does {policy_type} work?",
    "Tell me about the HR policy on {topic}",
]

# Define values to fill the placeholders
leave_types = ["sick", "casual", "medical", "annual"]
dates = ["Monday", "next week", "March 5th", "tomorrow"]
expenses = ["travel", "food", "office supplies"]
account_details = ["bank account", "email", "address"]
personal_details = ["phone number", "emergency contact", "home address"]
request_types = ["leave approval", "reimbursement", "promotion"]
benefits = ["health", "dental", "retirement"]
processes = ["resignation", "leave application", "performance review"]
actions = ["apply for leave", "request work-from-home"]
topics = ["overtime", "salary deductions", "maternity leave"]

# Generate queries
num_samples = 1_000_0
queries = []
labels = []

for _ in range(num_samples // 2):  # Half transactional, half informational
    transactional_query = random.choice(transactional_patterns).format(
        leave_type=random.choice(leave_types),
        date=random.choice(dates),
        expense=random.choice(expenses),
        account_detail=random.choice(account_details),
        personal_detail=random.choice(personal_details),
        request_type=random.choice(request_types),
        benefit=random.choice(benefits),
        reason=random.choice(expenses),
    )
    queries.append(transactional_query)
    labels.append("transactional")

    informational_query = random.choice(informational_patterns).format(
        policy_type=random.choice(topics),
        leave_type=random.choice(leave_types),
        process=random.choice(processes),
        action=random.choice(actions),
        benefit=random.choice(benefits),
        topic=random.choice(topics),
    )
    queries.append(informational_query)
    labels.append("informational")

# Save dataset
df = pd.DataFrame({"query": queries, "intent": labels})
df.to_csv("dataset.csv", index=False)

print("✅ Dataset generation complete: dataset.csv (1M queries)")
