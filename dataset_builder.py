import sqlite3
from openai import OpenAI

# ----------------------------
# OpenAI client
# ----------------------------
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")  # Or use st.secrets if inside Streamlit

# ----------------------------
# Database setup
# ----------------------------
def init_training_db():
    conn = sqlite3.connect("interviews.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS training_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  question TEXT,
                  answer TEXT,
                  label TEXT)''')  # label = correct/partial/incorrect
    conn.commit()
    conn.close()

def save_training_example(question, answer, label):
    conn = sqlite3.connect("interviews.db")
    c = conn.cursor()
    c.execute("INSERT INTO training_data (question, answer, label) VALUES (?,?,?)",
              (question, answer, label))
    conn.commit()
    conn.close()

# ----------------------------
# Generate synthetic answers
# ----------------------------
def generate_synthetic_answers(question):
    prompt = f"""
    For the following Excel interview question, generate 3 candidate answers:
    1. Correct answer
    2. Partially correct answer
    3. Incorrect answer

    Question: {question}
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return resp.choices[0].message.content

# ----------------------------
# Example question bank (sync with app.py)
# ----------------------------
QUESTIONS = [
    "Write a formula to add values in cells A1 through A10.",
    "What does VLOOKUP do in Excel?",
    "How would you use INDEX and MATCH together as an alternative to VLOOKUP?",
    "Write a formula that returns 'Pass' if the score in B2 is >=50, otherwise 'Fail'.",
    "How would you use nested IF to grade marks: >=90 'A', >=75 'B', >=60 'C', else 'F'?",
    "What formula would you use to extract the first 5 characters from a text in A1?",
    "Explain how you would highlight duplicate values in a column using conditional formatting.",
    "What chart type is best to show trends over time and why?"
]

# ----------------------------
# Main runner
# ----------------------------
if __name__ == "__main__":
    init_training_db()

    for q in QUESTIONS:
        print(f"\n🔹 Generating answers for: {q}")
        synthetic = generate_synthetic_answers(q)
        print(synthetic)

        # Parse answers into lines (correct, partial, wrong)
        lines = synthetic.split("\n")
        for line in lines:
            if "Correct" in line:
                save_training_example(q, line, "correct")
            elif "Partial" in line or "partially" in line:
                save_training_example(q, line, "partial")
            elif "Incorrect" in line or "wrong" in line:
                save_training_example(q, line, "incorrect")
