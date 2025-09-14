# 🤖 Excel Interview Agent

An AI-powered Streamlit app that simulates a structured Excel job interview.  
The agent (Alex) introduces itself, asks adaptive questions, evaluates answers,  
and generates a professional feedback report at the end.

---

## ✨ Features
- 🧑‍💼 **Structured interview flow** (intro → questions → feedback)
- 🤔 **Answer evaluation** (formula matching & semantic embeddings)
- 🔄 **Adaptive difficulty** (next question adjusts by score)
- 📊 **Final feedback report** (strengths, weaknesses, recommendations)
- 💾 **Interview history saved** in SQLite

---

<img width="1470" height="956" alt="Screenshot 2025-09-14 at 1 52 45 PM" src="https://github.com/user-attachments/assets/9a50a0fe-c15e-422c-b51d-e88198989e0b" />

## 🛠️ Setup

```bash
git clone https://github.com/<your-username>/excel-interview-agent.git
cd excel-interview-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt

