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

## 🛠️ Setup

```bash
git clone https://github.com/<your-username>/excel-interview-agent.git
cd excel-interview-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
