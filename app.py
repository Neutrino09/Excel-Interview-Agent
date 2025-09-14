import streamlit as st
import sqlite3
import numpy as np
import datetime
import json
import random
from openai import OpenAI

# ----------------------------
# OpenAI client
# ----------------------------
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not api_key:
    st.error("❌ OpenAI API key not found. Please set it in Streamlit Secrets or environment variable.")
    st.stop()

client = OpenAI(api_key=api_key)
# ----------------------------
# Load Questions from JSON
# ----------------------------
with open("questions.json") as f:
    ALL_QUESTIONS = json.load(f)

def get_questions_by_topic(topic: str):
    return [q for q in ALL_QUESTIONS if q.get("topic", "").lower() == topic.lower()]

# ----------------------------
# Database functions
# ----------------------------
def init_db():
    conn = sqlite3.connect("interviews.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interviews
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  candidate TEXT,
                  topic TEXT,
                  questions TEXT,
                  answers TEXT,
                  scores TEXT,
                  feedback TEXT,
                  date TEXT)''')
    conn.commit()
    conn.close()

def save_interview(candidate, topic, questions, answers, scores, feedback, date):
    conn = sqlite3.connect("interviews.db")
    c = conn.cursor()
    c.execute("INSERT INTO interviews (candidate, topic, questions, answers, scores, feedback, date) VALUES (?,?,?,?,?,?,?)",
              (candidate, topic, str(questions), str(answers), str(scores), feedback, date))
    conn.commit()
    conn.close()

def load_interviews():
    conn = sqlite3.connect("interviews.db")
    c = conn.cursor()
    c.execute("SELECT candidate, topic, date, feedback FROM interviews ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

# ----------------------------
# Evaluation helpers
# ----------------------------
def eval_formula(ans, expected_list):
    ans_norm = ans.strip().lower().replace(" ", "")
    for f in expected_list:
        if f.lower().replace(" ", "") in ans_norm:
            return 1.0
    return 0.0

def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(resp.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def eval_with_embeddings(answer, reference):
    if not answer.strip():
        return 0.0
    emb_ans = get_embedding(answer)
    emb_ref = get_embedding(reference)
    return cosine_similarity(emb_ans, emb_ref)

def acknowledge_response(answer):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Alex, a friendly interviewer. Be encouraging but concise."},
            {"role": "user", "content": f"The candidate answered: {answer}. Reply politely in one short sentence, no new question."}
        ],
        temperature=0.6
    )
    return resp.choices[0].message.content

def coaching_feedback(answer, reference):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Alex, an interviewer. Provide a short constructive feedback comparing the candidate’s answer with the reference. Be encouraging, max 2 sentences."},
            {"role": "user", "content": f"Candidate answer: {answer}\nCorrect/Expected: {reference}"}
        ],
        temperature=0.5
    )
    return resp.choices[0].message.content

def generate_feedback(candidate, topic, answers, scores, questions):
    qa_summary = []
    for q, a, s in zip(questions, answers, scores):
        qa_summary.append(f"Q: {q['prompt']}\nA: {a}\nScore: {s:.2f}")
    summary_text = "\n\n".join(qa_summary)

    today = datetime.date.today().strftime("%B %d, %Y")

    prompt = f"""
    You are Alex, a professional {topic} interviewer. Create a feedback report.

    Candidate Name: {candidate}
    Position: {topic} Specialist
    Date: {today}

    Candidate performance:
    {summary_text}

    Write the report with:
    - Strengths
    - Weaknesses
    - Overall Recommendation
    Keep it concise, professional, and role-relevant.
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return resp.choices[0].message.content

# ----------------------------
# Adaptive question selector
# ----------------------------
def select_next_question(last_score, last_level, questions, asked_ids):
    remaining = [q for q in questions if q["id"] not in asked_ids]
    if not remaining:
        return None

    if last_score >= 0.8:
        harder = [q for q in remaining if q["level"] == "hard"]
        if harder:
            return harder[0]
    elif last_score <= 0.4:
        easier = [q for q in remaining if q["level"] == "easy"]
        if easier:
            return easier[0]

    same = [q for q in remaining if q["level"] == last_level]
    return same[0] if same else remaining[0]

# ----------------------------
# Experience classifier
# ----------------------------
def classify_experience(exp_text: str) -> str:
    exp_text = exp_text.lower()
    if any(word in exp_text for word in ["beginner", "basic", "new", "learning"]):
        return "beginner"
    elif any(word in exp_text for word in ["intermediate", "some", "comfortable", "lookup", "formulas"]):
        return "intermediate"
    elif any(word in exp_text for word in ["advanced", "expert", "pivot", "vba", "macros", "dashboard"]):
        return "advanced"
    else:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Classify experience into beginner, intermediate, or advanced."},
                {"role": "user", "content": exp_text}
            ]
        )
        return resp.choices[0].message.content.strip().lower()

def pick_starting_question(exp_level, questions):
    if exp_level == "beginner":
        return next(q for q in questions if q["level"] == "easy")
    elif exp_level == "intermediate":
        return next(q for q in questions if q["level"] == "medium")
    elif exp_level == "advanced":
        return next(q for q in questions if q["level"] == "hard")
    else:
        return questions[0]

# ----------------------------
# Init DB
# ----------------------------
init_db()

# ----------------------------
# Session State
# ----------------------------
if "mode" not in st.session_state:
    st.session_state.mode = "intro"
if "asked_ids" not in st.session_state:
    st.session_state.asked_ids = []
    st.session_state.answers = []
    st.session_state.scores = []
    st.session_state.current_q = None
    st.session_state.feedback = None
    st.session_state.last_answer = None
    st.session_state.last_score = None
    st.session_state.exp_level = None
    st.session_state.topic = None
    st.session_state.questions = []

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
    <style>
    .chat-bubble {
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 8px;
        max-width: 80%;
        font-size: 15px;
        line-height: 1.5;
    }
    .assistant {
        background-color: #2b3e5c;
        color: #ffffff;
        margin-right: auto;
        font-size: 16px;
        font-weight: 500;
    }
    .candidate {
        background-color: #3a3a3a;
        color: #ffffff;
        margin-left: auto;
    }
    .feedback {
        background-color: #fff6e6;
        border-left: 4px solid #ffa726;
        padding: 10px;
        margin-top: 6px;
        border-radius: 6px;
        font-size: 15px;
    }
    .report-card {
        background-color: #2b3e5c;
        color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border-left: 6px solid #ffa726;
    }
    .report-card h3 {
        margin-top: 0;
        color: #ffa726;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("📋 Interview Info")
    if "candidate_name" in st.session_state and st.session_state.candidate_name:
        st.write(f"**Candidate:** {st.session_state.candidate_name}")
    st.write("**Topic:** Excel")
    st.write(f"**Questions Asked:** {len(st.session_state.asked_ids)}")
    if st.session_state.questions:
        progress = len(st.session_state.asked_ids) / len(st.session_state.questions)
        st.progress(progress)

# ----------------------------
# Main UI
# ----------------------------
st.title("🧑‍💼 Excel Interview")

# ---- Intro ----
if st.session_state.mode == "intro":
    st.markdown("""
    <div class='chat-bubble assistant'>
    👋 Hi, I’m <b>Alex</b>, and I’ll be conducting your interview today.<br><br>
    We’ll go through a few questions. After each answer, I’ll provide quick feedback. 
    At the end, you’ll also receive a summary report with strengths, weaknesses, and recommendations.<br><br>
    Before we begin, could you please tell me your name?
    </div>
    """, unsafe_allow_html=True)

    name = st.text_input("Your name:")
    exp = st.text_area("How would you describe your experience with Excel?")

    if st.button("I’m ready to start"):
        st.markdown(f"""
        <div class='chat-bubble assistant'>
        Great, {name}! Thanks for sharing your background. 
        Let’s begin your <b>Excel</b> interview. 🚀
        </div>
        """, unsafe_allow_html=True)

        st.session_state.candidate_name = name
        st.session_state.experience = exp
        st.session_state.topic = "excel"

        # Randomly select 5 questions from the Excel pool
        all_qs = get_questions_by_topic("excel")
        st.session_state.questions = random.sample(all_qs, min(5, len(all_qs)))

        exp_level = classify_experience(exp)
        st.session_state.exp_level = exp_level
        st.session_state.mode = "ask"
        st.session_state.current_q = pick_starting_question(exp_level, st.session_state.questions)
        st.rerun()

# ---- Ask ----
elif st.session_state.mode == "ask":
    q = st.session_state.current_q
    st.markdown(f"""
    <div class='chat-bubble assistant'>
    ❓ {q['prompt']}
    </div>
    """, unsafe_allow_html=True)
    ans = st.text_area("Your answer:", key=f"ans_{q['id']}")
    if st.button("Submit Answer"):
        if q["type"] == "formula":
            score = eval_formula(ans, q.get("expected", []))
        else:
            score = eval_with_embeddings(ans, q.get("reference", ""))
        st.session_state.answers.append(ans)
        st.session_state.scores.append(score)
        st.session_state.asked_ids.append(q["id"])
        st.session_state.last_answer = ans
        st.session_state.last_score = score
        st.session_state.mode = "acknowledge"
        st.rerun()

# ---- Acknowledge ----
elif st.session_state.mode == "acknowledge":
    q = next(q for q in st.session_state.questions if q["id"] == st.session_state.asked_ids[-1])
    ans = st.session_state.last_answer
    ack = acknowledge_response(ans)
    tip = coaching_feedback(ans, q.get("reference", q.get("expected", "")))

    st.markdown(f"""
    <div class='chat-bubble assistant'>💬 {ack}</div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class='feedback'>📝 Coaching Tip: {tip}</div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class='chat-bubble assistant'>(Score: {st.session_state.last_score:.2f})</div>
    """, unsafe_allow_html=True)

    if st.button("Next Question"):
        st.session_state.current_q = select_next_question(
            st.session_state.last_score,
            q["level"],
            st.session_state.questions,
            st.session_state.asked_ids
        )
        if st.session_state.current_q:
            st.session_state.mode = "ask"
        else:
            st.session_state.mode = "closing"
        st.session_state.last_answer = None
        st.session_state.last_score = None
        st.rerun()

# ---- Closing ----
elif st.session_state.mode == "closing":
    st.markdown(f"""
    <div class='chat-bubble assistant'>
    👏 Thanks {st.session_state.candidate_name}, that concludes our Excel interview!
    </div>
    """, unsafe_allow_html=True)

    st.write("### 📊 Results")
    for i, (q_id, ans, score) in enumerate(zip(st.session_state.asked_ids, st.session_state.answers, st.session_state.scores)):
        q = next(q for q in st.session_state.questions if q["id"] == q_id)
        st.write(f"- **Q{i+1}:** {q['prompt']}")
        st.write(f"  - Your Answer: {ans}")
        st.write(f"  - Score: {score:.2f}")

    if st.button("Generate Feedback Report"):
        with st.spinner("Generating feedback..."):
            qs = [next(q for q in st.session_state.questions if q["id"] == qid) for qid in st.session_state.asked_ids]
            raw_feedback = generate_feedback(
                st.session_state.candidate_name,
                "Excel",
                st.session_state.answers,
                st.session_state.scores,
                qs
            )
            st.session_state.feedback = raw_feedback

            st.markdown(f"""
            <div class='report-card'>
            <h3>📝 Interview Feedback Report</h3>
            {st.session_state.feedback}
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.feedback and st.button("Save Results"):
        today = datetime.date.today().strftime("%B %d, %Y")
        qs = [next(q for q in st.session_state.questions if q["id"] == qid) for qid in st.session_state.asked_ids]
        save_interview(st.session_state.candidate_name,
                       "Excel",
                       [q['prompt'] for q in qs],
                       st.session_state.answers,
                       st.session_state.scores,
                       st.session_state.feedback,
                       today)
        st.success("✅ Interview saved successfully!")

# ---- Past Interviews ----
if st.checkbox("📂 Show Past Interviews"):
    past = load_interviews()
    for cand, topic, date, fb in past:
        st.write(f"**Candidate:** {cand} | **Topic:** Excel | **Date:** {date}")
        st.markdown(fb)
        st.write("---")
