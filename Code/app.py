"""DataChat: chat with your CSV using a local LLM (Ollama) -> SQL (DuckDB) -> chart + summary.

Setup:
    ollama pull qwen2.5-coder:7b
    pip install -U streamlit duckdb pandas plotly ollama
Run:
    streamlit run app.py
"""
import json
import re
import uuid
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path

import duckdb
import ollama
import pandas as pd
import plotly.express as px
import streamlit as st

MODEL = "qwen2.5-coder:7b"  # try qwen2.5-coder:14b if you have the VRAM/RAM
MAX_RETRIES = 2
HISTORY_FILE = Path(__file__).parent / "chats.json"
FORBIDDEN = re.compile(
    r"\b(drop|delete|update|insert|alter|create|attach|copy|pragma|export|install|load)\b",
    re.IGNORECASE,
)

st.set_page_config(page_title="DataChat", page_icon="📊", layout="centered")


# ---------- Persistence ----------
def load_chats() -> dict:
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_chats() -> None:
    try:
        HISTORY_FILE.write_text(
            json.dumps(st.session_state.chats, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


# ---------- LLM + SQL ----------
def ask_llm(prompt: str) -> str:
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return resp["message"]["content"]


def extract_sql(text: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    sql = match.group(1) if match else text
    return sql.strip().rstrip(";").strip()


def is_safe(sql: str) -> bool:
    if ";" in sql:  # single statement only
        return False
    if not re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE):
        return False
    stripped = re.sub(r"'[^']*'|\"[^\"]*\"", "", sql)  # ignore quoted names/strings
    return not FORBIDDEN.search(stripped)


def build_schema(con: duckdb.DuckDBPyConnection) -> str:
    cols = con.execute("DESCRIBE data").fetchdf()[["column_name", "column_type"]]
    lines = [f'- "{r.column_name}" ({r.column_type})' for r in cols.itertuples()]
    sample = con.execute("SELECT * FROM data LIMIT 3").fetchdf().to_csv(index=False)
    return "Table: data\nColumns:\n" + "\n".join(lines) + f"\n\nSample rows (CSV):\n{sample}"


def history_context(messages: list, n: int = 3) -> str:
    """Previous question/SQL pairs so follow-ups like 'now by month' work."""
    pairs = []
    for i, m in enumerate(messages):
        if i > 0 and m["role"] == "assistant" and m.get("sql") and not m.get("error"):
            pairs.append((messages[i - 1]["content"], m["sql"]))
    return "\n".join(f"Q: {q}\nSQL: {s}" for q, s in pairs[-n:])


CHART_PATTERNS = [
    ("scatter", r"scatter|correlat|relationship between"),
    ("hist", r"histogram|distribution of"),
    ("box", r"box ?plot"),
    ("pie", r"\bpie\b"),
    ("line", r"line (chart|graph|plot)|trend|over time"),
    ("bar", r"\bbar\b"),
]

CHART_HINTS = {
    "scatter": "The user wants a SCATTER PLOT. Return raw rows with NO GROUP BY or aggregation: "
    "the x numeric column first, the y numeric column second, optionally a categorical column "
    "third for colour. Skip rows where x or y IS NULL. Use LIMIT 5000 at most.",
    "hist": "The user wants a HISTOGRAM. Return the raw values of ONE numeric column, no "
    "aggregation, skipping NULLs, LIMIT 5000 at most.",
    "box": "The user wants a BOX PLOT. Return a categorical column first and the raw numeric "
    "column second, no aggregation, skipping NULLs, LIMIT 5000 at most.",
    "pie": "The user wants a PIE CHART. Return two columns: the category, then a numeric total "
    "(SUM or COUNT) per category, at most 10 rows.",
    "line": "The user wants a LINE CHART. Return x first (a date/time or ordered column), then "
    "y aggregated per x, sorted by x.",
    "bar": "The user wants a BAR CHART. Return the category first, then a numeric aggregate, "
    "sorted descending, at most 20 rows.",
}


def detect_chart_type(question: str):
    for kind, pattern in CHART_PATTERNS:
        if re.search(pattern, question, re.IGNORECASE):
            return kind
    return None


def generate_sql(question, schema, history, chart=None, prev_sql="", error="") -> str:
    prompt = f"""You write DuckDB SQL. Use only the table and columns below.
Always wrap column names in double quotes. Return ONE read-only SELECT query
in a ```sql code block and nothing else.

{schema}
"""
    if history:
        prompt += f"\nEarlier in this conversation (use for follow-up questions):\n{history}\n"
    if chart:
        prompt += f"\nChart requirement: {CHART_HINTS[chart]}\n"
    prompt += f"\nQuestion: {question}\n"
    if error:
        prompt += f"\nYour previous query failed.\nQuery: {prev_sql}\nError: {error}\nFix it.\n"
    return extract_sql(ask_llm(prompt))


def summarise(question: str, result: pd.DataFrame) -> str:
    preview = result.head(25).to_csv(index=False)
    return ask_llm(
        f"Question: {question}\nQuery result (CSV):\n{preview}\n\n"
        "In 2-4 sentences, state what this result shows. Only use numbers that appear above."
    )


def answer(question: str, df: pd.DataFrame, schema: str, history: str) -> dict:
    msg = {"id": uuid.uuid4().hex, "role": "assistant", "content": "", "sql": "", "error": ""}
    try:
        con = duckdb.connect()  # in-memory
        con.register("data", df)
        sql, error, result = "", "", None
        chart = detect_chart_type(question)
        for _ in range(MAX_RETRIES + 1):
            sql = generate_sql(question, schema, history, chart, sql, error)
            if not is_safe(sql):
                error = "Query must be a single read-only SELECT statement."
                continue
            try:
                result = con.execute(sql).fetchdf()
                error = ""
                break
            except Exception as e:  # feed the error back to the model
                error = str(e)
        msg["sql"] = sql
        if result is None:
            msg["error"] = f"Couldn't produce a working query. Last error: {error}"
            return msg
        msg["chart"] = chart
        msg["result_json"] = result.head(2000).to_json(orient="table")
        msg["content"] = summarise(question, result)
    except Exception as e:
        msg["error"] = (
            f"Something went wrong: {e}. Is Ollama running, and have you run "
            f"`ollama pull {MODEL}`?"
        )
    return msg


# ---------- Charts ----------
def make_chart(df: pd.DataFrame, dark: bool, kind=None):
    if len(df) < 2:
        return None
    num = df.select_dtypes("number").columns.tolist()
    time = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    cat = [c for c in df.columns if c not in num and c not in time]
    template = "plotly_dark" if dark else "plotly_white"

    fig = None
    # If the user asked for a specific chart type, honour it first
    if kind == "scatter" and len(num) >= 2:
        colour = cat[0] if cat and df[cat[0]].nunique() <= 12 else None
        fig = px.scatter(df, x=num[0], y=num[1], color=colour, template=template)
    elif kind == "hist" and num:
        fig = px.histogram(df, x=num[0], template=template)
    elif kind == "box" and num:
        fig = px.box(df, x=cat[0] if cat else None, y=num[0], template=template)
    elif kind == "pie" and cat and num:
        fig = px.pie(df.head(15), names=cat[0], values=num[0], template=template)
    elif kind == "line" and num and (time or cat or len(num) >= 2):
        x = time[0] if time else (cat[0] if cat else num[0])
        y = [c for c in num if c != x][0]
        fig = px.line(df.sort_values(x), x=x, y=y, template=template)
    elif kind == "bar" and cat and num:
        top = df.sort_values(num[0], ascending=False).head(30)
        fig = px.bar(top, x=cat[0], y=num[0], template=template)
    # Otherwise guess from the shape of the result
    elif time and num:
        fig = px.line(df.sort_values(time[0]), x=time[0], y=num[0], template=template)
    elif cat and num:
        top = df.sort_values(num[0], ascending=False).head(30)
        fig = px.bar(top, x=cat[0], y=num[0], template=template)
    elif len(num) >= 2:
        fig = px.scatter(df, x=num[0], y=num[1], template=template)
    elif len(num) == 1:
        fig = px.histogram(df, x=num[0], template=template)
    if fig is not None:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
        )
    return fig


# ---------- Theme ----------
def theme_css(dark: bool) -> str:
    p = (
        {"bg": "#0b0b0c", "fg": "#ececec", "muted": "#9ca3af", "panel": "#161618", "border": "#2a2a2e"}
        if dark
        else {"bg": "#ffffff", "fg": "#111111", "muted": "#6b7280", "panel": "#f5f5f5", "border": "#e5e7eb"}
    )
    css = """
<style>
#MainMenu, footer {visibility: hidden;}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stBottom"] > div {background: %bg%;}
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
.stApp span, .stApp div[data-testid="stMarkdownContainer"] {color: %fg%;}
.block-container {max-width: 820px; padding-top: 2rem; padding-bottom: 6rem;}
[data-testid="stSidebar"] {background: %panel%; border-right: 1px solid %border%;}
[data-testid="stSidebar"] .stButton > button {width: 100%; justify-content: flex-start; text-align: left;}
.stButton > button {background: %bg%; color: %fg%; border: 1px solid %border%; border-radius: 10px;}
.stButton > button:hover {border-color: %muted%; color: %fg%;}
[data-testid="stChatInput"] {background: %panel%; border: 1px solid %border%; border-radius: 26px;}
[data-testid="stChatInput"] textarea {color: %fg%; background: transparent;}
[data-testid="stChatMessage"] {background: transparent;}
[data-testid="stExpander"] {background: %panel%; border: 1px solid %border%; border-radius: 12px;}
[data-testid="stFileUploaderDropzone"] {background: %bg%; border: 1px dashed %border%;}
pre, code {background: %panel% !important; color: %fg% !important;}
.hero {text-align: center; margin-top: 16vh;}
.hero h1 {font-weight: 600; letter-spacing: -0.02em;}
.hero p {color: %muted%;}
%extra%
</style>"""
    extra = "[data-testid='stDataFrame'] {filter: invert(1) hue-rotate(180deg);}" if dark else ""
    for k, v in {**p, "extra": extra}.items():
        css = css.replace(f"%{k}%", v)
    return css


# ---------- Rendering ----------
def render_message(m: dict, dark: bool) -> None:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.write(m["content"])
            return
        if m.get("error"):
            st.error(m["error"])
            if m.get("sql"):
                with st.expander("SQL"):
                    st.code(m["sql"], language="sql")
            return
        st.write(m["content"])
        if m.get("result_json"):
            result = pd.read_json(StringIO(m["result_json"]), orient="table")
            fig = make_chart(result, dark, m.get("chart"))
            if fig is not None:
                st.plotly_chart(fig, key=f"chart_{m['id']}", theme=None)
            with st.expander("Table and SQL"):
                st.code(m["sql"], language="sql")
                st.dataframe(result)


def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Parse date-looking columns so charts can draw time series
    for c in df.columns:
        if df[c].dtype == object and re.search(r"date|time", str(c), re.IGNORECASE):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.8:
                df[c] = parsed
    return df


# ---------- State ----------
ss = st.session_state
if "chats" not in ss:
    ss.chats = load_chats()
ss.setdefault("current_id", uuid.uuid4().hex)
ss.setdefault("dark", False)
ss.setdefault("df", None)
ss.setdefault("schema", "")
ss.setdefault("file_key", None)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 📊 DataChat")
    if st.button("＋ New chat"):
        ss.current_id = uuid.uuid4().hex
        st.rerun()

    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        key = (file.name, file.size)
        if ss.file_key != key:
            try:
                df = load_csv(file)
                con = duckdb.connect()
                con.register("data", df)
                ss.df, ss.schema, ss.file_key = df, build_schema(con), key
            except Exception as e:
                st.error(f"Couldn't read that CSV: {e}")
    else:
        ss.df, ss.schema, ss.file_key = None, "", None

    st.toggle("Dark mode", key="dark")

    st.markdown("**History**")
    for cid, c in sorted(ss.chats.items(), key=lambda kv: kv[1]["created"], reverse=True):
        col1, col2 = st.columns([5, 1])
        label = ("● " if cid == ss.current_id else "") + c["title"]
        if col1.button(label, key=f"open_{cid}"):
            ss.current_id = cid
            st.rerun()
        if col2.button("🗑", key=f"del_{cid}"):
            ss.chats.pop(cid, None)
            save_chats()
            if cid == ss.current_id:
                ss.current_id = uuid.uuid4().hex
            st.rerun()

st.markdown(theme_css(ss.dark), unsafe_allow_html=True)

# ---------- Main chat ----------
chat = ss.chats.get(ss.current_id) or {
    "id": ss.current_id,
    "title": "New chat",
    "created": datetime.now().isoformat(),
    "messages": [],
}

if not chat["messages"]:
    st.markdown(
        "<div class='hero'><h1>DataChat</h1>"
        "<p>Upload a CSV in the sidebar, then ask anything about it.</p></div>",
        unsafe_allow_html=True,
    )

for m in chat["messages"]:
    render_message(m, ss.dark)

prompt = st.chat_input(
    "Ask a question about your data" if ss.df is not None else "Upload a CSV in the sidebar to start",
    disabled=ss.df is None,
)
if prompt:
    history = history_context(chat["messages"])
    if not chat["messages"]:
        chat["title"] = prompt[:40]
    chat["messages"].append({"role": "user", "content": prompt})
    ss.chats[chat["id"]] = chat

    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = answer(prompt, ss.df, ss.schema, history)
    chat["messages"].append(reply)
    save_chats()
    st.rerun()

# Daleroy Chinomona