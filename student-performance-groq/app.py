"""Student Performance Analyzer — flexible student-performance multi-agent dashboard.

Run: streamlit run app.py
LLM provider: Groq through LangChain. No fixed spreadsheet schema is required.
"""
from __future__ import annotations

import json
import os
import re
from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="Student Performance Analyzer", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:#e8e4dc; }
.stApp { background:#0a0a0f; background-image:radial-gradient(ellipse 80% 50% at 20% -10%,rgba(255,140,50,.12) 0%,transparent 60%),radial-gradient(ellipse 60% 40% at 80% 110%,rgba(255,80,30,.08) 0%,transparent 55%); }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding:2rem 3rem 4rem; max-width:1220px; }
.hero { text-align:center; padding:3.1rem 0 2.35rem; }
.hero-eyebrow,.panel-label,.section-heading { font-family:'DM Mono',monospace; text-transform:uppercase; letter-spacing:.2em; }
.hero-eyebrow { font-size:.68rem; color:#ff8c32; margin-bottom:1rem; }
.hero h1 { font-family:'Syne',sans-serif; font-size:clamp(2.8rem,6vw,5rem); font-weight:800; line-height:1; letter-spacing:-.04em; color:#f0ebe0; margin:0 0 1rem; }
.hero h1 span { color:#ff8c32; }
.hero-sub { font-size:1.03rem; font-weight:300; color:#a09890; max-width:610px; margin:0 auto; line-height:1.65; }
.divider { height:1px; background:linear-gradient(90deg,transparent,rgba(255,140,50,.3),transparent); margin:1.5rem 0 2.2rem; }
.input-card,.result-panel,.report-panel { background:rgba(255,255,255,.03); border:1px solid rgba(255,140,50,.15); border-radius:16px; padding:2rem 2.25rem; backdrop-filter:blur(8px); }
.input-card { margin-bottom:1.1rem; }
.result-panel { border-color:rgba(255,255,255,.08); padding:1.4rem 1.6rem; }
.report-panel { border-color:rgba(255,140,50,.2); margin-top:1.2rem; }
.stTextInput > div > div > input, .stTextArea textarea { background:rgba(255,255,255,.05)!important; border:1px solid rgba(255,140,50,.25)!important; border-radius:10px!important; color:#f0ebe0!important; }
.stTextInput > label, .stFileUploader > label { font-family:'DM Mono',monospace!important; font-size:.7rem!important; letter-spacing:.15em!important; text-transform:uppercase!important; color:#ff8c32!important; }
[data-testid="stFileUploaderDropzone"] { background:rgba(255,255,255,.035); border:1px dashed rgba(255,140,50,.3); border-radius:10px; }
[data-testid="stFileUploaderDropzoneInstructions"] { color:#a09890; }
.stButton > button { background:linear-gradient(135deg,#ff8c32 0%,#ff5a1a 100%)!important; color:#0a0a0f!important; font-family:'Syne',sans-serif!important; font-weight:700!important; font-size:.93rem!important; letter-spacing:.03em!important; border:none!important; border-radius:10px!important; padding:.7rem 1.4rem!important; box-shadow:0 4px 20px rgba(255,140,50,.25)!important; transition:transform .15s,box-shadow .15s!important; }
.stButton > button:hover { transform:translateY(-2px)!important; box-shadow:0 8px 28px rgba(255,140,50,.38)!important; }
.step-card { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-radius:14px; padding:1.15rem 1.4rem; margin-bottom:.8rem; position:relative; overflow:hidden; }
.step-card.active { border-color:rgba(255,140,50,.45); background:rgba(255,140,50,.045); }
.step-card.done { border-color:rgba(80,200,120,.3); background:rgba(80,200,120,.03); }
.step-card::before { content:''; position:absolute; left:0;top:0;bottom:0;width:3px;background:rgba(255,255,255,.05); }
.step-card.active::before { background:#ff8c32; }.step-card.done::before { background:#50c878; }
.step-header { display:flex;align-items:center;gap:.75rem; }.step-num { font-family:'DM Mono';font-size:.66rem;color:#ff8c32;letter-spacing:.14em; }.step-title { font-family:'Syne';font-size:.91rem;font-weight:700;color:#f0ebe0; }.step-status { margin-left:auto;font-family:'DM Mono';font-size:.64rem;letter-spacing:.1em; }.status-waiting{color:#555}.status-running{color:#ff8c32}.status-done{color:#50c878}.step-desc{font-size:.76rem;color:#706860;margin:.35rem 0 0 1.65rem;line-height:1.45}
.section-heading { font-size:.68rem;color:#ff8c32;margin:1.8rem 0 .95rem; }.panel-label{font-size:.67rem;color:#ff8c32;border-bottom:1px solid rgba(255,140,50,.15);padding-bottom:.7rem;margin-bottom:1rem}.small-muted{font-size:.8rem;color:#817b73;line-height:1.65}.metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin:1.3rem 0}.metric{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:1rem}.metric b{display:block;color:#f0ebe0;font-family:'Syne';font-size:1.35rem}.metric span{font-family:'DM Mono';font-size:.62rem;color:#817b73;letter-spacing:.09em;text-transform:uppercase}.notice{font-family:'DM Mono';font-size:.68rem;color:#605850;text-align:center;margin-top:2.4rem;letter-spacing:.08em}
[data-testid="stDataFrame"] { border:1px solid rgba(255,140,50,.13); border-radius:10px; }
.stProgress > div > div > div > div { background:#ff8c32; }
@media (max-width: 768px) {
  .block-container { padding:1rem .85rem 2.5rem; }
  [data-testid="stHorizontalBlock"] { flex-direction:column!important; gap:1rem!important; }
  [data-testid="column"] { width:100%!important; flex:1 1 100%!important; min-width:0!important; }
  [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] { overflow-x:hidden!important; }
  .hero { padding:2rem 0 1.4rem; }
  .hero h1 { font-size:clamp(2rem,11vw,2.55rem); line-height:1.05; word-break:normal; }
  .hero-sub { font-size:.92rem; }
  .input-card,.result-panel,.report-panel { padding:1.15rem; border-radius:12px; }
  .metric-row { grid-template-columns:repeat(2,1fr); gap:.55rem; }
  .metric { padding:.75rem; }
  .metric b { font-size:1.05rem; }
  .step-card { padding:.9rem 1rem; }
  .step-title { font-size:.82rem; }
  .step-desc { font-size:.7rem; }
  .section-heading { margin-top:1.25rem; }
  [data-testid="stDataFrame"] { width:100%!important; max-width:100vw!important; overflow-x:auto!important; font-size:.72rem; }
  [data-testid="stDataFrame"] > div { max-width:100%!important; }
  .stButton > button { min-height:46px; width:100%; }
  .stFileUploader section { padding:.8rem!important; }
  .step-status { font-size:.55rem; }
  .notice { font-size:.56rem; line-height:1.6; }
}
</style>
""", unsafe_allow_html=True)

# -------------------- configuration and model helpers --------------------
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name)
        if value is not None:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)

def budget(name: str, default: int) -> int:
    try:
        return max(32, int(secret(name, str(default))))
    except ValueError:
        return default

def config() -> dict[str, Any]:
    return {"model":secret("GROQ_MODEL", DEFAULT_MODEL), "schema_tokens":budget("SCHEMA_MAX_TOKENS", 180), "eval_tokens":budget("EVAL_MAX_TOKENS", 420), "report_tokens":budget("REPORT_MAX_TOKENS", 220), "eval_batch_size":max(1, min(20, int(secret("EVAL_BATCH_SIZE", "8"))))}

def llm(api_key: str, model: str, max_tokens: int) -> ChatGroq:
    return ChatGroq(model=model, api_key=api_key, temperature=0, max_tokens=max_tokens, max_retries=2)

def json_object(text: str) -> dict[str, Any]:
    text = str(text).strip().replace("```json", "").replace("```", "").strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected an object")
    return value

# -------------------- flexible intake --------------------
def clean_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

def is_derived_column(value: Any) -> bool:
    """Exclude values generated by a previous run from score-column inference."""
    name = clean_key(value)
    derived = ("average", "avg", "mean", "total", "rank", "gap", "position", "priority", "level", "strength", "weakness", "action", "recommendation", "summary", "percentile", "classaverage")
    return any(term in name for term in derived)

def local_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Fallback mapping for arbitrary headers when the schema call fails."""
    work = df.copy().dropna(how="all").dropna(axis=1, how="all")
    cols = list(work.columns)
    keys = {c: clean_key(c) for c in cols}
    def find(words: list[str], used: set[Any] = set()) -> Any | None:
        return next((c for c in cols if c not in used and any(w in keys[c] for w in words)), None)
    student = find(["studentname","student","learner","pupil","candidate","name"])
    if student is None:
        non_num = [c for c in cols if not pd.api.types.is_numeric_dtype(pd.to_numeric(work[c], errors="coerce"))]
        student = non_num[0] if non_num else None
    roll = find(["rollno","rollnumber","roll","studentid","admissionno","registrationno","id"], {student} if student else set())
    attendance = find(["attendance","attend","present","presence"], {x for x in [student, roll] if x is not None})
    excluded = {x for x in [student, roll, attendance] if x is not None}
    numeric = [c for c in cols if c not in excluded and not is_derived_column(c) and pd.to_numeric(work[c], errors="coerce").notna().sum() > 0]
    return {"student_column":student, "roll_column":roll, "score_columns":numeric, "attendance_column":attendance, "ignore_columns":[c for c in cols if c not in set(numeric) | excluded]}

def infer_schema(df: pd.DataFrame, api_key: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Agent 1: infer meaning from arbitrary column names and sample values."""
    fallback = local_schema(df)
    if not api_key:
        return fallback
    sample = df.head(5).fillna("").astype(str).to_dict(orient="records")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data-intake agent. Infer the meaning of an unknown student-performance table. Return JSON only with exactly: student_column (string or null), roll_column (string or null), score_columns (array of existing column names containing grades, marks, scores, or measurable performance), attendance_column (string or null), ignore_columns (array). Prefer numeric performance columns and never invent column names. Any numeric column can be a score if it is not attendance, an ID, a date, or a total. Keep the response short."""),
        ("human", "Columns: {columns}\nSample rows: {sample}"),
    ])
    try:
        result = json_object((prompt | llm(api_key, cfg["model"], cfg["schema_tokens"])).invoke({"columns":json.dumps(list(df.columns)), "sample":json.dumps(sample)}).content)
        valid = set(df.columns)
        scores = [x for x in result.get("score_columns", []) if x in valid and not is_derived_column(x)]
        student = result.get("student_column") if result.get("student_column") in valid else fallback["student_column"]
        roll = result.get("roll_column") if result.get("roll_column") in valid else fallback.get("roll_column")
        attendance = result.get("attendance_column") if result.get("attendance_column") in valid else fallback["attendance_column"]
        return {"student_column":student, "roll_column":roll, "score_columns":scores or fallback["score_columns"], "attendance_column":attendance, "ignore_columns":[x for x in result.get("ignore_columns", []) if x in valid]}
    except Exception:
        return fallback

def orient_if_transposed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    first = df.columns[0]
    labels = {clean_key(x) for x in df[first].dropna().tolist()}
    known = {"student","studentname","name","math","maths","science","english","attendance","score","marks","grade"}
    if len(labels & known) >= 2 and len(df.columns) > 2:
        return df.set_index(first).T.reset_index().rename(columns={"index":"student"})
    return df

def prepare_data(raw: pd.DataFrame, mapping: dict[str, Any]) -> pd.DataFrame:
    """Convert any understood table into an internal long format."""
    df = orient_if_transposed(raw.copy()).dropna(how="all").dropna(axis=1, how="all")
    # If orientation changed, infer again from the now-normal row layout.
    if mapping.get("student_column") not in df.columns:
        mapping = local_schema(df)
    student_col = mapping.get("student_column")
    score_cols = [c for c in mapping.get("score_columns", []) if c in df.columns and c != student_col]
    if not score_cols:
        raise ValueError("No measurable score columns were detected. Include numeric marks, grades, scores, or performance columns.")
    if student_col and student_col in df.columns:
        df["student"] = df[student_col].astype(str).str.strip()
    else:
        df["student"] = [f"Record {i+1}" for i in range(len(df))]
    roll_col = mapping.get("roll_column")
    df["roll_no"] = df[roll_col].astype(str).str.strip() if roll_col in df.columns else df["student"]
    for col in score_cols:
        df[f"score::{col}"] = pd.to_numeric(df[col], errors="coerce")
    score_fields = [f"score::{c}" for c in score_cols]
    df["average"] = df[score_fields].mean(axis=1).round(2)
    attendance_col = mapping.get("attendance_column")
    df["attendance"] = pd.to_numeric(df[attendance_col], errors="coerce") if attendance_col in df.columns else 100.0
    df["attendance"] = df["attendance"].fillna(100).clip(0,100)
    df = df.dropna(subset=["average"]).copy()
    df["class_average"] = round(float(df["average"].mean()),2)
    df["rank"] = df["average"].rank(method="min", ascending=False).astype(int)
    df.attrs["score_columns"] = score_cols
    df.attrs["mapping"] = mapping
    return df.reset_index(drop=True)

# -------------------- agents --------------------
def evaluate_rows(df: pd.DataFrame, api_key: str, cfg: dict[str, Any], progress=None) -> list[dict[str, Any]]:
    """Performance Agent: evaluate records in batches instead of one API call per student."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Performance Agent. Return JSON only as an object with an evaluations array. For every input record, include its exact student value plus level (Excellent, Good, Needs Improvement, or At Risk), strengths (max 1 short string), weaknesses (max 1 short string), and recommendation (one short practical action). Return exactly one evaluation per input record and do not invent students. Use only supplied data."),
        ("human", "Student records: {records}"),
    ])
    chain = prompt | llm(api_key, cfg["model"], cfg["eval_tokens"])
    score_cols = df.attrs.get("score_columns", [])
    records=[]
    for row in df.to_dict(orient="records"):
        records.append({"student":row["student"],"roll_no":row.get("roll_no",row["student"]),"scores":{str(c):row.get(f"score::{c}") for c in score_cols},"average":row["average"],"attendance":row["attendance"]})
    output=[]; batch_size=cfg.get("eval_batch_size",8)
    for start in range(0, len(records), batch_size):
        batch=records[start:start+batch_size]
        try:
            parsed=json_object(chain.invoke({"records":json.dumps(batch)}).content)
            evaluations=parsed.get("evaluations",[])
            by_student={str(item.get("student")):item for item in evaluations if isinstance(item,dict)}
        except Exception:
            by_student={}
        for record in batch:
            result=by_student.get(str(record["student"]),{"level":"Needs Improvement","strengths":[],"weaknesses":[],"recommendation":"Review this learner's results with the teacher."})
            output.append({**record,**result})
        done=min(start+len(batch),len(records))
        if progress: progress.progress(.35 + .25*done/len(records), text=f"Performance Agent · batch {start//batch_size+1} complete ({done}/{len(records)} learners)")
    return output

def benchmark_rows(rows: list[dict[str, Any]], class_average: float, api_key: str, cfg: dict[str, Any], progress=None) -> list[dict[str, Any]]:
    """Benchmark Agent: deterministic comparison; no Groq call and no token cost."""
    output=[]
    for i, item in enumerate(rows,1):
        gap=round(float(item["average"])-class_average,2)
        if gap >= 5: position, priority = "Above class average", "Low"
        elif gap <= -5: position, priority = "Below class average", "High"
        else: position, priority = "Near class average", "Medium"
        output.append({**item,"gap":gap,"position":position,"gap_summary":f"{abs(gap):.2f} points {'above' if gap >= 0 else 'below'} the class average.","priority":priority})
        if progress: progress.progress(.6 + .18*i/len(rows), text=f"Benchmark Agent · local comparison {i} of {len(rows)}")
    return output

def final_report(rows: list[dict[str, Any]], class_average: float, api_key: str, cfg: dict[str, Any]) -> dict[str, Any]:
    prompt=ChatPromptTemplate.from_messages([("system", "You are the Report Agent. Return JSON only with summary (2 sentences max), class_strengths (array max 3), interventions (array max 3), and disclaimer (one sentence). Use only supplied facts and keep the report concise."),("human","Class average: {average}\nRows: {rows}")])
    compact=[{k:r.get(k) for k in ["student","average","attendance","level","position","priority","gap","recommendation"]} for r in rows]
    try: report=json_object((prompt | llm(api_key,cfg["model"],cfg["report_tokens"])).invoke({"average":class_average,"rows":json.dumps(compact)}).content)
    except Exception: report={"summary":"Review the student insights below.","class_strengths":[],"interventions":[],"disclaimer":"This is decision support and should be reviewed by a qualified teacher."}
    return {"class_average":class_average,"students":rows,"report":report}

# -------------------- UX helpers --------------------
STEPS=[("01","Intake Agent","Understands any rows, columns, headers, and data types"),("02","Calculation Agent","Computes scores, averages, ranks, and trends"),("03","Performance Agent","Finds strengths, weaknesses, and performance levels"),("04","Benchmark Agent","Compares every learner with the class"),("05","Report Agent","Creates a clear executive summary and next steps")]
def step_card(num: str, title: str, state: str, desc: str):
    label, cls = {"waiting":("WAITING","status-waiting"),"running":("● RUNNING","status-running"),"done":("✓ DONE","status-done")}.get(state,("WAITING","status-waiting"))
    card={"running":"active","done":"done"}.get(state,"")
    st.markdown(f'<div class="step-card {card}"><div class="step-header"><span class="step-num">{num}</span><span class="step-title">{title}</span><span class="step-status {cls}">{label}</span></div><div class="step-desc">{desc}</div></div>',unsafe_allow_html=True)
def render_steps(placeholder, states: dict[str,str]):
    with placeholder.container():
        for num,title,desc in STEPS: step_card(num,title,states.get(num,"waiting"),desc)

def render_activity(trace: list[dict[str, Any]]) -> None:
    """Show human-readable evidence of what each completed agent produced."""
    st.markdown('<div class="divider"></div><div class="section-heading">Agent activity report</div>', unsafe_allow_html=True)
    for item in trace:
        details = "".join(f"<li>{line}</li>" for line in item.get("details", []))
        st.markdown(f'<div class="result-panel"><div class="panel-label">{item["number"]} · {item["name"]} · COMPLETE</div><div class="small-muted">{item["summary"]}</div><ul>{details}</ul></div>', unsafe_allow_html=True)

def demo_data():
    return pd.DataFrame({"Learner Name":["Asha","Ben","Chitra","Dev","Elena","Farhan"],"Maths Marks":[88,62,95,48,76,69],"Science Score":[91,70,89,55,81,73],"English Grade":[84,66,93,51,79,64],"Attendance Rate":[96,82,98,68,91,87]})

def read_file(uploaded) -> pd.DataFrame:
    """Read common data-bearing files before Agent 01 infers their schema."""
    suffix = uploaded.name.lower().rsplit(".", 1)[-1] if "." in uploaded.name else ""
    payload = uploaded.getvalue()
    stream = BytesIO(payload)
    if suffix in {"csv", "tsv", "txt"}:
        if suffix == "tsv":
            return pd.read_csv(stream, sep="\t")
        # sep=None lets pandas detect comma, semicolon, pipe, or tab layouts.
        return pd.read_csv(stream, sep=None, engine="python")
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(stream)
    if suffix == "json":
        value = json.loads(payload.decode("utf-8-sig"))
        if isinstance(value, dict):
            value = value.get("data", value.get("records", value))
        return pd.json_normalize(value)
    if suffix == "parquet":
        return pd.read_parquet(stream)
    if suffix == "pdf":
        import pdfplumber
        tables = []
        with pdfplumber.open(stream) as pdf:
            for page in pdf.pages:
                tables.extend(page.extract_tables() or [])
        if not tables:
            raise ValueError("The PDF was readable, but no table was detected. Export the student table as CSV/XLSX or provide a PDF with a real table.")
        header, *rows = max(tables, key=len)
        return pd.DataFrame(rows, columns=header)
    if suffix == "docx":
        from docx import Document
        document = Document(stream)
        tables = []
        for table in document.tables:
            rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            if len(rows) >= 2:
                tables.append(pd.DataFrame(rows[1:], columns=rows[0]))
        if not tables:
            raise ValueError("The DOCX was readable, but no table was detected. Add the student data as a table or export it as CSV/XLSX.")
        return pd.concat(tables, ignore_index=True)
    raise ValueError("Unsupported file type. Use CSV, TSV, TXT, XLSX, XLS, JSON, Parquet, PDF, or DOCX.")

# -------------------- app --------------------
if "analysis" not in st.session_state: st.session_state.analysis=None
if "schema" not in st.session_state: st.session_state.schema=None
if "agent_trace" not in st.session_state: st.session_state.agent_trace=[]

st.markdown('<div class="hero"><div class="hero-eyebrow">Multi-Agent Student Intelligence</div><h1>Student Performance <span>Analyzer</span></h1><p class="hero-sub">Five specialized agents collaborate — understanding, calculating, evaluating, benchmarking, and reporting — to turn any student data into clear performance insight.</p></div><div class="divider"></div>',unsafe_allow_html=True)

col_input, spacer, col_pipeline = st.columns([5, .45, 4])
with col_input:
    st.markdown('<div class="input-card">',unsafe_allow_html=True)
    uploaded=st.file_uploader("Student performance file",type=["csv","tsv","txt","xlsx","xls","json","parquet","pdf","docx"],help="Any headers and most row/column layouts are supported.")
    use_demo=st.checkbox("Use demo data",value=uploaded is None)
    if uploaded is not None:
        try: raw=read_file(uploaded); source=uploaded.name
        except Exception as exc: st.error(f"Could not read file: {exc}"); raw=None; source=""
    elif use_demo: raw=demo_data(); source="Demo dataset"
    else: raw=None; source=""
    if raw is not None:
        st.markdown(f'<div class="small-muted" style="margin-top:.8rem">{len(raw):,} raw rows · {len(raw.columns):,} columns detected · schema will be inferred by Agent 01</div>',unsafe_allow_html=True)
    # Groq credentials are intentionally not exposed in the website UI.
    # Configure GROQ_API_KEY in the server environment or Streamlit secrets.
    api_key=secret("GROQ_API_KEY")
    run_btn=st.button("⚡  Run Performance Pipeline",use_container_width=True,disabled=not bool(raw is not None))
    st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="small-muted">The intake agent does not require fixed names like math or english. It reads CSV, Excel, JSON, text, Parquet, PDF tables, and DOCX tables, then uses headers, sample values, numeric patterns, and context to decide what each field means.</div>',unsafe_allow_html=True)

with col_pipeline:
    st.markdown('<div class="section-heading">Agent pipeline</div>',unsafe_allow_html=True)
    st.markdown('<div class="small-muted" style="margin-bottom:1rem">Sequential execution · one agent at a time · strict Groq budgets</div>',unsafe_allow_html=True)
    pipe=st.empty(); render_steps(pipe,st.session_state.get("pipeline_states",{}))

if raw is not None and not run_btn:
    with st.expander("Preview detected source data",expanded=False): st.dataframe(raw.head(20),use_container_width=True,hide_index=True)

if run_btn:
    if not api_key:
        st.error("The Run button is working, but GROQ_API_KEY is not configured on the server. Add it to the server environment or Streamlit secrets, then refresh the app.")
        st.stop()
    cfg=config(); states={num:"waiting" for num,_,_ in STEPS}; st.session_state.analysis=None; st.session_state.agent_trace=[]
    try:
        states["01"]="running"; render_steps(pipe,states)
        progress=st.progress(.04,text="Intake Agent · understanding your data")
        mapping=infer_schema(raw,api_key,cfg); df=prepare_data(raw,mapping); st.session_state.schema=mapping
        missing_total=int(raw.isna().sum().sum()); invalid_total=sum(int(raw[col].notna().sum()-pd.to_numeric(raw[col],errors="coerce").notna().sum()) for col in mapping.get("score_columns",[]) if col in raw.columns)
        st.session_state.agent_trace.append({"number":"01","name":"Intake Agent","summary":"Understood the uploaded table and converted it into a student-performance schema.","details":[f"Read {len(raw):,} rows and {len(raw.columns):,} source columns.",f"Identified {len(mapping.get('score_columns',[]))} performance columns: {', '.join(map(str,mapping.get('score_columns',[])))}.",f"Student identity column: {mapping.get('student_column') or 'generated record labels'}; roll column: {mapping.get('roll_column') or 'not supplied'}.",f"Attendance column: {mapping.get('attendance_column') or 'not supplied; neutral 100% fallback used'}.",f"Data quality: {missing_total} missing values, {invalid_total} wrong-format numeric values, and {len(df)} usable records."]})
        states["01"]="done"; states["02"]="running"; render_steps(pipe,states)
        progress.progress(.2,text="Calculation Agent · computing performance statistics")
        # Calculation is deterministic and intentionally consumes no LLM tokens.
        df.attrs["score_columns"]=df.attrs.get("score_columns",mapping.get("score_columns",[]))
        above=int((df["average"]>df["class_average"]).sum()); below=int((df["average"]<df["class_average"]).sum()); equal=len(df)-above-below
        st.session_state.agent_trace.append({"number":"02","name":"Calculation Agent","summary":"Calculated local statistics without spending Groq tokens.","details":[f"Computed an average across {len(df.attrs.get('score_columns',[]))} detected score columns for every learner.",f"Class average: {float(df['class_average'].iloc[0]):.2f}% · highest: {float(df['average'].max()):.2f}% · lowest: {float(df['average'].min()):.2f}%.",f"Class comparison: {above} above average, {below} below average, {equal} exactly at average.",f"Ranked {len(df)} learners and calculated attendance averages."]})
        states["02"]="done"; states["03"]="running"; render_steps(pipe,states)
        progress.progress(.3,text="Performance Agent · evaluating each learner")
        evaluated=evaluate_rows(df,api_key,cfg,progress)
        levels=pd.Series([r.get("level","Unknown") for r in evaluated]).value_counts().to_dict()
        good_count=int((df["average"]>=df["class_average"]).sum()); low_count=len(df)-good_count
        st.session_state.agent_trace.append({"number":"03","name":"Performance Agent","summary":"Reviewed each learner's scores and generated strengths, weaknesses, and a practical action.","details":[f"Evaluated {len(evaluated)} learner records with compact batched Groq prompts.","Performance levels: " + ", ".join(f"{k} ({v})" for k,v in levels.items()) + ".",f"Identified {good_count} good-performing learners and {low_count} learners needing improvement.",f"Generated {sum(bool(r.get('recommendation')) for r in evaluated)} learner recommendations."]})
        states["03"]="done"; states["04"]="running"; render_steps(pipe,states)
        compared=benchmark_rows(evaluated,float(df["class_average"].iloc[0]),api_key,cfg,progress)
        positions=pd.Series([r.get("position","Unknown") for r in compared]).value_counts().to_dict()
        st.session_state.agent_trace.append({"number":"04","name":"Benchmark Agent","summary":"Compared every learner with the computed class average and assigned a priority.","details":[f"Benchmarked {len(compared)} learners against {float(df['class_average'].iloc[0]):.2f}%.","Class positions: " + ", ".join(f"{k} ({v})" for k,v in positions.items()) + ".","Calculated a numeric gap for every learner."]})
        states["04"]="done"; states["05"]="running"; render_steps(pipe,states)
        progress.progress(.82,text="Report Agent · writing the class overview")
        result=final_report(compared,float(df["class_average"].iloc[0]),api_key,cfg)
        st.session_state.agent_trace.append({"number":"05","name":"Report Agent","summary":"Synthesized the completed analysis into an executive class report.","details":["Created a concise class summary.",f"Generated {len(result['report'].get('class_strengths',[]))} class strengths and {len(result['report'].get('interventions',[]))} recommended interventions.","Prepared the report and student insights for download."]})
        states["05"]="done"; render_steps(pipe,states); st.session_state.pipeline_states=states; st.session_state.analysis=result; st.session_state.cleaned=df
        progress.progress(1.0,text="All agents complete · insight report ready")
    except Exception as exc:
        st.error(f"Pipeline stopped: {exc}")

if st.session_state.analysis is not None:
    result=st.session_state.analysis; report=result["report"]; view=pd.DataFrame(result["students"]); df=st.session_state.cleaned
    render_activity(st.session_state.get("agent_trace", []))
    st.markdown('<div class="divider"></div><div class="section-heading">Performance overview</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="metric-row"><div class="metric"><span>Class average</span><b>{result["class_average"]:.1f}%</b></div><div class="metric"><span>Attendance average</span><b>{view["attendance"].mean():.1f}%</b></div><div class="metric"><span>Top performer</span><b>{view.loc[view.average.idxmax(),"student"]}</b></div><div class="metric"><span>Records analyzed</span><b>{len(view)}</b></div></div>',unsafe_allow_html=True)
    a,b=st.columns([1.2,1])
    with a: st.markdown(f'<div class="report-panel"><div class="panel-label">Executive report</div><div class="small-muted">{report.get("summary","")}</div><br><b>Class strengths</b><ul>{"".join(f"<li>{x}</li>" for x in report.get("class_strengths",[]))}</ul></div>',unsafe_allow_html=True)
    with b: st.markdown(f'<div class="result-panel"><div class="panel-label">Recommended next steps</div><ul>{"".join(f"<li>{x}</li>" for x in report.get("interventions",[]))}</ul><div class="small-muted">{report.get("disclaimer","")}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Student insights</div>',unsafe_allow_html=True)
    global_search=st.text_input("Search all student outputs by name or roll number",placeholder="Search name or roll number…",label_visibility="collapsed",key="global_student_search")
    if global_search.strip():
        query=global_search.strip()
        mask=view["student"].astype(str).str.contains(query,case=False,na=False) | view.get("roll_no",view["student"]).astype(str).str.contains(query,case=False,na=False)
        filtered_view=view[mask].copy()
    else:
        filtered_view=view.copy()
    st.caption(f"Search results: showing {len(filtered_view)} of {len(view)} students across the output tables.")
    ranked=filtered_view.sort_values("average",ascending=False).reset_index(drop=True)
    best, support=st.columns(2)
    with best:
        st.markdown('<div class="result-panel"><div class="panel-label">Best performers</div><div class="small-muted">Learners furthest above the class average.</div></div>',unsafe_allow_html=True)
        st.dataframe(ranked.head(min(5,len(ranked)))[["student","roll_no","average","gap","position"]].rename(columns={"student":"Student","roll_no":"Roll no.","average":"Average","gap":"Gap vs class","position":"Position"}),use_container_width=True,hide_index=True)
    with support:
        st.markdown('<div class="result-panel"><div class="panel-label">Priority support</div><div class="small-muted">Learners furthest below the class average.</div></div>',unsafe_allow_html=True)
        st.dataframe(ranked.tail(min(5,len(ranked))).sort_values("average")[["student","roll_no","average","gap","priority"]].rename(columns={"student":"Student","roll_no":"Roll no.","average":"Average","gap":"Gap vs class","priority":"Priority"}),use_container_width=True,hide_index=True)
    if len(filtered_view)==0:
        st.warning("No student matched that name or roll number.")
    st.markdown('<div class="section-heading">Detailed student table</div>',unsafe_allow_html=True)
    table_cols=["student","roll_no","average","attendance","level","position","gap","priority","recommendation"]
    st.dataframe(filtered_view[table_cols].rename(columns={"student":"Student","roll_no":"Roll no.","average":"Average","attendance":"Attendance","level":"Level","position":"Class position","gap":"Gap vs class","priority":"Priority","recommendation":"Recommended action"}),use_container_width=True,hide_index=True)
    chart_cols=[c for c in df.attrs.get("score_columns",[]) if f"score::{c}" in df.columns]
    st.markdown('<div class="section-heading">Performance distribution</div>',unsafe_allow_html=True)
    if "bell_curve_open" not in st.session_state: st.session_state.bell_curve_open=False
    if st.button("View Bell Curve",use_container_width=False): st.session_state.bell_curve_open=True
    if st.session_state.bell_curve_open:
        curve_subject=st.selectbox("Choose a distribution",["Overall average"]+list(map(str,chart_cols)),key="curve_subject")
        curve_search=st.text_input("Highlight student by name or roll number",placeholder="Search name or roll number…",key="curve_search")
        if curve_subject=="Overall average":
            curve_scores=view["average"].astype(float); curve_average=float(result["class_average"])
        else:
            curve_scores=df[f"score::{curve_subject}"].astype(float); curve_average=float(curve_scores.mean())
        curve_data=pd.DataFrame({"student":view["student"].astype(str),"roll_no":view.get("roll_no",view["student"]).astype(str),"score":curve_scores.values})
        curve_data["selected"]=False
        selected=None
        if curve_search.strip():
            matches=curve_data[curve_data["student"].str.contains(curve_search.strip(),case=False,na=False) | curve_data["roll_no"].str.contains(curve_search.strip(),case=False,na=False)]
            if not matches.empty:
                selected=matches.index[0]; curve_data.loc[selected,"selected"]=True
        density=alt.Chart(curve_data).transform_density("score",as_=["score","density"],extent=[0,100],steps=80).mark_area(opacity=.22,color="#ff8c32").encode(x=alt.X("score:Q",title="Score",scale=alt.Scale(domain=[0,100])),y=alt.Y("density:Q",title="Distribution"))
        ticks=alt.Chart(curve_data).mark_tick(color="#c9c0b6",thickness=2,size=35).encode(x=alt.X("score:Q",scale=alt.Scale(domain=[0,100])),y=alt.value(0),tooltip=["student","roll_no","score"])
        selected_tick=alt.Chart(curve_data[curve_data["selected"]]).mark_point(color="#ff8c32",filled=True,size=180,shape="diamond").encode(x="score:Q",y=alt.value(0),tooltip=["student","roll_no","score"])
        avg_rule=alt.Chart(pd.DataFrame({"class_average":[curve_average]})).mark_rule(color="#f0ebe0",strokeDash=[5,4]).encode(x="class_average:Q")
        st.altair_chart(density+ticks+selected_tick+avg_rule,use_container_width=True)
        st.caption(f"Class average for {curve_subject}: {curve_average:.2f} · {len(curve_data)} students represented.")
        if selected is not None:
            selected_row=curve_data.loc[selected]; gap=round(float(selected_row["score"])-curve_average,2)
            st.success(f"Selected: {selected_row['student']} · Roll no. {selected_row['roll_no']} · Score {selected_row['score']:.2f} · Difference from average {gap:+.2f}")
    with st.expander("View cleaned and understood data"): st.dataframe(df,use_container_width=True,hide_index=True)
    excel_buffer=BytesIO()
    with pd.ExcelWriter(excel_buffer,engine="openpyxl") as writer: view.to_excel(writer,index=False,sheet_name="Student Insights")
    d1,d2=st.columns(2)
    with d1: st.download_button("Download Detailed CSV",view.to_csv(index=False).encode(),"student_insights_detailed.csv","text/csv",use_container_width=True)
    with d2: st.download_button("Download Detailed Excel",excel_buffer.getvalue(),"student_insights_detailed.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)

st.markdown('<div class="notice">STUDENT PERFORMANCE ANALYZER · LOCAL CALCULATIONS · REVIEW WITH A QUALIFIED TEACHER</div>',unsafe_allow_html=True)
