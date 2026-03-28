"""
SchemaGain — JSON Schema Variant Evaluator for LLM Agents
A 3-column Streamlit app to measure the impact of JSON schema changes on agent performance.
"""

import streamlit as st
import json
import time
import traceback
from datetime import datetime
from openai import OpenAI
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="SchemaGain — JSON Schema Evaluator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;700&display=swap');

    .stApp {
        font-family: 'DM Sans', sans-serif;
    }

    .main-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .subtitle {
        color: #888;
        font-size: 0.9rem;
        margin-top: -8px;
        margin-bottom: 20px;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #334;
        border-radius: 12px;
        padding: 16px;
        margin: 6px 0;
        text-align: center;
    }

    .metric-card h3 {
        color: #aab;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0;
    }

    .metric-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #667eea;
        margin: 4px 0;
    }

    .schema-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 8px;
        font-weight: 600;
    }

    .recommendation-box {
        background: linear-gradient(135deg, #0d1117, #161b22);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 16px;
        margin: 10px 0;
        font-size: 0.9rem;
    }

    .chat-msg-assistant {
        background: #1a1a2e;
        border: 1px solid #334;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
    }

    .chat-msg-user {
        background: #16213e;
        border: 1px solid #445;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
        text-align: right;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #334;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────
@dataclass
class EvalResult:
    schema_name: str
    schema_json: dict
    # Efficiency
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    # Coverage
    coverage_score: float = 0.0
    missing_fields: list = field(default_factory=list)
    extra_fields: list = field(default_factory=list)
    schema_valid: bool = True
    # Quality
    quality_score: float = 0.0
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    relevance_score: float = 0.0
    quality_reasoning: str = ""
    # Raw
    raw_output: str = ""
    parsed_output: dict = field(default_factory=dict)
    error: str = ""


# ─────────────────────────────────────────────────
# Session State Initializers
# ─────────────────────────────────────────────────
if "schemas" not in st.session_state:
    st.session_state.schemas = {
        "Variant A (Flat)": json.dumps({
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The direct answer"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "source": {"type": "string"}
            },
            "required": ["answer", "confidence"]
        }, indent=2),
        "Variant B (Nested)": json.dumps({
            "type": "object",
            "properties": {
                "response": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["answer", "reasoning", "confidence"]
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "tokens_used": {"type": "integer"}
                    }
                }
            },
            "required": ["response"]
        }, indent=2)
    }

if "eval_results" not in st.session_state:
    st.session_state.eval_results = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "test_prompt" not in st.session_state:
    st.session_state.test_prompt = "What are the key benefits of microservices architecture over monolithic systems? Provide a structured technical answer conforming to this schema:\n\n{{schema}}"

if "eval_criteria" not in st.session_state:
    st.session_state.eval_criteria = [
        "Accuracy: Is the factual content correct?",
        "Completeness: Does the response cover all aspects of the question?",
        "Relevance: Is the response focused on the question asked?",
        "Reasoning Quality: Does the response show logical reasoning steps?",
    ]

if "num_trials" not in st.session_state:
    st.session_state.num_trials = 3

if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-4o-mini"

if "judge_model" not in st.session_state:
    st.session_state.judge_model = "gpt-4o-mini"

# ─────────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title">🔬 SchemaGain</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Measure the impact of JSON schema changes on your LLM agents</div>', unsafe_allow_html=True)
    st.divider()

    api_key = st.text_input("🔑 OpenAI API Key", type="password", placeholder="sk-...")

    st.markdown("#### ⚙️ Model Settings")
    model_name = st.selectbox(
        "Agent Model (generates output)",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1", "gpt-3.5-turbo"],
        index=0,
        key="model_select"
    )
    st.session_state.model_name = model_name

    judge_model = st.selectbox(
        "Judge Model (evaluates quality)",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
        index=0,
        key="judge_select"
    )
    st.session_state.judge_model = judge_model

    num_trials = st.slider("Trials per schema", 1, 10, 3)
    st.session_state.num_trials = num_trials

    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.1)

    st.divider()
    st.markdown("#### 📝 Test Prompt")
    st.caption("Use `{{schema}}` to inject the JSON schema variant.")
    st.session_state.test_prompt = st.text_area(
        "The prompt sent to the agent with each schema variant",
        value=st.session_state.test_prompt,
        height=120,
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("""
    <div style="color:#666; font-size:0.75rem;">
    Built for evaluating how JSON schema structure<br>
    affects LLM agent performance across<br>
    <b>Efficiency · Coverage · Quality</b>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# Helper: OpenAI calls
# ─────────────────────────────────────────────────
def get_client():
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def run_agent_with_schema(client, schema_json: dict, prompt: str, model: str, temp: float):
    """Run an LLM call with a given JSON schema and measure efficiency."""
    schema_str = json.dumps(schema_json, indent=2)
    
    if "{{schema}}" in prompt:
        # User-defined placeholder injection
        final_user_prompt = prompt.replace("{{schema}}", schema_str)
        system_prompt = "You are a helpful AI agent. You MUST respond with valid JSON. Respond ONLY with valid JSON. No markdown, no explanation outside the JSON."
    else:
        # Default behavior: inject into system prompt
        final_user_prompt = prompt
        system_prompt = f"""You are a helpful AI agent. You MUST respond with valid JSON that conforms to the following JSON schema:

```json
{schema_str}
```

Respond ONLY with valid JSON. No markdown, no explanation outside the JSON."""

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_user_prompt}
        ],
        temperature=temp,
        response_format={"type": "json_object"},
    )
    latency = (time.time() - start) * 1000

    content = response.choices[0].message.content
    usage = response.usage

    return {
        "content": content,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "latency_ms": latency,
    }


def evaluate_coverage(schema_json: dict, output_json: dict) -> dict:
    """Check if output covers all required fields in the schema."""
    def get_required_paths(schema, prefix=""):
        paths = []
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for field_name, field_schema in props.items():
            path = f"{prefix}.{field_name}" if prefix else field_name
            if field_name in required:
                paths.append(path)
            if field_schema.get("type") == "object":
                paths.extend(get_required_paths(field_schema, path))
        return paths

    def check_path(obj, path):
        parts = path.split(".")
        current = obj
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        return True

    required_paths = get_required_paths(schema_json)
    present = [p for p in required_paths if check_path(output_json, p)]
    missing = [p for p in required_paths if not check_path(output_json, p)]

    # Find extra fields
    def get_all_paths(obj, prefix=""):
        paths = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else k
                paths.append(path)
                paths.extend(get_all_paths(v, path))
        return paths

    def get_schema_paths(schema, prefix=""):
        paths = []
        props = schema.get("properties", {})
        for field_name, field_schema in props.items():
            path = f"{prefix}.{field_name}" if prefix else field_name
            paths.append(path)
            if field_schema.get("type") == "object":
                paths.extend(get_schema_paths(field_schema, path))
        return paths

    output_paths = get_all_paths(output_json)
    schema_paths = get_schema_paths(schema_json)
    extra = [p for p in output_paths if p not in schema_paths]

    score = len(present) / len(required_paths) if required_paths else 1.0

    return {
        "coverage_score": score,
        "missing_fields": missing,
        "extra_fields": extra,
        "required_count": len(required_paths),
        "present_count": len(present),
    }


def judge_quality(client, schema_json: dict, output_json: dict, prompt: str,
                  criteria: list, model: str) -> dict:
    """Use LLM-as-judge to evaluate quality."""
    criteria_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(criteria))

    judge_prompt = f"""You are an expert evaluator. Given:

1. A USER PROMPT that was sent to an LLM agent:
\"\"\"{prompt}\"\"\"

2. The JSON SCHEMA the agent was instructed to follow:
```json
{json.dumps(schema_json, indent=2)}
```

3. The agent's ACTUAL OUTPUT:
```json
{json.dumps(output_json, indent=2)}
```

Evaluate the output on these criteria:
{criteria_text}

Respond with ONLY valid JSON in this exact format:
{{
  "accuracy_score": <0.0-1.0>,
  "completeness_score": <0.0-1.0>,
  "relevance_score": <0.0-1.0>,
  "overall_quality_score": <0.0-1.0>,
  "reasoning": "<2-3 sentence explanation of strengths and weaknesses>",
  "criterion_scores": {{<criterion_name>: <0.0-1.0> for each criterion}}
}}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    return result


def suggest_criteria(client, schemas: dict, existing_criteria: list, model: str) -> str:
    """AI-powered suggestion of evaluation criteria based on schema analysis."""
    schemas_text = ""
    for name, schema_str in schemas.items():
        schemas_text += f"\n--- {name} ---\n{schema_str}\n"

    existing_text = "\n".join(f"- {c}" for c in existing_criteria)

    prompt = f"""You are an expert in LLM evaluation. Analyze these JSON schema variants and suggest additional evaluation criteria.

SCHEMA VARIANTS:
{schemas_text}

EXISTING CRITERIA:
{existing_text}

Based on the structural differences between these schemas (nesting depth, required fields, field types, descriptions, constraints), suggest 3-5 NEW evaluation criteria that would help measure the impact of these schema changes.

Focus on:
1. How schema structure affects reasoning quality
2. How nesting depth impacts information completeness
3. Whether field descriptions improve output specificity
4. Token efficiency implications
5. Error-proneness of each structure

Format each suggestion as a clear, measurable criterion. Explain WHY each criterion matters for comparing these specific schemas."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content


def chat_with_advisor(client, schemas: dict, results: list, criteria: list,
                      chat_history: list, user_msg: str, model: str) -> str:
    """Chat with the evaluation advisor bot."""
    # Build context
    schemas_text = ""
    for name, schema_str in schemas.items():
        schemas_text += f"\n--- {name} ---\n{schema_str}\n"

    results_summary = ""
    if results:
        for r in results:
            results_summary += f"\n[{r.schema_name}] Quality={r.quality_score:.2f}, Coverage={r.coverage_score:.2f}, Tokens={r.total_tokens}, Latency={r.latency_ms:.0f}ms"

    criteria_text = "\n".join(f"- {c}" for c in criteria)

    system = f"""You are SchemaGain Advisor — an expert in evaluating how JSON schema design affects LLM agent performance.

You have deep knowledge of:
- The "Let Me Speak Freely?" (Tam et al., EMNLP 2024) findings on format restriction degradation
- JSONSchemaBench benchmarking methodology
- The SLOT framework for decoupling formatting from reasoning
- Token efficiency, constrained decoding, and schema complexity impacts
- Practical evaluation frameworks (Promptfoo, DeepEval)

CURRENT SCHEMAS BEING EVALUATED:
{schemas_text}

CURRENT EVALUATION CRITERIA:
{criteria_text}

{"EVALUATION RESULTS SO FAR:" + results_summary if results_summary else "No evaluation results yet."}

Help the user understand:
1. EFFICIENCY: Token cost, latency implications of each schema
2. COVERAGE: Whether schemas capture required fields adequately
3. QUALITY: How schema structure affects content accuracy and reasoning

Be specific, cite research findings, and give actionable recommendations. If asked about metrics, provide concrete formulas or thresholds."""

    messages = [{"role": "system", "content": system}]
    for msg in chat_history[-10:]:  # Keep last 10 messages for context
        messages.append(msg)
    messages.append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
    )
    return response.choices[0].message.content


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Rough cost estimation per model."""
    pricing = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
        "gpt-4.1": (2.00, 8.00),
        "gpt-3.5-turbo": (0.50, 1.50),
    }
    input_rate, output_rate = pricing.get(model, (1.0, 3.0))
    cost = (prompt_tokens / 1_000_000 * input_rate) + (completion_tokens / 1_000_000 * output_rate)
    return cost


# ─────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────
st.markdown('<div class="main-title">🔬 SchemaGain</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Measure the gain from changing JSON schemas passed to your LLM agents</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.1, 1.2, 1.3], gap="large")

# ═════════════════════════════════════════════════
# COLUMN 1 — Schema Variants
# ═════════════════════════════════════════════════
with col1:
    st.markdown("### 📋 Schema Variants")
    st.caption("Define multiple JSON schema variants to compare")

    # Add new schema
    with st.expander("➕ Add New Variant", expanded=False):
        new_name = st.text_input("Variant name", placeholder="e.g., Variant C (Minimal)")
        new_schema = st.text_area("JSON Schema", height=150, placeholder='{"type": "object", ...}')
        if st.button("Add Variant", use_container_width=True, type="primary"):
            if new_name and new_schema:
                try:
                    json.loads(new_schema)
                    st.session_state.schemas[new_name] = new_schema
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON. Please check your schema.")
            else:
                st.warning("Please provide both a name and schema.")

    # Display existing schemas
    schema_tabs = st.tabs(list(st.session_state.schemas.keys()))
    for i, (name, schema_str) in enumerate(st.session_state.schemas.items()):
        with schema_tabs[i]:
            colors = ["#667eea", "#f093fb", "#4fd1c5", "#f6ad55", "#fc8181"]
            color = colors[i % len(colors)]
            st.markdown(f'<span class="schema-label" style="background:{color}22;color:{color};border:1px solid {color}44;">{name}</span>', unsafe_allow_html=True)

            edited = st.text_area(
                f"Edit {name}",
                value=schema_str,
                height=250,
                key=f"schema_{i}",
                label_visibility="collapsed"
            )
            st.session_state.schemas[name] = edited

            # Schema stats
            try:
                parsed = json.loads(edited)
                props = parsed.get("properties", {})
                required = parsed.get("required", [])
                depth = _get_depth(parsed) if 'parsed' else 1

                c1, c2, c3 = st.columns(3)
                c1.metric("Fields", len(props))
                c2.metric("Required", len(required))
                c3.metric("Tokens", len(edited.split()))
            except:
                st.error("⚠️ Invalid JSON")

            col_a, col_b = st.columns(2)
            with col_b:
                if st.button("🗑️ Remove", key=f"remove_{i}", use_container_width=True):
                    del st.session_state.schemas[name]
                    st.rerun()

    # Run evaluation button
    st.divider()
    run_eval = st.button(
        "🚀 Run Evaluation",
        use_container_width=True,
        type="primary",
        disabled=not api_key
    )
    if not api_key:
        st.caption("⬅️ Enter your OpenAI API key in the sidebar to begin")


def _get_depth(schema, current=0):
    """Get max nesting depth of a JSON schema."""
    if not isinstance(schema, dict):
        return current
    max_d = current
    for v in schema.get("properties", {}).values():
        if isinstance(v, dict) and v.get("type") == "object":
            max_d = max(max_d, _get_depth(v, current + 1))
    return max_d


# ═════════════════════════════════════════════════
# COLUMN 2 — Evaluation Criteria + Chat
# ═════════════════════════════════════════════════
with col2:
    st.markdown("### 🎯 Evaluation & Advisor")

    tab_criteria, tab_chat = st.tabs(["📏 Criteria", "💬 Advisor Chat"])

    with tab_criteria:
        st.caption("Define what to measure. The LLM judge evaluates each schema output against these criteria.")

        # Editable criteria
        for i, criterion in enumerate(st.session_state.eval_criteria):
            c_col1, c_col2 = st.columns([5, 1])
            with c_col1:
                st.session_state.eval_criteria[i] = st.text_input(
                    f"Criterion {i+1}",
                    value=criterion,
                    key=f"crit_{i}",
                    label_visibility="collapsed"
                )
            with c_col2:
                if st.button("✕", key=f"del_crit_{i}"):
                    st.session_state.eval_criteria.pop(i)
                    st.rerun()

        # Add criterion
        col_add1, col_add2 = st.columns([3, 2])
        with col_add1:
            new_crit = st.text_input("New criterion", placeholder="e.g., Conciseness: Is the response focused?", label_visibility="collapsed")
        with col_add2:
            if st.button("➕ Add", use_container_width=True):
                if new_crit:
                    st.session_state.eval_criteria.append(new_crit)
                    st.rerun()

        st.divider()

        # AI Suggest button
        if st.button("🤖 AI: Suggest More Criteria", use_container_width=True, type="secondary", disabled=not api_key):
            with st.spinner("Analyzing schema differences..."):
                client = get_client()
                if client:
                    suggestions = suggest_criteria(
                        client, st.session_state.schemas,
                        st.session_state.eval_criteria,
                        st.session_state.judge_model
                    )
                    st.markdown("#### 💡 Suggested Criteria")
                    st.markdown(f'<div class="recommendation-box">{suggestions}</div>', unsafe_allow_html=True)

    with tab_chat:
        st.caption("Ask the SchemaGain Advisor about your evaluation strategy, metrics, or results.")

        # Chat display
        chat_container = st.container(height=400)
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div class="chat-msg-assistant">
                👋 I'm the SchemaGain Advisor. I can help you understand:
                <br><br>
                • <b>Efficiency</b> — token costs & latency across schema variants<br>
                • <b>Coverage</b> — whether your schemas capture required fields<br>
                • <b>Quality</b> — how schema structure affects reasoning & accuracy<br>
                <br>
                Ask me anything about your schemas or evaluation results!
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="chat-msg-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-msg-assistant">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

        # Chat input
        user_input = st.chat_input("Ask about schemas, metrics, or results...", disabled=not api_key)
        if user_input and api_key:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            client = get_client()
            if client:
                with st.spinner("Thinking..."):
                    reply = chat_with_advisor(
                        client, st.session_state.schemas,
                        st.session_state.eval_results,
                        st.session_state.eval_criteria,
                        st.session_state.chat_history,
                        user_input,
                        st.session_state.judge_model
                    )
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

        # Quick questions
        st.markdown("**Quick questions:**")
        quick_qs = [
            "Which schema will minimize token cost?",
            "How does nesting depth affect reasoning?",
            "What's the two-step approach for my Q&A agents?",
            "Compare my schema variants on coverage risk",
        ]
        qcols = st.columns(2)
        for qi, q in enumerate(quick_qs):
            with qcols[qi % 2]:
                if st.button(q, key=f"qq_{qi}", use_container_width=True, disabled=not api_key):
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    client = get_client()
                    if client:
                        reply = chat_with_advisor(
                            client, st.session_state.schemas,
                            st.session_state.eval_results,
                            st.session_state.eval_criteria,
                            st.session_state.chat_history,
                            q,
                            st.session_state.judge_model
                        )
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                        st.rerun()


# ═════════════════════════════════════════════════
# EVALUATION EXECUTION
# ═════════════════════════════════════════════════
if run_eval:
    client = get_client()
    if not client:
        st.error("Please provide a valid OpenAI API key.")
    elif len(st.session_state.schemas) < 1:
        st.error("Please add at least one schema variant.")
    else:
        st.session_state.eval_results = []
        progress_bar = st.progress(0)
        status = st.status("Running evaluation...", expanded=True)

        total_steps = len(st.session_state.schemas) * st.session_state.num_trials
        step = 0

        for schema_name, schema_str in st.session_state.schemas.items():
            try:
                schema_json = json.loads(schema_str)
            except json.JSONDecodeError:
                st.session_state.eval_results.append(EvalResult(
                    schema_name=schema_name,
                    schema_json={},
                    error=f"Invalid JSON schema for {schema_name}"
                ))
                continue

            trial_results = []
            for trial in range(st.session_state.num_trials):
                step += 1
                progress_bar.progress(step / total_steps)
                status.write(f"⏳ {schema_name} — Trial {trial+1}/{st.session_state.num_trials}")

                try:
                    # Run agent
                    agent_result = run_agent_with_schema(
                        client, schema_json, st.session_state.test_prompt,
                        st.session_state.model_name, temperature
                    )

                    # Parse output
                    try:
                        parsed = json.loads(agent_result["content"])
                    except:
                        parsed = {}

                    # Coverage
                    coverage = evaluate_coverage(schema_json, parsed)

                    # Quality (LLM judge)
                    quality = judge_quality(
                        client, schema_json, parsed,
                        st.session_state.test_prompt,
                        st.session_state.eval_criteria,
                        st.session_state.judge_model
                    )

                    result = EvalResult(
                        schema_name=schema_name,
                        schema_json=schema_json,
                        prompt_tokens=agent_result["prompt_tokens"],
                        completion_tokens=agent_result["completion_tokens"],
                        total_tokens=agent_result["total_tokens"],
                        latency_ms=agent_result["latency_ms"],
                        estimated_cost_usd=estimate_cost(
                            agent_result["prompt_tokens"],
                            agent_result["completion_tokens"],
                            st.session_state.model_name
                        ),
                        coverage_score=coverage["coverage_score"],
                        missing_fields=coverage["missing_fields"],
                        extra_fields=coverage["extra_fields"],
                        quality_score=quality.get("overall_quality_score", 0),
                        accuracy_score=quality.get("accuracy_score", 0),
                        completeness_score=quality.get("completeness_score", 0),
                        relevance_score=quality.get("relevance_score", 0),
                        quality_reasoning=quality.get("reasoning", ""),
                        raw_output=agent_result["content"],
                        parsed_output=parsed,
                    )
                    trial_results.append(result)

                except Exception as e:
                    trial_results.append(EvalResult(
                        schema_name=schema_name,
                        schema_json=schema_json,
                        error=str(e)
                    ))

            # Average the trials
            valid_trials = [t for t in trial_results if not t.error]
            if valid_trials:
                avg = EvalResult(
                    schema_name=schema_name,
                    schema_json=schema_json,
                    prompt_tokens=int(sum(t.prompt_tokens for t in valid_trials) / len(valid_trials)),
                    completion_tokens=int(sum(t.completion_tokens for t in valid_trials) / len(valid_trials)),
                    total_tokens=int(sum(t.total_tokens for t in valid_trials) / len(valid_trials)),
                    latency_ms=sum(t.latency_ms for t in valid_trials) / len(valid_trials),
                    estimated_cost_usd=sum(t.estimated_cost_usd for t in valid_trials) / len(valid_trials),
                    coverage_score=sum(t.coverage_score for t in valid_trials) / len(valid_trials),
                    missing_fields=valid_trials[-1].missing_fields,
                    extra_fields=valid_trials[-1].extra_fields,
                    quality_score=sum(t.quality_score for t in valid_trials) / len(valid_trials),
                    accuracy_score=sum(t.accuracy_score for t in valid_trials) / len(valid_trials),
                    completeness_score=sum(t.completeness_score for t in valid_trials) / len(valid_trials),
                    relevance_score=sum(t.relevance_score for t in valid_trials) / len(valid_trials),
                    quality_reasoning=valid_trials[-1].quality_reasoning,
                    raw_output=valid_trials[-1].raw_output,
                    parsed_output=valid_trials[-1].parsed_output,
                )
                st.session_state.eval_results.append(avg)
            elif trial_results:
                st.session_state.eval_results.append(trial_results[-1])

        progress_bar.progress(1.0)
        status.update(label="✅ Evaluation complete!", state="complete")
        st.rerun()


# ═════════════════════════════════════════════════
# COLUMN 3 — Results & Charts
# ═════════════════════════════════════════════════
with col3:
    st.markdown("### 📊 Results & Recommendations")

    results = st.session_state.eval_results

    if not results:
        st.info("Run an evaluation to see results. Configure your schemas in Column 1, set criteria in Column 2, then click **Run Evaluation**.")

        # Show placeholder chart
        fig = go.Figure()
        fig.add_annotation(text="Run evaluation to see results", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#666"))
        fig.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        valid_results = [r for r in results if not r.error]
        error_results = [r for r in results if r.error]

        if error_results:
            for er in error_results:
                st.error(f"❌ {er.schema_name}: {er.error}")

        if valid_results:
            # ── Summary Metrics ──
            best_quality = max(valid_results, key=lambda x: x.quality_score)
            best_efficiency = min(valid_results, key=lambda x: x.total_tokens)
            best_coverage = max(valid_results, key=lambda x: x.coverage_score)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""<div class="metric-card">
                    <h3>🏆 Best Quality</h3>
                    <div class="value">{best_quality.quality_score:.0%}</div>
                    <div style="color:#aab;font-size:0.75rem;">{best_quality.schema_name}</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card">
                    <h3>⚡ Most Efficient</h3>
                    <div class="value">{best_efficiency.total_tokens}</div>
                    <div style="color:#aab;font-size:0.75rem;">{best_efficiency.schema_name}</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-card">
                    <h3>📋 Best Coverage</h3>
                    <div class="value">{best_coverage.coverage_score:.0%}</div>
                    <div style="color:#aab;font-size:0.75rem;">{best_coverage.schema_name}</div>
                </div>""", unsafe_allow_html=True)

            st.divider()

            # ── Charts ──
            result_tabs = st.tabs(["🎯 Quality", "⚡ Efficiency", "📋 Coverage", "🔍 Detail"])

            colors = ["#667eea", "#f093fb", "#4fd1c5", "#f6ad55", "#fc8181"]

            with result_tabs[0]:  # Quality
                # Radar chart
                categories = ["Accuracy", "Completeness", "Relevance", "Overall"]
                fig = go.Figure()
                for i, r in enumerate(valid_results):
                    fig.add_trace(go.Scatterpolar(
                        r=[r.accuracy_score, r.completeness_score, r.relevance_score, r.quality_score],
                        theta=categories,
                        fill='toself',
                        name=r.schema_name,
                        line_color=colors[i % len(colors)],
                        opacity=0.7,
                    ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor="#333"),
                        bgcolor="rgba(0,0,0,0)",
                        angularaxis=dict(gridcolor="#333"),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                    margin=dict(l=60, r=60, t=30, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                    font=dict(family="DM Sans", color="#ccc"),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Quality bar chart
                fig2 = go.Figure()
                names = [r.schema_name for r in valid_results]
                fig2.add_trace(go.Bar(
                    x=names,
                    y=[r.quality_score for r in valid_results],
                    marker_color=[colors[i % len(colors)] for i in range(len(valid_results))],
                    text=[f"{r.quality_score:.0%}" for r in valid_results],
                    textposition="outside",
                ))
                fig2.update_layout(
                    title="Overall Quality Score",
                    yaxis=dict(range=[0, 1.15], title="Score", gridcolor="#222"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    font=dict(family="DM Sans", color="#ccc"),
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig2, use_container_width=True)

            with result_tabs[1]:  # Efficiency
                # Tokens comparison
                fig3 = go.Figure()
                names = [r.schema_name for r in valid_results]
                fig3.add_trace(go.Bar(
                    name="Prompt Tokens",
                    x=names,
                    y=[r.prompt_tokens for r in valid_results],
                    marker_color="#667eea",
                ))
                fig3.add_trace(go.Bar(
                    name="Completion Tokens",
                    x=names,
                    y=[r.completion_tokens for r in valid_results],
                    marker_color="#f093fb",
                ))
                fig3.update_layout(
                    barmode="stack",
                    title="Token Usage by Schema",
                    yaxis_title="Tokens",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=300,
                    font=dict(family="DM Sans", color="#ccc"),
                    yaxis=dict(gridcolor="#222"),
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig3, use_container_width=True)

                # Latency & Cost
                fig4 = go.Figure()
                fig4.add_trace(go.Bar(
                    x=names,
                    y=[r.latency_ms for r in valid_results],
                    marker_color=[colors[i % len(colors)] for i in range(len(valid_results))],
                    text=[f"{r.latency_ms:.0f}ms" for r in valid_results],
                    textposition="outside",
                ))
                fig4.update_layout(
                    title="Latency (ms)",
                    yaxis_title="Milliseconds",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    font=dict(family="DM Sans", color="#ccc"),
                    yaxis=dict(gridcolor="#222"),
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig4, use_container_width=True)

                # Cost table
                st.markdown("**Estimated Cost per Call**")
                for r in valid_results:
                    st.markdown(f"- **{r.schema_name}**: ${r.estimated_cost_usd:.6f} ({r.total_tokens} tokens)")

            with result_tabs[2]:  # Coverage
                fig5 = go.Figure()
                fig5.add_trace(go.Bar(
                    x=[r.schema_name for r in valid_results],
                    y=[r.coverage_score for r in valid_results],
                    marker_color=[colors[i % len(colors)] for i in range(len(valid_results))],
                    text=[f"{r.coverage_score:.0%}" for r in valid_results],
                    textposition="outside",
                ))
                fig5.update_layout(
                    title="Schema Coverage Score",
                    yaxis=dict(range=[0, 1.15], title="Coverage", gridcolor="#222"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=300,
                    font=dict(family="DM Sans", color="#ccc"),
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig5, use_container_width=True)

                for r in valid_results:
                    with st.expander(f"📋 {r.schema_name} — Coverage Details"):
                        if r.missing_fields:
                            st.warning(f"Missing required fields: `{', '.join(r.missing_fields)}`")
                        else:
                            st.success("All required fields present ✓")
                        if r.extra_fields:
                            st.info(f"Extra fields in output: `{', '.join(r.extra_fields)}`")

            with result_tabs[3]:  # Detail
                for r in valid_results:
                    with st.expander(f"🔍 {r.schema_name} — Full Details", expanded=False):
                        st.markdown(f"**Judge Reasoning:** {r.quality_reasoning}")
                        st.divider()
                        st.markdown("**Raw Output:**")
                        st.json(r.parsed_output)

            # ── Recommendations ──
            st.divider()
            st.markdown("### 💡 Recommendations")

            if len(valid_results) >= 2:
                sorted_quality = sorted(valid_results, key=lambda x: x.quality_score, reverse=True)
                sorted_efficiency = sorted(valid_results, key=lambda x: x.total_tokens)

                winner_q = sorted_quality[0]
                runner_q = sorted_quality[1]
                winner_e = sorted_efficiency[0]

                quality_gap = winner_q.quality_score - runner_q.quality_score
                token_gap = sorted_efficiency[-1].total_tokens - winner_e.total_tokens

                recommendations = []

                if quality_gap > 0.1:
                    recommendations.append(
                        f"🏆 **{winner_q.schema_name}** is the clear quality winner "
                        f"({winner_q.quality_score:.0%} vs {runner_q.quality_score:.0%}). "
                        f"The {quality_gap:.0%} gap suggests meaningful structural advantages."
                    )
                elif quality_gap > 0:
                    recommendations.append(
                        f"📊 Quality scores are close ({winner_q.quality_score:.0%} vs {runner_q.quality_score:.0%}). "
                        f"Consider running more trials or adding criteria to differentiate."
                    )

                if winner_q.schema_name != winner_e.schema_name:
                    recommendations.append(
                        f"⚖️ **Quality-Efficiency tradeoff detected.** "
                        f"{winner_q.schema_name} wins quality but {winner_e.schema_name} is more token-efficient "
                        f"({winner_e.total_tokens} vs {winner_q.total_tokens} tokens). "
                        f"For high-volume agents, the token savings of {token_gap} tokens/call may justify the quality trade."
                    )

                for r in valid_results:
                    if r.missing_fields:
                        recommendations.append(
                            f"⚠️ **{r.schema_name}** has coverage gaps: `{', '.join(r.missing_fields)}` missing. "
                            f"Consider making these fields more prominent or adding descriptions."
                        )

                # Research-backed recommendation
                has_nested = any(_get_depth(r.schema_json) > 0 for r in valid_results)
                has_flat = any(_get_depth(r.schema_json) == 0 for r in valid_results)
                if has_nested and has_flat:
                    recommendations.append(
                        "📚 **Research insight (Tam et al., 2024):** Stricter/deeper schemas tend to degrade reasoning "
                        "by 10-15%. For your Q&A agents, consider the **two-step approach**: let the agent reason freely, "
                        "then format into schema. For classification agents, stricter schemas actually help."
                    )

                for rec in recommendations:
                    st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)

            elif len(valid_results) == 1:
                st.info("Add a second schema variant and re-run to get comparative recommendations.")
