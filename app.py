"""
Decision Support System - Main Application
Streamlit interface for the hybrid neuro-symbolic decision support system
"""

import streamlit as st
import streamlit.components.v1 as components
import os
from symbolic_engine import DecisionState, ConstraintViolation
from llm_interface import LLMInterface
import json

# ── API key ────────────────────────────────────────────────────────────────────
# Try Groq key from secrets; fall back to env var for local dev
try:
    GOOGLE_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    import os
    GOOGLE_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Decision Support System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state defaults ─────────────────────────────────────────────────────
if "state" not in st.session_state:
    st.session_state.state = DecisionState()
if "llm" not in st.session_state:
    st.session_state.llm = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_council" not in st.session_state:
    st.session_state.show_council = False
if "chat_locked" not in st.session_state:
    st.session_state.chat_locked = False  # Bug 2: lock chat after recommendation


# ── LLM init ───────────────────────────────────────────────────────────────────
def initialize_llm():
    try:
        import os as _os
        _app_dir  = _os.path.dirname(_os.path.abspath(__file__))
        _bls_path = _os.path.join(_app_dir, "bls_ooh_chunks.jsonl")
        st.session_state.llm = LLMInterface(api_key=GOOGLE_API_KEY, bls_path=_bls_path)
        return True
    except Exception as e:
        st.session_state.llm_error = str(e)
        return False


# ── Conversation-complete: ONLY when chat_locked ───────────────────────────────
def is_conversation_complete() -> bool:
    """
    Council button shows only when chat_locked=True.
    chat_locked is set in process_message() when the LLM sends
    a conclusion containing "Council of Experts".
    Removed field-count fallback — it was showing the button too early.
    """
    return st.session_state.get("chat_locked", False)


# ── Header ─────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div style='text-align:center;padding:20px;
                background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);
                border-radius:10px;margin-bottom:30px;'>
        <h1 style='color:white;margin:0;'>🧠 Decision Support System</h1>
        <p style='color:#f0f0f0;margin:10px 0 0 0;'>
            Neuro-Symbolic Reasoning for Complex Decisions
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Bug 4 fix: sidebar counts ALL categories ───────────────────────────────────
def render_sidebar_state():
    state_dict = st.session_state.state.to_dict()

    st.markdown("### 📊 Live State Tracking")

    # Decision mode
    mode = state_dict["decision_mode"]
    mode_icon = {
        "SURVIVAL_MODE": "🔴",
        "CAUTIOUS_MODE": "🟡",
        "GROWTH_MODE": "🟢",
        "INSUFFICIENT_DATA": "⚪",
    }
    st.markdown(f"**Mode:** {mode_icon.get(mode, 'circle')} {mode}")

    # Decision type + options
    meta = state_dict.get("decision_metadata", {})
    decision_type = meta.get("decision_type")
    options = meta.get("options_being_compared", [])
    if decision_type:
        label = decision_type
        if options:
            label += f": {' vs '.join(options)}"
        st.markdown(f"**Decision:** {label}")

    # Violations
    if state_dict["violations"]:
        st.markdown("#### Conflicts Detected")
        for v in state_dict["violations"]:
            st.error(f"**{v['type']}:** {v['description']}")

    # Recent updates — shows category.key so it's clear what changed
    if st.session_state.state.history:
        st.markdown("#### Recent Updates")
        for change in reversed(st.session_state.state.history[-3:]):
            st.caption(
                f"+ {change['category']}.{change['key']}: {change['new_value']}"
            )

    # Known facts — now counts ALL categories (was the bug)
    all_cats = [
        state_dict.get("financial", {}),
        state_dict.get("values", {}),
        state_dict.get("current", {}),
        state_dict.get("opportunity", {}),
        state_dict.get("interests", {}),
        state_dict.get("career_vision", {}),
        state_dict.get("strengths", {}),
        state_dict.get("decision_metadata", {}),
        state_dict.get("offer_a", {}),
        state_dict.get("offer_b", {}),
    ]
    known_count = sum(
        1 for cat in all_cats
        for v in cat.values()
        if v is not None and v is not False and v != "" and v != []
    )
    st.markdown(f"**Known facts:** {known_count}")

    # Per-category breakdown
    with st.expander("Fact breakdown", expanded=True):
        subtype = state_dict.get("decision_metadata", {}).get("decision_subtype", "general")
        options = state_dict.get("decision_metadata", {}).get("options_being_compared", ["Offer 1", "Offer 2"])

        if subtype == "offer_comparison":
            categories = [
                (f"{options[0] if options else 'Offer 1'} details", "offer_a"),
                (f"{options[1] if len(options)>1 else 'Offer 2'} details", "offer_b"),
                ("Values", "values"),
                ("Decision metadata", "decision_metadata"),
            ]
        else:
            categories = [
                ("Decision metadata", "decision_metadata"),
                ("Interests", "interests"),
                ("Career vision", "career_vision"),
                ("Strengths", "strengths"),
                ("Values", "values"),
                ("Financial", "financial"),
                ("Current situation", "current"),
            ]

        for label, key in categories:
            cat = state_dict.get(key, {})
            count = sum(
                1 for v in cat.values()
                if v is not None and v is not False and v != "" and v != []
            )
            if count > 0:
                st.caption(f"✅ {label}: {count} facts")
            else:
                st.caption(f"⬜ {label}: none yet")

    # Missing info
    if state_dict.get("missing_info"):
        with st.expander("Missing info", expanded=False):
            for m in state_dict["missing_info"]:
                st.caption(f"- {m}")


# ── Decision Tree ──────────────────────────────────────────────────────────────
# Master registry: (category, field) → (agent_id, display_label, note, impact_fn)
# impact_fn takes the raw value and returns an impact string
def _tree_impact_score(v, high_is_bad=True):
    """Generic 1-10 score → impact. high_is_bad=True means high score = more risk."""
    try:
        s = int(float(v))
        if high_is_bad:
            return "high-risk" if s >= 8 else ("moderate" if s >= 5 else "positive")
        else:
            return "positive" if s >= 7 else ("moderate" if s >= 4 else "risk")
    except:
        return "neutral"

def _tree_bool_impact(v, true_is_risk=False):
    if v is True:  return "high-risk" if true_is_risk else "positive"
    if v is False: return "positive"  if true_is_risk else "neutral"
    return "neutral"

# Registry: maps (category, field_key) → (agent_id, display_label, tooltip_note, impact_fn)
# impact_fn is a lambda that takes the raw value
_FACT_REGISTRY = {
    # ── Values ──────────────────────────────────────────────────────────────────
    ("values", "financial_security"):  ("financial", "Financial Security Priority",
        "How much salary and stability factor into the decision — higher = stronger need for income",
        lambda v: _tree_impact_score(v, high_is_bad=True)),
    ("values", "career_growth"):       ("growth", "Career Growth Priority",
        "How much fast career progression matters to the user",
        lambda v: _tree_impact_score(v, high_is_bad=False)),
    ("values", "work_life_balance"):   ("wellbeing", "Work-Life Balance Priority",
        "How much sustainable hours and personal time factor in",
        lambda v: _tree_impact_score(v, high_is_bad=False)),
    ("values", "impact"):              ("values_agent", "Impact / Purpose Priority",
        "How much the user wants their work to matter beyond just income",
        lambda v: _tree_impact_score(v, high_is_bad=False)),
    ("values", "learning"):            ("growth", "Learning Priority",
        "How important continuous skill development is to the user",
        lambda v: _tree_impact_score(v, high_is_bad=False)),
    ("values", "salary_importance"):   ("financial", "Salary Importance",
        "Explicit 1-10 score the user gave for how much salary matters",
        lambda v: _tree_impact_score(v, high_is_bad=True)),

    # ── Current situation ────────────────────────────────────────────────────────
    ("current", "current_satisfaction"): ("wellbeing", "Current Job Satisfaction",
        "Low satisfaction is a push factor away from current job — but not sufficient reason alone",
        lambda v: _tree_impact_score(v, high_is_bad=False)),
    ("current", "business_idea"):      ("growth", "Business Idea Clarity",
        "Vague ideas carry higher execution risk than a specific validated concept",
        lambda v: "positive" if str(v).lower() not in ("vague","none","") else "risk"),
    ("current", "business_validated"): ("growth", "Business Idea Validated?",
        "Has the user tested with real customers or earned side income — reduces startup risk significantly",
        lambda v: _tree_bool_impact(v, true_is_risk=False)),
    ("current", "financial_runway"):   ("financial", "Financial Runway",
        "How long the user can survive without income — thin runway with dependents is high risk",
        lambda v: "positive" if v and str(v).lower() not in ("minimal","none","no savings") else "risk"),
    ("current", "leave_reason"):       ("wellbeing", "Reason for Leaving",
        "Frustration-driven exits are riskier than opportunity-driven ones",
        lambda v: "caution" if "frustrat" in str(v).lower() else "positive"),
    ("current", "current_role"):       ("growth", "Current Role & Field",
        "Whether the business or new path is in the same domain — affects execution risk",
        lambda v: "neutral"),
    ("current", "concern"):            ("values_agent", "Biggest Concern",
        "The user's self-identified fear — often reveals the real deciding factor",
        lambda v: "caution"),
    ("current", "job_market_concern"): ("growth", "Industry Opportunity Cost",
        "Whether staying has real upside — promotions, growth, interesting work",
        lambda v: "neutral"),
    ("current", "financial_concern"):  ("financial", "Financial / Family Constraints",
        "Scholarships, family expectations, or cost differences affecting the decision",
        lambda v: "moderate"),
    ("current", "leaning"):            ("values_agent", "Current Leaning & Reason",
        "What the user is already leaning toward — gut instinct is often values-driven",
        lambda v: "neutral"),
    ("current", "current_year"):       ("values_agent", "Year in School",
        "Switching majors or paths later carries higher cost — affects decision urgency",
        lambda v: "neutral"),
    ("current", "business_experience"): ("growth", "Business Experience (1-10)",
        "Sales, marketing, accounting skills needed to run a business",
        lambda v: _tree_impact_score(v, high_is_bad=False)),

    # ── Personal ────────────────────────────────────────────────────────────────
    ("personal", "has_family"):        ("financial", "Has Family",
        "Partner, kids, or household obligations that are affected by income changes",
        lambda v: _tree_bool_impact(v, true_is_risk=True)),
    ("personal", "has_dependents"):    ("financial", "Has Dependents",
        "Financial dependents amplify the risk of any income reduction",
        lambda v: _tree_bool_impact(v, true_is_risk=True)),
    ("personal", "partner_employed"):  ("financial", "Partner Employed?",
        "A second income reduces personal financial risk significantly",
        lambda v: "positive" if v is True else "risk"),
    ("personal", "can_relocate"):      ("values_agent", "Can Relocate?",
        "Relocation feasibility — if required and not possible, this is a hard blocker",
        lambda v: "positive" if v is True else "high-risk"),
    ("personal", "relocation_concern"):("wellbeing", "Relocation Concern",
        "Specific worry about moving — affects wellbeing and family",
        lambda v: "caution"),
    ("personal", "current_city"):      ("values_agent", "Current City",
        "Location context for relocation assessment", lambda v: "neutral"),

    # ── Career vision ────────────────────────────────────────────────────────────
    ("career_vision", "post_graduation_goal"): ("growth", "Post-Graduation Goal",
        "Job, grad school, or startup — shapes which path makes more sense",
        lambda v: "neutral"),
    ("career_vision", "desired_role_5yr"):     ("growth", "5-Year Vision",
        "If the target role requires a specific credential or path, this is a strong signal",
        lambda v: "positive" if v and str(v).lower() != "undecided" else "neutral"),
    ("career_vision", "research_vs_applied"):  ("values_agent", "Research vs Applied Lean",
        "Research lean → PhD/academia aligns better; Applied → industry job likely sufficient",
        lambda v: "positive" if str(v).lower() == "research" else "caution" if str(v).lower() == "applied" else "neutral"),
    ("career_vision", "industry_preference"):  ("growth", "Industry Preference",
        "Target industry — affects which path provides better entry", lambda v: "neutral"),

    # ── Interests ────────────────────────────────────────────────────────────────
    ("interests", "hands_on_work"):     ("wellbeing", "Prefers Hands-On Work",
        "Hands-on preference aligns with industry/applied roles over research",
        lambda v: "positive" if v is True else "neutral"),
    ("interests", "enjoys_theory"):     ("wellbeing", "Enjoys Theoretical Work",
        "Theory enjoyment is a strong predictor of PhD or research path fit",
        lambda v: "positive" if v is True else "neutral"),
    ("interests", "enjoys_coding"):     ("growth", "Enjoys Coding",
        "Coding affinity — relevant when comparing technical paths", lambda v: "neutral"),
    ("interests", "enjoys_building_systems"): ("growth", "Enjoys Building Systems",
        "Systems-builder mindset — tends to favor engineering/applied over research",
        lambda v: "positive" if v is True else "neutral"),
    ("interests", "enjoys_working_with_data"): ("growth", "Enjoys Working with Data",
        "Data affinity — relevant when comparing data-related paths", lambda v: "neutral"),
    ("interests", "research"):          ("values_agent", "Research Interest",
        "Genuine research interest is a prerequisite for PhD success",
        lambda v: "positive" if v is True else "risk"),

    # ── Financial ────────────────────────────────────────────────────────────────
    ("financial", "expected_salary"):   ("financial", "Expected Salary",
        "Target salary range — gap between this and a stipend is the real PhD cost",
        lambda v: "neutral"),
    ("financial", "salary_importance"): ("financial", "Salary Importance Score",
        "Explicit 1-10 score for how much salary matters in this decision",
        lambda v: _tree_impact_score(v, high_is_bad=True)),

    # ── Offer A ──────────────────────────────────────────────────────────────────
    ("offer_a", "role"):           ("growth",    "Offer A — Role",        "Job title for the first offer", lambda v: "neutral"),
    ("offer_a", "salary_raw"):     ("financial", "Offer A — Salary",      "Compensation for the first offer", lambda v: "neutral"),
    ("offer_a", "work_location"):  ("wellbeing", "Offer A — Location",    "Remote/onsite/hybrid for first offer", lambda v: "neutral"),
    ("offer_a", "requires_relocation"): ("values_agent", "Offer A — Relocation?", "Whether first offer requires moving", lambda v: _tree_bool_impact(v, true_is_risk=True)),
    ("offer_a", "growth_potential"):    ("growth",    "Offer A — Growth Potential", "Career advancement path at first company", lambda v: "positive" if str(v).lower() in ("high","great") else "neutral"),
    ("offer_a", "work_life_balance"):   ("wellbeing", "Offer A — Work-Life Balance", "Balance expectations at first company", lambda v: "neutral"),
    ("offer_a", "concern"):        ("values_agent", "Offer A — Your Concern", "User's biggest hesitation about first offer", lambda v: "caution"),
    ("offer_a", "job_security"):   ("financial", "Offer A — Job Security", "Stability of first offer", lambda v: "positive" if str(v).lower() == "high" else "risk" if str(v).lower() == "low" else "moderate"),

    # ── Offer B ──────────────────────────────────────────────────────────────────
    ("offer_b", "role"):           ("growth",    "Offer B — Role",        "Job title for the second offer", lambda v: "neutral"),
    ("offer_b", "salary_raw"):     ("financial", "Offer B — Salary",      "Compensation for the second offer", lambda v: "neutral"),
    ("offer_b", "work_location"):  ("wellbeing", "Offer B — Location",    "Remote/onsite/hybrid for second offer", lambda v: "neutral"),
    ("offer_b", "requires_relocation"): ("values_agent", "Offer B — Relocation?", "Whether second offer requires moving", lambda v: _tree_bool_impact(v, true_is_risk=True)),
    ("offer_b", "growth_potential"):    ("growth",    "Offer B — Growth Potential", "Career advancement path at second company", lambda v: "positive" if str(v).lower() in ("high","great") else "neutral"),
    ("offer_b", "work_life_balance"):   ("wellbeing", "Offer B — Work-Life Balance", "Balance expectations at second company", lambda v: "neutral"),
    ("offer_b", "concern"):        ("values_agent", "Offer B — Your Concern", "User's biggest hesitation about second offer", lambda v: "caution"),
    ("offer_b", "job_security"):   ("financial", "Offer B — Job Security", "Stability of second offer", lambda v: "positive" if str(v).lower() == "high" else "risk" if str(v).lower() == "low" else "moderate"),

    # ── University A ─────────────────────────────────────────────────────────────
    ("uni_a", "name"):             ("growth",    "University A — Name",      "First university name", lambda v: "neutral"),
    ("uni_a", "tuition_raw"):      ("financial", "University A — Tuition",   "Cost for first university", lambda v: "neutral"),
    ("uni_a", "scholarship"):      ("financial", "University A — Scholarship","Financial aid reducing cost burden", lambda v: "positive"),
    ("uni_a", "ranking"):          ("growth",    "University A — Ranking",   "Reputation and standing in field", lambda v: "neutral"),
    ("uni_a", "job_placement"):    ("growth",    "University A — Placement", "Alumni network and placement rates", lambda v: "neutral"),
    ("uni_a", "requires_relocation"): ("values_agent", "University A — Relocation?", "Whether first university requires moving", lambda v: _tree_bool_impact(v, true_is_risk=True)),
    ("uni_a", "living_cost"):      ("financial", "University A — Living Cost","Cost of living in first university's city", lambda v: "neutral"),
    ("uni_a", "concern"):          ("values_agent", "University A — Concern","User's hesitation about first university", lambda v: "caution"),

    # ── University B ─────────────────────────────────────────────────────────────
    ("uni_b", "name"):             ("growth",    "University B — Name",      "Second university name", lambda v: "neutral"),
    ("uni_b", "tuition_raw"):      ("financial", "University B — Tuition",   "Cost for second university", lambda v: "neutral"),
    ("uni_b", "scholarship"):      ("financial", "University B — Scholarship","Financial aid reducing cost burden", lambda v: "positive"),
    ("uni_b", "ranking"):          ("growth",    "University B — Ranking",   "Reputation and standing in field", lambda v: "neutral"),
    ("uni_b", "job_placement"):    ("growth",    "University B — Placement", "Alumni network and placement rates", lambda v: "neutral"),
    ("uni_b", "requires_relocation"): ("values_agent", "University B — Relocation?", "Whether second university requires moving", lambda v: _tree_bool_impact(v, true_is_risk=True)),
    ("uni_b", "living_cost"):      ("financial", "University B — Living Cost","Cost of living in second university's city", lambda v: "neutral"),
    ("uni_b", "concern"):          ("values_agent", "University B — Concern","User's hesitation about second university", lambda v: "caution"),
}

# Values that should never appear in the tree — placeholders, not real answers
_PLACEHOLDER_VALUES = {"neutral", "unknown", "undecided", "n/a", "none", "", "null"}

def _paired_university_factors(state_dict, opt_a, opt_b, college_retriever=None):
    """
    For university comparisons: compare uni_a vs uni_b field-by-field.
    Returns list of factor dicts with real comparative directions.
    Also fetches College Scorecard data if retriever available.
    """
    uni_a = state_dict.get("uni_a", {})
    uni_b = state_dict.get("uni_b", {})
    personal = state_dict.get("personal", {})
    values   = state_dict.get("values", {})
    interests = state_dict.get("interests", {})
    career_vis = state_dict.get("career_vision", {})

    factors = []

    def add(name, val_a, val_b, interpret_fn, category="Comparison", source=None):
        """Add a comparative factor. interpret_fn(val_a, val_b) -> (direction, label)."""
        if val_a is None and val_b is None:
            return
        direction, label = interpret_fn(val_a, val_b)
        factors.append({
            "category":  category,
            "name":      name,
            "value":     f"{opt_a}: {val_a or '?'}  |  {opt_b}: {val_b or '?'}",
            "impact":    label,
            "direction": direction,
            "source":    source,   # "api", "bls", or None (user-provided)
        })

    def add_single(name, val, direction, label, category="User Input", source=None):
        if val in (None, False, "", []):
            return
        if isinstance(val, bool) and not val:
            return
        factors.append({
            "category":  category,
            "name":      name,
            "value":     str(val),
            "impact":    label,
            "direction": direction,
            "source":    source,
        })

    # ── Paired comparisons from user-provided data ────────────────────────────

    # Tuition comparison
    ta = uni_a.get("tuition")
    tb = uni_b.get("tuition")
    if ta and tb:
        def cmp_tuition(a, b):
            try:
                a, b = float(a), float(b)
                diff = abs(a - b)
                cheaper = opt_a if a < b else opt_b
                direction = "a" if a < b else "b"
                return direction, f"{cheaper} is ${diff:,.0f}/yr cheaper"
            except:
                return "neutral", f"Tuition: {opt_a} ${a} | {opt_b} ${b}"
        add("Tuition (annual)", ta, tb, cmp_tuition, "Cost", source=None)
    elif ta:
        add_single(f"{opt_a} Tuition", f"${ta:,}/yr" if isinstance(ta, (int,float)) else ta,
                   "neutral", "Only one tuition known", "Cost")
    elif tb:
        add_single(f"{opt_b} Tuition", f"${tb:,}/yr" if isinstance(tb, (int,float)) else tb,
                   "neutral", "Only one tuition known", "Cost")

    # Scholarship
    sa = uni_a.get("scholarship")
    sb = uni_b.get("scholarship")
    if sa and sa.lower() not in ("none", "no", "unknown"):
        add_single(f"{opt_a} — Scholarship", sa, "a", f"Financial support at {opt_a}", "Cost")
    if sb and sb.lower() not in ("none", "no", "unknown"):
        add_single(f"{opt_b} — Scholarship", sb, "b", f"Financial support at {opt_b}", "Cost")

    # Ranking / reputation
    ra = uni_a.get("ranking")
    rb = uni_b.get("ranking")
    rep_imp = values.get("reputation_importance", 0)
    try: rep_imp = float(rep_imp)
    except: rep_imp = 0
    if ra and rb:
        def cmp_rank(a, b):
            strong = ["top", "well known", "strong", "reputable", "better", "higher"]
            a_str = any(w in str(a).lower() for w in strong)
            b_str = any(w in str(b).lower() for w in strong)
            if a_str and not b_str: return "a", f"{opt_a} has stronger reputation"
            if b_str and not a_str: return "b", f"{opt_b} has stronger reputation"
            return "neutral", "Reputation roughly comparable"
        add("Program Reputation", ra, rb, cmp_rank, "Academic")
    elif ra:
        label = "Strong reputation" if any(w in str(ra).lower() for w in ["top","well known","strong"]) else f"Reputation: {ra}"
        direction = "a" if rep_imp >= 7 else "neutral"
        add_single(f"{opt_a} — Reputation", ra, direction, label, "Academic")
    elif rb:
        label = "Strong reputation" if any(w in str(rb).lower() for w in ["top","well known","strong"]) else f"Reputation: {rb}"
        direction = "b" if rep_imp >= 7 else "neutral"
        add_single(f"{opt_b} — Reputation", rb, direction, label, "Academic")

    # Location preference
    city_pref = personal.get("city_preference", "")
    la = uni_a.get("location", "")
    lb = uni_b.get("location", "")
    if city_pref and (la or lb):
        big_cities = ["corpus christi", "houston", "dallas", "austin", "san antonio",
                      "los angeles", "new york", "chicago", "boston", "seattle", "miami"]
        a_big = any(c in str(la).lower() for c in big_cities)
        b_big = any(c in str(lb).lower() for c in big_cities)
        if "big" in city_pref.lower():
            if a_big and not b_big: add_single("City Preference", f"Prefers big city ({la})", "a", f"{opt_a} is in a larger city — matches preference", "Personal")
            elif b_big and not a_big: add_single("City Preference", f"Prefers big city ({lb})", "b", f"{opt_b} is in a larger city — matches preference", "Personal")
            else: add_single("City Preference", city_pref, "neutral", "Both in comparable city sizes", "Personal")
        elif "small" in city_pref.lower() or "town" in city_pref.lower():
            if b_big and not a_big: add_single("City Preference", f"Prefers smaller town ({lb})", "b", f"{opt_b} is in a smaller town — matches preference", "Personal")
            else: add_single("City Preference", city_pref, "neutral", "Campus environment preference noted", "Personal")
        else:
            add_single("City Preference", city_pref, "neutral", "Campus environment preference noted", "Personal")

    # Social connection
    social = personal.get("social_connection", "")
    if social:
        direction = "b" if opt_b.lower() in social.lower() else ("a" if opt_a.lower() in social.lower() else "neutral")
        add_single("Social Connection", social, direction, f"Social support near {opt_b if direction=='b' else opt_a}", "Personal")

    # Field of interest
    field = interests.get("field_of_interest", "")
    if field:
        add_single("Field of Interest", field, "neutral", f"Both programs may offer {field} tracks", "Academic")

    # Career goal
    goal = career_vis.get("post_graduation_goal", "")
    if goal == "job":
        add_single("Career Goal", "Industry job", "neutral", "Both paths lead to industry — placement data matters", "Career")
    elif goal:
        add_single("Career Goal", goal, "neutral", f"Targeting: {goal}", "Career")

    desired = career_vis.get("desired_role_5yr", "")
    if desired:
        add_single("Target Role", desired, "neutral", f"Aiming for: {desired}", "Career")

    # Reputation importance
    if rep_imp >= 7:
        add_single("Reputation Priority", f"{rep_imp}/10", "neutral",
                   f"High priority ({rep_imp}/10) — program name matters for hiring", "Values")

    # Taking student debt
    if state_dict.get("financial", {}).get("taking_student_debt"):
        add_single("Financing", "Student loan", "neutral",
                   "Taking debt — net cost and post-grad earnings matter most", "Cost")

    return factors


def _api_university_factors(card_a, card_b, opt_a, opt_b):
    """
    Generate comparative factors from College Scorecard data.
    These are clearly marked as external data (source='api').
    """
    factors = []
    if card_a is None and card_b is None:
        return factors

    def add_cmp(name, val_a, val_b, fmt_fn, direction_fn):
        if val_a is None and val_b is None:
            return
        va = fmt_fn(val_a) if val_a else "no data"
        vb = fmt_fn(val_b) if val_b else "no data"
        direction = direction_fn(val_a, val_b)
        if val_a and val_b:
            diff = abs(val_a - val_b)
            higher = opt_a if val_a > val_b else opt_b
            lower  = opt_b if val_a > val_b else opt_a
        else:
            higher = lower = None
        factors.append({
            "category":  "College Scorecard Data",
            "name":      name,
            "value":     f"{opt_a}: {va}  |  {opt_b}: {vb}",
            "impact":    direction_fn(val_a, val_b, as_label=True),
            "direction": direction_fn(val_a, val_b),
            "source":    "api",
        })

    ta = card_a.tuition_in_state  if card_a else None
    tb = card_b.tuition_in_state  if card_b else None
    if ta or tb:
        va = f"${ta:,}/yr" if ta else "no data"
        vb = f"${tb:,}/yr" if tb else "no data"
        if ta and tb:
            cheaper = opt_a if ta < tb else opt_b
            direction = "a" if ta < tb else "b"
            label = f"{cheaper} is ${abs(ta-tb):,}/yr cheaper in-state"
        elif ta:
            direction, label = "neutral", f"{opt_a}: ${ta:,}/yr"
        else:
            direction, label = "neutral", f"{opt_b}: ${tb:,}/yr"
        factors.append({"category": "College Scorecard Data", "name": "In-State Tuition",
                        "value": f"{opt_a}: {va}  |  {opt_b}: {vb}",
                        "impact": label, "direction": direction, "source": "api"})

    ea = card_a.median_earnings_10yr if card_a else None
    eb = card_b.median_earnings_10yr if card_b else None
    if ea or eb:
        va = f"${ea:,}" if ea else "no data"
        vb = f"${eb:,}" if eb else "no data"
        if ea and eb:
            higher = opt_a if ea > eb else opt_b
            direction = "a" if ea > eb else "b"
            label = f"{higher} grads earn ${abs(ea-eb):,} more at 10 years"
        elif ea:
            direction, label = "neutral", f"{opt_a} median: ${ea:,}"
        else:
            direction, label = "neutral", f"{opt_b} median: ${eb:,}"
        factors.append({"category": "College Scorecard Data", "name": "Median Earnings (10yr)",
                        "value": f"{opt_a}: {va}  |  {opt_b}: {vb}",
                        "impact": label, "direction": direction, "source": "api"})

    ga = card_a.grad_rate if card_a else None
    gb = card_b.grad_rate if card_b else None
    if ga or gb:
        va = f"{ga*100:.0f}%" if ga else "no data"
        vb = f"{gb*100:.0f}%" if gb else "no data"
        if ga and gb:
            higher = opt_a if ga > gb else opt_b
            direction = "a" if ga > gb else "b"
            label = f"{higher} has higher graduation rate ({max(ga,gb)*100:.0f}%)"
        else:
            direction, label = "neutral", "Grad rate data partial"
        factors.append({"category": "College Scorecard Data", "name": "Graduation Rate (6yr)",
                        "value": f"{opt_a}: {va}  |  {opt_b}: {vb}",
                        "impact": label, "direction": direction, "source": "api"})

    return factors


def render_decision_tree(state_dict: dict, council_results: dict):
    """
    Decision factor tree — comparative analysis.
    For university comparisons: does paired field-by-field comparison.
    External data (College Scorecard / BLS) highlighted distinctly.
    """
    options  = state_dict.get("decision_metadata", {}).get("options_being_compared", ["A", "B"])
    subtype  = state_dict.get("decision_metadata", {}).get("decision_subtype", "general")
    opt_a    = options[0] if len(options) > 0 else "Option A"
    opt_b    = options[1] if len(options) > 1 else "Option B"
    avg_vote = council_results.get("avg_vote", {})
    avg_a    = avg_vote.get("option_a", 50)
    avg_b    = avg_vote.get("option_b", 50)
    winner   = opt_a if avg_a >= avg_b else opt_b
    win_pct  = max(avg_a, avg_b)
    win_color = "#16a34a" if avg_a >= avg_b else "#dc2626"
    agents   = council_results.get("agents", [])
    agent_votes = council_results.get("agent_votes", {})

    # ── Build factor list ─────────────────────────────────────────────────────
    bls_factors = []  # populated in major_choice branch; must exist for all branches
    if subtype == "university_comparison":
        # Paired comparative analysis
        factors = _paired_university_factors(state_dict, opt_a, opt_b)

        # Try to get College Scorecard data for the tree
        try:
            college = st.session_state.llm._college if st.session_state.llm else None
            if college:
                cards = college.get_cards_for_decision(opt_a, opt_b)
                ca, cb = cards.get("option_a"), cards.get("option_b")
                if ca or cb:
                    api_factors = _api_university_factors(ca, cb, opt_a, opt_b)
                    if api_factors:
                        factors = api_factors + factors
                        print(f"[TREE] Injected {len(api_factors)} scorecard factors")
                    else:
                        print(f"[TREE] Scorecard cards found but no comparable fields")
                else:
                    print(f"[TREE] Scorecard: no match for '{opt_a}' or '{opt_b}'")
            else:
                print("[TREE] College retriever not initialised")
        except Exception as e:
            print(f"[TREE] College Scorecard error: {type(e).__name__}: {e}")

    else:
        # Generic factor list for non-university decisions
        SKIP_VALS   = (None, False, "", [], "none", "null", "unknown", "n/a")
        SKIP_FIELDS = {"financial_runway_months","salary_raw","tuition_raw",
                       "work_life_balance_known","team_culture_known","job_market_concern"}
        READABLE = {
            "financial_security":    "Financial Security Priority",
            "career_growth":         "Career Growth Priority",
            "work_life_balance":     "Work-Life Balance Priority",
            "has_dependents":        "Has Dependents",
            "has_family":            "Has Family",
            "can_relocate":          "Can Relocate",
            "partner_employed":      "Partner Employed",
            "requires_relocation":   "Requires Relocation",
            "current_satisfaction":  "Current Satisfaction",
            "post_graduation_goal":  "Post-Graduation Goal",
            "desired_role_5yr":      "Desired Role (5yr)",
            "research_vs_applied":   "Research vs Applied",
            "hands_on_work":         "Prefers Hands-On Work",
            "concern":               "Main Concern",
            "financial_concern":     "Financial Concern",
            "leaning":               "Stated Lean",
            "financial_runway":      "Financial Runway",
            "current_salary":        "Current Salary",
            "current_income":        "Current Income",
            "current_savings":       "Savings / Runway",
            "debt_total":            "Total Debt",
            "leave_reason":          "Reason for Leaving",
            "business_validated":    "Business Tested",
            "business_idea":         "Business Idea",
            "current_satisfaction":  "Job Satisfaction",
            "city_preference":       "City Preference",
            "social_connection":     "Social Connection",
            "field_of_interest":     "Field of Interest",
            "reputation_importance": "Reputation Priority",
            "taking_student_debt":   "Taking Student Debt",
            "work_anywhere":         "Open to Work Anywhere",
        }
        CAT_LABELS = {
            "values":"Values & Priorities","interests":"Interests & Work Style",
            "career_vision":"Career Vision","current":"Current Situation",
            "personal":"Personal Context","financial":"Financial",
            "offer_a": opt_a,"offer_b": opt_b,
        }

        # Import impact_label inline (avoid circular ref)
        from app import render_decision_tree  # self-ref hack not needed — define inline

        def _impact(field, val, subtype_):
            val_str = str(val).lower()
            if field == "enjoys_coding" and val in (True, "True", "true", "yes"):
                return ("a","Enjoys coding -- favors CS path")
            if field == "enjoys_building_systems" and val in (True, "True", "true", "yes"):
                return ("a","Enjoys building systems -- favors CS")
            if field == "enjoys_analysis" and val in (True, "True", "true", "yes"):
                return ("b","Enjoys data analysis -- favors DS")
            if field == "financial_security" and isinstance(val,(int,float)):
                if int(val) >= 8:
                    return ("a", f"{val}/10 salary priority -- favors higher-paying path")
                elif int(val) >= 5:
                    return ("neutral", f"Financial security: {val}/10")
                else:
                    return ("b", f"{val}/10 -- low priority, more flexibility for risk")
            if field == "career_growth" and isinstance(val,(int,float)):
                return ("a", f"Career growth: {val}/10") if int(val)>=7 else ("neutral",f"Career growth: {val}/10")
            if field == "current_satisfaction" and isinstance(val,(int,float)):
                if int(val) >= 7: return ("a",f"Satisfied ({val}/10)")
                if int(val) <= 4: return ("b",f"Dissatisfied ({val}/10)")
                return ("neutral",f"Satisfaction: {val}/10")
            if field == "concern":
                if any(w in val_str for w in ["jobless","income","money","debt","stability"]): return ("a",f"Fear: {val}")
                if any(w in val_str for w in ["regret","miss","stuck","hate"]): return ("b",f"Fear: {val}")
                return ("neutral",f"Concern: {val}")
            if field in ("has_dependents","has_family") and val is True: return ("a","Has dependents -- stability matters")
            if field == "partner_employed" and val is True: return ("b","Partner employed -- shared safety net")
            if field in ("current_income","current_salary") and isinstance(val,(int,float)):
                return ("a",f"${val:,.0f}/yr -- high opportunity cost") if val>=80000 else ("neutral",f"${val:,.0f}/yr")
            if field in ("financial_runway","current_savings") and isinstance(val,(int,float)):
                return ("b",f"${val:,.0f} runway") if val>=100000 else ("neutral",f"${val:,.0f} runway")
            if field == "leaning": return ("b",f"Leans: {val}")
            if field == "desired_role_5yr":
                # For CS vs DS: engineering/dev/lead roles favor CS; analyst/scientist favor DS
                cs_roles = ["software","engineer","developer","lead","manager","architect","devops","sre","backend","frontend","fullstack"]
                ds_roles  = ["data scientist","analyst","ml engineer","machine learning","research scientist","statistician"]
                if any(r in val_str for r in cs_roles): return ("a", f"Target role '{val}' aligns with CS path")
                if any(r in val_str for r in ds_roles):  return ("b", f"Target role '{val}' aligns with DS path")
                return ("neutral", f"Target role: {val}")
            if field == "post_graduation_goal":
                return ("a","Targeting industry job -- CS has broader openings") if "job" in val_str else ("neutral",str(val))
            if field == "hands_on_work" and val is True: return ("a","Prefers hands-on building -- favors CS")
            if field == "research" and val is True: return ("b","Research-oriented -- favors DS/ML path")
            return ("neutral",str(val)[:45])

        seen = set()
        factors = []
        for cat, cat_label in CAT_LABELS.items():
            cat_data = state_dict.get(cat, {})
            if not isinstance(cat_data, dict): continue
            for field, val in cat_data.items():
                if field in SKIP_FIELDS or val in (None, False, "", []): continue
                if isinstance(val, bool) and not val: continue
                dk = (field, str(val).lower().strip())
                if dk in seen: continue
                seen.add(dk)
                display_name = READABLE.get(field, field.replace("_"," ").title())
                if isinstance(val, float): val = round(val, 1)
                elif isinstance(val, list): val = ", ".join(str(x) for x in val)
                direction, impact = _impact(field, str(val), subtype)
                factors.append({"category":cat_label,"name":display_name,
                                 "value":str(val),"impact":impact,
                                 "direction":direction,"source":None})

    factors = bls_factors + factors  # BLS data shown first

    if not factors:
        st.markdown("""
        <div style='background:#fef9c3;border:1px solid #fde047;border-radius:8px;
                    padding:14px 16px;color:#713f12;'>
            <b>🌳 Not enough data to build a decision tree.</b><br>
            Complete a full conversation first — the tree shows factors that actually shaped the decision.
        </div>""", unsafe_allow_html=True)
        return

    # ── Color scheme ──────────────────────────────────────────────────────────
    dir_color = {"a": "#16a34a", "b": "#2563eb", "neutral": "#64748b"}
    dir_bg    = {"a": "#f0fdf4", "b": "#eff6ff",  "neutral": "#f8fafc"}
    dir_arrow = {"a": f"→ {opt_a[:16]}", "b": f"→ {opt_b[:16]}", "neutral": "↔ Both"}

    # ── Build factor nodes HTML ───────────────────────────────────────────────
    factor_nodes_html = ""
    for i, f in enumerate(factors):
        dc = dir_color[f["direction"]]
        db = dir_bg[f["direction"]]
        arrow = dir_arrow[f["direction"]]
        is_api = f.get("source") == "api"
        is_bls = f.get("source") == "bls"

        # External data gets a distinct badge + highlight
        source_badge = ""
        border_style = f"1px solid {dc}30"
        bg_style     = db
        if is_api:
            source_badge = ("<span style='background:#0ea5e9;color:white;font-size:9px;"
                           "font-weight:700;padding:1px 5px;border-radius:3px;margin-left:6px;"
                           "vertical-align:middle;'>SCORECARD</span>")
            border_style = f"2px solid {dc}60"
            bg_style     = f"linear-gradient(135deg, {db}, #f0f9ff)"
        elif is_bls:
            source_badge = ("<span style='background:#8b5cf6;color:white;font-size:9px;"
                           "font-weight:700;padding:1px 5px;border-radius:3px;margin-left:6px;"
                           "vertical-align:middle;'>BLS DATA</span>")
            border_style = f"2px solid {dc}60"
            bg_style     = f"linear-gradient(135deg, {db}, #faf5ff)"

        factor_nodes_html += f"""
        <div style="display:flex;align-items:stretch;margin-bottom:6px;">
          <div style="width:24px;display:flex;flex-direction:column;align-items:center;flex-shrink:0;">
            <div style="width:2px;background:#cbd5e1;flex:1;"></div>
            <div style="width:12px;height:2px;background:#cbd5e1;"></div>
            <div style="width:2px;background:#cbd5e1;flex:1;{"display:none" if i==len(factors)-1 else ""}"></div>
          </div>
          <div style="flex:1;background:{bg_style};border:{border_style};border-radius:8px;
              padding:7px 12px;margin-left:4px;display:flex;align-items:center;gap:10px;">
            <div style="flex:1;">
              <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;">
                {f["category"]}{source_badge}
              </div>
              <div style="font-size:12px;font-weight:600;color:#1e293b;">
                {f["name"]}: <span style="color:{dc};">{f["value"][:60]}</span>
              </div>
              <div style="font-size:11px;color:#64748b;margin-top:1px;">{f["impact"]}</div>
            </div>
            <div style="background:{dc}18;color:{dc};border:1px solid {dc}40;border-radius:4px;
                padding:2px 8px;font-size:10px;font-weight:700;white-space:nowrap;">{arrow}</div>
          </div>
        </div>"""

    # ── Agent votes panel ─────────────────────────────────────────────────────
    agent_rows_html = ""
    for ag in agents:
        v   = agent_votes.get(ag["id"], {})
        va  = v.get("option_a", 50)
        vb  = v.get("option_b", 50)
        lean = opt_a if va >= vb else opt_b
        lc   = "#16a34a" if va >= vb else "#dc2626"
        agent_rows_html += f"""
        <div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid #f1f5f9;">
          <span style="font-size:16px;">{ag["emoji"]}</span>
          <span style="font-size:11px;font-weight:600;color:{ag["color"]};flex:1;">{ag["name"]}</span>
          <span style="font-size:11px;color:#64748b;">{opt_a[:10]} {va}% · {opt_b[:10]} {vb}%</span>
          <span style="background:{lc}18;color:{lc};border:1px solid {lc}40;
              border-radius:4px;padding:1px 7px;font-size:10px;font-weight:700;">→ {lean}</span>
        </div>"""

    # ── Legend note for external data ─────────────────────────────────────────
    has_api = any(f.get("source") == "api" for f in factors)
    has_bls = any(f.get("source") == "bls" for f in factors)
    ext_legend = ""
    if has_api:
        ext_legend += "<div style='font-size:10px;margin-top:4px;'><span style='background:#0ea5e9;color:white;font-size:9px;font-weight:700;padding:1px 4px;border-radius:3px;'>SCORECARD</span> = U.S. Dept. of Education verified data</div>"
    if has_bls:
        ext_legend += "<div style='font-size:10px;margin-top:4px;'><span style='background:#8b5cf6;color:white;font-size:9px;font-weight:700;padding:1px 4px;border-radius:3px;'>BLS DATA</span> = Bureau of Labor Statistics verified data</div>"

    # ── Final HTML ────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Inter",sans-serif;background:#f8fafc;padding:16px;}}</style>
</head><body>
<div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);border-radius:10px;padding:14px 18px;
    margin-bottom:4px;color:white;text-align:center;">
  <div style="font-size:13px;font-weight:700;">{opt_a} vs {opt_b}</div>
  <div style="font-size:11px;opacity:0.8;margin-top:2px;">{len(factors)} factors — {"paired comparison" if subtype=="university_comparison" else "factor analysis"}</div>
</div>
<div style="display:flex;justify-content:center;"><div style="width:2px;height:14px;background:#cbd5e1;"></div></div>
<div style="display:grid;grid-template-columns:1fr 270px;gap:12px;align-items:start;">
  <div style="background:white;border-radius:10px;padding:14px;border:1px solid #e2e8f0;">
    <div style="font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px;">
      Factors That Shaped This Decision
    </div>
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
      <div style="width:24px;text-align:center;">
        <div style="width:12px;height:12px;border-radius:50%;background:#2563eb;margin:auto;"></div>
      </div>
      <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:4px 10px;
          font-size:11px;font-weight:600;color:#2563eb;">Decision Root</div>
    </div>
    {factor_nodes_html}
    {ext_legend}
  </div>
  <div>
    <div style="background:{win_color}18;border:2px solid {win_color}50;border-radius:10px;
        padding:12px 14px;margin-bottom:10px;text-align:center;">
      <div style="font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:4px;">Council Outcome</div>
      <div style="font-size:16px;font-weight:700;color:{win_color};">{winner}</div>
      <div style="font-size:12px;color:{win_color};opacity:0.8;">{win_pct}% aggregate lean</div>
    </div>
    <div style="background:white;border-radius:10px;padding:12px 14px;border:1px solid #e2e8f0;">
      <div style="font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;">Agent Votes</div>
      {agent_rows_html}
    </div>
    <div style="background:#f8fafc;border-radius:8px;padding:10px 12px;margin-top:10px;border:1px solid #e2e8f0;">
      <div style="font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;margin-bottom:6px;">Legend</div>
      <div style="font-size:10px;color:#16a34a;margin-bottom:3px;">🟢 → {opt_a[:18]}: factor favors this option</div>
      <div style="font-size:10px;color:#2563eb;margin-bottom:3px;">🔵 → {opt_b[:18]}: factor favors this option</div>
      <div style="font-size:10px;color:#64748b;">⚫ ↔ Both: neutral or applies to both</div>
    </div>
  </div>
</div>
</body></html>"""

    height = max(520, len(factors) * 68 + 300)
    components.html(html, height=height, scrolling=True)


# ── Council view ───────────────────────────────────────────────────────────────
def render_council_perspectives():
    if not st.session_state.llm:
        st.warning("System not ready")
        return

    try:
        meta          = st.session_state.state.decision_metadata
        decision_type = meta.get("decision_type", "unknown")
        options       = meta.get("options_being_compared", [])
        print(f"[COUNCIL] Type: {decision_type}, Options: {options}")

        if len(options) >= 2:
            option_a, option_b = options[0], options[1]

            # Header
            st.markdown(f"""
            <div style='text-align:center;padding:20px;
                        background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);
                        border-radius:10px;margin-bottom:20px;'>
                <h2 style='color:white;margin:0;'>Council of Experts</h2>
                <p style='color:#f0f0f0;margin:8px 0 0 0;'>
                    {option_a} vs {option_b} — 3 independent analyses
                </p>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("The council is deliberating..."):
                p = st.session_state.llm.generate_council_perspectives(
                    st.session_state.state.to_dict()
                )

            agents   = p.get("agents", [])
            votes    = p.get("agent_votes", {})
            avg_vote = p.get("avg_vote", {})
            avg_a    = avg_vote.get("option_a", 50)
            avg_b    = avg_vote.get("option_b", 50)

            # ── 3 Agent analysis cards ────────────────────────────────────────
            st.markdown("### 🔍 Expert Analyses")
            cols = st.columns(len(agents))
            for i, agent in enumerate(agents):
                with cols[i]:
                    v      = votes.get(agent["id"], {})
                    vote_a = v.get("option_a", 50)
                    vote_b = v.get("option_b", 50)
                    lean   = option_a if vote_a >= vote_b else option_b
                    lean_pct = max(vote_a, vote_b)

                    st.markdown(f"""
                    <div style='background:{agent["bg"]};padding:12px;border-radius:10px;
                                border-left:5px solid {agent["border"]};margin-bottom:8px;'>
                        <div style='font-size:1.5em;'>{agent["emoji"]}</div>
                        <strong style='color:{agent["color"]};font-size:0.85em;'>{agent["name"]}</strong><br>
                        <span style='font-size:1em;font-weight:bold;'>Leans: {lean}</span><br>
                        <span style='color:#555;font-size:0.78em;'>{option_a}: {vote_a}% | {option_b}: {vote_b}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Extract ANALYSIS and KEY INSIGHT sections robustly
                    # The agent output can have multi-line ANALYSIS — don't split on \n
                    raw = v.get("raw", "")
                    analysis    = ""
                    key_insight = ""

                    import re as _re
                    # Extract ANALYSIS: block (everything up to next label or end)
                    m_analysis = _re.search(
                        r"ANALYSIS:\s*(.+?)(?=\nKEY INSIGHT:|\nLEAN:|$)",
                        raw, _re.DOTALL | _re.IGNORECASE
                    )
                    if m_analysis:
                        analysis = m_analysis.group(1).strip()

                    # Extract KEY INSIGHT: line
                    m_insight = _re.search(
                        r"KEY INSIGHT:\s*(.+?)(?=\n[A-Z]+:|$)",
                        raw, _re.DOTALL | _re.IGNORECASE
                    )
                    if m_insight:
                        key_insight = m_insight.group(1).strip()

                    # Fallback: if regex failed, try line-by-line
                    if not analysis:
                        in_analysis = False
                        lines_buf   = []
                        for line in raw.split("\n"):
                            if line.strip().upper().startswith("ANALYSIS:"):
                                in_analysis = True
                                rest = line.split(":", 1)[-1].strip()
                                if rest:
                                    lines_buf.append(rest)
                            elif in_analysis and line.strip().upper().startswith("KEY INSIGHT:"):
                                break
                            elif in_analysis and line.strip():
                                lines_buf.append(line.strip())
                        analysis = " ".join(lines_buf)

                    if analysis:
                        # Show full analysis text, not truncated
                        st.markdown(f"""
                        <div style='font-size:0.82em;color:#374151;line-height:1.5;
                                    padding:8px 0;border-top:1px solid #e5e7eb;margin-top:6px;'>
                            {analysis}
                        </div>
                        """, unsafe_allow_html=True)
                    if key_insight:
                        st.markdown(f"""
                        <div style='background:white;padding:6px 10px;border-radius:6px;
                                    border-left:3px solid {agent["border"]};margin-top:6px;
                                    font-size:0.78em;font-weight:600;color:{agent["color"]};'>
                            💡 {key_insight}
                        </div>
                        """, unsafe_allow_html=True)

     # ── Aggregate vote bar ──────────────────────────────────────────────
            st.markdown("---")
            winner    = option_a if avg_a >= avg_b else option_b
            win_color = "#16a34a" if avg_a >= avg_b else "#dc2626"
            st.markdown(f"""
            <div style='margin:8px 0 4px 0;'>
              <div style='display:flex;border-radius:8px;overflow:hidden;height:24px;'>
                <div style='width:{avg_a}%;background:#16a34a;display:flex;align-items:center;
                    justify-content:center;font-size:11px;font-weight:700;color:white;min-width:0;'>
                  {"" if avg_a < 20 else f"{avg_a}%"}
                </div>
                <div style='width:{avg_b}%;background:#dc2626;display:flex;align-items:center;
                    justify-content:center;font-size:11px;font-weight:700;color:white;min-width:0;'>
                  {"" if avg_b < 20 else f"{avg_b}%"}
                </div>
              </div>
              <div style='display:flex;justify-content:space-between;margin-top:5px;font-size:12px;'>
                <span style='color:#16a34a;font-weight:600;'>🟢 {option_a} {avg_a}%</span>
                <span style='background:{win_color}15;color:{win_color};border:1px solid {win_color}40;
                    font-weight:700;border-radius:4px;padding:2px 10px;font-size:11px;'>→ {winner} wins</span>
                <span style='color:#dc2626;font-weight:600;'>{avg_b}% {option_b} 🔴</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Synthesizer ruling ────────────────────────────────────────────
            st.markdown("---")
            st.markdown("""
            <div style='background:#fff3e0;padding:15px;border-radius:10px;
                        border-left:5px solid #ff9800;margin-bottom:10px;'>
                <h3 style='color:#e65100;margin:0;'>⚖️ The Synthesizer Rules</h3>
            </div>
            """, unsafe_allow_html=True)

            synth_text = p.get("synthesizer", "No ruling available")
            if "OPEN QUESTION:" in synth_text:
                parts = synth_text.split("OPEN QUESTION:")
                st.markdown(parts[0].strip())
                st.markdown(f"""
                <div style='background:#e8eaf6;padding:12px;border-radius:8px;
                            border-left:4px solid #5c6bc0;margin-top:12px;'>
                    <strong style='color:#3949ab;'>💭 Something to reflect on:</strong><br>
                    {parts[1].strip()}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(synth_text)

            # ── Decision reasoning tree ───────────────────────────────────────
            st.markdown("---")
            with st.expander("🌳 Decision Factor Tree", expanded=False):
                st.caption(
                    "Each branch is a fact collected from the conversation. "
                    "Green = favors the left option. Blue = favors the right option. "
                    "Grey = neutral or applies to both."
                )
                render_decision_tree(st.session_state.state.to_dict(), p)

        else:
            # No options — general 3-analyst fallback
            st.markdown("""
            <div style='text-align:center;padding:20px;
                        background:linear-gradient(90deg,#ff6b6b 0%,#4ecdc4 50%,#45b7d1 100%);
                        border-radius:10px;margin-bottom:20px;'>
                <h2 style='color:white;margin:0;'>Council of Experts</h2>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("The council is deliberating..."):
                p = st.session_state.llm.generate_council_perspectives(
                    st.session_state.state.to_dict()
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### ⚠️ Risk Analyst")
                st.markdown(p.get("risk", "No analysis"))
            with col2:
                st.markdown("#### 🚀 Opportunity Analyst")
                st.markdown(p.get("opportunity", "No analysis"))
            with col3:
                st.markdown("#### 🎯 Values Analyst")
                st.markdown(p.get("values", "No analysis"))

        # ── Navigation buttons ────────────────────────────────────────────────
        st.markdown("---")
        col_back, col_new = st.columns(2)
        with col_back:
            if st.button("← Back to Chat", use_container_width=True):
                st.session_state.show_council = False
                st.rerun()
        with col_new:
            if st.button("Start New Decision", use_container_width=True, type="primary"):
                st.session_state.state = DecisionState()
                st.session_state.messages = []
                st.session_state.show_council = False
                st.session_state.chat_locked = False
                if st.session_state.llm:
                    st.session_state.llm.reset_conversation()
                st.rerun()

    except Exception as e:
        st.error(f"Error generating council perspectives: {str(e)}")
        print(f"[COUNCIL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        col_back, col_new = st.columns(2)
        with col_back:
            if st.button("Back to Chat"):
                st.session_state.show_council = False
                st.rerun()
        with col_new:
            if st.button("Start New Decision"):
                st.session_state.state = DecisionState()
                st.session_state.messages = []
                st.session_state.show_council = False
                st.session_state.chat_locked = False
                if st.session_state.llm:
                    st.session_state.llm.reset_conversation()
                st.rerun()


def process_message(user_message: str):
    if not st.session_state.llm:
        st.error("Please ensure system is initialized")
        return

    print(f"\n=== PROCESSING MESSAGE: {user_message[:60]} ===")

    try:
        # Step 1: extract constraints
        extracted = st.session_state.llm.extract_constraints(
            user_message,
            st.session_state.state.to_dict()
        )
        print(f"Extracted: {json.dumps(extracted, indent=2)}")

        # Step 2: update symbolic state
        if extracted.get("extracted"):
            for category, updates in extracted["extracted"].items():
                if hasattr(st.session_state.state, category):
                    for key, value in updates.items():
                        print(f"  {category}.{key} = {value}")
                        st.session_state.state.update(category, key, value)
                else:
                    print(f"  WARNING: unknown category '{category}'")

        state_dict = st.session_state.state.to_dict()
        print(f"  decision_metadata: {state_dict.get('decision_metadata', {})}")
        print(f"  interests: {state_dict.get('interests', {})}")

        # Step 3: generate response
        response = st.session_state.llm.generate_response(
            user_message,
            st.session_state.state.to_dict(),
            mode="conversational"
        )

        st.session_state.messages.append({"role": "user", "content": user_message})
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Bug 2: lock chat if this is a conclusion response
        conclusion_phrases = [
            "Council of Experts", "council of experts",
            "'Council of Experts' button", "See Council",
        ]
        if any(phrase in response for phrase in conclusion_phrases):
            st.session_state.chat_locked = True
            print("[APP] Chat locked — conclusion reached")

        print("=== DONE ===\n")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        st.error(f"An error occurred: {str(e)}")


# ── Chat view ──────────────────────────────────────────────────────────────────
def render_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Bug 2: locked vs active input
    if st.session_state.chat_locked:
        st.info(
            "I have enough information. Click **'See Council of Experts'** above "
            "to get the full multi-perspective analysis.",
            icon="💡"
        )
    else:
        if prompt := st.chat_input("What decision are you working through?"):
            process_message(prompt)
            st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    render_header()

    with st.sidebar:
        st.markdown("### System")

        if not st.session_state.llm:
            with st.spinner("Initializing AI..."):
                if initialize_llm():
                    st.success("Ready")
                else:
                    st.error("System error")
                    if "llm_error" in st.session_state:
                        st.code(st.session_state.llm_error)
        else:
            st.success("System Ready")
            st.caption("Using Llama 3.3 70B · Groq")

        st.markdown("---")
        render_sidebar_state()
        st.markdown("---")

        if st.button("New Decision", use_container_width=True):
            st.session_state.state = DecisionState()
            st.session_state.messages = []
            st.session_state.show_council = False
            st.session_state.chat_locked = False
            if st.session_state.llm:
                st.session_state.llm.reset_conversation()
            st.rerun()

    # Council full-screen view
    if st.session_state.get("show_council", False):
        render_council_perspectives()
        return

    render_chat()

    # Council button appears BELOW the chat (fix: was showing at top)
    if is_conversation_complete():
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🎭 See Council of Experts",
                use_container_width=True,
                type="primary"
            ):
                st.session_state.show_council = True
                st.rerun()
        st.markdown("---")

    if len(st.session_state.messages) == 0:
        st.markdown("""
        ### Welcome!

        I help you think clearly about complex decisions. I won't tell you what to do, but I will:

        - Track your constraints and priorities
        - Catch logical inconsistencies
        - Show you different perspectives on your situation

        **How it works:**
        1. Tell me about your decision
        2. I'll ask a few targeted questions
        3. Once I have enough context, a **Council of Experts** will debate your options

        **Example:** *"I'm deciding between Computer Science and Data Science for my Masters."*
        """)


if __name__ == "__main__":
    main()