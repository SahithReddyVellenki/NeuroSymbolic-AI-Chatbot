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
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

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
        st.session_state.llm = LLMInterface(api_key=GOOGLE_API_KEY)
        return True
    except Exception as e:
        st.session_state.llm_error = str(e)
        return False


# ── Bug 1 fix: conversation-complete detection ─────────────────────────────────
def is_conversation_complete() -> bool:
    """Check if the system has gathered enough info to show the council."""
    if len(st.session_state.messages) < 4:
        return False

    state_dict = st.session_state.state.to_dict()
    decision_type = state_dict.get("decision_metadata", {}).get("decision_type")

    # Primary signal: last assistant message contains a conclusion phrase
    if st.session_state.messages:
        last_msg = st.session_state.messages[-1]
        if last_msg["role"] == "assistant":
            content = last_msg["content"]
            conclusion_phrases = [
                "Council of Experts",
                "council of experts",
                "Council of Expert",
                "See Council",
                "'Council of Experts' button",
            ]
            if any(phrase in content for phrase in conclusion_phrases):
                return True

    # Fallback: enough facts collected for the decision type
    if decision_type in ("career_choice", "education"):
        relevant_count = sum(
            1 for cat in [
                state_dict.get("interests", {}),
                state_dict.get("career_vision", {}),
                state_dict.get("values", {}),
            ]
            for v in cat.values()
            if v is not None and v is not False and v != ""
        )
        return relevant_count >= 4
    else:
        relevant_count = sum(
            1 for cat in [
                state_dict.get("financial", {}),
                state_dict.get("values", {}),
                state_dict.get("current", {}),
                state_dict.get("opportunity", {}),
            ]
            for v in cat.values()
            if v is not None
        )
        return relevant_count >= 12


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

def render_decision_tree(state_dict: dict, council_results: dict):
    """
    Renders a visual decision tree using the symbolic state facts.
    Fully dynamic — builds fact cards from whatever is actually in state_dict.
    Uses a master registry to assign each field to the right agent.
    """
    options   = state_dict.get("decision_metadata", {}).get("options_being_compared", ["Option A", "Option B"])
    option_a  = options[0] if len(options) > 0 else "Option A"
    option_b  = options[1] if len(options) > 1 else "Option B"
    avg_vote  = council_results.get("avg_vote", {})
    avg_a     = avg_vote.get("option_a", 50)
    avg_b     = avg_vote.get("option_b", 50)
    agent_votes = council_results.get("agent_votes", {})

    def fmt(v):
        """Format a raw value for display. Returns None if it's empty/placeholder."""
        if v is None:                        return None
        if isinstance(v, bool):              return "Yes" if v else "No"
        s = str(v).strip()
        if s == "":                          return None
        if s.lower() in _PLACEHOLDER_VALUES: return None
        # Convert float like 8.0 → "8"
        try:
            f = float(s)
            if f == int(f): return str(int(f))
        except: pass
        return s

    # ── Build agent_facts dynamically from state_dict + registry ──────────────
    # agent_id → list of fact dicts
    agent_facts = {"financial": [], "growth": [], "wellbeing": [], "values_agent": []}

    # Deduplicate: track (agent, label) pairs we've already added
    seen = set()

    # Iterate over every category and field in state_dict
    skip_categories = {"decision_metadata", "violations", "missing_info", "decision_mode", "history"}
    for category, cat_data in state_dict.items():
        if category in skip_categories:         continue
        if not isinstance(cat_data, dict):      continue
        for field, raw_value in cat_data.items():
            display_val = fmt(raw_value)
            if display_val is None:             continue  # nothing to show
            key = (category, field)
            if key not in _FACT_REGISTRY:       continue  # field not mapped
            agent_id, label, note, impact_fn = _FACT_REGISTRY[key]
            dedup_key = (agent_id, label)
            if dedup_key in seen:               continue  # already added
            seen.add(dedup_key)
            impact = impact_fn(raw_value)
            agent_facts[agent_id].append({
                "label": label,
                "value": display_val,
                "note":  note,
                "impact": impact,
            })

    # Rename values_agent → values for rendering
    agent_facts["values"] = agent_facts.pop("values_agent", [])

    # ── Agent metadata ─────────────────────────────────────────────────────────
    AGENTS = [
        {"id": "financial", "name": "💰 Financial Security", "border": "#22c55e", "text": "#16a34a", "bg": "#f0fdf4"},
        {"id": "growth",    "name": "📈 Career Growth",      "border": "#3b82f6", "text": "#1d4ed8", "bg": "#eff6ff"},
        {"id": "wellbeing", "name": "🧠 Mental Wellbeing",   "border": "#a855f7", "text": "#7e22ce", "bg": "#faf5ff"},
        {"id": "values",    "name": "🎯 Values Alignment",   "border": "#f97316", "text": "#c2410c", "bg": "#fff7ed"},
    ]

    IMPACT_COLORS = {
        "high-risk":   "#ef4444",
        "risk":        "#f97316",
        "conflict":    "#ef4444",
        "caution":     "#fb923c",
        "moderate":    "#eab308",
        "positive":    "#22c55e",
        "pull-factor": "#22c55e",
        "push-factor": "#f97316",
        "neutral":     "#9ca3af",
    }

    # ── Build HTML ─────────────────────────────────────────────────────────────
    winner       = option_a if avg_a >= avg_b else option_b
    winner_color = "#16a34a" if avg_a >= avg_b else "#dc2626"

    agent_cols_html = ""
    for ag in AGENTS:
        aid   = ag["id"]
        v     = agent_votes.get(aid, {})
        va    = v.get("option_a", 50)
        vb    = v.get("option_b", 50)
        facts = agent_facts.get(aid, [])

        facts_html = ""
        for f in facts:
            ic = IMPACT_COLORS.get(f["impact"], "#9ca3af")
            facts_html += f"""
            <div class="fact-card" style="border-left:3px solid {ag['border']}33;">
              <div class="fact-header">
                <span class="fact-label">{f['label']}</span>
                <span class="fact-badge" style="background:{ic}18;color:{ic};border:1px solid {ic}44;">{f['impact']}</span>
              </div>
              <div class="fact-value">{f['value']}</div>
              <div class="fact-note">{f['note']}</div>
            </div>"""

        if not facts_html:
            facts_html = '<div class="fact-card no-facts">No facts collected for this agent</div>'

        agent_cols_html += f"""
        <div class="agent-col">
          <div class="connector-v" style="background:linear-gradient(to bottom,#e2e8f0,{ag['border']});"></div>
          <div class="agent-node" style="border:2px solid {ag['border']};background:{ag['bg']};">
            <div class="agent-name" style="color:{ag['text']};">{ag['name']}</div>
            <div class="vote-bar-wrap">
              <div class="vote-bar-a" style="width:{va}%;"></div>
              <div class="vote-bar-b" style="width:{vb}%;"></div>
            </div>
            <div class="vote-labels">
              <span style="color:#16a34a;">{option_a[:14]} {va}%</span>
              <span style="color:#dc2626;">{option_b[:14]} {vb}%</span>
            </div>
          </div>
          <div class="connector-v" style="background:{ag['border']}88;height:16px;"></div>
          <div class="facts-col">{facts_html}</div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Inter', sans-serif; background: #f8fafc; padding: 20px; }}

  .tree-header {{ text-align: center; margin-bottom: 24px; }}
  .tree-title {{ font-size: 13px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }}
  .tree-subtitle {{ font-size: 11px; color: #94a3b8; }}

  .root-node {{
    background: white; border: 2px solid #e2e8f0; border-radius: 14px;
    padding: 18px 28px; max-width: 520px; margin: 0 auto;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  }}
  .root-label {{ font-size: 11px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }}
  .root-options {{ font-size: 16px; font-weight: 700; margin-bottom: 14px; }}
  .opt-a {{ color: #16a34a; }}
  .opt-sep {{ color: #cbd5e1; margin: 0 10px; }}
  .opt-b {{ color: #dc2626; }}
  .overall-bar {{ display: flex; border-radius: 6px; overflow: hidden; height: 10px; margin-bottom: 8px; }}
  .overall-bar-a {{ background: linear-gradient(to right, #16a34a, #22c55e); }}
  .overall-bar-b {{ background: linear-gradient(to right, #dc2626, #f87171); }}
  .overall-labels {{ display: flex; justify-content: space-between; font-size: 12px; font-family: 'JetBrains Mono', monospace; align-items: center; }}
  .winner-chip {{
    background: {winner_color}15; color: {winner_color}; border: 1px solid {winner_color}44;
    border-radius: 5px; padding: 1px 10px; font-size: 11px; font-weight: 700;
  }}

  .horiz-bar {{ max-width: 900px; margin: 0 auto; position: relative; height: 28px; }}
  .horiz-line {{ position: absolute; top: 0; left: 12.5%; right: 12.5%; height: 2px;
    background: linear-gradient(to right, #22c55e55, #3b82f655, #a855f755, #f9731655); }}
  .vert-tick {{ position: absolute; top: 0; width: 2px; height: 28px; transform: translateX(-50%); }}

  .agents-row {{ display: flex; gap: 12px; max-width: 960px; margin: 0 auto; align-items: flex-start; }}
  .agent-col {{ display: flex; flex-direction: column; align-items: center; flex: 1; min-width: 0; }}
  .connector-v {{ width: 2px; height: 28px; }}

  .agent-node {{ width: 100%; border-radius: 10px; padding: 12px 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .agent-name {{ font-size: 12px; font-weight: 700; margin-bottom: 8px; }}
  .vote-bar-wrap {{ display: flex; border-radius: 4px; overflow: hidden; height: 7px; margin-bottom: 5px; }}
  .vote-bar-a {{ background: #22c55e; }}
  .vote-bar-b {{ background: #ef4444; }}
  .vote-labels {{ display: flex; justify-content: space-between; font-size: 10px; font-family: 'JetBrains Mono', monospace; }}

  .facts-col {{ width: 100%; display: flex; flex-direction: column; gap: 6px; }}
  .fact-card {{
    background: white; border-radius: 8px; padding: 9px 11px;
    border: 1px solid #f1f5f9; transition: box-shadow 0.15s, transform 0.15s;
    cursor: default;
  }}
  .fact-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.10); transform: translateY(-1px); }}
  .no-facts {{ color: #9ca3af; font-size: 11px; border: 1px dashed #e2e8f0; text-align: center; padding: 12px; }}
  .fact-header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 6px; margin-bottom: 4px; }}
  .fact-label {{ font-size: 10px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; line-height: 1.3; }}
  .fact-badge {{ font-size: 9px; font-family: 'JetBrains Mono', monospace; border-radius: 4px; padding: 1px 6px; white-space: nowrap; flex-shrink: 0; }}
  .fact-value {{ font-size: 13px; font-weight: 600; color: #1e293b; margin-bottom: 3px; font-family: 'JetBrains Mono', monospace; }}
  .fact-note {{ font-size: 10px; color: #94a3b8; line-height: 1.4; display: none; }}
  .fact-card:hover .fact-note {{ display: block; }}

  .legend {{ display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; margin-top: 20px; padding-top: 16px; border-top: 1px solid #e2e8f0; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 10px; color: #64748b; }}
  .legend-dot {{ width: 8px; height: 8px; border-radius: 50%; }}
</style>
</head>
<body>
  <div class="tree-header">
    <div class="tree-title">🌳 Decision Reasoning Tree</div>
    <div class="tree-subtitle">Hover any fact card to see how it influenced the council's reasoning</div>
  </div>

  <div class="root-node">
    <div class="root-label">Decision</div>
    <div class="root-options">
      <span class="opt-a">{option_a}</span>
      <span class="opt-sep">vs</span>
      <span class="opt-b">{option_b}</span>
    </div>
    <div class="overall-bar">
      <div class="overall-bar-a" style="width:{avg_a}%;"></div>
      <div class="overall-bar-b" style="width:{avg_b}%;"></div>
    </div>
    <div class="overall-labels">
      <span style="color:#16a34a;">{avg_a}% {option_a}</span>
      <span class="winner-chip">Council → {winner}</span>
      <span style="color:#dc2626;">{avg_b}% {option_b}</span>
    </div>
  </div>

  <div class="horiz-bar">
    <div class="horiz-line"></div>
    <div class="vert-tick" style="left:12.5%;background:#22c55e88;"></div>
    <div class="vert-tick" style="left:37.5%;background:#3b82f688;"></div>
    <div class="vert-tick" style="left:62.5%;background:#a855f788;"></div>
    <div class="vert-tick" style="left:87.5%;background:#f9731688;"></div>
  </div>

  <div class="agents-row">
    {agent_cols_html}
  </div>

  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#ef4444;"></div>high-risk / conflict</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f97316;"></div>risk / caution</div>
    <div class="legend-item"><div class="legend-dot" style="background:#eab308;"></div>moderate</div>
    <div class="legend-item"><div class="legend-dot" style="background:#22c55e;"></div>positive</div>
    <div class="legend-item"><div class="legend-dot" style="background:#9ca3af;"></div>neutral</div>
  </div>
</body>
</html>"""

    max_facts = max((len(agent_facts.get(ag["id"], [])) for ag in AGENTS), default=3)
    estimated_height = 200 + 60 + 90 + (max_facts * 66) + 80
    components.html(html, height=estimated_height, scrolling=False)


# ── Council view ───────────────────────────────────────────────────────────────
def render_council_perspectives():
    if not st.session_state.llm:
        st.warning("System not ready")
        return

    try:
        meta = st.session_state.state.decision_metadata
        decision_type = meta.get("decision_type", "unknown")
        options = meta.get("options_being_compared", [])

        print(f"[COUNCIL] Type: {decision_type}, Options: {options}")

        if decision_type in ("career_choice", "education") and len(options) == 2:
            st.markdown(f"""
            <div style='text-align:center;padding:20px;
                        background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);
                        border-radius:10px;margin-bottom:20px;'>
                <h2 style='color:white;margin:0;'>Council of Experts</h2>
                <p style='color:#f0f0f0;margin:8px 0 0 0;'>
                    {options[0]} vs {options[1]} — 4 agents, one ruling
                </p>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("The council is deliberating..."):
                p = st.session_state.llm.generate_council_perspectives(
                    st.session_state.state.to_dict()
                )

            option_a, option_b = options[0], options[1]
            agents    = p.get("agents", [])
            votes     = p.get("agent_votes", {})
            avg_vote  = p.get("avg_vote", {})

            # ── Round 1: agent vote cards ─────────────────────────────────
            st.markdown("### Round 1 — Agent Votes")
            cols = st.columns(len(agents))
            for i, agent in enumerate(agents):
                with cols[i]:
                    v = votes.get(agent["id"], {})
                    vote_a = v.get("option_a", 50)
                    vote_b = v.get("option_b", 50)
                    leading = option_a if vote_a >= vote_b else option_b
                    pct     = max(vote_a, vote_b)

                    st.markdown(f"""
                    <div style='background:{agent["bg"]};padding:12px;border-radius:10px;
                                border-left:5px solid {agent["border"]};margin-bottom:8px;'>
                        <div style='font-size:1.5em;'>{agent["emoji"]}</div>
                        <strong style='color:{agent["color"]};font-size:0.85em;'>{agent["name"]}</strong><br>
                        <span style='font-size:1.1em;font-weight:bold;'>Votes: {leading}</span><br>
                        <span style='color:#555;font-size:0.8em;'>{option_a}: {vote_a}% | {option_b}: {vote_b}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                    raw = v.get("raw", "")
                    reasoning = ""
                    for line in raw.split("\n"):
                        if line.startswith("REASONING:"):
                            reasoning = line.replace("REASONING:", "").strip()
                    if reasoning:
                        st.caption(reasoning)

            # ── Vote tally bar ────────────────────────────────────────────
            st.markdown("---")
            avg_a = avg_vote.get("option_a", 50)
            avg_b = avg_vote.get("option_b", 50)
            st.markdown(f"**Aggregate vote: {option_a} {avg_a}% vs {option_b} {avg_b}%**")
            st.progress(avg_a / 100)

            # ── Round 2: debate ───────────────────────────────────────────
            debating = p.get("debating_agents", {})
            if debating and p.get("round2_a") and p.get("round2_b"):
                st.markdown("### Round 2 — Debate")
                agent_a_info = debating.get("a", {})
                agent_b_info = debating.get("b", {})
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div style='background:#e3f2fd;padding:12px;border-radius:8px;
                                border-left:4px solid #2196f3;margin-bottom:8px;'>
                        <strong style='color:#1565c0;'>{agent_a_info.get("emoji","")} {agent_a_info.get("name","Agent A")} — for {option_a}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(p.get("round2_a", ""))
                with col2:
                    st.markdown(f"""
                    <div style='background:#fce4ec;padding:12px;border-radius:8px;
                                border-left:4px solid #e91e63;margin-bottom:8px;'>
                        <strong style='color:#880e4f;'>{agent_b_info.get("emoji","")} {agent_b_info.get("name","Agent B")} — for {option_b}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(p.get("round2_b", ""))

            # ── Round 3 (if triggered) ────────────────────────────────────
            if p.get("has_round3"):
                st.markdown("### Round 3 — Tiebreaker")
                st.info("Close vote detected — triggering final round")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{debating.get('a',{}).get('emoji','')} {debating.get('a',{}).get('name','')}**")
                    st.markdown(p.get("round3_a", ""))
                with col2:
                    st.markdown(f"**{debating.get('b',{}).get('emoji','')} {debating.get('b',{}).get('name','')}**")
                    st.markdown(p.get("round3_b", ""))

            # ── Synthesizer ruling ────────────────────────────────────────
            st.markdown("---")
            st.markdown("""
            <div style='background:#fff3e0;padding:15px;border-radius:10px;
                        border-left:5px solid #ff9800;margin-bottom:10px;'>
                <h3 style='color:#e65100;margin:0;'>⚖️ The Synthesizer Rules</h3>
            </div>
            """, unsafe_allow_html=True)

            synth_text = p.get("synthesizer", "No ruling available")
            # Split out OPEN QUESTION if present and render it separately
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

            # ── Decision Tree ─────────────────────────────────────────────
            st.markdown("---")
            with st.expander("🌳 Decision Reasoning Tree — what facts shaped this outcome?", expanded=False):
                st.caption("Each branch shows the facts collected for that agent and how they influenced its vote. Hover a fact card to see the reasoning.")
                render_decision_tree(st.session_state.state.to_dict(), p)

        else:
            # General 3-analyst fallback
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
                st.markdown("#### Risk Analyst")
                st.markdown(p.get("risk", "No analysis"))
            with col2:
                st.markdown("#### Opportunity Analyst")
                st.markdown(p.get("opportunity", "No analysis"))
            with col3:
                st.markdown("#### Values Coach")
                st.markdown(p.get("values", "No analysis"))

        # ── Navigation buttons ────────────────────────────────────────────
        st.markdown("---")
        col_back, col_new = st.columns(2)
        with col_back:
            if st.button("Back to Chat", use_container_width=True):
                st.session_state.show_council = False
                st.rerun()
        with col_new:
            if st.button("Start New Decision", use_container_width=True):
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
            st.caption("Using Gemma 3 27B")

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