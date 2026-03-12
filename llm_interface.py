"""
LLM Interface - Natural language layer constrained by symbolic state
Uses Google Gemini SDK.
"""

import os
import json
import google.generativeai as genai
from typing import Dict, Optional, List, Tuple


class LLMInterface:
    """
    Handles natural language interaction with strict constraints.

    The LLM is used ONLY for:
    - Asking clarifying questions (2-3 lines, with humor, with contradiction callouts)
    - Presenting a score-based conclusion
    - Running a 4-agent structured debate with adaptive Round 3

    It CANNOT override symbolic facts or give direct recommendations.
    """

    # ── Fixed agent definitions ───────────────────────────────────────────────
    AGENTS = [
        {
            "id":    "financial",
            "name":  "Financial Security Agent",
            "emoji": "💰",
            "color": "#1b5e20",
            "bg":    "#e8f5e9",
            "border":"#4caf50",
            "lens":  "salary, financial stability, income trajectory, debt risk, earning potential",
        },
        {
            "id":    "growth",
            "name":  "Career Growth Agent",
            "emoji": "📈",
            "color": "#0d47a1",
            "bg":    "#e3f2fd",
            "border":"#2196f3",
            "lens":  "career trajectory, skill development, industry demand, path to leadership",
        },
        {
            "id":    "wellbeing",
            "name":  "Mental Wellbeing Agent",
            "emoji": "🧠",
            "color": "#4a148c",
            "bg":    "#f3e5f5",
            "border":"#9c27b0",
            "lens":  "stress, work-life balance, burnout risk, mental health, fulfillment",
        },
        {
            "id":    "values",
            "name":  "Values Alignment Agent",
            "emoji": "🎯",
            "color": "#bf360c",
            "bg":    "#fbe9e7",
            "border":"#ff5722",
            "lens":  "alignment with stated values, identity, authenticity, long-term purpose",
        },
    ]


    # ── Topic rotation for offer comparison ──────────────────────────────────
    # Each tuple: (field_to_check, category, topic_label, question)
    OFFER_QUESTION_TOPICS = [
        ("role",               "offer_a",  "offer A — company and role",
         "Tell me about the first offer — what's the company and the role?"),
        ("role",               "offer_b",  "offer B — company and role",
         "And the second offer — what company and what role?"),
        ("salary",             "offer_a",  "offer A — salary",
         "What's the salary or compensation for the first offer?"),
        ("salary",             "offer_b",  "offer B — salary",
         "And the salary for the second offer?"),
        ("work_location",      "offer_a",  "offer A — remote, onsite or hybrid?",
         "Is the first role remote, onsite, or hybrid?"),
        ("work_location",      "offer_b",  "offer B — remote, onsite or hybrid?",
         "And the second role — remote, onsite, or hybrid?"),
        ("requires_relocation","offer_a",  "offer A — relocation needed?",
         "Would you need to relocate for the first offer, or is it in your current city?"),
        ("requires_relocation","offer_b",  "offer B — relocation needed?",
         "Same question for the second offer — would it require relocating?"),
        ("can_relocate",       "personal", "personal — can you relocate?",
         "If relocation is involved, is that feasible for you right now? Any family or personal factors that affect that?"),
        ("growth_potential",   "offer_a",  "offer A — growth potential",
         "How does the growth potential look at the first company — clear paths to advance, or unclear?"),
        ("growth_potential",   "offer_b",  "offer B — growth potential",
         "And growth potential at the second company?"),
        ("work_life_balance",  "offer_a",  "offer A — work-life balance",
         "What's your sense of the work-life balance at the first company?"),
        ("work_life_balance",  "offer_b",  "offer B — work-life balance",
         "How about work-life balance at the second company?"),
        ("concern",            "offer_a",  "biggest concern about offer A",
         "What's your biggest concern or hesitation about the first offer?"),
        ("concern",            "offer_b",  "biggest concern about offer B",
         "And your biggest concern about the second offer?"),
        ("financial_security", "values",   "how important is financial security",
         "Overall, how important is financial security in this decision — on a scale of 1 to 10?"),
        ("career_growth",      "values",   "how important is career growth",
         "And career growth — how important is that to you right now?"),
        ("work_life_balance",  "values",   "how important is work-life balance",
         "Last one — how much does work-life balance weigh into this for you?"),
    ]

    # ── Topic rotation for university comparison ───────────────────────────────
    UNIVERSITY_QUESTION_TOPICS = [
        ("name",               "uni_a",   "university A — name and program",
         "Tell me about the first university — which school and what program?"),
        ("name",               "uni_b",   "university B — name and program",
         "And the second university — which school and program?"),
        ("tuition",            "uni_a",   "university A — tuition and cost",
         "What's the tuition or total cost for the first option? Any scholarships?"),
        ("tuition",            "uni_b",   "university B — tuition and cost",
         "And the cost for the second university? Any financial aid?"),
        ("requires_relocation","uni_a",   "university A — location and relocation",
         "Where is the first university located — would you need to move there?"),
        ("requires_relocation","uni_b",   "university B — location and relocation",
         "And the second university — same question, where is it and does it mean relocating?"),
        ("can_relocate",       "personal","personal — relocation feasibility",
         "If either requires moving, is that realistic for you right now — any family or life factors?"),
        ("living_cost",        "uni_a",   "university A — cost of living",
         "What's the cost of living like in the city where the first university is?"),
        ("living_cost",        "uni_b",   "university B — cost of living",
         "And cost of living for the second university's city?"),
        ("ranking",            "uni_a",   "university A — reputation and ranking",
         "How does the first university rank in your field — is it well known for this program?"),
        ("ranking",            "uni_b",   "university B — reputation and ranking",
         "And the second university's reputation in this field?"),
        ("job_placement",      "uni_a",   "university A — job placement",
         "Do you know anything about job placement or alumni network at the first school?"),
        ("job_placement",      "uni_b",   "university B — job placement",
         "Same question for the second school?"),
        ("concern",            "uni_a",   "biggest concern about university A",
         "What's your biggest concern or hesitation about the first university?"),
        ("concern",            "uni_b",   "biggest concern about university B",
         "And your biggest concern about the second university?"),
        ("financial_security", "values",  "how important is minimizing debt",
         "How important is minimizing student debt in this decision — 1 to 10?"),
        ("career_growth",      "values",  "how important is career reputation",
         "And how much does the university's reputation and career outcomes matter to you?"),
    ]

    # ── Topic rotation for major / field choice ──────────────────────────────
    MAJOR_CHOICE_QUESTION_TOPICS = [
        ("current_year",        "current",       "what year are you in",
         "What year are you in right now — are you still deciding before starting, or already a year or two in?"),
        ("leaning",             "current",       "current leaning and why",
         "Which one are you leaning toward currently, and what's pulling you that way?"),
        ("hands_on_work",       "interests",     "hands-on vs theoretical",
         "Do you prefer hands-on, practical work — building and tinkering — or more abstract, theoretical problem solving?"),
        ("post_graduation_goal","career_vision", "goal after graduation",
         "What do you want to do after graduating — industry job, further study, or something else?"),
        ("desired_role_5yr",    "career_vision", "5-year vision",
         "Where do you see yourself in 5 years — what kind of role or industry?"),
        ("financial_security",  "values",        "how important is salary",
         "How important is earning potential in this decision — on a scale of 1 to 10?"),
        ("enjoys_theory",       "interests",     "interest in theory vs building",
         "Which sounds more exciting — understanding how things work at a deep level, or building systems that solve real problems?"),
        ("career_growth",       "values",        "career growth priority",
         "Is fast career progression important to you, or are you more focused on doing work you genuinely find interesting?"),
        ("work_life_balance",   "values",        "work-life balance priority",
         "Which field do you think gives better work-life balance — and how much does that matter to you?"),
        ("job_market_concern",  "current",       "job market awareness",
         "How aware are you of the job market for each option — do you know which one tends to hire more or pay better right now?"),
        ("financial_concern",   "current",       "any financial or family constraints",
         "Are there any financial or family factors that might influence this choice — scholarships, family expectations, cost differences?"),
        ("concern",             "current",       "biggest concern about the choice",
         "What's your biggest concern about making the wrong choice here?"),
    ]

    MAJOR_CHOICE_TOPIC_CHECKS = {
        "what year are you in":               ("current",       ["current_year"]),
        "current leaning and why":            ("current",       ["leaning"]),
        "hands-on vs theoretical":            ("interests",     ["hands_on_work", "enjoys_theory", "enjoys_building_systems"]),
        "goal after graduation":              ("career_vision", ["post_graduation_goal"]),
        "5-year vision":                      ("career_vision", ["desired_role_5yr", "research_vs_applied"]),
        "how important is salary":            ("values",        ["financial_security"]),
        "interest in theory vs building":     ("interests",     ["enjoys_theory", "enjoys_building_systems", "hands_on_work"]),
        "career growth priority":             ("values",        ["career_growth", "impact"]),
        "work-life balance priority":         ("values",        ["work_life_balance"]),
        "job market awareness":               ("current",       ["job_market_concern"]),
        "any financial or family constraints":("current",       ["financial_concern"]),
        "biggest concern about the choice":   ("current",       ["concern"]),
    }

    # ── Topic rotation for job-vs-business decisions ─────────────────────────
    JOB_VS_BUSINESS_QUESTION_TOPICS = [
        ("current_satisfaction", "current",  "current job satisfaction",
         "How do you feel about your current job — are you enjoying it, just tolerating it, or actively miserable?"),
        ("business_idea",        "current",  "business idea clarity",
         "How developed is your business idea — do you have a specific concept, or is it still a vague feeling that you want to work for yourself?"),
        ("leave_reason",         "current",  "what's driving the urge to leave",
         "What's actually pushing you toward leaving — is it excitement about the business, frustration with the job, or both?"),
        ("current_role",         "current",  "current role and field",
         "What kind of work do you do now, and is the business idea in the same field or something different entirely?"),
        ("financial_runway",     "current",  "financial runway",
         "How long could you realistically survive financially without income — do you have savings set aside for this?"),
        ("has_family",           "personal", "family and financial obligations",
         "Do you have family dependents or significant financial commitments — mortgage, loans, kids — that factor into this?"),
        ("partner_employed",     "personal", "partner or support system",
         "Is there a partner or anyone else contributing to household income — or are you the sole earner?"),
        ("financial_security",   "values",   "risk tolerance",
         "How important is financial stability to you right now, on a scale of 1 to 10?"),
        ("career_growth",        "values",   "career growth priority",
         "Is the business idea about making more money, doing more meaningful work, or having more freedom — what's the main pull?"),
        ("business_validated",   "current",  "business idea validation",
         "Have you tested the business idea at all — any early customers, side income from it, or market research?"),
        ("work_life_balance",    "values",   "work-life balance priority",
         "Running a business often means longer hours, especially early on — how important is work-life balance to you?"),
        ("concern",              "current",  "biggest concern",
         "What's your biggest fear about making the wrong choice here?"),
    ]

    JOB_VS_BUSINESS_TOPIC_CHECKS = {
        "current job satisfaction":         ("current",  ["current_satisfaction"]),
        "business idea clarity":            ("current",  ["business_idea"]),
        "what's driving the urge to leave": ("current",  ["leave_reason"]),
        "current role and field":           ("current",  ["current_role"]),
        "financial runway":                 ("current",  ["financial_runway"]),
        "family and financial obligations": ("personal", ["has_family", "has_dependents"]),
        "partner or support system":        ("personal", ["partner_employed"]),
        "risk tolerance":                   ("values",   ["financial_security"]),
        "career growth priority":           ("values",   ["career_growth", "impact"]),
        "business idea validation":         ("current",  ["business_validated"]),
        "work-life balance priority":       ("values",   ["work_life_balance"]),
        "biggest concern":                  ("current",  ["concern"]),
    }

    # ── Topic rotation for data collection (education_path: PhD vs Job etc.) ────
    CAREER_QUESTION_TOPICS = [
        ("post_graduation_goal", "career_vision",
         "primary goal after graduation",
         "What's your primary goal after finishing your degree — are you looking to get a job, continue studying, or something else?"),
        ("hands_on_work", "interests",
         "work style preference",
         "Do you prefer hands-on practical work or more theoretical / research-oriented work?"),
        ("research_vs_applied", "career_vision",
         "applied vs research lean",
         "Do you lean more toward applied industry work, or do you enjoy deep research and publishing?"),
        ("current_satisfaction", "current",
         "current job satisfaction",
         "How satisfied are you with your current job on a scale of 1-10 — is something pushing you away, or is this purely about the PhD opportunity?"),
        ("desired_role_5yr", "career_vision",
         "5-year vision",
         "Where do you see yourself in 5 years — what kind of role or industry are you aiming for?"),
        ("financial_security", "values",
         "financial security priority",
         "How important is financial stability in this decision — a PhD stipend is significantly less than most industry salaries. Scale of 1-10?"),
        ("financial_runway", "current",
         "financial runway for PhD",
         "Do you have savings or financial support that would make living on a PhD stipend workable — any dependents or loans to consider?"),
        ("has_family", "personal",
         "family or personal obligations",
         "Do you have a partner, kids, or any personal obligations that would be affected by a PhD — location, time commitment, income drop?"),
        ("career_growth", "values",
         "growth vs balance",
         "Is fast career progression more important to you, or doing work you find genuinely interesting — even if it pays less or takes longer?"),
        ("work_life_balance", "values",
         "work-life balance priority",
         "PhDs can be demanding and isolating — how much does work-life balance factor into your thinking on a scale of 1-10?"),
        ("concern", "current",
         "biggest concern about the choice",
         "What's your biggest fear about making the wrong call here — financial, losing industry momentum, or something else?"),
        ("job_market_concern", "current",
         "industry opportunity cost",
         "Do you feel like staying in your current job has real upside — promotions, growth, interesting projects — or does it feel like it's plateauing?"),
    ]

    def __init__(self, api_key: Optional[str] = None, model: str = "gemma-3-27b-it"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key required.")
        genai.configure(api_key=self.api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)
        self.conversation_history: List[Dict] = []
        self.asked_topics: Dict[str, int] = {}  # label -> times asked, for loop guard

    # ── Core API call ─────────────────────────────────────────────────────────
    def _call_gemini(self, messages: List[Dict], system_prompt: str = "") -> str:
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            full_prompt += f"{role}: {msg['content']}\n\n"

        try:
            response = self.model.generate_content(full_prompt)
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            if hasattr(response, "parts") and response.parts:
                return response.parts[0].text.strip()
            return "I'm having trouble responding. Could you rephrase that?"
        except Exception as e:
            print(f"[API] Error: {e}")
            return "I encountered an error. Could you repeat that?"

    # ── Constraint extraction ─────────────────────────────────────────────────
    def extract_constraints(self, user_message: str, current_state: Dict) -> Dict:
        is_first = not current_state.get("decision_metadata", {}).get("decision_type")

        if is_first:
            system_prompt = f"""The user said: "{user_message}"

Detect the decision type, subtype, and options. Be precise.

Examples:

JOB OFFER COMPARISON — "2 job offers", "choosing between two offers", "job at Google vs startup":
→ {{"decision_metadata": {{"decision_type": "career_choice", "decision_subtype": "offer_comparison", "options_being_compared": ["<company/offer A name>", "<company/offer B name>"]}}}}
→ If companies not named yet, use ["Offer 1", "Offer 2"]

UNIVERSITY COMPARISON — "which university should I pick", "MIT vs Stanford for masters", "choosing between two schools":
→ {{"decision_metadata": {{"decision_type": "education", "decision_subtype": "university_comparison", "options_being_compared": ["<university A>", "<university B>"]}}}}
→ If universities not named yet, use ["University 1", "University 2"]

EDUCATION PATH — "PhD vs job", "should I do a PhD", "masters or work":
→ {{"decision_metadata": {{"decision_type": "education", "decision_subtype": "education_path", "options_being_compared": ["PhD", "Job"]}}}}

MAJOR/FIELD CHOICE — "CS vs DS", "computer science or data science", "which major":
→ {{"decision_metadata": {{"decision_type": "career_choice", "decision_subtype": "major_choice", "options_being_compared": ["CS", "DS"]}}}}

LOCATION — "move to NYC or stay", "which city should I move to":
→ {{"decision_metadata": {{"decision_type": "location", "decision_subtype": "location_choice", "options_being_compared": ["NYC", "Stay"]}}}}

JOB VS BUSINESS — "should I quit my job", "resign and start a business", "leave job to start startup", "job or entrepreneurship":
→ {{"decision_metadata": {{"decision_type": "career_choice", "decision_subtype": "job_vs_business", "options_being_compared": ["Start Business", "Continue Job"]}}}}
→ If they name the business or job specifically, use those names

ANYTHING ELSE:
→ {{"decision_metadata": {{"decision_type": "general", "decision_subtype": "general", "options_being_compared": []}}}}

Return ONLY valid JSON:
{{"extracted": {{"decision_metadata": {{...}}}}}}"""

        else:
            decision_type    = current_state.get("decision_metadata", {}).get("decision_type", "general")
            decision_subtype = current_state.get("decision_metadata", {}).get("decision_subtype", "general")
            options          = current_state.get("decision_metadata", {}).get("options_being_compared", [])
            opt_a = options[0] if options else "Offer 1"
            opt_b = options[1] if len(options) > 1 else "Offer 2"

            if decision_subtype == "offer_comparison":
                system_prompt = f"""Extract facts from this message about a job offer comparison ({opt_a} vs {opt_b}).

Message: "{user_message}"

OFFER A details (category: "offer_a") — facts about {opt_a}:
- company: string
- role: string (job title)
- salary: number ("80k"→80000, "6 LPA"→600000, "1.5 lakh/month"→1800000)
- salary_raw: string (original text)
- work_location: "remote" / "onsite" / "hybrid"
- city: string (where the job is located)
- requires_relocation: true / false
- growth_potential: "high" / "medium" / "low" or description
- work_life_balance: "great" / "ok" / "poor" or 1-10
- job_security: "high" / "medium" / "low"
- culture: string description
- concern: string (user's hesitation)

OFFER B details (category: "offer_b") — same fields as offer_a

PERSONAL CONTEXT (category: "personal"):
- has_family: true / false
- has_dependents: true / false
- partner_employed: true / false
- can_relocate: true / false
- relocation_concern: string description
- current_city: string

VALUES (category: "values"):
- financial_security (1-10), career_growth (1-10), work_life_balance (1-10)

MAPPING RULES:
- "first offer / offer 1 / {opt_a}" → offer_a; "second offer / offer 2 / {opt_b}" → offer_b
- "same salary / same pay / both pay the same" → offer_a: {{salary_raw: "same", salary: null}}, offer_b: {{salary_raw: "same", salary: null}}
- "same benefits / same compensation" → offer_a: {{benefits: "same"}}, offer_b: {{benefits: "same"}}
- "same work location / both onsite / both remote" → offer_a: {{work_location: <value>}}, offer_b: {{work_location: <value>}}
- "neither requires relocation / both in same city" → offer_a: {{requires_relocation: false}}, offer_b: {{requires_relocation: false}}
- CRITICAL: When user says "same X for both" — ALWAYS set that field on BOTH offer_a AND offer_b
- "work from home" / "remote" → work_location = "remote"
- "need to move" / "different city" → requires_relocation = true
- "same city" / "no relocation" → requires_relocation = false
- "I have a family" / "my spouse" / "my kids" → has_family = true, has_dependents = true
- "startup feels risky" → job_security = "low", concern = "startup risk"
- "lots of room to grow" → growth_potential = "high"

Return ONLY valid JSON. Omit empty categories:
{{"extracted": {{"offer_a": {{}}, "offer_b": {{}}, "personal": {{}}, "values": {{}}}}, "user_emotional_state": "uncertain"}}

If nothing extractable: {{"extracted": {{}}, "user_emotional_state": "neutral"}}"""

            elif decision_subtype == "university_comparison":
                system_prompt = f"""Extract facts from this message about a university comparison ({opt_a} vs {opt_b}).

Message: "{user_message}"

UNIVERSITY A details (category: "uni_a") — facts about {opt_a}:
- name: string (university name)
- program: string (degree program / major)
- tuition: number (annual, convert "40k/year"→40000, "20 lakhs total"→200000)
- tuition_raw: string (original)
- scholarship: string (amount or description)
- location: string (city / country)
- requires_relocation: true / false
- living_cost: "high" / "medium" / "low" or numeric monthly
- ranking: string (ranking or reputation note)
- job_placement: string (placement rate or reputation)
- research_interest: string (specific professor/lab interest)
- concern: string (user's hesitation)

UNIVERSITY B details (category: "uni_b") — same fields as uni_a

PERSONAL CONTEXT (category: "personal"):
- has_family: true / false
- can_relocate: true / false
- relocation_concern: string
- current_city: string

VALUES (category: "values"):
- financial_security (1-10), career_growth (1-10)

MAPPING RULES:
- "first university / {opt_a}" → uni_a; "second / {opt_b}" → uni_b
- "scholarship covers 50%" → scholarship = "50% covered"
- "need to move there" → requires_relocation = true
- "same city as me" → requires_relocation = false
- "top 10 school" → ranking = "top 10"
- "good placement" → job_placement = "good"

Return ONLY valid JSON. Omit empty categories:
{{"extracted": {{"uni_a": {{}}, "uni_b": {{}}, "personal": {{}}, "values": {{}}}}, "user_emotional_state": "uncertain"}}

If nothing extractable: {{"extracted": {{}}, "user_emotional_state": "neutral"}}"""

            elif decision_subtype == "major_choice":
                system_prompt = f"""Extract facts from this message about a major/field choice ({opt_a} vs {opt_b}).

Message: "{user_message}"

CURRENT SITUATION (category: "current"):
- current_year: string (e.g. "freshman", "pre-college", "2nd year", "high school senior")
- leaning: string (which option they lean toward and why)
- financial_concern: string (any financial/family constraints)
- concern: string (biggest worry about making the wrong choice)
- job_market_concern: true/false

INTERESTS (category: "interests"):
- hands_on_work: true/false ("building", "tinkering", "practical" → true)
- enjoys_theory: true/false ("abstract", "theory", "math-heavy" → true)
- enjoys_coding: true/false
- enjoys_building_systems: true/false
- research: true/false

CAREER VISION (category: "career_vision"):
- post_graduation_goal: "job" / "grad_school" / "undecided"
- desired_role_5yr: string (e.g. "software engineer", "hardware engineer", "researcher")
- research_vs_applied: "research" / "applied" / "both"
- industry_preference: string (e.g. "tech", "hardware", "defense", "any")

VALUES (category: "values"):
- financial_security (1-10)
- career_growth (1-10)
- learning (1-10)
- impact (1-10)

MAPPING RULES:
- "I like building things / circuits / hardware" → interests.enjoys_building_systems = true, interests.hands_on_work = true
- "I prefer coding / software" → interests.enjoys_coding = true
- "I like math / theory / abstract problems" → interests.enjoys_theory = true
- "want a job after college" → career_vision.post_graduation_goal = "job"
- "want to do grad school / masters / PhD" → career_vision.post_graduation_goal = "grad_school"
- "software engineer / developer" → career_vision.desired_role_5yr = "software engineer"
- "salary is important / want to earn well" → values.financial_security = 8
- "parents want me to" / "family pressure" / "parents forcing me" → current.financial_concern = "family expectation"
- "freshman / first year / just starting" → current.current_year = "freshman"
- "sophomore / second year" → current.current_year = "sophomore"
- "junior / third year" → current.current_year = "junior"
- "senior / fourth year / final year" → current.current_year = "senior"
- "still deciding / haven't started / not started yet / pre-college / deciding before starting / still in high school / about to start" → current.current_year = "pre-college"
- "just graduated / just finished / recently graduated" → current.current_year = "recent graduate"
- "leaning toward X because Y" → current.leaning = "X because Y"

CRITICAL — desired_role_5yr EXTRACTION:
The user often answers the "5-year vision" question vaguely or indirectly. You MUST extract something.
- "I want to be in arts" → desired_role_5yr = "arts professional"
- "something related to arts like graphic designing" → desired_role_5yr = "graphic designer"
- "I see myself in arts like X" → desired_role_5yr = X
- "I want to be an arts professor" → desired_role_5yr = "arts professor"
- "I don't know / still figuring out" → desired_role_5yr = "undecided" (do NOT leave null)
- "I want a good job in [field]" → desired_role_5yr = "[field] professional"
- ANY mention of a role, even vague → extract the best approximation
- If they mention a specific field (arts, engineering, tech) without a role → use "[field] professional"
- NEVER return null for desired_role_5yr if the user said anything about their future

Return ONLY valid JSON. Omit empty categories:
{{"extracted": {{"interests": {{}}, "career_vision": {{}}, "values": {{}}, "current": {{}}}}, "user_emotional_state": "uncertain"}}

If nothing extractable: {{"extracted": {{}}, "user_emotional_state": "neutral"}}"""

            elif decision_subtype == "job_vs_business":
                system_prompt = f"""Extract facts from this message about a job-vs-business decision.

Message: "{user_message}"

CURRENT SITUATION (category: "current"):
- current_satisfaction: 1-10 or "happy"/"ok"/"unhappy"/"miserable"
- business_idea: string (description of the business concept)
- business_validated: true/false (has tested with real customers or earned side income)
- financial_runway: string (e.g. "6 months savings", "no savings", "12 months")
- leave_reason: string (frustration with job / excitement about business / both / freedom)
- concern: string (biggest fear about making the wrong choice)
- job_market_concern: true/false

PERSONAL CONTEXT (category: "personal"):
- has_family: true/false
- has_dependents: true/false
- partner_employed: true/false
- can_relocate: true/false

VALUES (category: "values"):
- financial_security (1-10)
- career_growth (1-10)
- work_life_balance (1-10)
- impact (1-10)

MAPPING RULES:
- "I hate my job / bored / no growth" → current_satisfaction = 3, leave_reason = "job frustration"
- "I have a business idea" → business_idea = description
- "I've already got some customers / side income" → business_validated = true
- "no savings / can't afford to quit" → financial_runway = "minimal"
- "6 months runway / savings" → financial_runway = "6 months"
- "I want freedom / be my own boss" → leave_reason = "desire for autonomy"
- "I have a family / kids / mortgage" → has_family = true, has_dependents = true
- "salary / stability matters to me" → values.financial_security = 8

Return ONLY valid JSON. Omit empty categories:
{{"extracted": {{"current": {{}}, "personal": {{}}, "values": {{}}}}, "user_emotional_state": "uncertain"}}

If nothing extractable: {{"extracted": {{}}, "user_emotional_state": "neutral"}}"""

            else:
                system_prompt = f"""Extract facts from this message for a {decision_type} decision ({' vs '.join(options)}).

Message: "{user_message}"

Extract ANY of these mentioned (don't force all):

INTERESTS (category: "interests"):
- enjoys_coding, enjoys_analysis, enjoys_theory, enjoys_building_systems,
  enjoys_working_with_data, hands_on_work, research
- "hands-on", "practical", "building things" → hands_on_work = true
- "research", "theory", "academia" → research = true/false

CAREER VISION (category: "career_vision"):
- desired_role_5yr (string), research_vs_applied ("research"/"applied"/"both")
- post_graduation_goal: "job" / "phd" / "startup" / "undecided"
- "I want a job" → post_graduation_goal = "job"
- "applied", "real-world" → research_vs_applied = "applied"

VALUES (category: "values"):
- financial_security (1-10), career_growth (1-10), work_life_balance (1-10),
  learning (1-10), impact (1-10)

FINANCIAL (category: "financial"):
- salary_importance (1-10), expected_salary (number)

CURRENT (category: "current"):
- job_market_concern (true/false), current_satisfaction (1-10)

Return ONLY valid JSON. Omit empty categories:
{{"extracted": {{"interests": {{}}}}, "user_emotional_state": "uncertain"}}

If nothing extractable: {{"extracted": {{}}, "user_emotional_state": "neutral"}}"""

        # FIX: inject last assistant question so extractor understands bare answers
        last_bot_question = ""
        if self.conversation_history:
            for msg in reversed(self.conversation_history):
                if msg["role"] == "assistant":
                    last_bot_question = msg["content"]
                    break
        if last_bot_question:
            system_prompt = f"The assistant just asked the user: \"{last_bot_question}\"\nUse this to interpret bare or ambiguous answers (e.g. a lone number like '7' or 'yes').\n\n" + system_prompt

        response = self._call_gemini(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt
        )

        try:
            start = response.find("{")
            end   = response.rfind("}") + 1
            if start >= 0:
                data = json.loads(response[start:end])
                print(f"[EXTRACT] {data}")
                return data
        except Exception as e:
            print(f"[EXTRACT] Parse error: {e} | Response: {response[:200]}")

        return {"extracted": {}, "user_emotional_state": "unknown"}

    # ── Response generation ───────────────────────────────────────────────────
    def generate_response(self, user_message: str, state_dict: Dict, mode: str = "conversational") -> str:
        # Track which topic we're about to ask, so loop guard can count it
        # FIX: apply to ALL subtypes, not just offer/university comparison
        subtype = state_dict.get("decision_metadata", {}).get("decision_subtype", "general")
        topic_label, _ = self._get_next_topic(state_dict)
        self.asked_topics[topic_label] = self.asked_topics.get(topic_label, 0) + 1
        print(f"[ASKED] '{topic_label}' x{self.asked_topics[topic_label]}")

        system_prompt = self._get_conversational_mode_prompt(state_dict)
        messages = self.conversation_history + [{"role": "user", "content": user_message}]
        response = self._call_gemini(messages=messages, system_prompt=system_prompt)
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    # Multi-field topic definitions — a topic is "answered" if ANY of these fields is set
    # This prevents asking the same question when the user answered via a synonym field
    OFFER_TOPIC_CHECKS = {
        "offer A — company and role":       ("offer_a", ["company", "role"]),
        "offer B — company and role":       ("offer_b", ["company", "role"]),
        "offer A — salary":                 ("offer_a", ["salary", "salary_raw"]),
        "offer B — salary":                 ("offer_b", ["salary", "salary_raw"]),
        "offer A — remote, onsite or hybrid?":  ("offer_a", ["work_location"]),
        "offer B — remote, onsite or hybrid?":  ("offer_b", ["work_location"]),
        "offer A — relocation needed?":     ("offer_a", ["requires_relocation"]),
        "offer B — relocation needed?":     ("offer_b", ["requires_relocation"]),
        "personal — can you relocate?":     ("personal", ["can_relocate", "has_family", "relocation_concern"]),
        "offer A — growth potential":       ("offer_a", ["growth_potential"]),
        "offer B — growth potential":       ("offer_b", ["growth_potential"]),
        "offer A — work-life balance":      ("offer_a", ["work_life_balance"]),
        "offer B — work-life balance":      ("offer_b", ["work_life_balance"]),
        "biggest concern about offer A":    ("offer_a", ["concern"]),
        "biggest concern about offer B":    ("offer_b", ["concern"]),
        "how important is financial security": ("values", ["financial_security"]),
        "how important is career growth":   ("values", ["career_growth"]),
        "how important is work-life balance": ("values", ["work_life_balance"]),
    }

    UNIVERSITY_TOPIC_CHECKS = {
        "university A — name and program":  ("uni_a", ["name", "program"]),
        "university B — name and program":  ("uni_b", ["name", "program"]),
        "university A — tuition and cost":  ("uni_a", ["tuition", "tuition_raw", "scholarship"]),
        "university B — tuition and cost":  ("uni_b", ["tuition", "tuition_raw", "scholarship"]),
        "university A — location and relocation": ("uni_a", ["requires_relocation", "location"]),
        "university B — location and relocation": ("uni_b", ["requires_relocation", "location"]),
        "personal — relocation feasibility":("personal", ["can_relocate", "has_family"]),
        "university A — cost of living":    ("uni_a", ["living_cost"]),
        "university B — cost of living":    ("uni_b", ["living_cost"]),
        "university A — reputation and ranking": ("uni_a", ["ranking", "job_placement"]),
        "university B — reputation and ranking": ("uni_b", ["ranking", "job_placement"]),
        "biggest concern about university A": ("uni_a", ["concern"]),
        "biggest concern about university B": ("uni_b", ["concern"]),
        "how important is minimizing debt": ("values", ["financial_security"]),
        "how important is career reputation": ("values", ["career_growth"]),
    }

    def _is_topic_answered(self, label: str, state_dict: Dict, topic_checks: Dict) -> bool:
        """Check if a topic is answered by looking at all relevant fields."""
        if label not in topic_checks:
            return False
        category, fields = topic_checks[label]
        cat_data = state_dict.get(category, {})
        return any(
            cat_data.get(f) not in (None, False, "", [])
            for f in fields
        )

    def _get_next_topic(self, state_dict: Dict) -> Tuple[str, str]:
        """
        Return (topic_label, question_text) for the next unanswered topic.
        Uses multi-field topic checks so that synonym fields (salary vs salary_raw)
        correctly mark a topic as answered.
        Also respects asked_topics to avoid infinite loops.
        """
        subtype = state_dict.get("decision_metadata", {}).get("decision_subtype", "general")

        if subtype == "offer_comparison":
            topic_list   = self.OFFER_QUESTION_TOPICS
            topic_checks = self.OFFER_TOPIC_CHECKS
        elif subtype == "university_comparison":
            topic_list   = self.UNIVERSITY_QUESTION_TOPICS
            topic_checks = self.UNIVERSITY_TOPIC_CHECKS
        elif subtype == "major_choice":
            topic_list   = self.MAJOR_CHOICE_QUESTION_TOPICS
            topic_checks = self.MAJOR_CHOICE_TOPIC_CHECKS
        elif subtype == "job_vs_business":
            topic_list   = self.JOB_VS_BUSINESS_QUESTION_TOPICS
            topic_checks = self.JOB_VS_BUSINESS_TOPIC_CHECKS
        else:
            # education_path, general — CAREER_QUESTION_TOPICS with multi-field checks
            topic_list   = self.CAREER_QUESTION_TOPICS
            topic_checks = {
                "primary goal after graduation":    ("career_vision", ["post_graduation_goal"]),
                "work style preference":            ("interests",     ["hands_on_work", "enjoys_theory", "research"]),
                "applied vs research lean":         ("career_vision", ["research_vs_applied"]),
                "current job satisfaction":         ("current",       ["current_satisfaction"]),
                "5-year vision":                    ("career_vision", ["desired_role_5yr", "research_vs_applied"]),
                "financial security priority":      ("values",        ["financial_security"]),
                "financial runway for PhD":         ("current",       ["financial_runway", "has_dependents"]),
                "family or personal obligations":   ("personal",      ["has_family", "has_dependents", "partner_employed"]),
                "growth vs balance":                ("values",        ["career_growth", "work_life_balance"]),
                "work-life balance priority":       ("values",        ["work_life_balance"]),
                "biggest concern about the choice": ("current",       ["concern"]),
                "industry opportunity cost":        ("current",       ["job_market_concern"]),
            }

        for field_key, category, label, question in topic_list:
            # Skip if already answered (multi-field check)
            if self._is_topic_answered(label, state_dict, topic_checks):
                continue
            # Skip if we've already asked this topic twice without getting an answer
            # (loop guard — prevents infinite repeat)
            times_asked = self.asked_topics.get(label, 0)
            if times_asked >= 2:
                print(f"[LOOP GUARD] Skipping '{label}' — asked {times_asked} times, no answer")
                continue
            return label, question

        return "any remaining concerns", "Is there anything else that feels unresolved?"

    def _get_active_violations(self, state_dict: Dict) -> List[str]:
        """Return list of current violation descriptions."""
        return [v["description"] for v in state_dict.get("violations", [])]

    def _get_conversational_mode_prompt(self, state_dict: Dict) -> str:
        decision_type = state_dict.get("decision_metadata", {}).get("decision_type", "unknown")
        options       = state_dict.get("decision_metadata", {}).get("options_being_compared", [])

        option_a = options[0] if len(options) > 0 else "Option A"
        option_b = options[1] if len(options) > 1 else "Option B"

        interests  = state_dict.get("interests", {})
        values     = state_dict.get("values", {})
        career_vis = state_dict.get("career_vision", {})
        financial  = state_dict.get("financial", {})
        current    = state_dict.get("current", {})
        violations = self._get_active_violations(state_dict)

        subtype  = state_dict.get("decision_metadata", {}).get("decision_subtype", "general")
        offer_a  = state_dict.get("offer_a", {})
        offer_b  = state_dict.get("offer_b", {})
        uni_a    = state_dict.get("uni_a", {})
        uni_b    = state_dict.get("uni_b", {})
        personal = state_dict.get("personal", {})

        def _has(d, *keys):
            return any(d.get(k) not in (None, False, "", []) for k in keys)

        # ── Topic-based counting (not field counting) ─────────────────────────
        # Each meaningful topic = 1, regardless of how many fields it fills
        if subtype == "offer_comparison":
            topics = [
                _has(offer_a, "company", "role"),          # topic: what is offer A
                _has(offer_b, "company", "role"),          # topic: what is offer B
                _has(offer_a, "salary", "salary_raw"),     # topic: salary A
                _has(offer_b, "salary", "salary_raw"),     # topic: salary B
                _has(offer_a, "work_location"),            # topic: remote/onsite A
                _has(offer_b, "work_location"),            # topic: remote/onsite B
                _has(offer_a, "requires_relocation"),      # topic: relocation A
                _has(offer_b, "requires_relocation"),      # topic: relocation B
                _has(personal, "can_relocate", "has_family", "relocation_concern"),  # topic: personal/family
                _has(offer_a, "growth_potential"),         # topic: growth A
                _has(offer_b, "growth_potential"),         # topic: growth B
                _has(offer_a, "work_life_balance"),        # topic: WLB A
                _has(offer_b, "work_life_balance"),        # topic: WLB B
                _has(offer_a, "concern"),                  # topic: concern A
                _has(offer_b, "concern"),                  # topic: concern B
                _has(values, "financial_security", "career_growth", "work_life_balance"),  # topic: values
            ]
            relevant_count = sum(topics)
            threshold = 10  # must cover at least 10 meaningful topics

        elif subtype == "university_comparison":
            topics = [
                _has(uni_a, "name", "program"),            # topic: what is uni A
                _has(uni_b, "name", "program"),            # topic: what is uni B
                _has(uni_a, "tuition", "tuition_raw"),     # topic: cost A
                _has(uni_b, "tuition", "tuition_raw"),     # topic: cost B
                _has(uni_a, "requires_relocation", "location"),  # topic: location A
                _has(uni_b, "requires_relocation", "location"),  # topic: location B
                _has(personal, "can_relocate", "has_family"),    # topic: personal/family
                _has(uni_a, "living_cost"),                # topic: living cost A
                _has(uni_b, "living_cost"),                # topic: living cost B
                _has(uni_a, "ranking", "job_placement"),   # topic: reputation A
                _has(uni_b, "ranking", "job_placement"),   # topic: reputation B
                _has(uni_a, "concern"),                    # topic: concern A
                _has(uni_b, "concern"),                    # topic: concern B
                _has(values, "financial_security", "career_growth"),  # topic: values
            ]
            relevant_count = sum(topics)
            threshold = 9

        elif subtype == "major_choice":
            topics = [
                _has(current,    "current_year"),
                _has(current,    "leaning"),
                _has(interests,  "hands_on_work", "enjoys_theory", "enjoys_building_systems"),
                _has(career_vis, "post_graduation_goal"),
                _has(career_vis, "desired_role_5yr", "research_vs_applied"),
                _has(values,     "financial_security"),
                _has(values,     "career_growth", "impact"),
                _has(values,     "work_life_balance"),
                _has(current,    "job_market_concern"),
                _has(current,    "financial_concern"),
                _has(current,    "concern"),
            ]
            relevant_count = sum(topics)
            threshold = 9  # need 9 of 11 topics

        elif subtype == "job_vs_business":
            topics = [
                _has(current,    "current_satisfaction"),
                _has(current,    "business_idea"),
                _has(current,    "leave_reason"),
                _has(current,    "current_role"),
                _has(current,    "financial_runway"),
                _has(personal,   "has_family", "has_dependents"),
                _has(personal,   "partner_employed"),
                _has(values,     "financial_security"),
                _has(values,     "career_growth", "impact"),
                _has(current,    "business_validated"),
                _has(values,     "work_life_balance"),
                _has(current,    "concern"),
            ]
            relevant_count = sum(topics)
            threshold = 9  # need 9 of 12 topics

        elif decision_type in ("career_choice", "education"):
            # Topic-based counting — robust, matches other subtypes
            topics = [
                _has(career_vis, "post_graduation_goal"),
                _has(interests,  "hands_on_work", "enjoys_theory", "research"),
                _has(career_vis, "research_vs_applied"),
                _has(current,    "current_satisfaction"),
                _has(career_vis, "desired_role_5yr"),
                _has(values,     "financial_security"),
                _has(current,    "financial_runway"),
                _has(personal,   "has_family", "has_dependents"),
                _has(values,     "career_growth", "work_life_balance"),
                _has(current,    "concern"),
                _has(current,    "job_market_concern"),
            ]
            relevant_count = sum(topics)
            threshold = 9  # need 9 of 11 topics
        else:
            relevant_count = sum(
                1 for cat in [financial, values]
                for v in cat.values() if v is not None
            )
            threshold = 6

        print(f"[PROMPT] type={decision_type} facts={relevant_count}/{threshold} violations={len(violations)}")

        # ── CONCLUDE ─────────────────────────────────────────────────────────
        if relevant_count >= threshold:
            if subtype == "offer_comparison":
                known_facts = {
                    f"{option_a} details": {k: v for k, v in offer_a.items() if v is not None and v != ""},
                    f"{option_b} details": {k: v for k, v in offer_b.items() if v is not None and v != ""},
                    "personal context":    {k: v for k, v in personal.items() if v is not None and v != ""},
                    "your values":         {k: v for k, v in values.items()  if v is not None},
                }
            elif subtype == "university_comparison":
                known_facts = {
                    f"{option_a} details": {k: v for k, v in uni_a.items() if v is not None and v != ""},
                    f"{option_b} details": {k: v for k, v in uni_b.items() if v is not None and v != ""},
                    "personal context":    {k: v for k, v in personal.items() if v is not None and v != ""},
                    "your values":         {k: v for k, v in values.items()  if v is not None},
                }
            elif subtype == "major_choice":
                known_facts = {
                    "current situation": {k: v for k, v in current.items()    if v is not None and v != ""},
                    "interests":         {k: v for k, v in interests.items()  if v is not None and v is not False},
                    "career vision":     {k: v for k, v in career_vis.items() if v is not None and v != ""},
                    "values":            {k: v for k, v in values.items()     if v is not None},
                }
            elif subtype == "job_vs_business":
                known_facts = {
                    "current situation": {k: v for k, v in current.items()   if v is not None and v != ""},
                    "personal context":  {k: v for k, v in personal.items()  if v is not None and v != ""},
                    "values":            {k: v for k, v in values.items()     if v is not None},
                }
            else:
                known_facts = {
                    "interests":     {k: v for k, v in interests.items()  if v is not None and v is not False},
                    "values":        {k: v for k, v in values.items()     if v is not None},
                    "career_vision": {k: v for k, v in career_vis.items() if v is not None and v != ""},
                    "financial":     {k: v for k, v in financial.items()  if v is not None},
                    "current":       {k: v for k, v in current.items()    if v is not None},
                }

            violations_str = ""
            if violations:
                violations_str = f"\nContradictions noted: {violations}"

            return f"""TASK: Wrap up the conversation — do NOT give a score or recommendation.

Decision: {option_a} vs {option_b}
Collected facts:
{json.dumps(known_facts, indent=2)}{violations_str}

Write EXACTLY this (3-4 sentences total):

"Great, I think I have a solid picture of your situation. [1 sentence summarizing the key tension or tradeoff — e.g. 'You're weighing financial stability at {option_a} against the growth upside at {option_b}, with relocation being a key personal constraint.'] The council of expert agents will now analyze this from multiple angles and debate it. Click the **Council of Experts** button below to see the full analysis."

HARD RULES:
- Final sentence MUST contain "Council of Experts"
- Do NOT give a percentage score — the council will do that
- Do NOT say which option is better
- Do NOT give advice
- Maximum 4 sentences"""

        # ── COLLECTING ───────────────────────────────────────────────────────
        topic_label, next_question = self._get_next_topic(state_dict)

        # Build violation callout if any exist
        violation_note = ""
        if violations:
            violation_note = f"""
IMPORTANT: The user has a contradiction in their answers: "{violations[0]}"
Before asking your question, call this out in a friendly but direct way in 1 sentence.
"""

        return f"""You are a witty, warm decision-support assistant helping someone decide: {option_a} vs {option_b}.

You have {relevant_count}/{threshold} facts. Your job this turn: ask about "{topic_label}".
The exact question to ask: "{next_question}"{violation_note}

RULES:
- 2-3 sentences total
- Add a brief, dry observation or light humor before the question — something that shows you're paying attention to what they said, not just running through a checklist
- If there's a contradiction callout above, lead with that (still keep it warm, not accusatory)
- Ask exactly the one question above — do not invent different questions
- No advice, no summaries, no reflecting their words back at length
- Conversational tone, like a smart friend — not a therapist, not a robot"""

    # ── Council debate (4 fixed agents) ──────────────────────────────────────
    def generate_council_perspectives(self, state_dict: Dict) -> Dict:
        decision_type = state_dict.get("decision_metadata", {}).get("decision_type", "unknown")
        options       = state_dict.get("decision_metadata", {}).get("options_being_compared", [])

        if decision_type in ("career_choice", "education") and len(options) == 2:
            return self._run_agent_debate(state_dict, options)
        else:
            return self._generate_general_debate(state_dict)

    def _build_profile(self, state_dict: Dict) -> str:
        interests  = state_dict.get("interests", {})
        career_vis = state_dict.get("career_vision", {})
        strengths  = state_dict.get("strengths", {})
        values     = state_dict.get("values", {})
        financial  = state_dict.get("financial", {})
        current    = state_dict.get("current", {})
        offer_a    = state_dict.get("offer_a", {})
        offer_b    = state_dict.get("offer_b", {})
        uni_a      = state_dict.get("uni_a", {})
        uni_b      = state_dict.get("uni_b", {})
        personal   = state_dict.get("personal", {})
        options    = state_dict.get("decision_metadata", {}).get("options_being_compared", ["Option A", "Option B"])
        subtype    = state_dict.get("decision_metadata", {}).get("decision_subtype", "general")
        violations = self._get_active_violations(state_dict)

        opt_a = options[0] if options else "Option A"
        opt_b = options[1] if len(options) > 1 else "Option B"

        if subtype == "offer_comparison":
            profile = f"""User is comparing two job offers:

{opt_a}:
{json.dumps({k:v for k,v in offer_a.items() if v is not None and v != ""}, indent=2)}

{opt_b}:
{json.dumps({k:v for k,v in offer_b.items() if v is not None and v != ""}, indent=2)}

Personal / life context:
{json.dumps({k:v for k,v in personal.items() if v is not None and v != ""}, indent=2)}

Their values and priorities:
{json.dumps({k:v for k,v in values.items() if v is not None}, indent=2)}"""

        elif subtype == "university_comparison":
            profile = f"""User is comparing two universities:

{opt_a}:
{json.dumps({k:v for k,v in uni_a.items() if v is not None and v != ""}, indent=2)}

{opt_b}:
{json.dumps({k:v for k,v in uni_b.items() if v is not None and v != ""}, indent=2)}

Personal / life context:
{json.dumps({k:v for k,v in personal.items() if v is not None and v != ""}, indent=2)}

Their values and priorities:
{json.dumps({k:v for k,v in values.items() if v is not None}, indent=2)}"""

        elif subtype == "major_choice":
            profile = f"""User is choosing between {opt_a} and {opt_b} for their degree.

Current situation:
{json.dumps({k:v for k,v in current.items()   if v is not None and v != ""}, indent=2)}

Interests and work style:
{json.dumps({k:v for k,v in interests.items() if v is not None and v is not False}, indent=2)}

Career vision:
{json.dumps({k:v for k,v in career_vis.items() if v is not None and v != ""}, indent=2)}

Values and priorities:
{json.dumps({k:v for k,v in values.items()    if v is not None}, indent=2)}"""

        elif subtype == "job_vs_business":
            profile = f"""User is deciding whether to leave their job and start a business.

Current situation:
{json.dumps({k:v for k,v in current.items()   if v is not None and v != ""}, indent=2)}

Personal / family context:
{json.dumps({k:v for k,v in personal.items()  if v is not None and v != ""}, indent=2)}

Values and priorities:
{json.dumps({k:v for k,v in values.items()    if v is not None}, indent=2)}"""

        else:
            profile = f"""User profile (facts from conversation):
- Interests:      {json.dumps({k:v for k,v in interests.items()  if v is not None and v is not False})}
- Values:         {json.dumps({k:v for k,v in values.items()     if v is not None})}
- Career vision:  {json.dumps({k:v for k,v in career_vis.items() if v is not None and v != ""})}
- Financial:      {json.dumps({k:v for k,v in financial.items()  if v is not None})}
- Strengths:      {json.dumps({k:v for k,v in strengths.items()  if v is not None})}
- Current:        {json.dumps({k:v for k,v in current.items()    if v is not None})}"""

        if violations:
            profile += f"\n\nNoted contradictions: {violations}"

        return profile

    def _run_agent_debate(self, state_dict: Dict, options: List[str]) -> Dict:
        """
        Each agent:
        1. Analyzes profile from their lens
        2. Casts a vote (option_a %, option_b %) with 2-3 sentence reasoning
        3. Round 2: agents with opposing votes debate each other (2-3 sentences)
        4. Round 3 (adaptive): only if vote split is close (within 20 points)
        5. Synthesizer tallies weighted votes and rules
        """
        option_a, option_b = options[0], options[1]
        profile = self._build_profile(state_dict)
        results = {}

        # ── Round 1: Each agent votes ─────────────────────────────────────────
        agent_votes = {}  # agent_id -> {"option_a": X, "option_b": Y, "text": "..."}

        for agent in self.AGENTS:
            prompt = f"""{profile}

You are the {agent['name']} ({agent['emoji']}).
Your analytical lens: {agent['lens']}

The decision: {option_a} vs {option_b}

From YOUR specific lens only:
1. Cast your vote as a percentage split — e.g. "{option_a}: 70%, {option_b}: 30%"
2. Explain your vote in 2-3 sentences, citing specific facts from the profile
3. Name the single most important factor from your lens that swings this

Format your response EXACTLY like this:
VOTE: {option_a}: [X]% | {option_b}: [Y]%
REASONING: [2-3 sentences]
KEY FACTOR: [one phrase]

Rules:
- X + Y = 100
- Stay strictly within your lens — don't comment on things outside it
- Be direct and specific, not generic"""

            raw = self._call_gemini(messages=[{"role": "user", "content": prompt}])
            results[f"agent_{agent['id']}_round1"] = raw

            # Parse vote
            vote_a, vote_b = 50, 50  # defaults
            try:
                for line in raw.split("\n"):
                    if line.startswith("VOTE:"):
                        parts = line.replace("VOTE:", "").strip()
                        # e.g. "PhD: 30% | Job: 70%"
                        segs = parts.split("|")
                        if len(segs) == 2:
                            vote_a = int(''.join(c for c in segs[0].split(":")[-1] if c.isdigit()))
                            vote_b = int(''.join(c for c in segs[1].split(":")[-1] if c.isdigit()))
            except Exception as e:
                print(f"[VOTE PARSE] {agent['id']}: {e}")

            agent_votes[agent["id"]] = {"option_a": vote_a, "option_b": vote_b, "raw": raw}
            print(f"[VOTE] {agent['name']}: {option_a}={vote_a}% {option_b}={vote_b}%")

        results["agent_votes"] = agent_votes

        # ── Tally after Round 1 ───────────────────────────────────────────────
        total_a = sum(v["option_a"] for v in agent_votes.values())
        total_b = sum(v["option_b"] for v in agent_votes.values())
        avg_a   = total_a / len(self.AGENTS)
        avg_b   = total_b / len(self.AGENTS)
        spread  = abs(avg_a - avg_b)
        results["tally_after_r1"] = {"option_a": round(avg_a), "option_b": round(avg_b)}

        print(f"[TALLY R1] {option_a}={avg_a:.0f}% {option_b}={avg_b:.0f}% spread={spread:.0f}")

        # ── Round 2: Opposing agents debate ──────────────────────────────────
        # Find the agent most for A and the agent most for B
        top_a_agent = max(self.AGENTS, key=lambda ag: agent_votes[ag["id"]]["option_a"])
        top_b_agent = max(self.AGENTS, key=lambda ag: agent_votes[ag["id"]]["option_b"])

        if top_a_agent["id"] != top_b_agent["id"]:
            # Top-A agent rebuts top-B agent's Round 1 argument
            r2a_prompt = f"""{profile}

You are the {top_a_agent['name']} ({top_a_agent['emoji']}).
Your lens: {top_a_agent['lens']}

The {top_b_agent['name']} just argued:
"{results[f"agent_{top_b_agent['id']}_round1"]}"

Rebuttal (2-3 sentences max):
- Attack ONE specific weakness in their argument from your lens
- Reinforce your case for {option_a}
- Be sharp and direct
- Do NOT start with "I" """

            results["round2_a"] = self._call_gemini(messages=[{"role": "user", "content": r2a_prompt}])

            # Top-B agent rebuts top-A agent's Round 1 argument
            r2b_prompt = f"""{profile}

You are the {top_b_agent['name']} ({top_b_agent['emoji']}).
Your lens: {top_b_agent['lens']}

The {top_a_agent['name']} just argued:
"{results[f"agent_{top_a_agent['id']}_round1"]}"

Rebuttal (2-3 sentences max):
- Attack ONE specific weakness in their argument from your lens
- Reinforce your case for {option_b}
- Be sharp and direct
- Do NOT start with "I" """

            results["round2_b"] = self._call_gemini(messages=[{"role": "user", "content": r2b_prompt}])
            results["debating_agents"] = {
                "a": {"name": top_a_agent["name"], "emoji": top_a_agent["emoji"], "option": option_a},
                "b": {"name": top_b_agent["name"], "emoji": top_b_agent["emoji"], "option": option_b},
            }
        else:
            results["round2_a"] = ""
            results["round2_b"] = ""
            results["debating_agents"] = {}

        # ── Round 3 (adaptive): only if spread <= 20 ─────────────────────────
        results["has_round3"] = False
        if spread <= 20 and top_a_agent["id"] != top_b_agent["id"]:
            print(f"[ROUND 3] Close vote ({spread:.0f} pt spread), triggering Round 3")
            results["has_round3"] = True

            r3a_prompt = f"""{profile}

ROUND 3 — Final argument. You are the {top_a_agent['name']} ({top_a_agent['emoji']}).

Your opponent ({top_b_agent['name']}) just said in Round 2:
"{results['round2_b']}"

Give your FINAL, decisive argument for {option_a}. This is your closing statement.
- Address their Round 2 rebuttal directly
- Make your single strongest point
- 2-3 sentences, no more
- Do NOT start with "I" """

            results["round3_a"] = self._call_gemini(messages=[{"role": "user", "content": r3a_prompt}])

            r3b_prompt = f"""{profile}

ROUND 3 — Final argument. You are the {top_b_agent['name']} ({top_b_agent['emoji']}).

Your opponent ({top_a_agent['name']}) just said in Round 3:
"{results['round3_a']}"

Give your FINAL, decisive argument for {option_b}. This is your closing statement.
- Address their Round 3 argument directly
- Make your single strongest point
- 2-3 sentences, no more
- Do NOT start with "I" """

            results["round3_b"] = self._call_gemini(messages=[{"role": "user", "content": r3b_prompt}])

        # ── Synthesizer: reads all rounds, tallies, rules ─────────────────────
        all_rounds = f"""Round 1 votes:
{chr(10).join(f"  {ag['name']}: {agent_votes[ag['id']]['raw']}" for ag in self.AGENTS)}

Round 2 debate:
  {results.get('debating_agents', {}).get('a', {}).get('name', 'Agent A')}: {results.get('round2_a', 'N/A')}
  {results.get('debating_agents', {}).get('b', {}).get('name', 'Agent B')}: {results.get('round2_b', 'N/A')}"""

        if results["has_round3"]:
            all_rounds += f"""

Round 3 (tiebreaker):
  {results.get('debating_agents', {}).get('a', {}).get('name', 'Agent A')}: {results.get('round3_a', 'N/A')}
  {results.get('debating_agents', {}).get('b', {}).get('name', 'Agent B')}: {results.get('round3_b', 'N/A')}"""

        synth_prompt = f"""{profile}

The 4 agents just debated {option_a} vs {option_b}:
{all_rounds}

Weighted vote tally after Round 1:
  {option_a}: AVG_A_PCT% average across all agents
  {option_b}: AVG_B_PCT% average across all agents

You are the SYNTHESIZER. Rule on this debate.

Write your ruling in this format:
WINNING ARGUMENT: [Name the single strongest argument from any agent in any round, and why]
WEAKEST ARGUMENT: [Name the weakest argument, and why]
VOTE TALLY: {option_a} AVG_A_PCT% | {option_b} AVG_B_PCT%
RULING: [2-3 sentences — state which option the profile leans toward based on the weighted vote, and explain the single deciding factor clearly]
OPEN QUESTION: [One genuine unresolved question about the decision that the user should reflect on — NOT life advice, NOT "research X", NOT "talk to Y". Something like "The key tension here is whether financial stability now outweighs career fulfillment later — which matters more to you?"]

RULES:
- Do NOT say "you should"
- Use "the data suggests..." or "the profile leans toward..."
- Do NOT start with "I"
- Be decisive — if it's close, say so but still lean one way
- OPEN QUESTION must be a genuine dilemma question, NOT an action item or advice
- Do NOT suggest researching programs, networking, talking to advisors, or any external action"""

        synth_prompt = synth_prompt.replace("AVG_A_PCT", str(round(avg_a)))
        synth_prompt = synth_prompt.replace("AVG_B_PCT", str(round(avg_b)))

        results["synthesizer"] = self._call_gemini(
            messages=[{"role": "user", "content": synth_prompt}]
        )

        # Store summary data for rendering
        results["options"]  = [option_a, option_b]
        results["avg_vote"] = {"option_a": round(avg_a), "option_b": round(avg_b)}
        results["agents"]   = self.AGENTS

        return results

    def _generate_general_debate(self, state_dict: Dict) -> Dict:
        context = json.dumps(state_dict, indent=2, default=str)
        perspectives = {}
        for role, focus in [
            ("risk",        "risks, worst-case scenarios, missing safety nets"),
            ("opportunity", "upside potential, growth, what could go right"),
            ("values",      "alignment with stated values, identity, authenticity"),
        ]:
            perspectives[role] = self._call_gemini(messages=[{"role": "user", "content":
                f"""Decision state:\n{context}\n\nYou are the {role.upper()} ANALYST.
Focus on: {focus}
- Reference specific facts
- 4-5 sentences max
- Do NOT start with "I" """}])
        return perspectives

    def reset_conversation(self):
        self.conversation_history = []
        self.asked_topics = {}


if __name__ == "__main__":
    print("LLM Interface ready.")