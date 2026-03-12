"""
Symbolic State Engine - The authoritative source of truth for decision constraints
This is the deterministic logic layer that the LLM cannot override.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class ConstraintViolation:
    """Represents a logical inconsistency in the user's stated constraints"""
    
    def __init__(self, violation_type: str, description: str, severity: str):
        self.violation_type = violation_type
        self.description = description
        self.severity = severity  # "critical", "warning", "info"
        self.timestamp = datetime.now()
    
    def __repr__(self):
        emoji = {"critical": "ðŸ”´", "warning": "ðŸŸ¡", "info": "ðŸ”µ"}
        return f"{emoji.get(self.severity, 'âš ï¸')} {self.severity.upper()}: {self.description}"


class DecisionState:
    """
    Tracks all known constraints and facts about the decision.
    This is the symbolic layer - deterministic and authoritative.
    """
    
    def __init__(self):
        # Financial constraints
        self.financial = {
            "current_savings": None,
            "monthly_expenses": None,
            "current_income": None,
            "debt_total": None,
            "debt_monthly_payment": None,
            "new_opportunity_income": None,
            "expected_salary": None,       # expected/target salary
            "salary_importance": None,      # 1-10 how important salary is
            "financial_runway_months": None,
        }
        
        # Legal/practical constraints (binary)
        self.legal = {
            "visa_constrained": None,
            "non_compete": None,
            "health_insurance_needed": None,
            "contractual_obligations": None,
        }
        
        # Personal values (weighted 1-10)
        self.values = {
            "career_growth": None,
            "work_life_balance": None,
            "financial_security": None,
            "learning": None,
            "status": None,
            "impact": None,
        }
        
        # Relationship/family
        self.relationships = {
            "has_dependents": None,
            "partner_income_stable": None,
            "family_support": None,
            "geographic_constraints": None,
        }

        # Interests & Strengths (for career/academic decisions)
        self.interests = {
            "enjoys_coding": None,  # True/False
            "enjoys_analysis": None,
            "enjoys_theory": None,
            "enjoys_building_systems": None,
            "enjoys_working_with_data": None,
            "enjoys_visualization": None,
            "enjoys_algorithms": None,
        }

        # Career Vision
        self.career_vision = {
            "desired_role_5yr": None,  # "researcher", "engineer", "manager", "entrepreneur"
            "wants_specialization": None,  # True/False - specialist vs generalist
            "research_vs_applied": None,  # "research", "applied", "both"
            "industry_preference": None,  # "tech", "finance", "healthcare", "any"
            "post_graduation_goal": None,  # "job", "phd", "startup", "undecided"
        }

        # Strengths & Skills
        self.strengths = {
            "math_stats_comfort": None,  # 1-10 scale
            "programming_experience": None,  # 1-10 scale
            "communication_skills": None,  # 1-10 scale
            "problem_solving_approach": None,  # "analytical", "creative", "systematic"
        }

        # Decision metadata (stored as a dict so update() and to_dict() work uniformly)
        self.decision_metadata = {
            "decision_type": None,       # "career_choice", "education", "financial", "location"
            "options_being_compared": [], # ["Computer Science", "Data Science"]
            "decision_summary": None,    # free-text description of the decision
            "decision_subtype": None,   # "offer_comparison", "major_choice", "education_path", etc.
        }
        
        # Offer details — used when decision_subtype == "offer_comparison"
        self.offer_a = {
            "company":           None,  # company name
            "role":              None,  # job title / role description
            "salary":            None,  # numeric salary
            "salary_raw":        None,  # raw string e.g. "80k"
            "growth_potential":  None,  # "high" / "medium" / "low" or description
            "work_life_balance": None,  # "great" / "ok" / "poor" or 1-10
            "work_location":     None,  # "remote" / "onsite" / "hybrid"
            "city":              None,  # city where the job is
            "requires_relocation": None, # True / False
            "culture":           None,  # team/culture description
            "concern":           None,  # user's main concern about this offer
            "job_security":      None,  # "high" / "medium" / "low"
        }
        self.offer_b = {
            "company":           None,
            "role":              None,
            "salary":            None,
            "salary_raw":        None,
            "growth_potential":  None,
            "work_life_balance": None,
            "work_location":     None,
            "city":              None,
            "requires_relocation": None,
            "culture":           None,
            "concern":           None,
            "job_security":      None,
        }

        # Personal / life context — applies to any comparison
        self.personal = {
            "has_family":        None,  # True / False
            "has_dependents":    None,  # True / False
            "partner_employed":  None,  # True / False
            "can_relocate":      None,  # True / False
            "relocation_concern":None,  # description
            "current_city":      None,  # where they live now
        }

        # University comparison fields
        self.uni_a = {
            "name":              None,  # university name
            "program":           None,  # program / major
            "tuition":           None,  # numeric annual tuition
            "tuition_raw":       None,  # raw string
            "scholarship":       None,  # scholarship amount or description
            "location":          None,  # city / country
            "requires_relocation": None,
            "ranking":           None,  # ranking or reputation note
            "job_placement":     None,  # placement rate or reputation
            "living_cost":       None,  # "high" / "medium" / "low" or numeric
            "research_interest": None,  # specific professor / lab interest
            "concern":           None,
        }
        self.uni_b = {
            "name":              None,
            "program":           None,
            "tuition":           None,
            "tuition_raw":       None,
            "scholarship":       None,
            "location":          None,
            "requires_relocation": None,
            "ranking":           None,
            "job_placement":     None,
            "living_cost":       None,
            "research_interest": None,
            "concern":           None,
        }

        # Opportunity details
        self.opportunity = {
            "role_description": None,
            "company": None,
            "work_life_balance_known": False,
            "team_culture_known": False,
            "reversibility": None,  # "high", "medium", "low"
        }
        
        # Current situation
        self.current = {
            "current_employer":   None,
            "current_role":       None,
            "current_satisfaction": None,
            "current_wlb":        None,
            # Major choice fields
            "current_year":       None,  # "freshman", "sophomore", "pre-college" etc.
            "leaning":            None,  # which option and why
            "financial_concern":  None,  # financial/family constraint
            "concern":            None,  # biggest worry about the choice
            "job_market_concern": None,  # true/false
            # Job vs business fields
            "business_idea":     None,  # description of business idea
            "business_validated":None,  # true/false — tested with real customers
            "financial_runway":  None,  # months or description of savings
            "leave_reason":      None,  # what is driving desire to leave
        }
        
        # State tracking
        self.history = []  # List of state changes
        self.violations = []  # List of constraint violations
        self.missing_critical_info = []
        
    def update(self, category: str, key: str, value: Any) -> None:
        """Update a state variable and record the change"""
        cat_obj = getattr(self, category, None)
        if cat_obj is None:
            print(f"[STATE] WARNING: unknown category '{category}', skipping")
            return

        old_value = cat_obj.get(key) if isinstance(cat_obj, dict) else None

        # ── financial: robust string-to-number conversion ──────────────────
        if category == "financial" and value is not None:
            if isinstance(value, str):
                cleaned = (value.lower()
                           .replace('usd', '').replace('inr', '')
                           .replace('$', '').replace(',', '').strip())
                if 'lakh' in cleaned or 'lac' in cleaned:
                    num_part = cleaned.replace('lakh', '').replace('lac', '').strip()
                    try:
                        value = float(num_part) * 100_000
                    except Exception:
                        pass
                else:
                    try:
                        value = float(cleaned) if cleaned else None
                    except Exception:
                        pass

        # ── values: convert string numbers to int ──────────────────────────
        if category == "values" and value is not None:
            if isinstance(value, str):
                try:
                    value = int(value)
                except Exception:
                    pass

        # ── decision_metadata: options_being_compared must stay a list ─────
        if category == "decision_metadata" and key == "options_being_compared":
            if isinstance(value, str):
                value = [v.strip() for v in value.split(",") if v.strip()]

        cat_obj[key] = value
        
        # Record state change
        change = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "key": key,
            "old_value": old_value,
            "new_value": value,
        }
        self.history.append(change)
        
        # Recalculate derived values
        self._recalculate_derived()
        
        # Check for violations
        self._check_violations()
    
    def _recalculate_derived(self) -> None:
        """Calculate derived values from base constraints"""
        # Calculate financial runway
        savings = self.financial.get("current_savings")
        expenses = self.financial.get("monthly_expenses")
        
        if savings is not None and expenses is not None:
            # Ensure both are numbers
            try:
                savings = float(savings) if isinstance(savings, str) else savings
                expenses = float(expenses) if isinstance(expenses, str) else expenses
                
                if expenses > 0:
                    self.financial["financial_runway_months"] = savings / expenses
            except (ValueError, TypeError):
                # If conversion fails, skip calculation
                pass
    
    def _check_violations(self) -> None:
        """Check for logical inconsistencies"""
        new_violations = []
        
        # Financial violations
        if (self.financial["debt_monthly_payment"] and 
            self.financial["current_income"] and
            self.financial["debt_monthly_payment"] > 0.5 * self.financial["current_income"]):
            new_violations.append(
                ConstraintViolation(
                    "financial_stress",
                    "Debt payments exceed 50% of income - extremely high risk",
                    "critical"
                )
            )
        
        # Value-action misalignment
        if (self.values.get("work_life_balance") and 
            self.values["work_life_balance"] >= 8 and
            self.current.get("current_wlb") == "great" and
            not self.opportunity.get("work_life_balance_known")):
            new_violations.append(
                ConstraintViolation(
                    "missing_critical_data",
                    "You highly value work-life balance and have it now, but haven't verified new opportunity offers the same",
                    "critical"
                )
            )
        
        # Risk tolerance violations
        if (self.values.get("financial_security") and
            self.values["financial_security"] >= 8 and
            self.financial.get("financial_runway_months") and
            self.financial["financial_runway_months"] < 6):
            new_violations.append(
                ConstraintViolation(
                    "values_mismatch",
                    "You rate financial security as high priority (8+) but have less than 6 months runway",
                    "warning"
                )
            )
        
        # ── Career-specific contradiction checks ───────────────────────────
        decision_type = self.decision_metadata.get("decision_type")
        if decision_type in ("career_choice", "education"):

            # WLB vs leadership ambition conflict
            if (self.values.get("work_life_balance") and
                self.values["work_life_balance"] >= 8 and
                self.career_vision.get("desired_role_5yr") and
                any(role in str(self.career_vision["desired_role_5yr"]).lower()
                    for role in ["lead", "manager", "director", "executive", "cto"])):
                new_violations.append(ConstraintViolation(
                    "career_wlb_conflict",
                    "You rate work-life balance 8+ but are targeting a leadership role — "
                    "those roles typically demand long hours, especially early on",
                    "warning"
                ))

            # Research aversion vs PhD interest
            if (self.interests.get("research") is False and
                self.decision_metadata.get("options_being_compared") and
                "PhD" in str(self.decision_metadata.get("options_being_compared", []))):
                new_violations.append(ConstraintViolation(
                    "research_phd_conflict",
                    "You said you dislike research, but PhD is still an option — "
                    "a PhD is almost entirely research for 4-5 years",
                    "critical"
                ))

            # High financial security value but choosing low-income path
            if (self.values.get("financial_security") and
                self.values["financial_security"] >= 8 and
                self.career_vision.get("post_graduation_goal") == "phd"):
                new_violations.append(ConstraintViolation(
                    "financial_phd_conflict",
                    "You rate financial security 8+ but are leaning toward a PhD — "
                    "most PhD stipends are significantly below industry salaries",
                    "warning"
                ))

            # Hands-on preference vs research path
            if (self.interests.get("hands_on_work") is True and
                self.career_vision.get("research_vs_applied") == "research"):
                new_violations.append(ConstraintViolation(
                    "handson_research_conflict",
                    "You said you prefer hands-on work but also indicated a research path — "
                    "academic research is often more theoretical than applied",
                    "warning"
                ))

            # Impact value vs low-visibility role
            if (self.values.get("impact") and
                self.values["impact"] >= 8 and
                self.career_vision.get("research_vs_applied") == "applied" and
                self.interests.get("research") is False):
                # This is actually alignment, not conflict — skip
                pass

        # ── Offer comparison contradiction checks ─────────────────────────
        if self.decision_metadata.get("decision_subtype") == "offer_comparison":
            sal_a = self.offer_a.get("salary")
            sal_b = self.offer_b.get("salary")

            # Both salaries known and user rates financial security high
            if (sal_a and sal_b and
                self.values.get("financial_security") and
                self.values["financial_security"] >= 8):
                try:
                    diff = abs(float(sal_a) - float(sal_b))
                    if diff < 5000:
                        new_violations.append(ConstraintViolation(
                            "salary_similar",
                            f"Both offers have similar salaries (within $5k) — "
                            f"salary shouldn't be the deciding factor here",
                            "info"
                        ))
                except (TypeError, ValueError):
                    pass

            # Growth matters but no growth info collected
            if (self.values.get("career_growth") and
                self.values["career_growth"] >= 8 and
                not self.offer_a.get("growth_potential") and
                not self.offer_b.get("growth_potential")):
                new_violations.append(ConstraintViolation(
                    "growth_unknown",
                    "Career growth is a top priority (8+) but you haven't shared "
                    "the growth potential for either offer",
                    "warning"
                ))

            # WLB matters but not asked
            if (self.values.get("work_life_balance") and
                self.values["work_life_balance"] >= 8 and
                not self.offer_a.get("work_life_balance") and
                not self.offer_b.get("work_life_balance")):
                new_violations.append(ConstraintViolation(
                    "wlb_unknown",
                    "Work-life balance is a top priority but hasn't been compared "
                    "across the two offers",
                    "warning"
                ))

        # Update violations list
        self.violations = new_violations
    
    def get_missing_critical_info(self) -> List[str]:
        """Identify what critical information is still unknown"""
        missing = []

        decision_type = self.decision_metadata.get("decision_type")

        # Career / education decisions: don't require financial data
        if decision_type in ("career_choice", "education"):
            interests_known = sum(1 for v in self.interests.values() if v is not None)
            if interests_known == 0:
                missing.append("What kind of work they enjoy")
            if not self.career_vision.get("desired_role_5yr"):
                missing.append("Career vision / desired role in 5 years")
        else:
            # General / financial decisions: need basic financials
            if self.financial["current_savings"] is None:
                missing.append("Current savings amount")
            if self.financial["monthly_expenses"] is None:
                missing.append("Monthly expenses")

            # If considering a specific opportunity, need some details
            if self.opportunity["company"]:
                if not self.opportunity["work_life_balance_known"]:
                    missing.append("Work-life balance at new opportunity")
                if not self.opportunity["team_culture_known"]:
                    missing.append("Team culture and expectations")
        
        self.missing_critical_info = missing
        return missing
    
    def can_analyze(self) -> bool:
        """Determine if we have enough information to provide analysis"""
        critical_missing = self.get_missing_critical_info()
        return len(critical_missing) == 0
    
    def get_decision_mode(self) -> str:
        """
        Determine the analysis mode based on constraints.
        This is NOT a recommendation - it's a description of the constraint environment.
        """
        decision_type = self.decision_metadata.get("decision_type")

        # Career / education decisions: mode is based on clarity of preferences, not finances
        if decision_type in ("career_choice", "education"):
            interests_known = sum(1 for v in self.interests.values() if v is not None)
            career_known = sum(1 for v in self.career_vision.values() if v is not None)
            total = interests_known + career_known
            if total == 0:
                return "INSUFFICIENT_DATA"
            elif total < 3:
                return "CAUTIOUS_MODE"
            else:
                return "GROWTH_MODE"

        # General decisions: mode is based on financial runway
        if not self.financial["financial_runway_months"]:
            return "INSUFFICIENT_DATA"
        
        # Critical financial stress
        if (self.financial.get("debt_total") and 
            self.financial["debt_total"] > 0 and
            self.financial.get("debt_monthly_payment") and
            self.financial["current_income"] and
            self.financial["debt_monthly_payment"] > 0.3 * self.financial["current_income"]):
            return "SURVIVAL_MODE"
        
        # Cautious conditions
        runway = self.financial["financial_runway_months"]
        if (runway < 6 or 
            self.legal.get("visa_constrained") or
            (self.relationships.get("has_dependents") and 
             self.values.get("financial_security", 0) >= 7)):
            return "CAUTIOUS_MODE"
        
        # Growth conditions
        if runway >= 12:
            return "GROWTH_MODE"
        
        return "CAUTIOUS_MODE"  # Default conservative
    
    def to_dict(self) -> Dict:
        """Export current state as dictionary"""
        return {
            # ── core categories ──────────────────────────────────────────────
            "decision_metadata": self.decision_metadata,
            "offer_a": self.offer_a,
            "offer_b": self.offer_b,
            "personal": self.personal,
            "uni_a": self.uni_a,
            "uni_b": self.uni_b,
            "interests": self.interests,
            "career_vision": self.career_vision,
            "strengths": self.strengths,
            # ── existing categories ──────────────────────────────────────────
            "financial": self.financial,
            "legal": self.legal,
            "values": self.values,
            "relationships": self.relationships,
            "opportunity": self.opportunity,
            "current": self.current,
            # ── derived / meta ───────────────────────────────────────────────
            "violations": [
                {
                    "type": v.violation_type,
                    "description": v.description,
                    "severity": v.severity,
                }
                for v in self.violations
            ],
            "missing_info": self.missing_critical_info,
            "decision_mode": self.get_decision_mode(),
        }
    
    def get_state_summary(self) -> str:
        """Get a human-readable state summary"""
        lines = ["â•" * 60]
        lines.append("CURRENT STATE")
        lines.append("â•" * 60)
        
        # Show what we know
        known_count = sum(
            1 for cat in [self.financial, self.legal, self.values,
                         self.relationships, self.opportunity, self.current,
                         self.interests, self.career_vision, self.strengths,
                         self.decision_metadata]
            for v in cat.values() if v is not None and v != [] and v != False and v != ""
        )
        
        lines.append(f"Known constraints: {known_count}")
        lines.append(f"Decision mode: {self.get_decision_mode()}")
        
        # Show violations
        if self.violations:
            lines.append("\nâš ï¸ CONSTRAINT VIOLATIONS:")
            for v in self.violations:
                lines.append(f"  â€¢ {v.description}")
        
        # Show missing info
        missing = self.get_missing_critical_info()
        if missing:
            lines.append("\nâŒ MISSING CRITICAL INFORMATION:")
            for m in missing:
                lines.append(f"  â€¢ {m}")
        
        lines.append("â•" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    # Test the engine
    state = DecisionState()
    
    print("Initial state:")
    print(state.get_state_summary())
    
    print("\n\nUpdating state...")
    state.update("current", "current_employer", "Google")
    state.update("financial", "current_savings", 50000)
    state.update("financial", "monthly_expenses", 4000)
    state.update("values", "work_life_balance", 9)
    state.update("current", "current_wlb", "great")
    state.update("opportunity", "company", "Apple")
    
    print("\n\nUpdated state:")
    print(state.get_state_summary())
    
    print("\n\nState dict:")
    print(json.dumps(state.to_dict(), indent=2, default=str))