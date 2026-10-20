"""Variable Role Resolution — deterministic, research-question-aware variable classification.

Resolves variable roles using:
1. Deterministic heuristics (identifier patterns, type analysis)
2. Research question parsing (mapping language to variables)
3. Conflict detection and resolution

Every role decision has provenance and is auditable.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Variable Roles
# ---------------------------------------------------------------------------

class VariableRole(str, Enum):
    IDENTIFIER = "identifier"
    OUTCOME = "outcome"
    EXPOSURE = "exposure"
    PREDICTOR = "predictor"
    GROUPING_VARIABLE = "grouping_variable"
    CONFOUNDER_CANDIDATE = "confounder_candidate"
    TIME_VARIABLE = "time_variable"
    DATE_VARIABLE = "date_variable"
    COVARIATE = "covariate"
    UNKNOWN = "unknown"
    SURVIVAL_TIME = "survival_time"
    EVENT_INDICATOR = "event_indicator"


class RoleSource(str, Enum):
    """Provenance for role assignments."""
    DETERMINISTIC_RULE = "deterministic_rule"
    RESEARCH_QUESTION_MATCH = "research_question_match"
    RESEARCHER_INPUT = "researcher_input"
    AI_INTERPRETATION = "ai_interpretation"
    INFERRED_FROM_DATA = "inferred_from_data"


class RoleStatus(str, Enum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"
    REVIEW_REQUIRED = "review_required"
    RESEARCHER_CONFIRMED = "researcher_confirmed"


# ---------------------------------------------------------------------------
# Analysis Problem Types
# ---------------------------------------------------------------------------

class AnalysisProblemType(str, Enum):
    """Statistical problem families — what question is the researcher asking?"""
    TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME = "two_independent_groups_continuous_outcome"
    TWO_PAIRED_GROUPS_CONTINUOUS_OUTCOME = "two_paired_groups_continuous_outcome"
    MULTI_GROUP_CONTINUOUS_OUTCOME = "multi_group_continuous_outcome"
    CONTINUOUS_CONTINUOUS_ASSOCIATION = "continuous_continuous_association"
    BINARY_OUTCOME_SINGLE_PREDICTOR = "binary_outcome_single_predictor"
    BINARY_OUTCOME_MULTIVARIABLE = "binary_outcome_multivariable"
    CONTINUOUS_OUTCOME_MULTIVARIABLE = "continuous_outcome_multivariable"
    CATEGORICAL_OUTCOME = "categorical_outcome"
    TIME_TO_EVENT = "time_to_event"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Variable Role Resolution Model
# ---------------------------------------------------------------------------

class VariableRoleResolution(BaseModel):
    """Structured resolution of a single variable's role."""
    variable_name: str
    proposed_role: VariableRole
    role_source: RoleSource
    confidence: str = "moderate"  # high, moderate, low
    linked_research_question_term: str | None = None
    candidate_roles: list[VariableRole] = []
    conflicts: list[str] = []
    status: RoleStatus = RoleStatus.RESOLVED
    requires_researcher_review: bool = False
    reasoning_summary: str = ""


class VariableRoleResolutionSet(BaseModel):
    """Complete set of variable role resolutions for a dataset."""
    resolutions: list[VariableRoleResolution] = []
    research_question: str | None = None
    problem_type: AnalysisProblemType = AnalysisProblemType.UNKNOWN
    has_conflicts: bool = False
    unresolved_variables: list[str] = []


# ---------------------------------------------------------------------------
# Research Question Parser
# ---------------------------------------------------------------------------

class ResearchQuestionParser:
    """Parse a research question to extract statistical intent.

    This is deterministic — no LLM. Uses keyword patterns to identify:
    - Outcome terms (what is being measured)
    - Exposure/grouping terms (what is being compared)
    - Comparison structure (two groups, multiple groups, correlation, etc.)
    """

    # Patterns that suggest a group comparison
    GROUP_COMPARISON_PATTERNS = [
        r"diff(?:er)?\s+(?:between|across|among|in|from)",
        r"compar(?:e|ing|ison)\s+(?:between|across|among|of)",
        r"difference\s+(?:in|of|between|among)",
        r"effect\s+(?:of|on)",
        r"affect(?:s|ed)?\s+(?:\w+\s+)*by",
        r"affect(?:s|ed)?\s+(?:the\s+)?(?:\w+\s+)*\w+(?:\s|$|\?)",
        r"(?:treatment|group|arm)\s+(?:effect|difference|comparison)",
        r"(?:higher|lower|greater|less|differ)\s+(?:in|than|among|from)",
        r"mean\s+(?:difference|comparison)",
    ]

    # Patterns that suggest correlation/association
    CORRELATION_PATTERNS = [
        r"assoc(?:iation|iated)?\s+(?:between|with)",
        r"relat(?:ionship|ed|ion)\s+(?:between|with|to|of)",
        r"correlat(?:ion|ed|es)\s+(?:with|between)",
        r"predict(?:or|ing|s)?\s+(?:of|for|on)",
        r"linear\s+relationship",
        r"linked\s+(?:to|with|between)",
    ]

    # Patterns that suggest paired/repeated measures
    PAIRED_PATTERNS = [
        r"pre[\s-]?post",
        r"before[\s-]?after",
        r"paired",
        r"repeated[\s-]?measures",
        r"within[\s-]?subject",
        r"longitudinal",
        r"change\s+(?:from|between)\s+.+\s+(?:and|to)",
    ]

    # Patterns that suggest survival/time-to-event analysis
    SURVIVAL_PATTERNS = [
        r"time\s+to\s+(?:event|death|recurrence|relapse|failure|onset|progression|diagnosis)",
        r"(?:overall|cause[\s-]specific)?\s*survival",
        r"kaplan[\s-]?meier",
        r"cox\s+(?:proportional|hazard)",
        r"hazard\s+ratio",
        r"censored",
        r"censoring",
        r"time[\s-]to[\s-]event",
        r"(?:event[\s-])?free\s+survival",
        r"mortality",
        r"(?:time|duration)\s+(?:of|to)\s+(?:follow[\s-]?up|observation)",
    ]

    # Survival time variable name patterns
    SURVIVAL_TIME_PATTERNS = [
        r"^time$", r"^t$", r"^survival[_\s]?time$", r"^time[_\s]?to[_\s]?(?:event|death|failure)",
        r"^follow[\s-]?up$", r"^duration$", r"^survival$", r"^t[_\s]?(?:obs|fu|follow)",
    ]

    # Event indicator name patterns
    EVENT_INDICATOR_PATTERNS = [
        r"^event$", r"^status$", r"^censor(?:ed|ing)?$", r"^delta$",
        r"^observed$", r"^failed$", r"^event[_\s]?indicator$",
    ]

    # Outcome-related keywords
    OUTCOME_KEYWORDS = [
        "outcome", "response", "effect", "result", "endpoint",
        "dependent", "measure", "score", "level", "concentration",
        "pressure", "weight", "height", "temperature", "rate",
        "diagnosis", "event", "mortality", "survival", "death",
        "improvement", "change", "difference", "reduction",
        "disease", "illness", "condition",
    ]

    # Exposure/grouping keywords
    EXPOSURE_KEYWORDS = [
        "treatment", "intervention", "exposure", "group", "arm",
        "cohort", "condition", "therapy", "drug", "dose",
        "factor", "independent", "predictor", "variable",
    ]

    @classmethod
    def parse(cls, research_question: str) -> dict[str, Any]:
        """Parse a research question into structured intent.

        Returns dict with:
        - outcome_terms: list of terms likely referring to the outcome
        - exposure_terms: list of terms likely referring to the exposure/grouping
        - comparison_type: "group_comparison", "correlation", "paired", "survival", or "unknown"
        - paired: bool
        - survival: bool (True if survival analysis is indicated)
        """
        if not research_question:
            return {
                "outcome_terms": [],
                "exposure_terms": [],
                "comparison_type": "unknown",
                "paired": False,
                "survival": False,
            }

        rq_lower = research_question.lower()

        # Detect comparison type
        paired = any(re.search(p, rq_lower) for p in cls.PAIRED_PATTERNS)

        # Detect survival analysis
        survival = any(re.search(p, rq_lower) for p in cls.SURVIVAL_PATTERNS)

        is_group_comparison = any(
            re.search(p, rq_lower) for p in cls.GROUP_COMPARISON_PATTERNS
        )
        is_correlation = any(
            re.search(p, rq_lower) for p in cls.CORRELATION_PATTERNS
        )

        if paired:
            comparison_type = "paired"
        elif survival:
            comparison_type = "survival"
        elif is_group_comparison:
            comparison_type = "group_comparison"
        elif is_correlation:
            comparison_type = "correlation"
        else:
            comparison_type = "unknown"

        # Extract potential outcome terms
        outcome_terms = []
        # Pattern 1: "effect of X on Y" or "affect X" — positional: term after "on" or "affect"
        effect_on_match = re.findall(
            r"(?:effect\s+of\s+\w+\s+on|affect(?:s|ed)?)\s+(?:the\s+)?(.+?)(?:\?|$|\.|\s+after\s+adjusting|\s+and\s+|\s+versus)",
            rq_lower,
        )
        for match in effect_on_match:
            stripped = match.strip()
            if stripped:
                outcome_terms.append(stripped)

        # Pattern 1b: causal/reduction verbs — "reduce/lower/prevent/increase X".
        # The plan-create error message tells researchers to name the outcome
        # column in the research question; without this pattern the common
        # "Does <exposure> reduce <outcome>?" phrasing extracted no outcome
        # term at all and naming the column had no effect.
        if not outcome_terms:
            causal_match = re.findall(
                r"(?:reduc(?:e|es|ed|ing|tion)|lower(?:s|ed|ing)?|"
                r"decreas(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|"
                r"prevent(?:s|ed|ing)?|improv(?:e|es|ed|ing)|"
                r"rais(?:e|es|ed|ing)|caus(?:e|es|ed|ing))\s+(?:the\s+)?"
                r"(?:(?:risk|rate|chance|odds|probability|incidence|prevalence|level)"
                r"\s+(?:of|in)\s+)?"
                r"(.+?)(?:\?|$|\.|,|\s+after\s+adjusting|\s+after\s+controlling"
                r"|\s+versus|\s+compared\s+(?:with|to)|\s+and\s+|\s+from\s+|\s+to\s+)",
                rq_lower,
            )
            for match in causal_match:
                stripped = match.strip()
                if stripped:
                    outcome_terms.append(stripped)

        # Pattern 2: "in/of/between X" with keyword filter
        if not outcome_terms:
            in_of_match = re.findall(
                r"(?:in|of|between)\s+(?:the\s+)?(.+?)(?:\?|$|\.|\s+and\s+|\s+versus)",
                rq_lower,
            )
            for match in in_of_match:
                words = match.split()
                for kw in cls.OUTCOME_KEYWORDS:
                    for word in words:
                        if kw in word:
                            outcome_terms.append(match.strip())
                            break

        # For correlation questions, be more precise: only the term after "associated with" is the outcome
        if comparison_type == "correlation":
            corr_outcome_match = re.findall(
                r"(?:associated|related|correlated|linked)\s+(?:with|to)\s+(\w[\w\s]*?)(?:\?|$|\.)",
                rq_lower,
            )
            if corr_outcome_match:
                # Override outcome_terms with just the correlation outcome
                outcome_terms = [t.strip() for t in corr_outcome_match if t.strip()]

        # Also look for specific outcome patterns — only if no positional outcome terms found
        if not outcome_terms:
            for kw in cls.OUTCOME_KEYWORDS:
                if kw in rq_lower and comparison_type != "correlation":
                    idx = rq_lower.find(kw)
                    start = max(0, rq_lower.rfind(" ", 0, max(0, idx - 30)))
                    end = min(len(rq_lower), rq_lower.find(" ", idx + len(kw) + 15))
                    if end <= start:
                        end = min(len(rq_lower), idx + len(kw) + 10)
                    term = rq_lower[start:end].strip()
                    if term and term not in outcome_terms:
                        outcome_terms.append(term)

        # Extract exposure/grouping terms — look for specific keywords, not whole phrases
        exposure_terms = []
        for kw in cls.EXPOSURE_KEYWORDS:
            if kw in rq_lower:
                # Just add the keyword itself, not surrounding context
                if kw not in exposure_terms:
                    exposure_terms.append(kw)

        # For correlation questions, extract the non-outcome term as a predictor
        if comparison_type == "correlation":
            # Pattern: "Is X associated with Y?" or "X related to Y?"
            corr_match = re.findall(
                r"(?:is\s+)?(\w[\w\s]*?)\s+(?:associated|related|correlated|linked)\s+(?:with|to)\s+(\w[\w\s]*?)(?:\?|$|\.)",
                rq_lower,
            )
            for term1, term2 in corr_match:
                t1 = term1.strip()
                t2 = term2.strip()
                t1_is_outcome = any(kw in t1 for kw in cls.OUTCOME_KEYWORDS)
                t2_is_outcome = any(kw in t2 for kw in cls.OUTCOME_KEYWORDS)

                if t1_is_outcome and t2_is_outcome:
                    # Both match outcome keywords — in "Is X related to Y?", X=predictor, Y=outcome
                    # Ensure outcome_terms has t2, exposure_terms has t1
                    if t2 and t2 not in outcome_terms:
                        outcome_terms.append(t2)
                    if t1 and t1 not in exposure_terms:
                        exposure_terms.append(t1)
                elif t1_is_outcome:
                    # t1 is outcome, t2 is predictor
                    if t2 and t2 not in exposure_terms:
                        exposure_terms.append(t2)
                elif t2_is_outcome:
                    # t2 is outcome, t1 is predictor
                    if t1 and t1 not in exposure_terms:
                        exposure_terms.append(t1)
                else:
                    # Neither matches outcome keywords — first is predictor, second is outcome
                    if t1 and t1 not in exposure_terms:
                        exposure_terms.append(t1)
                    if t2 and t2 not in outcome_terms:
                        outcome_terms.append(t2)

        return {
            "outcome_terms": outcome_terms,
            "exposure_terms": exposure_terms,
            "comparison_type": comparison_type,
            "paired": paired,
            "survival": survival,
        }


# ---------------------------------------------------------------------------
# Role Resolver
# ---------------------------------------------------------------------------

class RoleResolver:
    """Resolve variable roles using deterministic rules and research question context.

    Architecture:
    1. Deterministic identifier exclusion (always first)
    2. Research question mapping (if available)
    3. Heuristic fallback (name patterns, data characteristics)
    4. Conflict detection
    """

    # Identifier patterns — always excluded from statistical roles
    ID_PATTERNS = [
        r"\bid\b", r"_id$", r"^id[_\b]", r"idx", r"index",
        r"patient", r"subject", r"record", r"sample", r"participant",
        r"identifier", r"key", r"uuid",
    ]

    # Date/time patterns — only match standalone date/datetime fields, not duration names like recovery_time
    DATE_PATTERNS = [
        r"^date$", r"^time$", r"^datetime$", r"dob", r"^created$", r"^updated$",
        r"^timestamp$", r"^birthday$", r"^admission$", r"^discharge$",
        r"^date_", r"^created_", r"^updated_",
    ]

    # Predictor/exposure name patterns
    EXPOSURE_PATTERNS = [
        r"treatment", r"intervention", r"exposure", r"group", r"arm",
        r"cohort", r"condition", r"therapy", r"drug", r"dose",
        r"factor", r"assign",
    ]

    # Outcome name patterns — only names that are unambiguously outcomes
    OUTCOME_PATTERNS = [
        r"outcome", r"result", r"response", r"endpoint", r"death", r"died",
        r"mortality", r"survival", r"event", r"diagnosis", r"diagnosed",
        r"effect", r"dependent", r"^y_", r"target", r"label",
        r"score", r"concentration", r"pressure",
        r"temperature", r"rate", r"improvement", r"change",
        r"hospitalization", r"biomarker", r"marker",
        r"disease", r"recovery", r"illness",
        r"pss", r"stress", r"anxiety", r"depression", r"pain",
    ]

    # Survival time variable name patterns (used by RoleResolver)
    SURVIVAL_TIME_PATTERNS = [
        r"^time$", r"^t$", r"^survival[_\s]?time$", r"^time[_\s]?to[_\s]?(?:event|death|failure)",
        r"^follow[\s-]?up$", r"^duration$", r"^survival$", r"^t[_\s]?(?:obs|fu|follow)",
    ]

    # Event indicator name patterns (used by RoleResolver)
    EVENT_INDICATOR_PATTERNS = [
        r"^event$", r"^status$", r"^censor(?:ed|ing)?$", r"^delta$",
        r"^observed$", r"^failed$", r"^event[_\s]?indicator$",
        r"^died$", r"^recurrence$", r"^hospitalized$", r"^relapse$",
    ]

    @classmethod
    def resolve_all(
        cls,
        variables: list[dict[str, Any]],
        research_question: str | None = None,
    ) -> VariableRoleResolutionSet:
        """Resolve roles for all variables in a dataset.

        Args:
            variables: list of dicts with 'name', 'dtype', 'n_unique', 'n_total', 'n_missing'
            research_question: optional research question string

        Returns:
            VariableRoleResolutionSet with resolutions for each variable
        """
        resolutions = []

        # Parse the research question
        rq_intent = ResearchQuestionParser.parse(research_question)

        # Track assigned roles to detect conflicts
        assigned_outcomes = []
        assigned_exposures = []
        assigned_survival_time = []
        assigned_event_indicator = []

        for var in variables:
            resolution = cls._resolve_single(
                var, rq_intent, assigned_outcomes, assigned_exposures
            )
            resolutions.append(resolution)

            if resolution.proposed_role == VariableRole.OUTCOME:
                assigned_outcomes.append(var["name"])
            elif resolution.proposed_role == VariableRole.SURVIVAL_TIME:
                assigned_survival_time.append(var["name"])
            elif resolution.proposed_role == VariableRole.EVENT_INDICATOR:
                assigned_event_indicator.append(var["name"])
            elif resolution.proposed_role in (VariableRole.EXPOSURE, VariableRole.GROUPING_VARIABLE):
                assigned_exposures.append(var["name"])

        # Detect conflicts
        has_conflicts = any(r.conflicts for r in resolutions)
        unresolved = [r.variable_name for r in resolutions if r.status in (RoleStatus.AMBIGUOUS, RoleStatus.REVIEW_REQUIRED)]

        # Classify the statistical problem
        problem_type = cls._classify_problem(resolutions, rq_intent)

        return VariableRoleResolutionSet(
            resolutions=resolutions,
            research_question=research_question,
            problem_type=problem_type,
            has_conflicts=has_conflicts,
            unresolved_variables=unresolved,
        )

    @classmethod
    def _resolve_single(
        cls,
        var: dict[str, Any],
        rq_intent: dict[str, Any],
        assigned_outcomes: list[str],
        assigned_exposures: list[str],
    ) -> VariableRoleResolution:
        """Resolve the role for a single variable."""
        name = var.get("name", "")
        dtype = var.get("dtype", "unknown")
        n_unique = var.get("n_unique", 0)
        n_total = var.get("n_total", 0) or var.get("n_unique", 0) + var.get("n_missing", 0)
        n_missing = var.get("n_missing", 0)
        lower = name.lower()

        conflicts = []

        # Check for pre-assigned role (from caller context)
        existing_role = var.get("role")
        if existing_role:
            # Map existing role strings to VariableRole enum
            role_mapping = {
                "outcome": VariableRole.OUTCOME,
                "outcome_candidate": VariableRole.OUTCOME,
                "dependent": VariableRole.OUTCOME,
                "predictor": VariableRole.PREDICTOR,
                "predictor_candidate": VariableRole.PREDICTOR,
                "exposure": VariableRole.EXPOSURE,
                "independent": VariableRole.PREDICTOR,
                "grouping_variable": VariableRole.GROUPING_VARIABLE,
                "group": VariableRole.GROUPING_VARIABLE,
                "covariate": VariableRole.COVARIATE,
                "confounder": VariableRole.CONFOUNDER_CANDIDATE,
                "confounder_candidate": VariableRole.CONFOUNDER_CANDIDATE,
                "potential_confounder": VariableRole.CONFOUNDER_CANDIDATE,
                "identifier": VariableRole.IDENTIFIER,
                "temporal": VariableRole.TIME_VARIABLE,
                "date": VariableRole.DATE_VARIABLE,
            }
            mapped_role = role_mapping.get(existing_role)
            if mapped_role:
                return VariableRoleResolution(
                    variable_name=name,
                    proposed_role=mapped_role,
                    role_source=RoleSource.RESEARCHER_INPUT,
                    confidence="high",
                    status=RoleStatus.RESOLVED,
                    reasoning_summary=f"Pre-assigned role: {existing_role}",
                )

        # --- Step 1: Deterministic identifier check (highest priority) ---
        is_identifier = cls._is_identifier(name, n_unique, n_total, n_missing)
        if is_identifier:
            return VariableRoleResolution(
                variable_name=name,
                proposed_role=VariableRole.IDENTIFIER,
                role_source=RoleSource.DETERMINISTIC_RULE,
                confidence="high",
                status=RoleStatus.RESOLVED,
                reasoning_summary=f"Identifier detected: n_unique={n_unique}, n_total={n_total}, pattern match",
            )

        # --- Step 1a: Survival variable detection (when RQ indicates survival) ---
        # This must happen before date detection because "time" is a common survival variable name
        if rq_intent.get("survival", False):
            for pattern in cls.SURVIVAL_TIME_PATTERNS:
                if re.search(pattern, lower):
                    return VariableRoleResolution(
                        variable_name=name,
                        proposed_role=VariableRole.SURVIVAL_TIME,
                        role_source=RoleSource.RESEARCH_QUESTION_MATCH,
                        confidence="high",
                        status=RoleStatus.RESOLVED,
                        reasoning_summary=f"Survival time pattern detected: {pattern}",
                    )

            for pattern in cls.EVENT_INDICATOR_PATTERNS:
                if re.search(pattern, lower):
                    return VariableRoleResolution(
                        variable_name=name,
                        proposed_role=VariableRole.EVENT_INDICATOR,
                        role_source=RoleSource.RESEARCH_QUESTION_MATCH,
                        confidence="high",
                        status=RoleStatus.RESOLVED,
                        reasoning_summary=f"Event indicator pattern detected: {pattern}",
                    )

        # --- Step 2: Date/time check ---
        if any(re.search(p, lower) for p in cls.DATE_PATTERNS):
            return VariableRoleResolution(
                variable_name=name,
                proposed_role=VariableRole.DATE_VARIABLE,
                role_source=RoleSource.DETERMINISTIC_RULE,
                confidence="high",
                status=RoleStatus.RESOLVED,
                reasoning_summary=f"Date/time pattern detected in variable name",
            )

        # --- Step 3: Research question mapping ---
        rq_outcome_match = cls._match_rq_to_variable(name, rq_intent.get("outcome_terms", []))
        rq_exposure_match = cls._match_rq_to_variable(name, rq_intent.get("exposure_terms", []))

        proposed_role = None
        role_source = None
        confidence = "moderate"
        linked_term = None
        candidate_roles = []

        if rq_outcome_match:
            proposed_role = VariableRole.OUTCOME
            role_source = RoleSource.RESEARCH_QUESTION_MATCH
            confidence = "high"
            linked_term = rq_outcome_match

        if rq_exposure_match:
            if proposed_role is not None:
                # Conflict: RQ maps to both outcome and exposure
                conflicts.append(
                    f"Research question maps '{name}' to both outcome and exposure"
                )
                proposed_role = None
                role_source = RoleSource.RESEARCH_QUESTION_MATCH
                confidence = "low"
            else:
                # Determine if it's an exposure or grouping variable
                if n_unique <= 20 and n_unique >= 2:
                    proposed_role = VariableRole.EXPOSURE
                else:
                    proposed_role = VariableRole.PREDICTOR
                role_source = RoleSource.RESEARCH_QUESTION_MATCH
                confidence = "high"
                linked_term = rq_exposure_match

        # --- Step 4: Heuristic fallback (name patterns) ---
        if proposed_role is None:
            proposed_role, confidence = cls._heuristic_role(
                name, dtype, n_unique, n_total, assigned_outcomes
            )
            role_source = RoleSource.DETERMINISTIC_RULE

        # --- Step 5: Candidate roles ---
        if proposed_role == VariableRole.OUTCOME:
            candidate_roles = [VariableRole.OUTCOME]
        elif proposed_role in (VariableRole.EXPOSURE, VariableRole.GROUPING_VARIABLE):
            candidate_roles = [VariableRole.EXPOSURE, VariableRole.GROUPING_VARIABLE, VariableRole.PREDICTOR]
        elif proposed_role == VariableRole.PREDICTOR:
            candidate_roles = [VariableRole.PREDICTOR, VariableRole.COVARIATE]
        else:
            candidate_roles = [proposed_role]

        # --- Step 6: Conflict detection with previously assigned roles ---
        if proposed_role == VariableRole.OUTCOME and assigned_outcomes:
            conflicts.append(
                f"Multiple outcome candidates: {', '.join(assigned_outcomes + [name])}"
            )

        status = RoleStatus.RESOLVED
        requires_review = False
        if conflicts:
            status = RoleStatus.REVIEW_REQUIRED
            requires_review = True
        elif confidence == "low":
            status = RoleStatus.AMBIGUOUS

        reasoning = cls._build_reasoning(
            name, proposed_role, role_source, rq_outcome_match,
            rq_exposure_match, is_identifier=False,
        )

        return VariableRoleResolution(
            variable_name=name,
            proposed_role=proposed_role,
            role_source=role_source,
            confidence=confidence,
            linked_research_question_term=linked_term,
            candidate_roles=candidate_roles,
            conflicts=conflicts,
            status=status,
            requires_researcher_review=requires_review,
            reasoning_summary=reasoning,
        )

    @classmethod
    def _is_identifier(
        cls, name: str, n_unique: int, n_total: int, n_missing: int
    ) -> bool:
        """Determine if a variable is an identifier."""
        lower = name.lower()

        # Pattern match (name-based)
        for pattern in cls.ID_PATTERNS:
            if re.search(pattern, lower):
                return True

        # Every value is unique (and not missing) AND looks like sequential integers
        if n_total > 0 and n_unique == n_total and n_missing == 0:
            # Only flag as identifier if the name suggests it or values are sequential
            # Continuous variables with all unique values are NOT identifiers
            name_suggests_id = any(
                re.search(p, lower)
                for p in [r"\bid\b", r"_id$", r"^id", r"idx", r"index",
                           r"patient", r"subject", r"record", r"sample",
                           r"participant", r"identifier", r"key", r"number", r"num"]
            )
            if name_suggests_id:
                return True

        return False

    # Common abbreviations for outcome terms
    OUTCOME_ABBREVIATIONS = {
        "pss": "perceived stress scale",
        "pss10": "perceived stress scale 10",
        "phq": "patient health questionnaire",
        "gad": "generalized anxiety disorder",
        "bmi": "body mass index",
        "bp": "blood pressure",
        "hr": "heart rate",
        "glucose": "blood glucose",
        "cholesterol": "total cholesterol",
        "hba1c": "hemoglobin a1c",
        "creatinine": "serum creatinine",
        "alt": "alanine aminotransferase",
        "ast": "aspartate aminotransferase",
    }

    @classmethod
    def _match_rq_to_variable(
        cls, var_name: str, rq_terms: list[str]
    ) -> str | None:
        """Check if a variable name matches any research question terms.

        Uses multiple matching strategies:
        1. Direct substring match
        2. Word overlap (>= 1 common word)
        3. Underscore/hyphen normalized match
        4. Partial word match (e.g., "systolic" matches "systolic_bp")
        5. Abbreviation match (e.g., "stress" matches "PSS10")
        """
        # Normalize variable name: replace underscores/hyphens with spaces
        lower = var_name.lower().replace("_", " ").replace("-", " ")
        var_words = set(lower.split())

        for term in rq_terms:
            term_lower = term.lower().strip()
            # Remove trailing punctuation
            term_clean = term_lower.rstrip("?!.,")

            # Direct substring match
            if term_clean in lower or lower in term_clean:
                return term

            # Word overlap
            term_words = set(term_clean.split())
            overlap = term_words & var_words
            if len(overlap) >= 1 and len(overlap) / max(len(term_words), 1) >= 0.3:
                return term

            # Partial word match: check if any word in the term matches any word in the variable
            for tw in term_words:
                for vw in var_words:
                    if len(tw) >= 4 and len(vw) >= 4 and (tw.startswith(vw) or vw.startswith(tw)):
                        return term

            # Abbreviation match: check if variable name contains a known abbreviation
            # for the research question term
            for abbr, full_term in cls.OUTCOME_ABBREVIATIONS.items():
                if abbr in lower and any(word in term_clean for word in full_term.split()):
                    return term

        return None

    @classmethod
    def _heuristic_role(
        cls, name: str, dtype: str, n_unique: int, n_total: int,
        assigned_outcomes: list[str] | None = None,
    ) -> tuple[VariableRole, str]:
        """Fallback heuristic role assignment based on name patterns and data characteristics."""
        lower = name.lower()

        # Exposure/grouping patterns
        for pattern in cls.EXPOSURE_PATTERNS:
            if re.search(pattern, lower):
                if n_unique <= 20 and n_unique >= 2:
                    return VariableRole.EXPOSURE, "moderate"
                return VariableRole.PREDICTOR, "moderate"

        # Outcome patterns
        for pattern in cls.OUTCOME_PATTERNS:
            if re.search(pattern, lower):
                return VariableRole.OUTCOME, "moderate"

        # Binary variables with few levels are often grouping variables
        if n_unique == 2 and dtype in ("numeric", "categorical", "int64", "int32"):
            return VariableRole.GROUPING_VARIABLE, "low"

        # High-cardinality numeric: if outcome already assigned, this is likely a predictor
        if dtype in ("numeric", "float64", "float32", "int64", "int32"):
            if assigned_outcomes:
                return VariableRole.PREDICTOR, "low"
            # Don't auto-assign outcome by cardinality alone — name patterns
            # (OUTCOME_PATTERNS) are authoritative. Use PREDICTOR as safe default.
            return VariableRole.PREDICTOR, "low"
            return VariableRole.PREDICTOR, "low"

        return VariableRole.UNKNOWN, "low"

    @classmethod
    def _classify_problem(
        cls,
        resolutions: list[VariableRoleResolution],
        rq_intent: dict[str, Any],
    ) -> AnalysisProblemType:
        """Classify the statistical problem type from variable roles and RQ intent."""
        outcomes = [r for r in resolutions if r.proposed_role == VariableRole.OUTCOME]
        exposures = [r for r in resolutions if r.proposed_role in (VariableRole.EXPOSURE, VariableRole.GROUPING_VARIABLE)]
        identifiers = [r for r in resolutions if r.proposed_role == VariableRole.IDENTIFIER]

        # Check for survival analysis
        survival_times = [r for r in resolutions if r.proposed_role == VariableRole.SURVIVAL_TIME]
        event_indicators = [r for r in resolutions if r.proposed_role == VariableRole.EVENT_INDICATOR]
        if survival_times and event_indicators:
            return AnalysisProblemType.TIME_TO_EVENT

        if not outcomes:
            return AnalysisProblemType.UNKNOWN

        comparison = rq_intent.get("comparison_type", "unknown")
        paired = rq_intent.get("paired", False)

        # Check if any outcome is likely binary (only 2 unique values in data)
        # We don't have data here, so check by name patterns suggesting binary
        binary_outcome_names = {"hospitalization", "hospitalized", "death", "died", "event", "diagnosis", "diagnosed", "disease"}
        has_binary_outcome = any(
            any(kw in r.variable_name.lower() for kw in binary_outcome_names)
            for r in outcomes
        )

        # Binary outcome problems take priority over correlation
        if has_binary_outcome:
            if len(exposures) >= 1:
                return AnalysisProblemType.BINARY_OUTCOME_MULTIVARIABLE if len(exposures) > 1 else AnalysisProblemType.BINARY_OUTCOME_SINGLE_PREDICTOR
            non_id_vars = [r for r in resolutions if r.proposed_role != VariableRole.IDENTIFIER]
            if len(non_id_vars) == 2:
                return AnalysisProblemType.BINARY_OUTCOME_SINGLE_PREDICTOR

        if comparison == "paired":
            return AnalysisProblemType.TWO_PAIRED_GROUPS_CONTINUOUS_OUTCOME

        if comparison == "group_comparison":
            if len(exposures) == 1:
                return AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME
            elif len(exposures) > 1:
                return AnalysisProblemType.MULTI_GROUP_CONTINUOUS_OUTCOME

        if comparison == "correlation":
            non_id_vars = [r for r in resolutions if r.proposed_role != VariableRole.IDENTIFIER]
            if len(non_id_vars) == 2:
                return AnalysisProblemType.CONTINUOUS_CONTINUOUS_ASSOCIATION

        # Default: if we have outcome + exposure, it's likely a group comparison
        if outcomes and exposures:
            if paired:
                return AnalysisProblemType.TWO_PAIRED_GROUPS_CONTINUOUS_OUTCOME
            if len(exposures) == 1:
                return AnalysisProblemType.TWO_INDEPENDENT_GROUPS_CONTINUOUS_OUTCOME
            return AnalysisProblemType.MULTI_GROUP_CONTINUOUS_OUTCOME

        return AnalysisProblemType.UNKNOWN

    @classmethod
    def _build_reasoning(
        cls,
        name: str,
        role: VariableRole,
        source: RoleSource,
        rq_outcome: str | None,
        rq_exposure: str | None,
        is_identifier: bool,
    ) -> str:
        """Build a concise reasoning summary."""
        parts = []
        if is_identifier:
            parts.append("Identifier detected by pattern/unicity")
        if rq_outcome:
            parts.append(f"Matches RQ outcome term '{rq_outcome}'")
        if rq_exposure:
            parts.append(f"Matches RQ exposure term '{rq_exposure}'")
        if source == RoleSource.DETERMINISTIC_RULE:
            parts.append("Assigned by heuristic rule")
        if role == VariableRole.UNKNOWN:
            parts.append("No clear role identified")
        return "; ".join(parts) if parts else f"Role: {role.value}"


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def resolve_variable_roles(
    variables: list[dict[str, Any]],
    research_question: str | None = None,
) -> VariableRoleResolutionSet:
    """Convenience function for variable role resolution."""
    return RoleResolver.resolve_all(variables, research_question)


def is_identifier(var: dict[str, Any]) -> bool:
    """Quick check if a variable is an identifier."""
    return RoleResolver._is_identifier(
        var.get("name", ""),
        var.get("n_unique", 0),
        var.get("n_total", 0) or var.get("n_unique", 0) + var.get("n_missing", 0),
        var.get("n_missing", 0),
    )


def get_non_identifier_roles() -> set[VariableRole]:
    """Roles that identifiers must never have."""
    return {
        VariableRole.OUTCOME,
        VariableRole.EXPOSURE,
        VariableRole.PREDICTOR,
        VariableRole.GROUPING_VARIABLE,
        VariableRole.CONFOUNDER_CANDIDATE,
    }
