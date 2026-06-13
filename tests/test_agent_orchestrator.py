"""
Tests for the Agent Orchestrator (ReAct Reasoning Loop).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent_orchestrator import (
    ContextRetriever,
    CorporatePolicy,
    CustomerProfile,
    ReActOrchestrator,
    RetentionProposal,
    RetentionProposalGenerator,
    RetentionStrategy,
    RiskLevel,
    RiskPredictor,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def risk_predictor() -> RiskPredictor:
    """Load the risk predictor tool."""
    return RiskPredictor()


@pytest.fixture(scope="module")
def context_retriever() -> ContextRetriever:
    """Create a context retriever tool."""
    return ContextRetriever()


@pytest.fixture(scope="module")
def proposal_generator() -> RetentionProposalGenerator:
    """Create a retention proposal generator tool."""
    return RetentionProposalGenerator()


@pytest.fixture(scope="module")
def orchestrator() -> ReActOrchestrator:
    """Create the full orchestrator."""
    return ReActOrchestrator()


HIGH_RISK_CUSTOMER = CustomerProfile(
    customer_id="TEST-HIGH",
    gender="Male",
    SeniorCitizen=1,
    Partner="No",
    Dependents="No",
    tenure=2,
    PhoneService="Yes",
    MultipleLines="Yes",
    InternetService="Fiber optic",
    OnlineSecurity="No",
    OnlineBackup="No",
    DeviceProtection="No",
    TechSupport="No",
    StreamingTV="Yes",
    StreamingMovies="Yes",
    Contract="Month-to-month",
    PaperlessBilling="Yes",
    PaymentMethod="Electronic check",
    MonthlyCharges=95.50,
    TotalCharges=191.00,
)

LOW_RISK_CUSTOMER = CustomerProfile(
    customer_id="TEST-LOW",
    gender="Female",
    SeniorCitizen=0,
    Partner="Yes",
    Dependents="Yes",
    tenure=60,
    PhoneService="Yes",
    MultipleLines="No",
    InternetService="DSL",
    OnlineSecurity="Yes",
    OnlineBackup="Yes",
    DeviceProtection="Yes",
    TechSupport="Yes",
    StreamingTV="No",
    StreamingMovies="No",
    Contract="Two year",
    PaperlessBilling="No",
    PaymentMethod="Bank transfer (automatic)",
    MonthlyCharges=45.00,
    TotalCharges=2700.00,
)


# ─── Test: Risk Predictor (Tool 1) ─────────────────────────────────────────────

class TestRiskPredictor:
    """Tests for the Risk Predictor tool."""

    def test_model_loaded(self, risk_predictor: RiskPredictor) -> None:
        """Model is loaded successfully."""
        assert risk_predictor.model is not None

    def test_high_risk_prediction(self, risk_predictor: RiskPredictor) -> None:
        """High-risk customer gets elevated churn probability."""
        result = risk_predictor.predict(HIGH_RISK_CUSTOMER)
        assert result.churn_probability >= 0.3
        assert result.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert len(result.top_risk_factors) > 0

    def test_low_risk_prediction(self, risk_predictor: RiskPredictor) -> None:
        """Low-risk customer gets low churn probability."""
        result = risk_predictor.predict(LOW_RISK_CUSTOMER)
        assert result.churn_probability <= 0.7
        assert result.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)

    def test_probability_range(self, risk_predictor: RiskPredictor) -> None:
        """Predicted probability is always in [0, 1]."""
        result = risk_predictor.predict(HIGH_RISK_CUSTOMER)
        assert 0.0 <= result.churn_probability <= 1.0

    def test_model_metadata(self, risk_predictor: RiskPredictor) -> None:
        """Model metadata is populated."""
        result = risk_predictor.predict(HIGH_RISK_CUSTOMER)
        assert result.model_name != ""
        assert result.model_confidence in ("Low", "Medium", "High")


# ─── Test: Context Retriever (Tool 2) ──────────────────────────────────────────

class TestContextRetriever:
    """Tests for the Context Retriever tool."""

    def test_context_returned(
        self,
        context_retriever: ContextRetriever,
        risk_predictor: RiskPredictor,
    ) -> None:
        """Context is returned with all required fields."""
        risk = risk_predictor.predict(HIGH_RISK_CUSTOMER)
        context = context_retriever.retrieve(HIGH_RISK_CUSTOMER, risk)
        assert context.customer_id == "TEST-HIGH"
        assert context.loyalty_tier in ("Bronze", "Silver", "Gold", "Platinum")
        assert context.lifetime_value > 0

    def test_loyalty_tier_by_tenure(
        self,
        context_retriever: ContextRetriever,
        risk_predictor: RiskPredictor,
    ) -> None:
        """Loyalty tier correlates with tenure."""
        risk_high = risk_predictor.predict(HIGH_RISK_CUSTOMER)
        risk_low = risk_predictor.predict(LOW_RISK_CUSTOMER)

        ctx_high = context_retriever.retrieve(HIGH_RISK_CUSTOMER, risk_high)
        ctx_low = context_retriever.retrieve(LOW_RISK_CUSTOMER, risk_low)

        # Short tenure -> lower tier
        assert ctx_high.loyalty_tier in ("Bronze", "Silver")
        # Long tenure -> higher tier
        assert ctx_low.loyalty_tier in ("Gold", "Platinum")

    def test_high_risk_has_tickets(
        self,
        context_retriever: ContextRetriever,
        risk_predictor: RiskPredictor,
    ) -> None:
        """High-risk customers get more support tickets."""
        risk = risk_predictor.predict(HIGH_RISK_CUSTOMER)
        context = context_retriever.retrieve(HIGH_RISK_CUSTOMER, risk)
        assert len(context.recent_tickets) >= 1

    def test_deterministic_output(
        self,
        context_retriever: ContextRetriever,
        risk_predictor: RiskPredictor,
    ) -> None:
        """Same customer produces same context (deterministic seed)."""
        risk = risk_predictor.predict(HIGH_RISK_CUSTOMER)
        ctx1 = context_retriever.retrieve(HIGH_RISK_CUSTOMER, risk)
        ctx2 = context_retriever.retrieve(HIGH_RISK_CUSTOMER, risk)
        assert ctx1.loyalty_tier == ctx2.loyalty_tier
        assert ctx1.nps_score == ctx2.nps_score


# ─── Test: Retention Proposal Generator (Tool 3) ──────────────────────────────

class TestRetentionProposalGenerator:
    """Tests for the Retention Proposal Generator tool."""

    def test_high_risk_proposal(
        self,
        proposal_generator: RetentionProposalGenerator,
        risk_predictor: RiskPredictor,
        context_retriever: ContextRetriever,
    ) -> None:
        """High-risk customer receives a proposal with immediate urgency."""
        risk = risk_predictor.predict(HIGH_RISK_CUSTOMER)
        context = context_retriever.retrieve(HIGH_RISK_CUSTOMER, risk)
        proposal = proposal_generator.generate(HIGH_RISK_CUSTOMER, risk, context)

        assert isinstance(proposal, RetentionProposal)
        assert proposal.urgency in ("Immediate", "Within 48 hours")
        assert len(proposal.personalized_offer) > 20
        assert len(proposal.recommended_actions) >= 2
        assert "%" in proposal.estimated_retention_lift

    def test_low_risk_proposal(
        self,
        proposal_generator: RetentionProposalGenerator,
        risk_predictor: RiskPredictor,
    ) -> None:
        """Low-risk customer receives a lighter proposal."""
        risk = risk_predictor.predict(LOW_RISK_CUSTOMER)
        proposal = proposal_generator.generate(LOW_RISK_CUSTOMER, risk, None)

        assert isinstance(proposal, RetentionProposal)
        assert proposal.urgency in ("This week", "This month")
        assert len(proposal.personalized_offer) > 10

    def test_policy_compliance_flag(
        self,
        proposal_generator: RetentionProposalGenerator,
        risk_predictor: RiskPredictor,
    ) -> None:
        """Proposals include a policy compliance flag."""
        risk = risk_predictor.predict(HIGH_RISK_CUSTOMER)
        proposal = proposal_generator.generate(HIGH_RISK_CUSTOMER, risk, None)
        assert isinstance(proposal.policy_compliant, bool)
        assert len(proposal.policy_notes) >= 1

    def test_discount_capping(self) -> None:
        """Discount validation caps values above the policy maximum."""
        gen = RetentionProposalGenerator()
        capped, note = gen._validate_discount(50.0)
        assert capped == gen.policy.max_discount_pct
        assert note is not None
        assert "capped" in note.lower()

    def test_discount_within_policy(self) -> None:
        """Discount within policy range passes validation cleanly."""
        gen = RetentionProposalGenerator()
        value, note = gen._validate_discount(10.0)
        assert value == 10.0
        assert note is None

    def test_free_months_capping(self) -> None:
        """Free months validation caps values above the policy maximum."""
        gen = RetentionProposalGenerator()
        capped, note = gen._validate_free_months(12)
        assert capped == gen.policy.max_free_months
        assert note is not None
        assert "capped" in note.lower()

    def test_free_months_within_policy(self) -> None:
        """Free months within policy range passes validation cleanly."""
        gen = RetentionProposalGenerator()
        value, note = gen._validate_free_months(2)
        assert value == 2
        assert note is None

    def test_corporate_policy_defaults(self) -> None:
        """Corporate policy has sensible default constraints."""
        policy = CorporatePolicy()
        assert policy.max_discount_pct == 25.0
        assert policy.max_free_months == 3
        assert policy.min_contract_months == 12
        assert policy.requires_contract_commitment is True
        assert len(policy.eligible_upgrades) >= 3

    def test_senior_citizen_action(
        self,
        proposal_generator: RetentionProposalGenerator,
        risk_predictor: RiskPredictor,
    ) -> None:
        """Senior citizen customers get a senior support action."""
        risk = risk_predictor.predict(HIGH_RISK_CUSTOMER)
        proposal = proposal_generator.generate(HIGH_RISK_CUSTOMER, risk, None)
        senior_actions = [a for a in proposal.recommended_actions if "senior" in a.lower()]
        assert len(senior_actions) >= 1


# ─── Test: ReAct Orchestrator ──────────────────────────────────────────────────

class TestReActOrchestrator:
    """Tests for the full reasoning loop."""

    def test_high_risk_strategy(self, orchestrator: ReActOrchestrator) -> None:
        """High-risk customer gets a complete retention strategy."""
        strategy = orchestrator.run(HIGH_RISK_CUSTOMER)
        assert isinstance(strategy, RetentionStrategy)
        assert strategy.customer_id == "TEST-HIGH"
        assert strategy.churn_probability >= 0.3
        assert strategy.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert len(strategy.personalized_offer) > 20
        assert len(strategy.recommended_actions) >= 2
        assert strategy.urgency in ("Immediate", "Within 48 hours")

    def test_low_risk_strategy(self, orchestrator: ReActOrchestrator) -> None:
        """Low-risk customer gets a lighter engagement strategy."""
        strategy = orchestrator.run(LOW_RISK_CUSTOMER)
        assert isinstance(strategy, RetentionStrategy)
        assert strategy.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)
        assert len(strategy.personalized_offer) > 10
        assert len(strategy.recommended_actions) >= 1

    def test_reasoning_trace_populated(self, orchestrator: ReActOrchestrator) -> None:
        """The reasoning trace has at least 4 steps."""
        strategy = orchestrator.run(HIGH_RISK_CUSTOMER)
        assert len(strategy.reasoning_trace) >= 4
        # Check that we have all three phases
        phases = {step.phase for step in strategy.reasoning_trace}
        assert "Thought" in phases
        assert "Action" in phases
        assert "Observation" in phases

    def test_high_risk_triggers_context_retrieval(
        self, orchestrator: ReActOrchestrator
    ) -> None:
        """High-risk customers trigger the context retrieval step."""
        strategy = orchestrator.run(HIGH_RISK_CUSTOMER)
        action_texts = [
            s.content for s in strategy.reasoning_trace if s.phase == "Action"
        ]
        # Should have at least 2 actions: risk predictor + context retriever
        assert len(action_texts) >= 2

    def test_strategy_has_reasoning_summary(self, orchestrator: ReActOrchestrator) -> None:
        """Strategy includes a reasoning summary."""
        strategy = orchestrator.run(HIGH_RISK_CUSTOMER)
        assert len(strategy.reasoning_summary) > 30
        assert strategy.customer_id in strategy.reasoning_summary

    def test_estimated_retention_lift(self, orchestrator: ReActOrchestrator) -> None:
        """Strategy includes an estimated retention lift."""
        strategy = orchestrator.run(HIGH_RISK_CUSTOMER)
        assert strategy.estimated_retention_lift != "N/A"
        assert "%" in strategy.estimated_retention_lift

    def test_orchestrator_has_three_tools(self, orchestrator: ReActOrchestrator) -> None:
        """Orchestrator has all three registered tools."""
        assert hasattr(orchestrator, "risk_predictor")
        assert hasattr(orchestrator, "context_retriever")
        assert hasattr(orchestrator, "proposal_generator")
        assert isinstance(orchestrator.risk_predictor, RiskPredictor)
        assert isinstance(orchestrator.context_retriever, ContextRetriever)
        assert isinstance(orchestrator.proposal_generator, RetentionProposalGenerator)

    def test_high_risk_trace_has_proposal_step(
        self, orchestrator: ReActOrchestrator
    ) -> None:
        """High-risk trace includes the Retention Proposal Generator action."""
        strategy = orchestrator.run(HIGH_RISK_CUSTOMER)
        action_contents = [
            s.content for s in strategy.reasoning_trace if s.phase == "Action"
        ]
        proposal_actions = [
            a for a in action_contents
            if "Proposal" in a or "Retention" in a or "proposal" in a or "retention" in a
        ]
        assert len(proposal_actions) >= 1, (
            f"Expected a Proposal Generator action in trace, got: {action_contents}"
        )
