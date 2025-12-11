# Credit Scoring Business Understanding

This section summarizes the business and regulatory context for the credit-scoring project and explains key design choices documented in this repository.

---

## 1. Basel II and the Need for an Interpretable, Well-Documented Model

Basel II requires banks to measure credit risk systematically and to hold capital proportional to that risk. It emphasizes three pillars: (a) rigorous estimation of risk parameters such as Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD); (b) supervisory review; and (c) transparency and market discipline. As a result, any model influencing credit decisions must be **transparent, auditable, and well documented**, enabling regulators and internal risk teams to validate data sources, model assumptions, segmentation logic, and how outputs link to provisioning and capital calculations.

Opaque or poorly documented models increase regulatory, capital, and operational risk.

### Practical Implications for This Project

* Log all data transformations, feature definitions (R, F, M, S), snapshot date, and grouping logic.
* Maintain reproducible pipelines, seed values, and versioned datasets.
* Provide clear validation artifacts (performance by segment, calibration, stability over time) to support supervisory review.

---

## 2. Why Create a Proxy Default Label, and What Are the Business Risks?

### Why a Proxy Is Necessary

Because the provided dataset does not include a direct loan default label, we cannot train a PD model using true repayment outcomes. A well-defined proxy (e.g., an RFM-based "disengaged" cluster labeled `is_high_risk`) provides a supervised target constructed from behavioral signals. International industry and development guidance shows that alternative data and proxy targets are common in credit assessment when traditional bureau data is limited.

### Business Risks of Using a Proxy

* **Label mismatch / imperfect proxy:** Proxy-based labels may not match true default outcomes, producing biased PD estimates.
* **Operational risk:** Decisions based on proxy labels may lead to higher losses or reputational damage.
* **Regulatory risk:** If the proxy is not justified or documented, regulators may challenge its use in credit decisions or capital models.
* **Fairness & discrimination risk:** Proxy construction may inadvertently encode biases.
* **Feedback loop risk:** Decisions informed by the proxy may alter customer behavior, degrading future model reliability.

### Mitigations and Controls

* Validate the proxy using any available ground truth (pilot loans, collections outcomes).
* Treat the model as experimental initially; run controlled pilots or shadow deployments.
* Fully document proxy creation, rationale, and limitations in this README and governance artifacts.

---

## 3. Trade-Offs: Interpretable Models vs. Complex Models in Regulated Settings

### Advantages of Simple, Interpretable Models (e.g., Logistic Regression + WoE)

* **Transparency:** Easy to explain coefficients and monotonic relationships.
* **Ease of validation:** Straightforward to test for leakage, stability, or unexpected driver behavior.
* **Regulatory acceptance:** Preferred for use cases tied to capital or provisioning.
* **Operational simplicity:** Faster to deploy and maintain.

### Advantages of Complex Models (e.g., GBM, XGBoost)

* **Higher predictive power:** Capture nonlinearities and interactions.
* **Improved segmentation:** Often yield higher AUC/ROC and better ranking performance.

### Recommended Approach

* Prioritize interpretable models for production if outputs feed credit decisions or regulatory capital.
* Use complex models experimentally, adding strong explainability (SHAP/ICE) and strict governance.
* Implement robust monitoring for model drift, calibration degradation, and feature stability.
* Combine approaches when appropriate: e.g., GBM for ranking paired with an interpretable scorecard for decisions.

---


