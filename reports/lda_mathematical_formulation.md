# Mathematical formulation: Linear Discriminant Analysis

Let observations be vectors xᵢ∈Rᵖ with class yᵢ∈{0,1}; here class 1 denotes phishing. For K classes, nₖ is the count, μₖ the sample mean of class k, and μ the overall mean.

Classification LDA is a probabilistic classifier: it models each class distribution and applies Bayes' rule. Dimensionality-reduction LDA instead uses class labels to find projections that maximize class separation; these views are related but their outputs and objectives are not interchangeable.

Within-class scatter is S_W=ΣₖΣ_{i:yᵢ=k}(xᵢ−μₖ)(xᵢ−μₖ)ᵀ. Between-class scatter is S_B=Σₖnₖ(μₖ−μ)(μₖ−μ)ᵀ. A projection w is scored by Fisher's criterion J(w)=(wᵀS_Bw)/(wᵀS_Ww). Stationary directions satisfy S_Bw=λS_Ww. With two classes, rank(S_B)≤1, so there is at most one nonzero discriminant direction.

For binary classification, under Gaussian class-conditional densities with a shared covariance Σ, the discriminant is δₖ(x)=xᵀΣ⁻¹μₖ−(1/2)μₖᵀΣ⁻¹μₖ+log πₖ. Predict the class with maximum δₖ(x); equivalently phishing is selected when δ₁(x)>δ₀(x). Here πₖ is the prior probability. The equal-covariance assumption makes the boundary linear. LDA does not assume conditional feature independence; correlated features are allowed, although severe collinearity can make covariance estimates ill-conditioned. Singular covariance arises when predictors are redundant or p is large relative to sample count. Shrinkage blends the empirical covariance toward a better-conditioned target.

In this application x contains 30 encoded web/URL indicators and classes mean phishing versus legitimate. Predictive associations are not causal explanations. Strong nonlinear interactions can require nonlinear models, at a cost to the assigned linear baseline.

scikit-learn `LinearDiscriminantAnalysis` implements the classification formulation. `solver="svd"` uses a rank-aware SVD approach and does not support shrinkage. `solver="lsqr", shrinkage="auto"` estimates a shrinkage covariance and is suitable for classification but does not expose the same transform behavior. The implementation standardizes with train-fold-fitted preprocessing, compares these two classifiers by training-only stratified CV F1, then evaluates the chosen complete pipeline once on the held-out test data. This pipeline-level comparison is the operational selection criterion; it is not a direct numerical solution of the displayed generalized eigenproblem.
