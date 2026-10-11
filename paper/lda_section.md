## 3.5 Linear Discriminant Analysis

Linear Discriminant Analysis (LDA) is a supervised classifier that models each class with a Gaussian distribution and assumes a shared covariance matrix. Given class means μₖ and pooled covariance Σ, its class score is δₖ(x)=xᵀΣ⁻¹μₖ−½μₖᵀΣ⁻¹μₖ+log πₖ, where πₖ is the class prior. A sample is assigned to the class with the highest score. Equal covariance yields a linear decision boundary.

The geometric motivation is Fisher's criterion J(w)=(wᵀS_Bw)/(wᵀS_Ww), where S_B is between-class scatter and S_W is within-class scatter. Its stationary projections satisfy S_Bw=λS_Ww. Since binary S_B has rank at most one, phishing-versus-legitimate discrimination has at most one nonzero discriminant direction. Classification LDA should be distinguished from using LDA only as a dimensionality-reduction step.

For this study, class `-1` is mapped to phishing and class `+1` to legitimate. Thirty verified structured features form x; the sequential row index is removed. The scikit-learn implementation compares its SVD classifier to LSQR with automatic covariance shrinkage. This accommodates potential covariance ill-conditioning. Five-fold stratified cross-validation on training data selects a configuration; the untouched stratified test set is used for final reporting. Shared Gaussian covariance is an approximation: correlated features are permitted, but nonlinear class boundaries and dataset shift can limit the model. Probabilities are model estimates rather than certainty.

### References cited

[1] I. Hannousse and S. Yahiouche, “Phishing website prediction using base and ensemble classifier techniques with cross-validation,” *Cybersecurity*, vol. 5, art. 19, 2022, doi: 10.1186/s42400-022-00126-9. This study evaluates LDA as one base classifier alongside other algorithms on phishing website datasets; it reports 92.87% accuracy for its experiment 1 LDA result. Its split and data handling differ from this project, so the result is contextual, not directly comparable.
