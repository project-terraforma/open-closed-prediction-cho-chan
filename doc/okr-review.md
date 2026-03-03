okr review
v1 feb 25 2026
Restate Your Original OKRs
Objective 1
Deliver a production-viable model that accurately predicts whether a real-world place is open or closed at a global scale.
Key Results:
KR 1.1: Achieve ≥ 85% F1 score and outperform a rule-based baseline by ≥ 10%.
KR 1.2: Maintain ≤ 8% performance variance across major place categories.
KR 1.3: Meet scalability constraints for 100M+ places with acceptable inference cost and latency.

Objective 2
Make a data-driven recommendation on the optimal model and feature set.
Key Results:
KR 2.1: Identify and rank the top 5 predictive features.
KR 2.2: Benchmark model architectures and produce a cost vs accuracy comparison.
KR 2.3: Deliver a final deployment recommendation supported by empirical evidence.

Evaluate Progress
KR 1.1
F1 ≥ 85%, ≥ 10% over baseline
Status: Behind
Best model performance:
```
Model              AUC-ROC  AUC-PR    F1   Prec  Recall
--------------------------------------------------------
GBM + OHE           0.686   0.255  0.277  0.173   0.698
XGBoost + OHE       0.720   0.293  0.311  0.206   0.635
MLP + NCM           0.721   0.243  0.302  0.218   0.492
MLP + SLDA          0.726   0.253  0.308  0.235   0.444
MLP head            0.734   0.263  0.315  0.313   0.318   <- best
```
The MLP models do slightly better than GBM on both AUC-ROC and F1, which tells us there is a real signal in the data. But the gap to 0.85 F1 seems to be more about the limits of the dataset than the model itself, given the heavy class imbalance, small number of positive examples, and overlap between open and closed features.

KR 1.2
≤ 8% category variance
Status: Not yet evaluated
We only have 313 closed places in total, and many categories have very few positives. That makes per-category metrics unstable. We will evaluate category performance only if we have at least 20 closed examples.

KR 1.3
Scalability to 100M+ places
Status: On track
The model is lightweight and designed for scale. Inference is constant time, and updates can be done incrementally instead of retraining from scratch. A full pass over 100M places is feasible in minutes on a CPU. From a systems perspective, this objective is in good shape.

KR 2.1
Top 5 predictive features
Status: Complete
The strongest signals are address completeness, confidence, source confidence, phone presence, and overall completeness score. One especially interesting finding is source staleness. Closed places tend to have Microsoft data that is about 2.5 times older than open places. Time-based signals look promising.



KR 2.2
Architecture benchmark
Status: In progress
We’ve completed the accuracy comparison. What’s left is the operational side like inference speed, memory usage, and update cost per release.

KR 2.3
Deployment recommendation
Status: Done in the final phase

Challenges & Insights
Challenge 1: Severe class imbalance
The dataset contains 3,425 samples with only 9% closed places. The validation set contains roughly 63 closed examples, making F1 highly unstable and threshold-sensitive.

Challenge 2: Limited dataset scope
The data is US-only and drawn from two primary sources. That limits diversity and generalization. There is only so much signal we can extract from this setup.

Challenge 3: Indirect signal
Most features are proxies for closure, like missing phone numbers or stale data. Even the strongest feature still overlaps heavily between open and closed classes. That overlap likely puts a ceiling on performance.

Key insights:
AUC-ROC around 0.73 confirms we are learning a meaningful ranking signal. Completeness-related features and source staleness are the most informative.
Continual learning approaches do not overcome data scarcity but provide a structural operational advantage through incremental updates at scale.
Given how much the feature distributions overlap and how imbalanced the dataset is, it might be very difficult to reach an F1 of 0.85 with this data, even with better tuning.

Refining Your OKRs
Revised Objective 1
Deliver a scalable, continual-learning model that improves over baseline for open/closed prediction.
Key Results:
KR 1.1: Achieve AUC-ROC ≥ 0.78 and F1 ≥ 0.40, with ≥ 5% AUC-ROC improvement over GBM.
KR 1.2: Evaluate per-category performance only if at least 20 closed samples exist.
KR 1.3: Maintain scalability to 100M+ places with efficient inference and incremental updates per release.

Keep Objective 2
Strategic Next Steps
Tune decision thresholds to improve F1.
Add interaction and time-based features.
Run category-level evaluations where sample size allows.
Simulate multi-release updates to show the benefit of continual learning.
Finish the cost comparison table.
Run structured error analysis to understand failure patterns.
Deliver a clear deployment recommendation that covers model choice, feature priorities, operational integration, and what additional data would most improve performance.

