Skill Recommendation Engine — Algorithm Approach
Problem Summary

Given a student's history of completed skill levels (with scores and time spent), recommend the top 3 next levels they are most likely to attempt and pass (estimated pass rate > 70%).

Algorithm: Similarity-Based Collaborative Filtering

The engine follows a classic user-based collaborative filtering pipeline adapted for skill progression data.
Step 1 — Feature Engineering
Each student is represented as a vector of length 20 (one dimension per skill level). The encoding captures three states:
StateEncoded ValueNot attempted0.0Attempted but failed (score < 70)0.5Passed (score ≥ 70)score / 100 ∈ [0.70, 1.0]
This preserves partial progress information — a student who tried and failed is closer to a student who passed than to one who never attempted at all.
Step 2 — Cosine Similarity
For a target student t, cosine similarity is computed against every other student u:
sim(t, u) = (v_t · v_u) / (||v_t|| × ||v_u||)
Cosine similarity is chosen over Euclidean distance because it captures relative learning patterns regardless of total levels completed. A beginner who passed 3 levels and an advanced student who passed 18 levels can still be "similar" if their profiles overlap proportionally.
The full N × N similarity matrix is pre-computed at fit() time using scikit-learn's cosine_similarity, enabling O(1) lookups at inference time.
Step 3 — Neighbour Selection
The top-K nearest neighbours (default K = 20) are selected — excluding the student themselves. Only neighbours with positive similarity are considered.
Step 4 — Candidate Level Collection
From each neighbour, all levels they completed that the target student has not yet attempted are gathered. Each entry records: (neighbour similarity, whether they passed).
Levels with fewer than min_neighbour_support (default 2) endorsements are discarded to avoid noisy recommendations.
Step 5 — Composite Scoring
Each candidate level is scored by:
score = avg_sim × blended_pass_rate × coverage_factor
Where:

avg_sim — average cosine similarity of neighbours who completed the level (weights more similar peers higher)
blended_pass_rate — Bayesian blend of neighbour pass rate (70%) and global pass rate (30%), reducing variance from small neighbour samples
coverage_factor — min(n_endorsements / K, 1.0), rewarding levels with broad neighbour support

Levels with blended pass rate below the 70% threshold are filtered out before ranking.
Step 6 — Output
Candidates are sorted descending by composite score. The top 3 are returned with their score, estimated pass rate, and neighbour support count.

Design Decisions
DecisionRationaleCosine similarity over PearsonNo need to centre scores; sparser vectors benefit from cosinePre-compute similarity matrixAmortises cost; O(1) per student inference70/30 Bayesian blendPrevents extreme estimates when a student has few similar peersmin_neighbour_support=2Balances recall vs. precision for cold-start studentssuccess_rate_threshold=0.70Matches the evaluation criterion directly

Complexity

Fit: O(N² × L) where N = students, L = levels (20). For 1,000 students: ~20M ops — runs in < 1 second.
Inference: O(K × levels) after matrix is built — effectively O(1) per student.
Memory: Stores N × N float matrix (~8 MB for 1,000 students).


Evaluation Strategy
The engine is evaluated using a leave-one-out proxy: for a sample of students, mask their most recently completed level and check if the engine recommended it. The hit_rate metric measures what fraction of the time the held-out level appears in the top-3. Additionally, avg_estimated_pass_rate validates that recommendations consistently exceed the 70% success threshold.
