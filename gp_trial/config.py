A_bound = (0, 1000)
R_bound = (0, 20)
D_bound = (0, 20)
F_bound = (0, 20)

target_dc = 0.5

period = 3.734

lhs_points = 10
stopping_threshold = 0.001
n_restart = 13
lambda_uniformity = 0.3
raw_samples = 130

# weight on spatial non-uniformity (diagnostics['problem_fraction'] from
# compute_duty_cycle -- fraction of the sample that measured as overpoled/
# unstable rather than clean). A parameter combination that produces a
# partially-overpoled sample should be penalized, not just averaged over the
# clean region as if the rest of the sample didn't exist.
lambda_spatial = 0.3

# floor for the GP's per-point observation noise (heteroskedastic train_Yvar) -- guards against a
# measured std of ~0 producing a near-zero noise variance, which can make the GP's kernel matrix
# numerically singular
min_observation_variance = 1e-6