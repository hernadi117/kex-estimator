import estimation
import auxmodels
import geometricbrownianmotion as gbm

ts_real = gbm.gbm_naive(0.2, 0.5, 100, 253, 1, seed=True)

est = estimation.indirect_inference_estimate(ts_real, gbm.gbm_naive, auxmodels.naive_indirect_ml_estimator,
                                             method="local",
                                             replications=10, kwargs={"gbm_kwargs": {"m": 1000, "n": 253, "s0": 1},
                                                                      "aux_kwargs": {}})
print(est)

print(estimation.two_stage_estimate(ts_real))
