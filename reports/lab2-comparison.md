# Lab 2 — Run comparison

Experiment `itcs355-lab2` · 12 trials · total spend 1.3400 THB

`thb_per_point` is cost per percentage point of val_roc_auc above the worst trial. Cheap improvements rank low; expensive improvements rank high, however good the headline number is.

| run_id   |   val_roc_auc |   cost_thb |   n_estimators |   max_depth |   min_samples_leaf |   thb_per_point |
|:---------|--------------:|-----------:|---------------:|------------:|-------------------:|----------------:|
| fc8eba44 |        0.8426 |     0.107  |            100 |           4 |                  5 |          0.0663 |
| 9ee96e89 |        0.8424 |     0.1398 |            100 |           4 |                  1 |          0.0877 |
| 52b39842 |        0.8411 |     0.1072 |            300 |           4 |                  5 |          0.0732 |
| 0c9b36c8 |        0.8404 |     0.1069 |            300 |           4 |                  1 |          0.0767 |
| 205154d7 |        0.8397 |     0.1066 |            100 |           8 |                  5 |          0.0805 |
| c3f5c571 |        0.8377 |     0.1549 |            300 |           8 |                  5 |          0.1378 |
| 3ce051de |        0.8354 |     0.1067 |            300 |          12 |                  5 |          0.1194 |
| b7e56027 |        0.8338 |     0.1066 |            300 |           8 |                  1 |          0.1453 |
| 564b2830 |        0.8322 |     0.0905 |            100 |          12 |                  5 |          0.1577 |
| 09c8bd1b |        0.8312 |     0.1074 |            100 |           8 |                  1 |          0.2267 |
| 7dabfd4c |        0.8268 |     0.1074 |            100 |          12 |                  1 |          3.1738 |
| 7ced4dbf |        0.8265 |     0.099  |            300 |          12 |                  1 |         25.7811 |

## Which model did you register, and why?

**My answer:**\
I selected the Random Forest configuration with 100 estimators, max depth 4, and minimum samples per leaf 5. It achieved the highest validation ROC-AUC in the original 12-trial study at 0.8426. However, the margin over the next-best result, 0.8424, is very small compared with the seed variation, so I do not treat this difference as strong evidence of better performance. I still selected this configuration because it was among the best-performing settings and had a lower estimated training cost of 0.107 THB.

To evaluate stability, I repeated this configuration with three seeds. The validation ROC-AUC values were 0.8426, 0.8479, and 0.8492, giving a mean of approximately 0.8466 and a standard deviation of about 0.0029. The corresponding test ROC-AUC values were 0.8533, 0.8643, and 0.8426.

The average estimated training cost across the three runs was approximately 0.118 THB. Assuming one retraining run per month, the estimated monthly retraining cost is also approximately 0.118 THB.

This choice could still be wrong if the current validation and test splits are not representative of future production data. Dataset shift could cause another configuration to generalize better after deployment.
