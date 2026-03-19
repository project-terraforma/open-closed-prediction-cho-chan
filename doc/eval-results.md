3k labeled, with conf, no aug
Results saved to models/evals/eval_results_1773668231.json
cost table(ran first)
Results saved to models/cost_table_1773668278.json

3k labeled, wo conf, no aug
Results saved to models/evals/eval_results_1773668349.json
cost table
Results saved to models/cost_table_1773668366.json

3k labeled, w conf, aug: sf 0 75 85
n --include-conf true
Loading data/project_c_samples.json ...
Loading augment records from sf_aug_0_75_85.json ...
  +9,547 records  (444 closed, 9,103 open)
  Excluded features: ['has_only_meta', 'n_sources_with_update_time', 'min_update_age_days', 'max_update_age_days']

Split complete  (seed=42)  (include_conf=true)  (val = original benchmark only)
  Train: 12,287  |  closed=694  open=11,593
  Val:      685  |  closed=63  open=622
  X shape: 24 features  (23 numeric + 1 categorical)
  Category vocab size: 862 (+1 OOV)
Results saved to models/evals/eval_results_1773668500.json
cost table
Results saved to models/cost_table_1773668517.json

3k labeled, wo conf, aug: sf 0 75 85
+9,547 records  (444 closed, 9,103 open)
  Excluded features: ['has_only_meta', 'n_sources_with_update_time', 'min_update_age_days', 'max_update_age_days']

Split complete  (seed=42)  (no-conf)  (val = original benchmark only)
  Train: 12,287  |  closed=694  open=11,593
  Val:      685  |  closed=63  open=622
  X shape: 19 features  (18 numeric + 1 categorical)
  Category vocab size: 862 (+1 OOV)

Results saved to models/evals/eval_results_1773668429.json
cost table
Results saved to models/cost_table_1773668448.json

sf
Schema: sf  (10 numeric + 1 categorical features)
Loading data/sf_open_dataset_20260309.geojson ...

Split complete  (seed=42)  (include_conf=false)
  Train: 285,080  |  closed=153,654  open=131,426
  Val:   71,271  |  closed=38,414  open=32,857
  X shape: 12 features  (11 numeric + 1 categorical)
  Category vocab size: 19 (+1 OOV)
Results saved to models/evals/eval_results_1773654090.json
cost table
Results saved to models/cost_table_1773654114.json

sf, spatial

