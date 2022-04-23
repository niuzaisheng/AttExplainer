# %%
import pandas as pd
import wandb
from pandas_profiling import ProfileReport

api = wandb.Api()

# %%
explain_method = "LIME"  # "ShapleyValueSampling" , "KernelShap" or "LIME"
data_set_name = "snli"  # "emotion", "sst2" or "snli"

runs = api.runs("attexplaner", filters={
    # "config.explain_method": explain_method,
    "config.discribe": "LIME",
    "config.data_set_name": data_set_name})

summary_list = []
for run in runs:
    summary = run.summary._json_dict
    run_config = run.config
    dic = {
        # "Average Victim Model Query Times": summary["Average Victim Model Query Times"],
        "Average Victim Model Query Times": run_config["max_sample_num"],
        "Attack Success Rate": summary["Attack Success Rate"],
        "Fidelity": summary["Fidelity+"],
        "Token Modification Rate": summary["Token Modification Rate"],
        "delta_prob": summary["delta_prob"],
    }
    summary_list.append(dic)

df = pd.DataFrame(summary_list)
df.describe()

profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file(f"reports/Report_for_baseline_{data_set_name}_by_{explain_method}.html")

# %%
