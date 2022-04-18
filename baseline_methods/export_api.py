# %%
import pandas as pd
import wandb
from pandas_profiling import ProfileReport

api = wandb.Api()

explain_method = "ShapleyValueSampling" # "ShapleyValueSampling" or "KernelShap"
data_set_name = "sst2" # "emotion", "sst2" or "snli"

runs = api.runs("attexplaner", filters={"config.explain_method": explain_method,
                                        "config.data_set_name": data_set_name})

summary_list = []
for run in runs:
    summary = run.summary._json_dict
    dic = {
        "Average Victim Model Query Times": summary["Average Victim Model Query Times"],
        "Attack Success Rate": summary["Attack Success Rate"],
        "Fidelity": summary["Fidelity"],
        "Token Modification Rate": summary["Token Modification Rate"],
        "delta_prob": summary["delta_prob"],
    }
    summary_list.append(dic)

df = pd.DataFrame(summary_list)
df.describe()

profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file(f"Report_for_baseline_{explain_method}_by_{explain_method}.html")

# %%
