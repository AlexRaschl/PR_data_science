import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.model.utils import handle_savefig

stats = {'KNNR': [39484792.42411988, 0.0668295243863195],
         'RF': [39109248.927510925, 0.17523268661905567],
         'SVR': [39483066.334116116, 0.1144045808502393],
         'MLP': [46598782.664061956, -0.23486510051492226]}
path_spec = ['fitted_models', 'comparison']

stats_bl = {'KNNR': [39971652.30015526, -0.40254226422102035],
            'RF': [39574046.472325444, 0.027156038905454816],
            'SVR': [39597164.765587814, 0.03418382173672674]
            }
path_spec_bl = ['basic_models', 'error_comparison']


def compare_models(stats, path_spec, ylimr=(-0.3, 0.25), title="Final Model Comparison"):
    df = pd.DataFrame.from_dict(stats, orient='index')
    df.columns = ['mean absolute error', '$R^2$']

    sns.set_style(style='whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'top': 0.92})

    plt.suptitle(title)
    bars = sns.barplot(x=df.index, y=df['mean absolute error'], ax=ax1, alpha=0.8)
    for p in bars.patches:
        bars.annotate(format(int(p.get_height()), ',d'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                      va='center', xytext=(0, 10), textcoords='offset points')
    ax1.set_ylim((3.5e7, 5e7))
    bars = sns.barplot(x=df.index, y=df['$R^2$'], ax=ax2, alpha=0.8)
    for p in bars.patches:
        if p.get_height() > 0:
            bars.annotate(format(p.get_height(), '.4f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                          va='center', xytext=(0, 10), textcoords='offset points')
        else:
            bars.annotate(format(p.get_height(), '.4f'), (p.get_x() + p.get_width() / 2., p.get_height() - 0.04),
                          ha='center',
                          va='center', xytext=(0, 10), textcoords='offset points')
    ax2.set_ylim(ylimr)

    handle_savefig(path_spec)


compare_models(stats, path_spec)
compare_models(stats_bl, path_spec_bl, ylimr=(-0.48, 0.25), title='Basic Model Comparison')
