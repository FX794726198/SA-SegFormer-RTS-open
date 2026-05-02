import pandas as pd
import matplotlib.pyplot as plt

# 读取所有 sheet
excel_path = '/Users/fengxiao/Desktop/Work/语义分割/Results/各模型的各项指标2.xlsx'
sheet_dict = pd.read_excel(excel_path, sheet_name=None)

# 1. 单独绘图并保存（300 dpi），添加 top-x 轴和 right-y 轴但不显示刻度
for metric, df in sheet_dict.items():
    epochs = df.iloc[:, 0]
    models = df.columns[1:]

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap('tab10')
    colors = cmap(range(len(models)))

    for model, color in zip(models, colors):
        ax.plot(epochs, df[model], label=model, color=color, linewidth=2)

    # 显示 top 和 right 轴，但不显示其刻度
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(top=False, right=False)

    ax.set_title(f'{metric} per Epoch', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    ax.legend(title='Model', fontsize=10)


    fig.tight_layout()
    fig.savefig(f'{metric}_per_epoch.png', dpi=300)
    plt.close(fig)

# 2. 汇总大图：3行2列排布，最后一个位置留空放图例
metrics = list(sheet_dict.keys())
fig, axes = plt.subplots(3, 2, figsize=(12, 16), sharex=False, sharey=False)

for idx, metric in enumerate(metrics):
    r, c = divmod(idx, 2)
    ax = axes[r, c]
    df = sheet_dict[metric]
    epochs = df.iloc[:, 0]
    models = df.columns[1:]
    cmap = plt.get_cmap('tab10')
    colors = cmap(range(len(models)))

    for model, color in zip(models, colors):
        ax.plot(epochs, df[model], label=model, color=color, linewidth=1.5)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(top=False, right=False)
    ax.set_title(metric, fontsize=12)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)

# 隐藏第3行第2列（索引 [2,1]）的子图
axes[2, 1].axis('off')

# 在空白位置添加统一图例
handles, labels = axes[0, 0].get_legend_handles_labels()
#axes[2, 1].legend(handles, labels, loc='center', ncol=2, fontsize=10, title='Model')
axes[2, 1].legend(
    handles, labels,
    loc='center',
    ncol=2,
    fontsize=18,          # 放大图例文字
    title='Model',
    title_fontsize=20,    # 放大标题
    labelspacing=1,     # 可选：调大行间距
    columnspacing=2     # 可选：调大列间距
)

fig.tight_layout()
fig.savefig('all_metrics_summary.png', dpi=300)
plt.close(fig)

print("已生成并保存各单图及汇总图（300 dpi）")
