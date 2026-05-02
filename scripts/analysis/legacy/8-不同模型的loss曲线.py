import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss_with_internal_multiline_legend(excel_path: str, output_path: str):
    """
    单图 + 内部多行图例（放在绘图区顶部中央）+ 300dpi 保存
    """
    # 1. 读数据
    df_train = pd.read_excel(excel_path, sheet_name='Train', index_col=0)
    df_test  = pd.read_excel(excel_path, sheet_name='Test',  index_col=0)
    if df_train.shape != df_test.shape:
        raise ValueError("Train/Test sheets 尺寸不一致！")

    models   = df_train.columns.tolist()
    n_epochs = df_train.shape[0]
    x = np.arange(n_epochs)
    epoch_labels = df_train.index.tolist()

    # 2. 调色板
    palette = sns.color_palette("colorblind", n_colors=len(models))

    # 3. 创建画布
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    # 4. 绘制所有曲线并带 label
    for color, model in zip(palette, models):
        ax.plot(x, df_train[model].values,
                color=color, linestyle='-',
                linewidth=1.8, alpha=0.9,
                label=f'{model} (Train)')
        ax.plot(x, df_test[model].values,
                color=color, linestyle='--',
                linewidth=1.2, alpha=0.6,
                label=f'{model} (Test)')

    # 5. 坐标轴 & 网格
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss',  fontsize=12)
    ax.grid(linestyle=':', linewidth=0.5, alpha=0.7)

    # 6. x 轴刻度
    step = max(1, n_epochs // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(epoch_labels[i]) for i in x[::step]],
                       rotation=45, fontsize=8)

    # 7. 图例：内部上方中央，多列分行
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              loc='upper center',
              bbox_to_anchor=(0.5, 0.98),
              ncol=6,                # adjust columns to control number of rows
              frameon=True,
              fontsize=9.5,
              borderaxespad=0.5)

    # 8. 调整留白，让曲线不被图例遮挡
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
    ax.margins(y=0.1)  # 增加上下边距

    # 9. 保存高分辨率图
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    excel_file = r'/Users/fengxiao/Desktop/Work/语义分割/Results/各模型loss2.xlsx'
    out_png    = r'/Users/fengxiao/Desktop/Work/语义分割/Results/loss_internal_legend_300dpi.png'
    plot_loss_with_internal_multiline_legend(excel_file, out_png)
