import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_per_model_uniform_colors_with_gray_grid(excel_path: str,
                                                output_path: str,
                                                cols: int = 3,
                                                figsize_per_plot=(4, 3),
                                                dpi: int = 300):
    """
    每个模型一个子图，Train: 淡蓝色实线；Test: 淡红色虚线；
    网格为淡灰色虚线；保存 300dpi。
    """
    # 1. 读数据
    df_train = pd.read_excel(excel_path, sheet_name='Train', index_col=0)
    df_test  = pd.read_excel(excel_path, sheet_name='Test',  index_col=0)
    if df_train.shape != df_test.shape:
        raise ValueError("Train/Test sheets 尺寸不一致！")

    epochs   = df_train.index
    models   = df_train.columns.tolist()
    n_models = len(models)
    rows     = math.ceil(n_models / cols)

    # 2. 风格
    sns.set_style("whitegrid")
    train_color = 'lightblue'
    test_color  = 'lightcoral'

    # 3. 创建子图
    fig, axes = plt.subplots(rows, cols,
                             figsize=(figsize_per_plot[0] * cols,
                                      figsize_per_plot[1] * rows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    # 4. 绘制
    for idx, model in enumerate(models):
        ax = axes[idx]
        # 训练：淡蓝色实线
        ax.plot(epochs, df_train[model],
                color=train_color,
                linestyle='-',
                linewidth=1.8,
                label='Train')
        # 测试：淡红色虚线
        ax.plot(epochs, df_test[model],
                color=test_color,
                linestyle='-',
                linewidth=1.8,
                label='Test')

        # 5. 淡灰色虚线网格
        ax.grid(color='lightgray', linestyle='--', linewidth=0.5)

        ax.set_title(model, fontsize=10, pad=6)
        if idx % cols == 0:
            ax.set_ylabel('Loss', fontsize=9)
        if idx // cols == rows - 1:
            ax.set_xlabel('Epoch', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc='upper right', frameon=False)

    # 6. 删除多余子图
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    # 7. 保存高清图
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    excel_file = r'/Users/fengxiao/Desktop/Work/语义分割/Results/各模型loss.xlsx'
    out_file   = r'/Users/fengxiao/Desktop/Work/语义分割/Results/per_model_uniform_graygrid_300dpi.png'
    plot_per_model_uniform_colors_with_gray_grid(excel_file, out_file)
