import matplotlib.pyplot as plt


def plot_metrics_comparison(rmse: float, precision_at_k: float) -> None:
    """Gráfico de barras comparando RMSE (CF) e Precision@K (zero-shot)."""
    metrics = ["RMSE (CF)", "Precision@5 (BERT)"]
    values = [rmse, precision_at_k]

    plt.figure(figsize=(10, 5))
    plt.bar(metrics, values, color=["skyblue", "salmon"])
    plt.title("Comparativo de Métricas do Sistema Híbrido")
    plt.ylabel("Valor da Métrica")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")
    plt.show()
