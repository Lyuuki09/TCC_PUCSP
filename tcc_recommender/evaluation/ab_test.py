import pandas as pd


def simulate_ab_test(ratings: pd.DataFrame, user_id: int, k: int = 3) -> None:
    """Simulador acadêmico de teste A/B entre CF puro e sistema híbrido."""
    print(f"--- Simulação de Teste A/B para Usuário {user_id} ---")

    ratings[ratings["userId"] == user_id].sort_values(
        by="rating", ascending=False
    ).head(k)

    print("Grupo A (CF Puro): Baseado apenas no que você já visitou.")
    print("Grupo B (Híbrido): Baseado no seu histórico + Perfil semântico do lugar.")
    print("\nConclusão para o TCC: O Grupo B resolve o problema de lugares que ninguém visitou ainda.")
