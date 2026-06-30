import pandas as pd

LUGARES_PUC = [
    {"nome": "Teatro TUCA", "desc": "Espaço cultural histórico com apresentações teatrais, shows e debates acadêmicos de vanguarda."},
    {"nome": "Tradição Lanches", "desc": "Lanchonete clássica de bairro, famosa pelo custo-benefício e sanduíches rápidos para estudantes."},
    {"nome": "Bar e Lanchonete Luar De Paraty", "desc": "Ponto de encontro tradicional dos alunos da PUC, ideal para cerveja gelada e petiscos pós-aula."},
    {"nome": "Q-Burger", "desc": "Hamburgueria artesanal com ambiente moderno e opções de burgers suculentos em Perdizes."},
    {"nome": "Da Pá Virada Gelateria Perdizes", "desc": "Sorveteria artesanal com sabores naturais e ambiente tranquilo para um café ou sobremesa."},
    {"nome": "The New York Coffee - Perdizes", "desc": "Cafeteria inspirada em NYC com donuts, cafés especiais e espaço para estudo com Wi-Fi."},
    {"nome": "Zé do Hamburger", "desc": "Hamburgueria temática estilo anos 50 com decoração retrô, milkshakes e clima nostálgico."},
    {"nome": "Hobby Hamburger", "desc": "Hamburgueria tradicional da Cardoso de Almeida, focada em lanches clássicos e rápidos."},
    {"nome": "Torino", "desc": "Restaurante italiano aconchegante com massas frescas e pratos executivos de qualidade."},
    {"nome": "Wei Lai Restaurante Ltda", "desc": "Culinária chinesa autêntica com porções generosas e ambiente familiar próximo à universidade."},
    {"nome": "Padaria La Plaza", "desc": "Padaria premium com buffet de café da manhã, pães artesanais e balcão para refeições rápidas."},
    {"nome": "Nova Charmosa Casa de Pães", "desc": "Padaria e confeitaria com ampla variedade de doces, salgados e refeições a quilo."},
    {"nome": "Bacio di Latte", "desc": "Gelateria italiana famosa pela cremosidade e ingredientes selecionados em ambiente sofisticado."},
    {"nome": "Noul Padaria Brunch & Café", "desc": "Espaço moderno especializado em brunch, pães de fermentação natural e cafés filtrados."},
    {"nome": "Flying Sushi Perdizes", "desc": "Restaurante japonês com sistema de rodízio e combinados, popular entre o público jovem."},
    {"nome": "Arabesco", "desc": "Culinária árabe tradicional com esfihas famosas e ambiente climatizado excelente para grupos."},
    {"nome": "Pizza Hut", "desc": "Rede internacional de pizzaria com massas pan e ambiente casual para grupos grandes."},
    {"nome": "Shopping West Plaza", "desc": "Centro comercial completo com cinemas, praça de alimentação variada e opções de lazer."},
    {"nome": "Panegiorno Pães Artesanais", "desc": "Boutique de pães focada em longa fermentação e ambiente acolhedor para café da manhã."},
    {"nome": "Milo Garage", "desc": "Balada underground com estilo rock, indie e eletrônico, atraindo público alternativo e universitário."},
    {"nome": "Sadiek Bar e Lounge", "desc": "Lounge com narguilé, drinks coloridos e ambiente com luz baixa para socialização noturna."},
    {"nome": "O Sobrado", "desc": "Bar com deck ao ar livre, drinks autorais e música ambiente, muito frequentado para happy hour."},
    {"nome": "Bar do Baixo", "desc": "Bar cultural com grafites nas paredes, música ao vivo e clima de boteco raiz da Vila Madalena/Perdizes."},
    {"nome": "Canto da Ema", "desc": "Casa de forró tradicional próxima à região, ideal para dançar e socializar ao som de trios pé de serra."},
    {"nome": "Legítimo Bar", "desc": "Espetinhos variados, chopp gelado e ambiente descontraído com mesas na calçada."},
    {"nome": "Armazém Piola", "desc": "Mistura de bar e pizzaria com decoração rústica e clima vibrante para as noites de fim de semana."},
    {"nome": "Desmanche", "desc": "Clube com decoração industrial e festas de música brasileira, pop e nostalgia anos 2000."},
    {"nome": "Caos Augusta", "desc": "Bar que vira balada com decoração de antiquário e sets de DJ focados em eletrônico e rock."},
    {"nome": "Pratododia", "desc": "Espaço cultural e bar focado em vinis, música brasileira e black music em ambiente intimista."},
    {"nome": "Bebo Sim", "desc": "Bar e balada focado em brasilidades, marchinhas de carnaval e drinks tropicais."},
    {"nome": "Tiki na Calçada", "desc": "Bar com temática tropical/tiki, drinks exóticos e ambiente animado para começar a noite."},
]

DEMO_LUGARES = [
    {"nome": "Café Arte & Som", "desc": "Café acolhedor com Wi-Fi, vibe artística e eventos musicais"},
    {"nome": "Bar do Zé", "desc": "Bar descontraído com cervejas artesanais e música ao vivo"},
    {"nome": "Espaço Cultural Vila", "desc": "Espaço para exposições, workshops e networking"},
]


def load_puc_places() -> pd.DataFrame:
    """Retorna DataFrame com locais reais da região PUC-SP Monte Alegre."""
    df = pd.DataFrame(LUGARES_PUC)
    print(f"Sucesso: {len(df)} locais reais integrados ao ecossistema PUC-SP.")
    return df
