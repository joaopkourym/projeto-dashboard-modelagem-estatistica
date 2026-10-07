"""
Dashboard - Desigualdade socioeconômica e acesso ao ensino superior (ENEM 2023)

Como rodar:
    pip install -r requirements.txt
    streamlit run app.py

Estrutura do arquivo (de cima para baixo):
    1. Configurações e constantes (cores, ordem das faixas)
    2. Carregamento dos dados  -> funções com @st.cache_data (lê o disco uma vez só)
    3. Filtros da barra lateral -> devolvem um DataFrame filtrado
    4. Funções de gráfico       -> recebem dados agregados e devolvem uma figura Plotly
    5. Abas                     -> uma por pergunta do trabalho + uma tabela dinâmica livre

Regra do projeto: o painel não traz textos com conclusões. Cada gráfico tem um objetivo (escrito na
legenda abaixo dele) e o usuário tira as conclusões explorando filtros e seletores.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# 1. Configurações e constantes
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ENEM 2023 · Desigualdade e acesso", page_icon="📊", layout="wide")

PASTA = Path(__file__).parent / "data" / "processed"


# Cores do tema escuro (as mesmas do .streamlit/config.toml), usadas nos gráficos e no CSS
FUNDO, FUNDO_CARD, BORDA = "#0f141c", "#1a212c", "#2b3442"
APAGADO = "#3a4352"  # barras fora da seleção

FAIXAS = ["Até 1/4 SM", "1/4 a 1/2 SM", "1/2 a 1 SM", "1 a 2 SM", "Acima de 2 SM"]
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]

# Paleta categórica validada para daltonismo (ordem fixa: azul, laranja, verde-água, amarelo, magenta).
# A cor segue a ENTIDADE: Pública é sempre azul, Privada sempre laranja, em qualquer gráfico.
AZUL, LARANJA, AQUA, AMARELO, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
COR_ESCOLA = {"Pública": AZUL, "Privada": LARANJA}
COR_INTERNET = {"Sem internet": AZUL, "Com internet": LARANJA}
COR_REGIAO = dict(zip(REGIOES, [AZUL, LARANJA, AQUA, AMARELO, MAGENTA]))
CINZA = "#898781"
EXTRAS = ["#8a5cd6", "#4d9b9b", CINZA]  # para dimensões com mais de 5 categorias
# Renda é ordinal: escala sequencial (claro = mais pobre, escuro = mais rico)
COR_FAIXA = dict(zip(FAIXAS, ["#b9d3f3", "#86b2ea", "#4f8fdc", "#2a6cc0", "#1a4a8a"]))
CORES_RACA = ["Branca", "Preta", "Parda", "Amarela", "Indígena", "Não declarado"]

# Notas que o usuário pode escolher analisar (coluna -> rótulo)
NOTAS = {"NU_NOTA_MT": "Matemática", "NU_NOTA_CN": "Ciências da Natureza", "NU_NOTA_CH": "Ciências Humanas",
         "NU_NOTA_LC": "Linguagens", "NU_NOTA_REDACAO": "Redação", "media_geral": "Média das 5 provas"}
MIN_N = 30  # grupos com menos alunos que isso ficam fora dos gráficos de comparação

# Rótulos amigáveis para os coeficientes da regressão
ANOS_ESCOLARIDADE = {
    0: "Nunca estudou", 2: "Fundamental I incompleto", 6: "Fundamental I completo",
    10: "Fundamental II completo", 12: "Médio completo", 16: "Superior completo", 18: "Pós-graduação",
}


# ---------------------------------------------------------------------------
# 2. Carregamento dos dados
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando 1 milhão de inscritos...")
def carregar_base() -> pd.DataFrame:
    """Lê o parquet exportado pelo notebook e reduz a memória (int64 -> int8, texto -> category)."""
    colunas = [*NOTAS, "faltou", "faixa_renda", "escol_pais", "internet",
               "privada", "escola", "regiao", "SG_UF_PROVA", "cor", "feminino", "atraso"]
    df = pd.read_parquet(PASTA / "enem2023_concluintes.parquet", columns=colunas)
    for c in NOTAS:
        df[c] = df[c].astype("float32")
    for c in ["faltou", "internet", "privada", "feminino", "atraso"]:
        df[c] = df[c].astype("int8")
    for c in ["escola", "regiao", "SG_UF_PROVA", "cor"]:
        df[c] = df[c].astype("category")
    df["faixa_renda"] = pd.Categorical(df.faixa_renda, categories=FAIXAS, ordered=True)
    df["sexo"] = np.where(df.feminino == 1, "Feminino", "Masculino")
    df["internet_txt"] = np.where(df.internet == 1, "Com internet", "Sem internet")
    df["escol_txt"] = pd.Categorical(df.escol_pais.round().map(ANOS_ESCOLARIDADE),
                                     categories=list(ANOS_ESCOLARIDADE.values()), ordered=True)
    df["atraso_txt"] = np.where(df.atraso == 1, "Com atraso (19+)", "Sem atraso")
    # quem fez a prova e não zerou MT (mesmo critério do notebook)
    df["tem_nota"] = (df.faltou == 0) & (df.NU_NOTA_MT > 0)
    return df


@st.cache_data
def carregar_resultados() -> dict[str, pd.DataFrame]:
    """CSVs pequenos com os resultados da amostra (ICs, regressão)."""
    return {
        "coef": pd.read_csv(PASTA / "coeficientes_regressao.csv", index_col=0),
        "metricas": pd.read_csv(PASTA / "metricas_regressao.csv", index_col=0),
        "ic_nota": pd.read_csv(PASTA / "ic_nota_por_renda.csv", index_col=0),
        "ic_faltas": pd.read_csv(PASTA / "ic_faltas_por_renda.csv", index_col=0),
    }


base = carregar_base()
res = carregar_resultados()


# ---------------------------------------------------------------------------
# 3. Filtros
# ---------------------------------------------------------------------------
UF_REGIAO = base[["SG_UF_PROVA", "regiao"]].drop_duplicates().astype(str).set_index("SG_UF_PROVA").regiao.to_dict()
UFS_POR_REGIAO = {r: sorted(u for u, rr in UF_REGIAO.items() if rr == r) for r in REGIOES}
TODAS_UFS = [u for r in REGIOES for u in UFS_POR_REGIAO[r]]  # agrupadas por região
ESCOLAS, SEXOS = ["Pública", "Privada"], ["Feminino", "Masculino"]
FILTROS = ["f_reg", "f_uf", "f_esc", "f_cor", "f_sexo"]

# Os filtros vivem no session_state para que região, UF e o clique no gráfico de UFs fiquem sincronizados.
# O painel começa sem filtro: seleção vazia = todos. O usuário afunila escolhendo o que quer ver.
for chave in FILTROS:
    st.session_state.setdefault(chave, [])


def ao_mudar_regiao():
    """Tirar uma região tira as UFs escolhidas dela."""
    regs = st.session_state.f_reg
    st.session_state.f_uf = [u for u in st.session_state.f_uf if not regs or UF_REGIAO[u] in regs]


def ao_mudar_uf():
    """Escolher uma UF de uma região desligada religa a região (se houver regiões escolhidas)."""
    if st.session_state.f_reg:
        regs = {UF_REGIAO[u] for u in st.session_state.f_uf} | set(st.session_state.f_reg)
        st.session_state.f_reg = [r for r in REGIOES if r in regs]


def ao_clicar_uf():
    """Clique/seleção no gráfico de UFs filtra o painel; duplo clique (seleção vazia) volta para todas."""
    escolhidas = {p["x"] for p in st.session_state.graf_uf["selection"]["points"]}
    st.session_state.f_uf = [u for u in TODAS_UFS if u in escolhidas]


def restaurar():
    for chave in FILTROS:
        st.session_state[chave] = []


def ufs_efetivas(regs: list, ufs: list) -> list:
    """Região sem UF escolhida vale inteira; região com UFs escolhidas vale só por elas. Nada escolhido = Brasil."""
    if not regs:
        return [u for u in TODAS_UFS if not ufs or u in ufs]
    return [u for r in REGIOES if r in regs
            for u in ([x for x in UFS_POR_REGIAO[r] if x in ufs] or UFS_POR_REGIAO[r])]


# O cache usa os FILTROS como chave (tuplas pequenas), não o DataFrame inteiro.
# Assim, voltar para um filtro já usado é instantâneo.
@st.cache_data(max_entries=20)
def filtrar(ufs: tuple, escolas: tuple, cores: tuple, sexos: tuple) -> pd.DataFrame:
    m = base.SG_UF_PROVA.isin(ufs) & base.escola.isin(escolas) & base.cor.isin(cores) & base.sexo.isin(sexos)
    return base[m]


with st.sidebar:
    NOTA = st.selectbox("Nota analisada", list(NOTAS), format_func=NOTAS.get, key="f_nota")
    NOTA_NOME = NOTAS[NOTA]
    st.header("Filtros")
    st.caption("Valem para as abas descritivas. O modelo de regressão e os ICs da amostra (Matemática) não mudam. "
               "Nada selecionado = todos.")
    st.pills("Região", REGIOES, selection_mode="multi", key="f_reg", on_change=ao_mudar_regiao)
    st.multiselect("UF", TODAS_UFS, key="f_uf", on_change=ao_mudar_uf, placeholder="Todas as UFs")
    st.caption(f"{len(ufs_efetivas(st.session_state.f_reg, st.session_state.f_uf))} de {len(TODAS_UFS)} UFs"
               " · dica: clique nas barras do gráfico de UFs")
    st.pills("Escola", ESCOLAS, selection_mode="multi", key="f_esc")
    st.pills("Cor/raça", CORES_RACA, selection_mode="multi", key="f_cor")
    st.pills("Sexo", SEXOS, selection_mode="multi", key="f_sexo")
    st.divider()
    st.button("Restaurar filtros", on_click=restaurar, icon=":material/restart_alt:", width="stretch",
              disabled=not any(st.session_state[k] for k in FILTROS))

# Valores efetivos (seleção vazia = todos), usados daqui pra baixo
f_reg = st.session_state.f_reg or list(REGIOES)
f_uf = ufs_efetivas(st.session_state.f_reg, st.session_state.f_uf)
f_esc = st.session_state.f_esc or ESCOLAS
f_cor = st.session_state.f_cor or list(CORES_RACA)
f_sexo = st.session_state.f_sexo or SEXOS
df = filtrar(tuple(f_uf), tuple(sorted(f_esc)), tuple(sorted(f_cor)), tuple(sorted(f_sexo)))
notas = df[df.tem_nota & (df[NOTA] > 0)]  # nota zero = prova anulada/em branco, fica fora das médias

if len(notas) < 30:
    st.warning("Poucos alunos com esses filtros. Amplie a seleção na barra lateral.")
    st.button("Restaurar filtros", on_click=restaurar, icon=":material/restart_alt:", key="restaurar_aviso")
    st.stop()


# Agregados por UF ignorando o filtro de UF (só os outros filtros), para o gráfico clicável
# mostrar todas as UFs das regiões escolhidas e destacar as selecionadas.
@st.cache_data(max_entries=20)
def agregar_uf(nota: str, escolas: tuple, cores: tuple, sexos: tuple) -> pd.DataFrame:
    d = base[base.escola.isin(escolas) & base.cor.isin(cores) & base.sexo.isin(sexos)]
    g = d.groupby("SG_UF_PROVA", observed=True)
    t = d[d.tem_nota & (d[nota] > 0)].groupby("SG_UF_PROVA", observed=True)
    out = pd.DataFrame({
        "inscritos": g.size(), "faltou": g.faltou.mean(), "privada": g.privada.mean(),
        "sem_net": 1 - g.internet.mean(), "nota": t[nota].mean(),
    })
    out.index = out.index.astype(str)
    out["regiao"] = out.index.map(UF_REGIAO)
    return out


# ---------------------------------------------------------------------------
# 4. Funções auxiliares e de gráfico
# ---------------------------------------------------------------------------
def fmt(x: float, casas: int = 0) -> str:
    """Formata número no padrão brasileiro: 1.234,5"""
    if pd.isna(x):
        return "–"
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, casas: int = 1) -> str:
    return "–" if pd.isna(x) else fmt(100 * x, casas) + "%"


def rgba(cor: str, alfa: float) -> str:
    return f"rgba({int(cor[1:3], 16)},{int(cor[3:5], 16)},{int(cor[5:7], 16)},{alfa})"


# Dimensões que o usuário pode escolher para comparar grupos: rótulo -> (coluna, ordem das categorias)
DIMENSOES = {
    "Renda per capita": ("faixa_renda", FAIXAS),
    "Região": ("regiao", REGIOES),
    "Escola": ("escola", ["Pública", "Privada"]),
    "Internet": ("internet_txt", ["Sem internet", "Com internet"]),
    "Cor/raça": ("cor", CORES_RACA),
    "Sexo": ("sexo", ["Feminino", "Masculino"]),
    "Escolaridade dos pais": ("escol_txt", list(ANOS_ESCOLARIDADE.values())),
    "Atraso escolar": ("atraso_txt", ["Sem atraso", "Com atraso (19+)"]),
}
COR_FIXA = {"faixa_renda": COR_FAIXA, "regiao": COR_REGIAO, "escola": COR_ESCOLA, "internet_txt": COR_INTERNET}


def cores_de(dim: str) -> dict:
    """Cor por categoria: entidades com cor fixa mantêm a mesma cor em todo o painel."""
    col, ordem = DIMENSOES[dim]
    return COR_FIXA.get(col, dict(zip(ordem, [AZUL, LARANJA, AQUA, AMARELO, MAGENTA, *EXTRAS])))


def escolher_dim(rotulo: str, chave: str, padrao: str, sem: tuple = (), nenhum: bool = False) -> str | None:
    opcoes = (["Nenhum"] if nenhum else []) + [d for d in DIMENSOES if d not in sem]
    escolha = st.selectbox(rotulo, opcoes, index=opcoes.index(padrao), key=chave)
    return None if escolha == "Nenhum" else escolha


def layout(fig: go.Figure, titulo: str = "", altura: int = 380, **kw) -> go.Figure:
    """Estilo comum: grade discreta, legenda em cima, tooltip unificado."""
    fig.update_layout(
        # título fica no topo do "container" e a legenda logo abaixo dele, sem se sobrepor
        title=dict(text=titulo, font=dict(size=15), y=0.98, yref="container", yanchor="top"),
        height=altura, margin=dict(l=10, r=10, t=85 if titulo else 10, b=10),
        legend=dict(orientation="h", y=1.01, yanchor="bottom", x=0, title=None),
        hoverlabel=dict(font_size=13, bgcolor=FUNDO_CARD, bordercolor=BORDA, font_family="Inter, sans-serif"),
        font=dict(family="Inter, sans-serif"), separators=",.", **kw,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1, zeroline=False)
    return fig


CONFIG_PLOTLY = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "autoScale2d"],
                 "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "enem2023"}}


def grafico(fig: go.Figure, **kw):
    """st.plotly_chart com barra de ferramentas enxuta (zoom, pan, seleção, baixar PNG)."""
    return st.plotly_chart(fig, width="stretch", config=CONFIG_PLOTLY, **kw)


def halteres(dados: pd.DataFrame, dim: str, grupo: str, a: str, b: str, cores: dict, titulo: str) -> go.Figure:
    """Gráfico de halteres: média de dois grupos (a, b) em cada categoria de `dim`, ligadas por uma linha."""
    col, ordem = DIMENSOES[dim]
    agg = dados.groupby([col, grupo], observed=True)[NOTA].agg(["mean", "size"])
    media = agg["mean"].unstack().where(agg["size"].unstack() >= MIN_N).reindex(ordem).reindex(columns=[a, b])
    n = agg["size"].unstack().reindex(ordem).reindex(columns=[a, b]).fillna(0)
    media = media.dropna(how="all")
    fig = go.Figure()
    for cat, linha in media.iterrows():
        fig.add_trace(go.Scatter(x=[linha[a], linha[b]], y=[cat, cat], mode="lines",
                                 line=dict(color=CINZA, width=2), showlegend=False, hoverinfo="skip"))
    for g in (a, b):
        fig.add_trace(go.Scatter(
            x=media[g], y=media.index, mode="markers", name=g, customdata=n.loc[media.index, g],
            marker=dict(size=13, color=cores[g], line=dict(width=2, color=FUNDO)),
            hovertemplate=f"<b>%{{y}}</b><br>{g}: %{{x:.1f}}<br>Alunos: %{{customdata:,}}<extra></extra>",
        ))
    # diferença (b - a) à direita de cada haltere
    for cat, linha in media.iterrows():
        if linha.notna().all():
            fig.add_annotation(x=linha.max(), y=cat, text=f"  {linha[b] - linha[a]:+.0f}",
                               showarrow=False, xanchor="left", font=dict(color=CINZA, size=12))
    fig.update_yaxes(categoryorder="array", categoryarray=ordem[::-1], title=dim)
    fig.update_xaxes(title=f"Média de {NOTA_NOME}")
    return layout(fig, titulo, altura=140 + 48 * len(media))


def barras_prop(serie: pd.Series, n: pd.Series, cores: dict, titulo: str, rotulo: str, ref: float | None = None):
    """Barras horizontais de uma proporção por categoria, com linha de referência opcional."""
    fig = go.Figure(go.Bar(
        x=serie.values, y=serie.index, orientation="h", marker_color=[cores.get(c, AZUL) for c in serie.index],
        text=[pct(v) for v in serie.values], textposition="outside", customdata=n.values,
        hovertemplate="<b>%{y}</b><br>" + rotulo + ": %{x:.1%}<br>Inscritos: %{customdata:,}<extra></extra>",
    ))
    if ref is not None:
        fig.add_vline(x=ref, line_dash="dot", line_color=CINZA,
                      annotation_text=f"seleção {pct(ref)}", annotation_position="top")
    fig.update_yaxes(categoryorder="array", categoryarray=list(serie.index)[::-1])
    fig.update_xaxes(tickformat=".0%", range=[0, max(serie.max() * 1.3, 0.05)])
    return layout(fig, titulo, altura=140 + 42 * len(serie), showlegend=False)


def d_cohen(a: pd.Series, b: pd.Series) -> float:
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)


# ---------------------------------------------------------------------------
# 5. Página
# ---------------------------------------------------------------------------
# Ajustes finos de estilo que o tema (.streamlit/config.toml) não cobre
st.markdown(f"""<style>
.block-container {{padding-top: 2.2rem;}}
[data-testid="stMetric"] {{background: {FUNDO_CARD}; border: none !important;}}
[data-testid="stMetricLabel"] p {{font-size: 0.82rem; font-weight: 500; opacity: .8;}}
[data-testid="stMetricValue"] {{font-weight: 650; letter-spacing: -0.01em; font-size: 1.7rem;}}
[data-testid="stVerticalBlockBorderWrapper"] {{box-shadow: 0 1px 3px rgba(16, 24, 40, .05);}}
h1 {{letter-spacing: -0.02em;}}
</style>""", unsafe_allow_html=True)

st.title("Desigualdade socioeconômica e acesso ao ensino superior")
st.caption(f"ENEM 2023 · concluintes do ensino médio · nota analisada: **{NOTA_NOME}** (troque na barra lateral)")

# Resumo dos filtros ativos (só o que o usuário escolheu)
ativos = [f":blue-badge[:material/map: {r}]" for r in st.session_state.f_reg]
ativos += [f":blue-badge[:material/location_on: {u}]" for u in st.session_state.f_uf]
for chave, icone in [("f_esc", "school"), ("f_cor", "group"), ("f_sexo", "wc")]:
    ativos += [f":orange-badge[:material/{icone}: {v}]" for v in st.session_state[chave]]
topo1, topo2 = st.columns([5, 1], vertical_alignment="center")
topo1.markdown("**Filtrando:** " + " ".join(ativos) if ativos else ":gray-badge[Brasil inteiro, sem filtros]")
if ativos:
    topo2.button("Limpar", on_click=restaurar, icon=":material/filter_alt_off:", type="tertiary", key="limpar_topo")

# ---------------- KPIs (topo, valem para todas as abas) ----------------
# Só números, sem texto interpretativo. A comparação com o Brasil aparece quando há filtro.
filtrado = len(df) != len(base)
nota_br = base.loc[base.tem_nota & (base[NOTA] > 0), NOTA].mean()


def vs_br(valor: float, ref: float, em_pct: bool) -> str | None:
    if not filtrado:
        return None
    dif = (valor - ref) * (100 if em_pct else 1)
    return f"{fmt(dif, 1) if dif < 0 else '+' + fmt(dif, 1)}{' p.p.' if em_pct else ' pts'} vs Brasil"


publ_k, priv_k = notas.loc[notas.privada == 0, NOTA], notas.loc[notas.privada == 1, NOTA]
c = st.columns(6)
c[0].metric("Inscritos", fmt(len(df)), border=True, help="Concluintes do ensino médio inscritos no ENEM 2023.",
            delta=f"{pct(len(df) / len(base))} do Brasil" if filtrado else None, delta_color="off")
c[1].metric("Média da nota", fmt(notas[NOTA].mean(), 1), vs_br(notas[NOTA].mean(), nota_br, False),
            delta_color="off", border=True, help=f"Média de {NOTA_NOME} de quem fez a prova.")
c[2].metric("Faltaram", pct(df.faltou.mean()), vs_br(df.faltou.mean(), base.faltou.mean(), True),
            delta_color="off", border=True)
c[3].metric("Escola privada", pct(df.privada.mean()), vs_br(df.privada.mean(), base.privada.mean(), True),
            delta_color="off", border=True)
c[4].metric("Sem internet", pct(1 - df.internet.mean()),
            vs_br(1 - df.internet.mean(), 1 - base.internet.mean(), True), delta_color="off", border=True,
            help="Sem acesso à internet em casa.")
c[5].metric("Privada − pública", f"{priv_k.mean() - publ_k.mean():+.0f} pts" if len(priv_k) and len(publ_k) else "–",
            border=True, help=f"Diferença entre as médias de {NOTA_NOME} da rede privada e da pública na seleção.")

abas = st.tabs([":material/dashboard: Visão geral", ":material/payments: Renda", ":material/school: Escola",
                ":material/wifi: Internet", ":material/event_busy: Faltas", ":material/function: Modelo",
                ":material/table_chart: Explorar"])

# ---------------- Visão geral ----------------
with abas[0]:
    col1, col2 = st.columns(2)
    with col1, st.container(border=True):
        separar = escolher_dim("Comparar a distribuição por", "hist_grupo", "Nenhum", nenhum=True)
        # Histograma pré-calculado com numpy: mandar 900 mil pontos para o navegador travaria a página.
        passo = 20 if NOTA == "NU_NOTA_REDACAO" else 10  # redação só tem múltiplos de 20
        bins = np.arange(0, 1000 + passo, passo)
        fig = go.Figure()
        if separar:
            col, _ = DIMENSOES[separar]
            for nome, cor in cores_de(separar).items():
                x = notas.loc[notas[col] == nome, NOTA]
                if len(x) < MIN_N:
                    continue
                cont, _ = np.histogram(x, bins=bins)
                fig.add_trace(go.Scatter(
                    x=bins[:-1] + passo / 2, y=cont / len(x), mode="lines",
                    line=dict(color=cor, width=2.5, shape="spline"), fill="tozeroy", fillcolor=rgba(cor, 0.12),
                    name=f"{nome} · média {x.mean():.0f}",
                    hovertemplate=f"<b>{nome}</b><br>%{{x:.0f}} pts: %{{y:.1%}} do grupo<extra></extra>",
                ))
                fig.add_vline(x=x.mean(), line_dash="dot", line_color=cor, line_width=1.5)
            fig.update_yaxes(title="% do grupo", tickformat=".0%")
            fig.update_layout(hovermode="x unified")
        else:
            cont, _ = np.histogram(notas[NOTA], bins=bins)
            fig.add_trace(go.Bar(x=bins[:-1] + passo / 2, y=cont, marker_color=AZUL, marker_line_width=0,
                                 hovertemplate="%{x:.0f} pts: %{y:,} alunos<extra></extra>"))
            fig.add_vline(x=notas[NOTA].mean(), line_dash="dot", line_color=CINZA,
                          annotation_text=f"média {notas[NOTA].mean():.0f}", annotation_position="top right")
            fig.update_layout(bargap=0.08)
            fig.update_yaxes(title="Alunos")
        fig.update_xaxes(title=NOTA_NOME, range=[notas[NOTA].quantile(0.001) - 20, notas[NOTA].quantile(0.999) + 20])
        grafico(layout(fig, f"Distribuição da nota de {NOTA_NOME}"))
        st.caption("**Objetivo:** ver como as notas se espalham e comparar o formato da distribuição entre grupos. "
                   "Ao separar por grupo, cada curva soma 100%.")

    with col2, st.container(border=True):
        INDICADORES = {  # nome: (coluna, é proporção?, valor de referência dos filtros atuais)
            "Nota": ("nota", False, notas[NOTA].mean()),
            "Faltaram": ("faltou", True, df.faltou.mean()),
            "Escola privada": ("privada", True, df.privada.mean()),
            "Sem internet": ("sem_net", True, 1 - df.internet.mean()),
            "Inscritos": ("inscritos", False, None),
        }
        ind = st.segmented_control("Indicador", list(INDICADORES), default="Nota", key="uf_ind") or "Nota"
        col, eh_pct, ref = INDICADORES[ind]
        rotulo_ind = f"Média de {NOTA_NOME}" if ind == "Nota" else ind

        uf = agregar_uf(NOTA, tuple(sorted(f_esc)), tuple(sorted(f_cor)), tuple(sorted(f_sexo)))
        uf = uf[uf.regiao.isin(f_reg)].dropna(subset=[col]).sort_values(col)
        sel = set(f_uf)
        fig = go.Figure()
        for reg in f_reg:
            d = uf[uf.regiao == reg]
            if len(d):
                fig.add_trace(go.Bar(
                    x=d.index, y=d[col], name=reg,
                    marker=dict(color=[COR_REGIAO[reg] if u in sel else APAGADO for u in d.index], line_width=0),
                    text=[pct(v, 0) if eh_pct else fmt(v) for v in d[col]] if len(uf) <= 12 else None,
                    textposition="outside", customdata=d.inscritos,
                    hovertemplate="<b>%{x}</b> · " + reg + "<br>" + rotulo_ind + ": "
                                  + ("%{y:.1%}" if eh_pct else "%{y:,.1f}")
                                  + "<br>Inscritos: %{customdata:,}<extra></extra>",
                ))
        if ref is not None:
            fig.add_hline(y=ref, line_dash="dot", line_color=CINZA,
                          annotation_text=f"seleção {pct(ref) if eh_pct else fmt(ref, 0)}",
                          annotation_position="top left")
        fig.update_xaxes(categoryorder="array", categoryarray=uf.index.tolist())
        if eh_pct:
            fig.update_yaxes(title=rotulo_ind, tickformat=".0%", range=[0, uf[col].max() * 1.18])
        elif ind == "Nota":
            fig.update_yaxes(title=rotulo_ind, range=[uf[col].min() - 30, uf[col].max() + 20])
        else:
            fig.update_yaxes(title=rotulo_ind)
        grafico(layout(fig, f"{rotulo_ind} por UF", bargap=0.15, clickmode="event+select"),
                key="graf_uf", on_select=ao_clicar_uf, selection_mode=("points", "box"))
        st.caption("**Objetivo:** comparar as UFs no indicador escolhido. Clique numa barra (ou arraste uma caixa) "
                   "para filtrar o painel por essas UFs; duplo clique volta para todas. Cinza = fora da seleção.")

    with st.container(border=True):
        st.markdown("**Tabela por região** · indicadores das UFs selecionadas")
        por_reg = df.groupby("regiao", observed=True).agg(
            inscritos=("faltou", "size"), faltou=("faltou", "mean"), privada=("privada", "mean"),
            internet=("internet", "mean"),
        ).reindex(REGIOES).dropna()
        por_reg["nota"] = notas.groupby("regiao", observed=True)[NOTA].mean()
        por_reg["sem_net"] = 1 - por_reg.internet
        por_reg["ufs"] = [", ".join(u for u in f_uf if UF_REGIAO[u] == r) for r in por_reg.index]
        st.dataframe(
            por_reg[["ufs", "inscritos", "nota", "faltou", "privada", "sem_net"]],
            width="stretch",
            column_config={
                "_index": st.column_config.TextColumn("Região"),
                "ufs": st.column_config.TextColumn("UFs"),
                "inscritos": st.column_config.NumberColumn("Inscritos", format="localized"),
                "nota": st.column_config.ProgressColumn(
                    f"Média {NOTA_NOME}", format="%.1f",
                    min_value=float(np.floor(por_reg.nota.min() / 50) * 50 - 50),
                    max_value=float(np.ceil(por_reg.nota.max() / 50) * 50)),
                "faltou": st.column_config.ProgressColumn("Faltaram", format="percent", min_value=0, max_value=1),
                "privada": st.column_config.ProgressColumn("Privada", format="percent", min_value=0, max_value=1),
                "sem_net": st.column_config.ProgressColumn("Sem internet", format="percent", min_value=0, max_value=1),
            },
        )

    with st.expander("Sobre os dados"):
        st.markdown(
            "- **População:** concluintes do ensino médio em 2023 (só para eles o INEP informa o tipo de escola). "
            "Treineiros e eliminados ficam de fora.\n"
            "- **Notas:** só de quem fez a prova; notas zero (prova anulada ou em branco) ficam fora das médias. "
            "Em Matemática, o menor valor possível pela TRI fica perto de 320.\n"
            "- **Renda per capita:** ponto médio da faixa de renda familiar ÷ moradores, em salários mínimos (R$ 1.320).\n"
            f"- **Grupos pequenos:** grupos com menos de {MIN_N} alunos ficam fora dos gráficos de comparação.\n"
            "- **Amostra:** testes de hipótese, ICs e regressão foram feitos numa amostra aleatória de 10 mil inscritos "
            "(nota de Matemática) e não mudam com os filtros."
        )

# ---------------- Renda (P2) ----------------
with abas[1]:
    st.subheader("A nota muda entre faixas de renda?")
    g = notas.groupby("faixa_renda", observed=True)[NOTA]
    media_faixa = g.mean().reindex(FAIXAS)
    # eta² = variância entre grupos / variância total (quanto da nota a faixa de renda "explica")
    m = notas[NOTA].mean()
    eta2 = (g.size() * (g.mean() - m) ** 2).sum() / ((notas[NOTA] - m) ** 2).sum()

    c = st.columns(4)
    c[0].metric(f"Média · {FAIXAS[0]}", fmt(media_faixa.iloc[0], 1), border=True)
    c[1].metric(f"Média · {FAIXAS[-1]}", fmt(media_faixa.iloc[-1], 1), border=True,
                delta=f"{media_faixa.iloc[-1] - media_faixa.iloc[0]:+.0f} pts" if media_faixa.notna().all() else None,
                delta_color="off")
    c[2].metric("Variância explicada pela renda (η²)", pct(eta2), border=True,
                help="Parcela da variação da nota que fica entre as faixas de renda (0% a 100%).")
    c[3].metric(f"Concluintes · {FAIXAS[0]}", pct((df.faixa_renda == FAIXAS[0]).mean()), border=True)

    col1, col2 = st.columns(2)
    with col1, st.container(border=True):
        sep = escolher_dim("Separar cada faixa por", "renda_box", "Nenhum", sem=("Renda per capita",), nenhum=True)
        # Boxplot a partir de quartis já calculados (leve para o navegador)
        quantis = [0.05, 0.25, 0.5, 0.75, 0.95]
        fig = go.Figure()
        grupos = [(None, AZUL)] if sep is None else list(cores_de(sep).items())
        for nome, cor in grupos:
            d = notas if nome is None else notas[notas[DIMENSOES[sep][0]] == nome]
            gg = d.groupby("faixa_renda", observed=True)[NOTA]
            q = gg.quantile(quantis).unstack()[gg.size() >= MIN_N].reindex(FAIXAS).dropna()
            if len(q):
                fig.add_trace(go.Box(
                    x=q.index.astype(str), q1=q[0.25], median=q[0.5], q3=q[0.75], lowerfence=q[0.05],
                    upperfence=q[0.95], marker_color=cor, name=nome or "", showlegend=nome is not None,
                ))
        fig.update_layout(boxmode="group")
        fig.update_xaxes(categoryorder="array", categoryarray=FAIXAS, title="Renda per capita")
        fig.update_yaxes(title=NOTA_NOME)
        grafico(layout(fig, f"{NOTA_NOME} por renda per capita", altura=420))
        st.caption("**Objetivo:** comparar a distribuição da nota entre faixas de renda, e dentro delas por outro "
                   "fator. Caixa = 50% do meio dos alunos; linhas vão de 5% a 95%.")
    with col2, st.container(border=True):
        ic = res["ic_nota"].reindex(FAIXAS)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ic.index, y=ic.media, mode="markers", name="Amostra (IC 95%)",
            marker=dict(size=11, color=AZUL),
            error_y=dict(type="data", array=ic.ic_sup - ic.media, arrayminus=ic.media - ic.ic_inf, thickness=2, width=8),
            customdata=np.c_[ic.ic_inf, ic.ic_sup, ic.n],
            hovertemplate="<b>%{x}</b><br>Média: %{y:.1f}<br>IC: %{customdata[0]:.1f} a %{customdata[1]:.1f}"
                          "<br>n = %{customdata[2]:.0f}<extra></extra>",
        ))
        pop = base[base.tem_nota].groupby("faixa_renda", observed=True).NU_NOTA_MT.mean()
        fig.add_trace(go.Scatter(x=pop.index.astype(str), y=pop.values, mode="markers", name="População",
                                 marker=dict(size=10, symbol="diamond-open", color=LARANJA, line=dict(width=2)),
                                 hovertemplate="<b>%{x}</b><br>População: %{y:.1f}<extra></extra>"))
        fig.update_xaxes(title="Renda per capita")
        fig.update_yaxes(title="Média de Matemática")
        grafico(layout(fig, "Matemática: IC 95% vs população", altura=420 + 60))
        st.caption("**Objetivo:** conferir se os intervalos estimados na amostra contêm a média real da população. "
                   "Fixo (não muda com filtros nem com a nota escolhida).")

    with st.container(border=True):
        comp = escolher_dim("Composição de renda por", "renda_comp", "Região", sem=("Renda per capita",))
        col, ordem = DIMENSOES[comp]
        tab = pd.crosstab(df[col], df.faixa_renda, normalize="index").reindex(index=ordem, columns=FAIXAS).dropna(how="all")
        n_cat = df[col].value_counts().reindex(tab.index)
        fig = go.Figure()
        for faixa in FAIXAS:
            fig.add_trace(go.Bar(
                y=tab.index.astype(str), x=tab[faixa], name=faixa, orientation="h", marker_color=COR_FAIXA[faixa],
                text=[pct(v, 0) if v >= 0.06 else "" for v in tab[faixa]], textposition="inside",
                customdata=n_cat.values,
                hovertemplate="<b>%{y}</b><br>" + faixa + ": %{x:.1%}<br>Inscritos no grupo: %{customdata:,}<extra></extra>",
            ))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        fig.update_yaxes(categoryorder="array", categoryarray=[str(o) for o in ordem][::-1])
        grafico(layout(fig, "Faixas de renda em cada grupo",
                               altura=160 + 40 * len(tab), legend_traceorder="normal"))
        st.caption("**Objetivo:** ver quantos alunos de cada faixa de renda existem em cada grupo, para entender "
                   "o perfil por trás das comparações das outras abas.")

# ---------------- Escola (P1 e P5) ----------------
with abas[2]:
    st.subheader("Escola privada tira nota maior? E entre alunos de mesmo perfil?")
    priv, publ = notas.loc[notas.privada == 1, NOTA], notas.loc[notas.privada == 0, NOTA]
    c = st.columns(3)
    c[0].metric("Média pública", fmt(publ.mean(), 1), border=True)
    c[1].metric("Média privada", fmt(priv.mean(), 1), border=True, delta_color="off",
                delta=f"{priv.mean() - publ.mean():+.0f} pts" if len(priv) and len(publ) else None)
    c[2].metric("Tamanho do efeito (d de Cohen)", fmt(d_cohen(priv, publ), 2) if len(priv) > 1 and len(publ) > 1 else "–",
                border=True, help="Diferença das médias em desvios-padrão. Referência: 0,2 pequeno · 0,5 médio · 0,8 grande.")

    if {"Pública", "Privada"} <= set(f_esc):
        col1, col2 = st.columns([3, 2])
        with col1, st.container(border=True):
            dim = escolher_dim("Comparar pública e privada dentro de", "esc_dim", "Renda per capita", sem=("Escola",))
            grafico(halteres(notas, dim, "escola", "Pública", "Privada", COR_ESCOLA,
                                     f"{NOTA_NOME}: pública vs privada"))
            st.caption("**Objetivo:** comparar as redes entre alunos do mesmo grupo. O número à direita é a "
                       "diferença (privada − pública).")
        with col2, st.container(border=True):
            col, ordem = DIMENSOES[dim]
            agg = df.groupby(col, observed=True).privada.agg(["mean", "size"]).reindex(ordem).dropna()
            grafico(barras_prop(agg["mean"], agg["size"], cores_de(dim),
                                        "% em escola privada", "Em escola privada",
                                        ref=df.privada.mean()))
            st.caption("**Objetivo:** ver quantos alunos de cada grupo estão na rede privada (tamanho dos grupos "
                       "comparados ao lado).")
    else:
        st.info("Selecione as duas redes de ensino no filtro para comparar.")

# ---------------- Internet (P4) ----------------
with abas[3]:
    st.subheader("Internet em casa faz diferença entre alunos de mesmo perfil?")
    sem_n, com_n = notas.loc[notas.internet == 0, NOTA], notas.loc[notas.internet == 1, NOTA]
    c = st.columns(3)
    c[0].metric("Média sem internet", fmt(sem_n.mean(), 1), border=True)
    c[1].metric("Média com internet", fmt(com_n.mean(), 1), border=True, delta_color="off",
                delta=f"{com_n.mean() - sem_n.mean():+.0f} pts" if len(sem_n) and len(com_n) else None)
    c[2].metric("Sem internet em casa", pct(1 - df.internet.mean()), border=True)

    col1, col2 = st.columns([3, 2])
    with col1, st.container(border=True):
        dim = escolher_dim("Comparar sem e com internet dentro de", "net_dim", "Renda per capita", sem=("Internet",))
        grafico(halteres(notas, dim, "internet_txt", "Sem internet", "Com internet", COR_INTERNET,
                                 f"{NOTA_NOME}: sem vs com internet"))
        st.caption(f"**Objetivo:** comparar quem tem e quem não tem internet dentro do mesmo grupo. "
                   f"Grupos com menos de {MIN_N} alunos não aparecem (passe o mouse para ver o n).")
    with col2, st.container(border=True):
        col, ordem = DIMENSOES[dim]
        agg = df.groupby(col, observed=True).internet.agg(["mean", "size"]).reindex(ordem).dropna()
        grafico(barras_prop(1 - agg["mean"], agg["size"], cores_de(dim),
                                    "% sem internet em casa", "Sem internet",
                                    ref=1 - df.internet.mean()))
        st.caption("**Objetivo:** ver em quais grupos falta internet em casa e quantos alunos sustentam a "
                   "comparação ao lado.")

# ---------------- Faltas (P6) ----------------
with abas[4]:
    st.subheader("A falta na prova muda com a renda e o perfil?")
    falta = df.groupby("faixa_renda", observed=True).faltou.agg(["mean", "size"]).reindex(FAIXAS)
    taxa_geral = df.faltou.mean()

    c = st.columns(3)
    c[0].metric(f"Faltas · {FAIXAS[0]}", pct(falta["mean"].iloc[0]), border=True)
    c[1].metric(f"Faltas · {FAIXAS[-1]}", pct(falta["mean"].iloc[-1]), border=True)
    c[2].metric("Faltas na seleção", pct(taxa_geral), border=True,
                delta=f"{fmt(df.faltou.sum())} alunos", delta_color="off")

    col1, col2 = st.columns(2)
    with col1, st.container(border=True):
        dim_a = escolher_dim("Taxa de falta por", "falta_a", "Renda per capita")
        col, ordem = DIMENSOES[dim_a]
        agg = df.groupby(col, observed=True).faltou.agg(["mean", "size"]).reindex(ordem).dropna()
        agg = agg[agg["size"] >= MIN_N]
        grafico(barras_prop(agg["mean"], agg["size"], cores_de(dim_a),
                                    "Proporção que faltou", "Faltaram", ref=taxa_geral))
        st.caption("**Objetivo:** comparar a taxa de falta entre os grupos do fator escolhido. "
                   "Linha pontilhada = taxa da seleção atual.")
    with col2, st.container(border=True):
        dim_b = escolher_dim("Cruzar com", "falta_b", "Região", sem=(dim_a,))
        col_b, ordem_b = DIMENSOES[dim_b]
        agg = df.groupby([col, col_b], observed=True).faltou.agg(["mean", "size"])
        mapa = agg["mean"].where(agg["size"] >= MIN_N).unstack().reindex(index=ordem, columns=ordem_b)
        mapa = mapa.dropna(how="all").dropna(axis=1, how="all")
        n_mapa = agg["size"].unstack().reindex(index=mapa.index, columns=mapa.columns).fillna(0)
        fig = go.Figure(go.Heatmap(
            z=mapa.values, x=mapa.columns.astype(str), y=mapa.index.astype(str), customdata=n_mapa.values,
            colorscale=[[0, "#262c36"], [0.5, "#8f4426"], [1, "#e0662f"]], textfont=dict(color="#f1f3f6"), colorbar=dict(tickformat=".0%", title=""),
            text=[[pct(v, 0) for v in linha] for linha in mapa.values], texttemplate="%{text}",
            hovertemplate="<b>%{y} · %{x}</b><br>Faltaram: %{z:.1%}<br>Inscritos: %{customdata:,}<extra></extra>",
            xgap=2, ygap=2,
        ))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        grafico(layout(fig, "Taxa de falta cruzada",
                               altura=max(340, 140 + 48 * len(mapa))))
        st.caption("**Objetivo:** ver se a diferença de faltas entre os grupos se mantém quando se cruza com um "
                   "segundo fator. Células com menos de 30 alunos ficam vazias.")

    with st.expander("IC 95% das proporções de falta na amostra (Wilson)"):
        icf = res["ic_faltas"].reindex(FAIXAS)
        st.dataframe(pd.DataFrame({
            "Amostra": icf.prop.map(pct), "IC inferior": icf.ic_inf.map(pct), "IC superior": icf.ic_sup.map(pct),
            "População": icf.prop_pop.map(pct), "População dentro do IC?": icf.pop_no_ic.map({True: "Sim", False: "Não"}),
        }))
        st.caption("Fixo: calculado na amostra de 10 mil, não muda com os filtros.")

# ---------------- Modelo (P3) ----------------
with abas[5]:
    st.subheader("Quanto da nota é explicado pelo perfil socioeconômico?")
    st.caption("Regressão linear múltipla da nota de Matemática na amostra de 10 mil (treino 80% / teste 20%, "
               "erros-padrão HC3). Não muda com os filtros.")
    met = res["metricas"]
    c = st.columns(4)
    c[0].metric("R² no teste", fmt(met.loc["teste", "R2"], 2), border=True)
    c[1].metric("R² no resto da população", fmt(met.loc["pop_fora_amostra", "R2"], 2), border=True)
    c[2].metric("Erro típico do modelo (RMSE)", fmt(met.loc["teste", "RMSE"], 0) + " pts", border=True)
    c[3].metric("Erro chutando a média", fmt(met.loc["so_media", "RMSE"], 0) + " pts", border=True)

    coef = res["coef"].drop(index="Intercept").copy()

    def rotulo(nome: str) -> str:
        if "[T." in nome:
            nivel = nome.split("[T.")[1].rstrip("]")
            if "faixa_renda" in nome:
                return f"Renda: {nivel}"
            if "regiao" in nome:
                return f"Região: {nivel}"
            if "cor" in nome:
                return f"Cor: {nivel}"
        return {"escol_pais": "Escolaridade dos pais (+1 ano)", "internet": "Internet em casa",
                "privada": "Escola privada", "feminino": "Sexo feminino", "atraso": "Atraso escolar (19+ anos)"}.get(nome, nome)

    coef["rotulo"] = [rotulo(n) for n in coef.index]
    coef["grupo"] = [r.split(":")[0] if ":" in r else "Outros" for r in coef.rotulo]
    with st.container(border=True):
        grupos_coef = st.pills("Mostrar variáveis", ["Renda", "Região", "Cor", "Outros"], selection_mode="multi",
                               default=["Renda", "Região", "Cor", "Outros"], key="coef_grupos")
        so_sig = st.toggle("Só efeitos significativos (p < 0,05)", key="coef_sig")
        ver = coef[coef.grupo.isin(grupos_coef or [])]
        if so_sig:
            ver = ver[ver.p < 0.05]
        ver = ver.sort_values("coef")
        fig = go.Figure()
        fig.add_vline(x=0, line_color=CINZA, line_width=1)
        fig.add_trace(go.Scatter(
            x=ver.coef, y=ver.rotulo, mode="markers", name="Amostra (IC 95%)",
            marker=dict(size=10, color=np.where(ver.coef > 0, LARANJA, AZUL)),
            error_x=dict(type="data", array=ver.ic_sup - ver.coef, arrayminus=ver.coef - ver.ic_inf,
                         thickness=2, width=0, color=CINZA),
            customdata=np.c_[ver.ic_inf, ver.ic_sup, ver.p],
            hovertemplate="<b>%{y}</b><br>%{x:+.1f} pts<br>IC: %{customdata[0]:.1f} a %{customdata[1]:.1f}"
                          "<br>p = %{customdata[2]:.3g}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(x=ver.coef_pop, y=ver.rotulo, mode="markers", name="Mesmo modelo na população",
                                 marker=dict(size=8, symbol="diamond-open", color=CINZA, line=dict(width=1.5)),
                                 hovertemplate="<b>%{y}</b><br>População: %{x:+.1f}<extra></extra>"))
        fig.update_xaxes(title="Efeito na nota de MT (pontos), mantendo o resto constante", showgrid=True)
        grafico(layout(fig, "Efeito de cada variável", altura=max(300, 140 + 30 * len(ver))))
        st.caption("**Objetivo:** comparar o efeito de cada variável com as outras fixas, e conferir a estimativa da "
                   "amostra contra a população. Referências: renda até 1/4 SM, Sudeste, cor Branca, escola pública, "
                   "sexo masculino. Barra = IC 95%.")

    # ---- Simulador ----
    with st.container(border=True):
        st.subheader("Simulador de perfil")
        st.caption("**Objetivo:** montar um perfil e ver a nota média de Matemática que o modelo prevê para ele.")
        cf = res["coef"]["coef"]
        s1, s2, s3 = st.columns(3)
        with s1:
            sim_renda = st.selectbox("Renda per capita", FAIXAS)
            sim_esc = st.selectbox("Escolaridade dos pais (maior entre pai e mãe)", list(ANOS_ESCOLARIDADE),
                                   index=4, format_func=lambda a: ANOS_ESCOLARIDADE[a])
        with s2:
            sim_reg = st.selectbox("Região", REGIOES, index=3)
            sim_cor = st.selectbox("Cor/raça", CORES_RACA)
        with s3:
            sim_escola = st.radio("Escola", ["Pública", "Privada"], horizontal=True)
            sim_sexo = st.radio("Sexo", ["Feminino", "Masculino"], horizontal=True)
            sim_net = st.toggle("Tem internet em casa", value=True)
            sim_atraso = st.toggle("Atraso escolar (19 anos ou mais)")

        def termo(chave: str) -> float:
            return cf.get(chave, 0.0)  # categoria de referência não tem coeficiente -> 0

        previsto = (cf["Intercept"]
                    + termo(f"C(faixa_renda)[T.{sim_renda}]")
                    + termo(f"C(regiao, Treatment('Sudeste'))[T.{sim_reg}]")
                    + termo(f"C(cor, Treatment('Branca'))[T.{sim_cor}]")
                    + cf["escol_pais"] * sim_esc + cf["internet"] * sim_net
                    + cf["privada"] * (sim_escola == "Privada") + cf["feminino"] * (sim_sexo == "Feminino")
                    + cf["atraso"] * sim_atraso)
        margem = 1.96 * met.loc["teste", "RMSE"]
        media_geral = base.loc[base.tem_nota, "NU_NOTA_MT"].mean()

        r1, r2, r3 = st.columns(3)
        r1.metric("Nota média prevista (MT)", fmt(previsto, 0), border=True,
                  delta=f"{previsto - media_geral:+.0f} vs média geral", delta_color="off")
        r2.metric("Faixa provável de um aluno (95%)", f"{fmt(previsto - margem, 0)} a {fmt(previsto + margem, 0)}",
                  border=True, help="Previsão ± 1,96 × RMSE do teste.")
        r3.metric("Média geral de MT", fmt(media_geral, 0), border=True)

# ---------------- Explorar ----------------
with abas[6]:
    st.subheader("Tabela dinâmica")
    st.caption("**Objetivo:** cruzar livremente dois fatores e um indicador. Respeita todos os filtros da barra lateral.")
    METRICAS = {
        f"Média de {NOTA_NOME}": (lambda d: d[d.tem_nota & (d[NOTA] > 0)], NOTA, "mean", False),
        "% que faltou": (lambda d: d, "faltou", "mean", True),
        "% em escola privada": (lambda d: d, "privada", "mean", True),
        "% com internet": (lambda d: d, "internet", "mean", True),
        "Nº de inscritos": (lambda d: d, "faltou", "size", False),
    }
    e1, e2, e3 = st.columns(3)
    with e1:
        linhas = escolher_dim("Linhas", "piv_lin", "Renda per capita")
    with e2:
        colunas = escolher_dim("Colunas", "piv_col", "Escola", sem=(linhas,), nenhum=True)
    with e3:
        metrica = st.selectbox("Indicador", list(METRICAS), key="piv_met")
    recorte, alvo, func, eh_pct = METRICAS[metrica]
    d = recorte(df)
    lin_col, lin_ordem = DIMENSOES[linhas]
    chaves = [lin_col] + ([DIMENSOES[colunas][0]] if colunas else [])
    agg = d.groupby(chaves, observed=True)[alvo].agg(valor=func, n="size")
    valor = agg.valor if func == "size" else agg.valor.where(agg.n >= MIN_N)
    piv = valor.unstack() if colunas else valor.to_frame(metrica)
    piv = piv.reindex(index=lin_ordem)
    if colunas:
        piv = piv.reindex(columns=DIMENSOES[colunas][1])
    piv = piv.dropna(how="all").dropna(axis=1, how="all")
    piv.index, piv.columns = piv.index.astype(str), piv.columns.astype(str)
    piv.index.name = linhas

    # escala do fundo do tema até o azul da paleta ("Blues" deixaria células quase brancas no fundo escuro)
    MAPA_TABELA = LinearSegmentedColormap.from_list("tabela", [FUNDO_CARD, AZUL])
    formato = (lambda v: pct(v)) if eh_pct else (lambda v: fmt(v, 0 if func == "size" else 1))
    st.dataframe(piv.style.format(formato, na_rep="–").background_gradient(cmap=MAPA_TABELA, axis=None))
    st.download_button("Baixar CSV", piv.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
                       file_name="enem2023_tabela.csv", mime="text/csv", icon=":material/download:")
