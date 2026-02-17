import pandas as pd
import os
import google.generativeai as genai
import sys
import re
import numpy as np

# ==============================================================================
# 1. CONFIGURACIÓN ESPECIAL COPA DEL REY
# ==============================================================================
MODEL_NAME = "gemini-2.5-flash"
FILE_PATH = "data/BoxScore_Copa_2025_Cumulative.csv"

# Capturamos la fase que nos envía check_status.py (ej: "Cuartos de Final")
# Si lo ejecutas a mano y no le pasas nada, usará "Copa del Rey" por defecto.
FASE_ACTUAL = sys.argv[1] if len(sys.argv) > 1 else "Copa del Rey"

# Mapa de Equipos (Solo los 8 clasificados a la Copa)
TEAM_MAP = {
    'UNI': 'Unicaja', 'JOV': 'Joventut Badalona', 'RMB': 'Real Madrid', 
    'BKN': 'Baskonia', 'TEN': 'La Laguna Tenerife', 'UCM': 'UCAM Murcia', 
    'VBC': 'Valencia Basket', 'BAR': 'Barça'
}

# ==============================================================================
# 2. DICCIONARIO MAESTRO DE JUGADORES (REVISADO AL MILÍMETRO - COPA DEL REY)
# ==============================================================================
CORRECCIONES_VIP = {
    # --- BARÇA (BAR) ---
    "D. Brizuela": "Darío Brizuela",
    "D. González": "Dani González",
    "J. Marcos": "Juani Marcos",
    "J. Parra": "Joel Parra",
    "J. Vesely": "Jan Vesely",
    "K. Punter": "Kevin Punter",
    "M. Cale": "Myles Cale",
    "M. Norris": "Miles Norris",
    "N. Kusturica": "Nikola Kusturica",
    "N. Laprovittola": "Nico Laprovittola",
    "S. Keita": "Sayon Keita",
    "T. Satoransky": "Tomas Satoransky",
    "T. Shengelia": "Toko Shengelia",
    "W. Clyburn": "Will Clyburn",
    "W. Hernangómez": "Willy Hernangómez",
    "Y. Fall": "Youssoupha Fall",

    # --- BASKONIA (BKN) ---
    "C. Frisch": "Clément Frisch",
    "E. Omoruyi": "Eugene Omoruyi",
    "G. Radzevicius": "Gytis Radzevicius",
    "H. Diallo": "Hamidou Diallo",
    "K. Diop": "Khalifa Diop",
    "K. Simmons": "Kobi Simmons",
    "L. Samanic": "Luka Samanic",
    "Luwawu-Cabarrot": "Timothé Luwawu-Cabarrot",
    "M. Diakite": "Mamadi Diakite",
    "M. Howard": "Markus Howard",
    "M. Nowell": "Markquis Nowell",
    "M. Spagnolo": "Matteo Spagnolo",
    "R. Kurucs": "Rodions Kurucs",
    "R. Villar": "Rafa Villar",
    "S. Joksimovic": "Stefan Joksimovic",
    "T. Forrest": "Trent Forrest",
    "T. Sedekerskis": "Tadas Sedekerskis",

    # --- JOVENTUT BADALONA (JOV) ---
    "A. Hanga": "Adam Hanga",
    "A. Tomic": "Ante Tomic",
    "A. Torres": "Adrià Torres",
    "C. Hunt": "Cameron Hunt",
    "F. Mauri": "Ferran Mauri",
    "G. Vives": "Guillem Vives",
    "H. Drell": "Henri Drell",
    "L. Hakanson": "Ludde Hakanson",
    "M. Allen": "Miguel Allen",
    "M. Ruzic": "Michael Ruzic",
    "R. Rubio": "Ricky Rubio",
    "S. Birgander": "Simon Birgander",
    "Y. Kraag": "Yannick Kraag",

    # --- LA LAGUNA TENERIFE (TEN) ---
    "A. Doornekamp": "Aaron Doornekamp",
    "B. Fitipaldo": "Bruno Fitipaldo",
    "D. Bordón": "Diego Bordón",
    "F. Guerra": "Fran Guerra",
    "G. Shermadini": "Giorgi Shermadini",
    "H. Alderete": "Hector Alderete",
    "J. Fernández": "Jaime Fernández",
    "J. Sastre": "Joan Sastre",
    "K. Kostadinov": "Konstantin Kostadinov",
    "L. Costa": "Lluís Costa",
    "M. Huertas": "Marcelinho Huertas",
    "R. Giedraitis": "Rokas Giedraitis",
    "T. Abromaitis": "Tim Abromaitis",
    "T. Scrubb": "Thomas Scrubb",
    "W. Van Beck": "Wesley Van Beck",

    # --- REAL MADRID (RMB) ---
    "A. Abalde": "Alberto Abalde",
    "A. Feliz": "Andrés Feliz",
    "A. Len": "Alex Len",
    "C. Okeke": "Chuma Okeke",
    "D. Kramer": "David Kramer",
    "F. Campazzo": "Facundo Campazzo",
    "G. Deck": "Gabriel Deck",
    "G. Grinvalds": "Gunars Grinvalds",
    "G. Procida": "Gabriele Procida",
    "I. Almansa": "Izan Almansa",
    "M. Hezonja": "Mario Hezonja",
    "S. Llull": "Sergio Llull",
    "T. Lyles": "Trey Lyles",
    "T. Maledon": "Théo Maledon",
    "U. Garuba": "Usman Garuba",
    "W. Tavares": "Edy Tavares",

    # --- UCAM MURCIA (UCM) ---
    "D. Cacok": "Devontae Cacok",
    "D. DeJulius": "David DeJulius",
    "D. Ennis": "Dylan Ennis",
    "D. García": "Dani García",
    "E. Cate": "Emanuel Cate",
    "H. Sant-Roos": "Howard Sant-Roos",
    "J. Radebaugh": "Jonah Radebaugh",
    "M. Diagné": "Moussa Diagné",
    "M. Forrest": "Michael Forrest",
    "R. López": "Rubén López de la Torre",
    "S. Raieste": "Sander Raieste",
    "T. Nakic": "Toni Nakic",
    "W. Falk": "Wilhelm Falk",
    "Z. Hicks": "Zach Hicks",

    # --- UNICAJA (UNI) ---
    "A. Butajevas": "Arturas Butajevas",
    "A. Díaz": "Alberto Díaz",
    "A. Rubit": "Augustine Rubit",
    "C. Audige": "Chase Audige",
    "C. Duarte": "Chris Duarte",
    "D. Kravish": "David Kravish",
    "E. Sulejmanovic": "Emir Sulejmanovic",
    "J. Barreiro": "Jonathan Barreiro",
    "J. Webb": "James Webb III",
    "K. Perry": "Kendrick Perry",
    "K. Tillie": "Killian Tillie",
    "N. Djedovic": "Nihad Djedovic",
    "O. Balcerowski": "Olek Balcerowski",
    "T. Kalinoski": "Tyler Kalinoski",
    "T. Pérez": "Tyson Pérez",
    "X. Castañeda": "Xavier Castañeda",

    # --- VALENCIA BASKET (VBC) ---
    "B. Badio": "Brancou Badio",
    "B. Key": "Braxton Key",
    "D. Thompson": "Darius Thompson",
    "I. Iroegbu": "Ike Iroegbu",
    "I. Nogués": "Isaac Nogués",
    "J. Montero": "Jean Montero",
    "J. Pradilla": "Jaime Pradilla",
    "J. Puerto": "Josep Puerto",
    "K. Taylor": "Kameron Taylor",
    "López-Arostegui": "Xabi López-Arostegui",
    "M. Costello": "Matt Costello",
    "N. Reuvers": "Nathan Reuvers",
    "N. Sako": "Neal Sako",
    "O. Moore": "Omari Moore",
    "S. de Larrea": "Sergio de Larrea",
    "Y. Sima": "Yankuba Sima"
}

# ==============================================================================
# 3. FUNCIONES AUXILIARES
# ==============================================================================
def guardar_salida(mensaje, nombre_archivo="newsletter_borrador.md"):
    print(mensaje)
    try:
        with open(nombre_archivo, "w", encoding="utf-8") as f:
            f.write(mensaje)
        print(f"\n✅ Newsletter guardada: {nombre_archivo}")
    except Exception as e:
        print(f"❌ Error guardando archivo: {e}")
    sys.exit(0)

def b(val, decimals=0, is_percent=False):
    if pd.isna(val) or val == np.inf or val == -np.inf: val = 0
    suffix = "%" if is_percent else ""
    if isinstance(val, (int, float)):
        if val % 1 == 0 and decimals == 0: return f"**{int(val)}**{suffix}"
        return f"**{val:.{decimals}f}**{suffix}"
    return f"**{val}**{suffix}"

def get_team_name(abbr, use_full=True):
    return TEAM_MAP.get(abbr, abbr) if use_full else abbr

def clean_name(name_raw):
    return CORRECCIONES_VIP.get(name_raw, name_raw)

# ==============================================================================
# 4. CARGA DE DATOS
# ==============================================================================
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key: guardar_salida("❌ Error: Falta GEMINI_API_KEY.")
genai.configure(api_key=api_key)

if not os.path.exists(FILE_PATH): guardar_salida(f"❌ No hay CSV en {FILE_PATH}.")
df = pd.read_csv(FILE_PATH)

cols_num = ['VAL', 'PTS', 'Reb_T', 'AST', 'Win', 'Game_Poss', 'TO', 'TS%', 'USG%']
for col in cols_num:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Filtramos SOLO los partidos de la fase actual para el análisis de destacados
df_fase = df[df['Week'] == FASE_ACTUAL]

if df_fase.empty:
    guardar_salida(f"❌ No hay datos para la fase: {FASE_ACTUAL}.")

print(f"🤖 Analizando {FASE_ACTUAL}...")

# ==============================================================================
# 5. PREPARACIÓN DE DATOS (CON LÓGICA DE TOP PERFORMERS VS MVP)
# ==============================================================================

# A. MEJORES JUGADORES (SOPORTA EMPATES)
ganadores = df_fase[df_fase['Win'] == 1]
pool = ganadores if not ganadores.empty else df_fase

max_val = pool['VAL'].max()
mejores = pool[pool['VAL'] == max_val]

txt_mejores = ""
mejores_ids = []

for _, row in mejores.iterrows():
    m_name = clean_name(row['Name'])
    txt_mejores += (f"- {m_name} ({get_team_name(row['Team'])}): {b(row['VAL'])} VAL, "
                f"{b(row['PTS'])} PTS (TS%: {b(row['TS%'], 1, True)}), {b(row['Reb_T'])} REB.\n")
    mejores_ids.append(row['PlayerID'])

# B. DESTACADOS SECUNDARIOS
resto = df_fase[~df_fase['PlayerID'].isin(mejores_ids)]
top_rest = resto.sort_values('VAL', ascending=False).head(3)
txt_rest = ""
for _, row in top_rest.iterrows():
    r_name = clean_name(row['Name'])
    txt_rest += f"- {r_name} ({get_team_name(row['Team'])}): {b(row['VAL'])} VAL.\n"

# C. EQUIPOS (En la fase actual)
team_agg = df_fase.groupby('Team').agg({
    'PTS': 'sum', 'Game_Poss': 'mean', 'Reb_T': 'sum', 'AST': 'sum', 'TO': 'sum'
}).reset_index()
team_agg['ORTG'] = (team_agg['PTS'] / team_agg['Game_Poss']) * 100
team_agg['AST_Ratio'] = (team_agg['AST'] / team_agg['Game_Poss']) * 100
team_agg['TO_Ratio'] = (team_agg['TO'] / team_agg['Game_Poss']) * 100

best_offense = team_agg.sort_values('ORTG', ascending=False).iloc[0]
best_passing = team_agg.sort_values('AST_Ratio', ascending=False).iloc[0]
most_careful = team_agg.sort_values('TO_Ratio', ascending=True).iloc[0]

txt_teams = f"""
- Mejor Ataque: {get_team_name(best_offense['Team'])} ({b(best_offense['ORTG'], 1)} pts/100).
- Fluidez: {get_team_name(best_passing['Team'])} ({b(best_passing['AST_Ratio'], 1)} ast/100).
- Control: {get_team_name(most_careful['Team'])} ({b(most_careful['TO_Ratio'], 1)} perdidas/100).
"""

# D. LÍDERES ACUMULADOS DE LA COPA 
means = df.groupby(['Name', 'Team'])[['VAL', 'PTS', 'AST', 'TS%']].mean().reset_index()
hot = means.sort_values('VAL', ascending=False).head(5)
txt_trends = ""
for _, row in hot.iterrows():
    t_name = clean_name(row['Name'])
    txt_trends += (f"- {t_name} ({get_team_name(row['Team'], False)}): "
                   f"{b(row['VAL'], 1)} VAL, {b(row['PTS'], 1)} PTS, {b(row['AST'], 1)} AST.\n")

# ==============================================================================
# 6. LÓGICA DE TÍTULOS (ESTRELLAS VS MVP Y GRAMÁTICA)
# ==============================================================================
if FASE_ACTUAL == "Final":
    titulo_seccion_1 = "### 👑 MVP de la Copa del Rey"
    etiqueta_jugador = "MVP:"
elif FASE_ACTUAL == "Semifinales":
    titulo_seccion_1 = "### 🌟 Estrellas de las Semifinales"
    etiqueta_jugador = "TOP PERFORMER(S):"
else:
    titulo_seccion_1 = "### 🌟 Estrellas de los Cuartos de Final"
    etiqueta_jugador = "TOP PERFORMER(S):"

# ==============================================================================
# 7. GENERACIÓN IA 
# ==============================================================================

prompt = f"""
Actúa como Periodista Deportivo experto en la Copa del Rey de Baloncesto (ACB) y Copywriter de Email Marketing viral.
Estás escribiendo la crónica de la fase: {FASE_ACTUAL}. El torneo es eliminatorio (a vida o muerte).

DATOS DE LA FASE ({FASE_ACTUAL}):
{etiqueta_jugador}
{txt_mejores}
DESTACADOS:
{txt_rest}
EQUIPOS EN ESTA FASE:
{txt_teams}
LÍDERES ESTADÍSTICOS DE TODO EL TORNEO HASTA AHORA:
{txt_trends}

INSTRUCCIONES:
1. **PRIMERA LÍNEA OBLIGATORIA**: Escribe una frase corta (máx 50 caracteres), impactante y estilo "clickbait" que resuma lo más loco de estos partidos. EMPIEZA LA LÍNEA CON "ASUNTO:".
2. **RESPETA LOS NOMBRES**: Úsalos tal cual aparecen arriba.
3. **NARRATIVA**: Crónica vibrante, habla de la tensión del torneo del KO.

ESTRUCTURA DE SALIDA (ESTRICTA):
ASUNTO: [Aquí tu frase clickbait]

## 🏆 Especial Copa del Rey 2026: {FASE_ACTUAL}

{titulo_seccion_1}
[Análisis vibrante de los mejores jugadores de esta fase basándote en los datos aportados]

### 🚀 Radar de Eficiencia y Contexto
[Análisis de destacados y rendimiento de equipos (ataque, fluidez, control)]

### 🔥 Dominadores del Torneo (Promedios Acumulados)
{txt_trends}
"""

try:
    print(f"🚀 Generando crónica para {FASE_ACTUAL}...")
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    texto = response.text.replace(":\n-", ":\n\n-")
    guardar_salida(texto)
except Exception as e:
    guardar_salida(f"❌ Error Gemini: {e}")
