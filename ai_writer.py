import pandas as pd
import os
import google.generativeai as genai
import sys
import numpy as np

# ==============================================================================
# 1. CONFIGURACIÓN ESPECIAL COPA DEL REY
# ==============================================================================
MODEL_NAME = "gemini-2.5-flash"
FILE_PATH = "data/BoxScore_Copa_2025_Cumulative.csv"

# Capturamos la fase que nos envía check_status.py
FASE_ACTUAL = sys.argv[1] if len(sys.argv) > 1 else "Copa del Rey"

# Mapa de Equipos (Solo los 8 clasificados a la Copa)
TEAM_MAP = {
    'UNI': 'Unicaja', 'JOV': 'Joventut Badalona', 'RMB': 'Real Madrid', 
    'BKN': 'Baskonia', 'TEN': 'La Laguna Tenerife', 'UCM': 'UCAM Murcia', 
    'VBC': 'Valencia Basket', 'BAR': 'Barça'
}

# Mapa de Entrenadores (Temporada 2025/2026 - ACTUALIZADO OFICIAL)
COACH_MAP = {
    'BAR': 'Xavi Pascual', 'RMB': 'Sergio Scariolo', 'UNI': 'Ibon Navarro',
    'BKN': 'Paolo Galbiati', 'VBC': 'Pedro Martínez', 'UCM': 'Sito Alonso',
    'GCA': 'Jaka Lakovic', 'TEN': 'Txus Vidorreta', 'JOV': 'Dani Miret',
    'MAN': 'Diego Ocampo', 'SBB': 'Jaume Ponsarnau', 'CAZ': 'Joan Plaza',
    'GIR': 'Moncho Fernández', 'BRE': 'Luis Casimiro', 'LLE': 'Gerard Encuentra',
    'COV': 'Arturo Ruíz', 'MBA': 'Zan Tabak', 'BUR': 'Porfi Fisac'
}

# ==============================================================================
# 2. DICCIONARIO MAESTRO DE JUGADORES (COPA DEL REY)
# ==============================================================================
CORRECCIONES_VIP = {
    # --- BARÇA (BAR) ---
    "D. Brizuela": "Darío Brizuela", "D. González": "Dani González", "J. Marcos": "Juani Marcos", "J. Parra": "Joel Parra", "J. Vesely": "Jan Vesely", "K. Punter": "Kevin Punter", "M. Cale": "Myles Cale", "M. Norris": "Miles Norris", "N. Kusturica": "Nikola Kusturica", "N. Laprovittola": "Nico Laprovittola", "S. Keita": "Sayon Keita", "T. Satoransky": "Tomas Satoransky", "T. Shengelia": "Toko Shengelia", "W. Clyburn": "Will Clyburn", "W. Hernangómez": "Willy Hernangómez", "Y. Fall": "Youssoupha Fall",
    # --- BASKONIA (BKN) ---
    "C. Frisch": "Clément Frisch", "E. Omoruyi": "Eugene Omoruyi", "G. Radzevicius": "Gytis Radzevicius", "H. Diallo": "Hamidou Diallo", "K. Diop": "Khalifa Diop", "K. Simmons": "Kobi Simmons", "L. Samanic": "Luka Samanic", "Luwawu-Cabarrot": "Timothé Luwawu-Cabarrot", "M. Diakite": "Mamadi Diakite", "M. Howard": "Markus Howard", "M. Nowell": "Markquis Nowell", "M. Spagnolo": "Matteo Spagnolo", "R. Kurucs": "Rodions Kurucs", "R. Villar": "Rafa Villar", "S. Joksimovic": "Stefan Joksimovic", "T. Forrest": "Trent Forrest", "T. Sedekerskis": "Tadas Sedekerskis",
    # --- JOVENTUT BADALONA (JOV) ---
    "A. Hanga": "Adam Hanga", "A. Tomic": "Ante Tomic", "A. Torres": "Adrià Torres", "C. Hunt": "Cameron Hunt", "F. Mauri": "Ferran Mauri", "G. Vives": "Guillem Vives", "H. Drell": "Henri Drell", "L. Hakanson": "Ludde Hakanson", "M. Allen": "Miguel Allen", "M. Ruzic": "Michael Ruzic", "R. Rubio": "Ricky Rubio", "S. Birgander": "Simon Birgander", "Y. Kraag": "Yannick Kraag",
    # --- LA LAGUNA TENERIFE (TEN) ---
    "A. Doornekamp": "Aaron Doornekamp", "B. Fitipaldo": "Bruno Fitipaldo", "D. Bordón": "Diego Bordón", "F. Guerra": "Fran Guerra", "G. Shermadini": "Giorgi Shermadini", "H. Alderete": "Hector Alderete", "J. Fernández": "Jaime Fernández", "J. Sastre": "Joan Sastre", "K. Kostadinov": "Konstantin Kostadinov", "L. Costa": "Lluís Costa", "M. Huertas": "Marcelinho Huertas", "R. Giedraitis": "Rokas Giedraitis", "T. Abromaitis": "Tim Abromaitis", "T. Scrubb": "Thomas Scrubb", "W. Van Beck": "Wesley Van Beck",
    # --- REAL MADRID (RMB) ---
    "A. Abalde": "Alberto Abalde", "A. Feliz": "Andrés Feliz", "A. Len": "Alex Len", "C. Okeke": "Chuma Okeke", "D. Kramer": "David Kramer", "F. Campazzo": "Facundo Campazzo", "G. Deck": "Gabriel Deck", "G. Grinvalds": "Gunars Grinvalds", "G. Procida": "Gabriele Procida", "I. Almansa": "Izan Almansa", "M. Hezonja": "Mario Hezonja", "S. Llull": "Sergio Llull", "T. Lyles": "Trey Lyles", "T. Maledon": "Théo Maledon", "U. Garuba": "Usman Garuba", "W. Tavares": "Edy Tavares",
    # --- UCAM MURCIA (UCM) ---
    "D. Cacok": "Devontae Cacok", "D. DeJulius": "David DeJulius", "D. Ennis": "Dylan Ennis", "D. García": "Dani García", "E. Cate": "Emanuel Cate", "H. Sant-Roos": "Howard Sant-Roos", "J. Radebaugh": "Jonah Radebaugh", "M. Diagné": "Moussa Diagné", "M. Forrest": "Michael Forrest", "R. López": "Rubén López de la Torre", "S. Raieste": "Sander Raieste", "T. Nakic": "Toni Nakic", "W. Falk": "Wilhelm Falk", "Z. Hicks": "Zach Hicks",
    # --- UNICAJA (UNI) ---
    "A. Butajevas": "Arturas Butajevas", "A. Díaz": "Alberto Díaz", "A. Rubit": "Augustine Rubit", "C. Audige": "Chase Audige", "C. Duarte": "Chris Duarte", "D. Kravish": "David Kravish", "E. Sulejmanovic": "Emir Sulejmanovic", "J. Barreiro": "Jonathan Barreiro", "J. Webb": "James Webb III", "K. Perry": "Kendrick Perry", "K. Tillie": "Killian Tillie", "N. Djedovic": "Nihad Djedovic", "O. Balcerowski": "Olek Balcerowski", "T. Kalinoski": "Tyler Kalinoski", "T. Pérez": "Tyson Pérez", "X. Castañeda": "Xavier Castañeda",
    # --- VALENCIA BASKET (VBC) ---
    "B. Badio": "Brancou Badio", "B. Key": "Braxton Key", "D. Thompson": "Darius Thompson", "I. Iroegbu": "Ike Iroegbu", "I. Nogués": "Isaac Nogués", "J. Montero": "Jean Montero", "J. Pradilla": "Jaime Pradilla", "J. Puerto": "Josep Puerto", "K. Taylor": "Kameron Taylor", "López-Arostegui": "Xabi López-Arostegui", "M. Costello": "Matt Costello", "N. Reuvers": "Nate Reuvers", "N. Sako": "Neal Sako", "O. Moore": "Omari Moore", "S. de Larrea": "Sergio de Larrea", "Y. Sima": "Yankuba Sima"
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

df_fase = df[df['Week'] == FASE_ACTUAL]

if df_fase.empty:
    guardar_salida(f"❌ No hay datos para la fase: {FASE_ACTUAL}.")

print(f"🤖 Analizando {FASE_ACTUAL}...")

# ==============================================================================
# 5. PREPARACIÓN DE DATOS
# ==============================================================================

# A. MEJORES JUGADORES
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

# C. EQUIPOS
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
- Mejor Ataque: {get_team_name(best_offense['Team'])} (Entrenador: {COACH_MAP.get(best_offense['Team'], 'su técnico')}) con {b(best_offense['ORTG'], 1)} pts/100.
- Fluidez: {get_team_name(best_passing['Team'])} (Entrenador: {COACH_MAP.get(best_passing['Team'], 'su técnico')}) con {b(best_passing['AST_Ratio'], 1)} ast/100.
- Control: {get_team_name(most_careful['Team'])} (Entrenador: {COACH_MAP.get(most_careful['Team'], 'su técnico')}) con {b(most_careful['TO_Ratio'], 1)} perdidas/100.
"""

# D. LÍDERES ACUMULADOS
means = df.groupby(['Name', 'Team'])[['VAL', 'PTS', 'AST', 'TS%']].mean().reset_index()
hot = means.sort_values('VAL', ascending=False).head(5)
txt_trends = ""
for _, row in hot.iterrows():
    t_name = clean_name(row['Name'])
    txt_trends += (f"- {t_name} ({get_team_name(row['Team'], False)}): "
                   f"{b(row['VAL'], 1)} VAL, {b(row['PTS'], 1)} PTS, {b(row['AST'], 1)} AST.\n")

# ==============================================================================
# 6. LÓGICA DE TÍTULOS E INSTRUCCIONES DE BÚSQUEDA
# ==============================================================================
instrucciones_especificas = ""

if FASE_ACTUAL == "Final":
    titulo_seccion_1 = "### El MVP Oficial y las Claves de la Final"
    instrucciones_especificas = """
    INSTRUCCIONES EXCLUSIVAS PARA LA FINAL (USO OBLIGATORIO DE GOOGLE SEARCH):
    1. BÚSQUEDA DEL MVP: Usa tu herramienta de búsqueda en Google para confirmar quién ha sido el MVP Oficial de la Copa del Rey de baloncesto 2026. Nómbralo en el primer párrafo y añade sus estadísticas destacadas o por qué se lo han dado.
    2. EL CAMPEÓN Y SU CAMINO: Menciona explícitamente y con emoción al equipo que ha ganado la Final y haz un brevísimo apunte sobre cómo ha sido su camino hasta levantar el título.
    3. JUGADAS DETERMINANTES: Usa tu búsqueda en Internet para encontrar 1 o 2 jugadas o momentos clave del partido (un triple decisivo para romper un parcial, un tapón, una actuación *clutch* en los últimos minutos) e intégralos en la crónica para dar contexto real a los fríos datos.
    4. Analiza el RITMO DEL PARTIDO basándote en los datos estadísticos proporcionados (ORTG, posesiones).
    5. Tono de "Gran Final": Transmite la tensión y el prestigio de levantar la Copa, combinando la épica periodística con tus datos de analítica avanzada.
    """
elif FASE_ACTUAL == "Semifinales":
    titulo_seccion_1 = "### Estrellas de las Semifinales"
else:
    titulo_seccion_1 = "### Estrellas de los Cuartos de Final"

# ==============================================================================
# 7. GENERACIÓN IA CON GOOGLE SEARCH ACTIVADO
# ==============================================================================

prompt = f"""
    Actúa como un analista de baloncesto profesional y periodista deportivo de élite.
    Vas a escribir la newsletter 'Analyzing Basketball' sobre la Copa del Rey.
    
    FASE ACTUAL: {FASE_ACTUAL}
    
    DATOS DE LOS JUGADORES (Top Performers Estadísticos):
    {txt_mejores}
    {txt_rest}
    
    DATOS DE LOS EQUIPOS (Eficiencia y Entrenadores):
    {txt_teams}
    
    LÍDERES ACUMULADOS DE LA COPA:
    {txt_trends}
    
    {instrucciones_especificas}
    
    REGLAS DE ESTILO (¡MUY ESTRICTAS Y DE OBLIGADO CUMPLIMIENTO!):
    1. TONO Y AUDIENCIA: Profesional, analítico y estrictamente periodístico. Escribes para expertos en baloncesto en ESPAÑA.
    2. IDIOMA (ESPAÑOL DE ESPAÑA PURO): Tienes TERMINANTEMENTE PROHIBIDO usar vocabulario latinoamericano. Usa "mate" (nunca volcada), "parqué/cancha" (nunca duela), y "tiros libres" (nunca lanzamiento de personal).
    3. CERO EMOJIS (CRÍTICO): Está TOTALMENTE PROHIBIDO usar emojis en cualquier parte del texto. NI UNO SOLO en el asunto, NI en los títulos, NI en el cuerpo.
    4. TRATO AL LECTOR (IMPERSONAL): NO te dirijas al lector bajo ningún concepto. Tienes PROHIBIDO usar "tú", PROHIBIDO usar "vosotros" y PROHIBIDO tratar de "usted". Escribe exclusivamente en tercera persona o usando formas impersonales ("se observa", "el equipo logró", "destaca"). Cero preguntas retóricas.
    5. ENTRENADORES Y ALUCINACIONES: Usa estrictamente los nombres de los entrenadores proporcionados en los datos. No inventes nada.
    6. RITMO Y VOZ ACTIVA: Cero dramatismos ("a vida o muerte", "clavo en el ataúd"). Escribe en voz activa usando vocabulario técnico real (pick & roll, spacing, mismatch, lado débil, colapso defensivo).
    7. VOCABULARIO DE PARQUÉ: Usa terminología técnica real de baloncesto con naturalidad (spacing, pick & roll, mismatch, IQ, colapso defensivo, tiro tras bote, generación de ventajas, lado débil).

    ESTRUCTURA DE SALIDA (ESTRICTA):
    ASUNTO: [Escribe aquí un asunto atractivo, muy profesional, que denote que es la Final, basado en el MVP/Campeón y ESTRICTAMENTE SIN NINGÚN EMOJI]

    ## Especial Copa del Rey 2026: {FASE_ACTUAL}

    {titulo_seccion_1}
    [Redacta la crónica principal siguiendo las instrucciones (buscando al MVP real y las jugadas clave en internet). Combina la narrativa épica con el análisis del rendimiento estadístico aportado en los datos.]

    ### Radar de Eficiencia y Pizarra Táctica
    [Redacta el análisis del rendimiento de los equipos. Usa los datos de Puntos por 100 posesiones, Asistencias o Pérdidas y menciona a sus entrenadores reales proporcionados. Traduce esto a cómo fue el ritmo y el control táctico del partido.]

    ### Dominadores del Torneo (Promedios Totales)
    [Enumera a los 5 jugadores con mayor valoración acumulada de toda la Copa en este formato exacto, usando guiones:]
    {txt_trends}
"""

try:
    print(f"🚀 Generando crónica premium para {FASE_ACTUAL}...")
    # ATENCIÓN AQUÍ: Corrección de la herramienta de Google Search aplicada
    model = genai.GenerativeModel(model_name=MODEL_NAME, tools="google_search")
    response = model.generate_content(prompt)
    texto = response.text.replace(":\n-", ":\n\n-")
    guardar_salida(texto)
except Exception as e:
    guardar_salida(f"❌ Error Gemini: {e}")
