import requests
import os
import re
import datetime
import subprocess 
import time
import random
from bs4 import BeautifulSoup

# ==============================================================================
# CONFIGURACIÓN ESPECIAL COPA DEL REY
# ==============================================================================
TEMPORADA = '2025'
COMPETICION = '2'   # ID de la Copa del Rey
LOG_FILE = "data/log.txt"

# API Key y Headers
API_KEY = '0dd94928-6f57-4c08-a3bd-b1b2f092976e'
HEADERS_API = {
    'x-apikey': API_KEY,
    'origin': 'https://live.acb.com',
    'referer': 'https://live.acb.com/',
    'user-agent': 'Mozilla/5.0'
}

# Definimos las 3 Fases de envío (Sábado, Domingo y Lunes)
FASES_COPA = [
    {"paso": 1, "nombre": "Cuartos de Final", "jornada_id": "1", "partidos_terminados_requeridos": 4},
    {"paso": 2, "nombre": "Semifinales", "jornada_id": "2", "partidos_terminados_requeridos": 2},
    {"paso": 3, "nombre": "Final", "jornada_id": "3", "partidos_terminados_requeridos": 1}
]

# ==============================================================================
# ZONA 1: FUNCIONES DE SCRAPING
# ==============================================================================

def get_last_fase_from_log():
    """Lee el log para saber por qué envío de la Copa vamos (0 a 3)"""
    if not os.path.exists(LOG_FILE):
        return 0
    last_paso = 0
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                match = re.search(r'Paso\s+(\d+)', line, re.IGNORECASE)
                if match:
                    num = int(match.group(1))
                    if num > last_paso:
                        last_paso = num
    except Exception as e:
        print(f"Error leyendo log: {e}")
    return last_paso

def get_game_ids(temp_id, comp_id, jornada_id):
    url = f"https://www.acb.com/resultados-clasificacion/ver/temporada_id/{temp_id}/competicion_id/{comp_id}/jornada_numero/{jornada_id}"
    ids = []
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(r.content, 'html.parser')
        for a in soup.find_all('a', href=True):
            if "/partido/estadisticas/id/" in a['href']:
                try:
                    pid = int(a['href'].split("/id/")[1].split("/")[0])
                    ids.append(pid)
                except: pass
        return list(set(ids))
    except: return []

def is_game_finished(game_id):
    url = "https://api2.acb.com/api/matchdata/Result/boxscores"
    try:
        r = requests.get(url, params={'matchId': game_id}, headers=HEADERS_API, timeout=5)
        if r.status_code != 200: return False
        data = r.json()
        if 'teamBoxscores' not in data or len(data['teamBoxscores']) < 2: return False
        return True
    except: return False

# ==============================================================================
# ZONA 2: SECUENCIA DE ENVÍO
# ==============================================================================

def ejecutar_secuencia_completa(nombre_fase):
    print(f"🔄 Iniciando secuencia para: {nombre_fase}...")

    NOMBRE_SCRIPT_DATOS = "boxscore_COPA_headless.py"
    print(f"📥 0. Ejecutando {NOMBRE_SCRIPT_DATOS}...")
    try:
        subprocess.run(["python", NOMBRE_SCRIPT_DATOS], check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error actualizando datos: {e}")
        return False

    print("🤖 1. Ejecutando ai_writer.py...")
    try:
        # Le pasaremos el nombre de la fase como argumento para que la IA sepa de qué escribir
        subprocess.run(["python", "ai_writer.py", nombre_fase], check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en ai_writer: {e}")
        return False

    print("📧 2. Ejecutando email_sender.py...")
    try:
        subprocess.run(["python", "email_sender.py"], check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en email_sender: {e}")
        return False

# ==============================================================================
# MAIN 
# ==============================================================================

def main():
    pasos_completados = get_last_fase_from_log()
    
    if pasos_completados >= len(FASES_COPA):
        print("🏆 La cobertura de la Copa del Rey ha finalizado por completo. No hay más envíos pendientes.")
        return

    fase_actual = FASES_COPA[pasos_completados]
    print(f"--- SCRIPT DE CONTROL: COPA DEL REY ---")
    print(f"Objetivo actual: Paso {fase_actual['paso']} -> {fase_actual['nombre']}")

    game_ids = get_game_ids(TEMPORADA, COMPETICION, fase_actual['jornada_id'])
    
    # COMPROBACIÓN: ¿Cuántos partidos de esta jornada han terminado?
    finished_count = sum(1 for gid in game_ids if is_game_finished(gid))
    
    print(f"📊 Estado: {finished_count} de {fase_actual['partidos_terminados_requeridos']} partidos terminados.")

    if finished_count >= fase_actual['partidos_terminados_requeridos']:
        print(f"✅ Requisitos cumplidos para {fase_actual['nombre']}.")
        
        # --- EL TRUCO DEL FACTOR HUMANO ---
        minutos_espera = random.randint(5, 20)
        print(f"☕ Simulando comportamiento humano... Esperando {minutos_espera} minutos.")
        time.sleep(minutos_espera * 60)
        print("⏰ ¡Despierta! Enviando ahora.")
        # ----------------------------------

        exito = ejecutar_secuencia_completa(fase_actual['nombre'])
        
        if exito:
            # Aseguramos que existe la carpeta data/ antes de escribir el log
            if not os.path.exists("data"):
                os.makedirs("data")

            fecha_log = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            linea_log = f"{fecha_log} : ✅ Paso {fase_actual['paso']} completado y enviado ({fase_actual['nombre']}).\n"
            
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(linea_log)
            print("🏁 Newsletter enviada con éxito.")
    else:
        print(f"⚽ Aún faltan partidos por terminar para completar la fase de {fase_actual['nombre']}.")

if __name__ == "__main__":
    main()
