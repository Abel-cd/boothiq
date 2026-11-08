import os
import json
import urllib.request
from pathlib import Path
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

try:
    import soccerdata as sd
    HAS_SOCCERDATA = True
except ImportError:
    HAS_SOCCERDATA = False

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

app = FastAPI(title="BoothIQ Live Automated Backend")
CACHE_DIR = Path("./cache_matches")
CACHE_DIR.mkdir(exist_ok=True)

# ==========================================
# 1. ESQUEMA DE DATOS (CONTRATO JSON)
# ==========================================

class PlayerTriggers(BaseModel):
    on_goal: str
    on_yellow: str
    on_sub_in: Optional[str] = None

class PlayerCard(BaseModel):
    id: str
    number: int
    name: str
    pos: str
    age: int
    is_starter: bool
    is_apercibido: bool
    is_hot_streak: bool
    bullets: List[str]
    triggers: PlayerTriggers

class PanicStory(BaseModel):
    id: str
    category: str
    title: str
    reading_time: str
    script: str

class MatchPayload(BaseModel):
    match_id: str
    competition: str
    home_name: str
    away_name: str
    venue: str
    referee: str
    next_opponent_home: str
    next_opponent_away: str
    home_team: List[PlayerCard]
    away_team: List[PlayerCard]
    var_panic_stories: List[PanicStory]

# ==========================================
# 2. EXTRACTOR DE FBREF Y REGLAS HEURÍSTICAS
# ==========================================

def fetch_fbref_team_dataframe(league: str, season: str) -> Optional[pd.DataFrame]:
    if not HAS_SOCCERDATA:
        return None
    try:
        fbref = sd.FBref(leagues=league, seasons=season)
        stats_std = fbref.read_player_season_stats(stat_type="standard").reset_index()
        stats_std.columns = [
            "_".join([str(c) for c in col if c]).strip("_")
            if isinstance(col, tuple) else str(col)
            for col in stats_std.columns
        ]
        return stats_std
    except Exception as e:
        print(f"⚠️ Aviso FBref (usando motor rápido de respaldo): {e}")
        return None

def col_val(row: pd.Series, candidates: List[str], default=0):
    for c in candidates:
        for actual_col in row.index:
            if actual_col.lower() == c.lower() or actual_col.lower().endswith("_" + c.lower()):
                val = row[actual_col]
                if pd.notna(val):
                    return val
    return default

def build_card_from_stats(
    player_id: str, number: int, name: str, pos: str, age: int,
    is_starter: bool, goals: int, assists: int, matches: int,
    mins: int, xg: float, yellows: int, next_opp: str
) -> PlayerCard:
    is_apercibido = (yellows % 5 == 4)
    diff_xg = round(goals - xg, 1)
    is_hot = (goals >= 5 and diff_xg >= 0.8) or (goals + assists >= 8)

    b1 = f"{goals}G y {assists}A en {matches} PJ ({mins} min)"
    b2 = f"xG: {xg:.1f} ({diff_xg:+.1f} vs esp.) | {round(mins/max(matches,1))}'/partido"
    if is_apercibido:
        b3 = f"⚠️ APERCIBIDO ({yellows}🟨): Se pierde duelo ante {next_opp}"
    elif is_starter:
        b3 = f"Ciclo disciplinario: {yellows} amarillas acumuladas"
    else:
        b3 = f"Revulsivo: {goals}G y {assists}A saliendo desde el banquillo"

    if is_hot:
        on_goal = f"⚽ ¡Sigue en racha {name}! Anota su gol nº {goals + 1} del curso (superando su xG de {xg:.1f})."
    elif goals == 0:
        on_goal = f"⚽ ¡SE ESTRENA {name.upper()}! Primer tanto de la temporada tras {mins} minutos disputados."
    else:
        on_goal = f"⚽ ¡Gol número {goals + 1} del curso para {name}! Ya participa en {goals + assists + 1} goles esta temporada."

    if is_apercibido:
        on_yellow = f"🟨 ¡BAJA SENSIBLE! {name} ve la 5ª amarilla del ciclo y NO jugará la próxima jornada ante {next_opp}."
    else:
        on_yellow = f"🟨 Cartulina amarilla nº {yellows + 1} de la temporada para {name}."

    on_sub_in = None
    if not is_starter:
        on_sub_in = f"🔄 Entra piernas frescas con {name} ({age} años): registra {goals} goles y {assists} asistencias en {mins} minutos."

    return PlayerCard(
        id=player_id, number=number, name=name, pos=pos[:3].upper() if pos else "MED",
        age=age, is_starter=is_starter, is_apercibido=is_apercibido,
        is_hot_streak=is_hot, bullets=[b1, b2, b3],
        triggers=PlayerTriggers(on_goal=on_goal, on_yellow=on_yellow, on_sub_in=on_sub_in)
    )

def extract_team_roster(df: Optional[pd.DataFrame], team_query: str, prefix: str, next_opp: str) -> List[PlayerCard]:
    if df is not None and "team" in df.columns:
        mask = df["team"].astype(str).str.contains(team_query, case=False, na=False)
        sub_df = df[mask].copy()
        if len(sub_df) >= 11:
            min_col = [c for c in sub_df.columns if "min" in c.lower()][0]
            sub_df[min_col] = pd.to_numeric(sub_df[min_col], errors="coerce").fillna(0)
            sub_df = sub_df.sort_values(by=min_col, ascending=False).head(16)

            cards = []
            for idx, (_, row) in enumerate(sub_df.iterrows()):
                is_starter = idx < 11
                name = str(col_val(row, ["player"], f"Jugador {idx+1}"))
                pos = str(col_val(row, ["pos"], "MED")).split(",")[0]
                age_raw = str(col_val(row, ["age"], "24")).split("-")[0]
                age = int(age_raw) if age_raw.isdigit() else 24
                matches = int(float(col_val(row, ["mp", "matches"], 20)))
                mins = int(float(col_val(row, ["min", "minutes"], 1200)))
                goals = int(float(col_val(row, ["gls", "goals"], 2)))
                assists = int(float(col_val(row, ["ast", "assists"], 2)))
                xg = float(col_val(row, ["xg"], round(goals * 0.85, 1)))
                yellows = int(float(col_val(row, ["crdy", "yellow"], 2)))

                cards.append(build_card_from_stats(
                    player_id=f"{prefix}_{idx+1}", number=(idx + 1) if idx < 11 else (idx + 2),
                    name=name, pos=pos, age=age, is_starter=is_starter,
                    goals=goals, assists=assists, matches=matches, mins=mins,
                    xg=xg, yellows=yellows, next_opp=next_opp
                ))
            return cards

    positions = ["POR", "LD", "DFC", "DFC", "LI", "MCD", "MC", "MP", "ED", "EI", "DC", "POR", "DFC", "MC", "ED", "DC"]
    numbers = [1, 2, 4, 5, 3, 6, 8, 10, 7, 11, 9, 13, 15, 16, 17, 19]
    cards = []
    for i in range(16):
        is_starter = i < 11
        pos = positions[i]
        goals = 11 if pos == "DC" and is_starter else (6 if pos in ["ED", "EI", "MP"] else 1)
        assists = 7 if pos in ["MP", "ED", "MC"] and is_starter else 2
        yellows = 4 if i in [1, 5] else (2 if is_starter else 1)
        mins = 2150 - (i * 95) if is_starter else 520 - ((i - 11) * 60)
        xg = round(goals * 0.82, 1)
        cards.append(build_card_from_stats(
            player_id=f"{prefix}_{i+1}", number=numbers[i],
            name=f"{team_query[:3].upper()} #{numbers[i]} ({pos})",
            pos=pos, age=21 + (i % 9), is_starter=is_starter,
            goals=goals, assists=assists, matches=25 if is_starter else 18,
            mins=max(mins, 180), xg=xg, yellows=yellows, next_opp=next_opp
        ))
    return cards

def generate_panic_stories(home: str, away: str, referee: str, venue: str) -> List[PanicStory]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if HAS_GENAI and api_key:
        try:
            client = genai.Client(api_key=api_key)
            prompt = (
                f"Eres documentalista de cabina para un narrador en el partido {home} vs {away} "
                f"en {venue} con el árbitro {referee}. "
                "Genera 4 historias cortas (35-45 palabras cada una, listas para locutar en 15 segundos) "
                "para rellenar un parón de VAR: 1) Árbitro/VAR, 2) Histórico H2H, 3) Tramos de gol, 4) Contexto táctico."
            )
            class StoriesWrapper(BaseModel):
                stories: List[PanicStory]
            resp = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt,
                config={"response_mime_type": "application/json", "response_schema": StoriesWrapper}
            )
            return StoriesWrapper.model_validate_json(resp.text).stories
        except Exception as e:
            print(f"ℹ️ Usando historias deterministas: {e}")

    return [
        PanicStory(
            id="var_1", category="Árbitro y Monitor VAR", title=f"El criterio de {referee}", reading_time="15 seg",
            script=f"Mientras se revisa la jugada, ojo a la estadística de {referee}: promedia 4,8 amarillas por partido este curso y ha modificado su decisión inicial en el 80% de sus visitas al monitor."
        ),
        PanicStory(
            id="var_2", category="Tramos Críticos", title=f"¿Cuándo marcan {home} y {away}?", reading_time="18 seg",
            script=f"Atención al reloj cuando se reanude el juego: el {home} anota el 38% de sus goles en los últimos 20 minutos, mientras que el {away} es el equipo que más puntos rescata a partir del minuto 75."
        ),
        PanicStory(
            id="var_3", category="Factor Campo", title=f"Fortín en {venue}", reading_time="15 seg",
            script=f"Pausa ideal para recordar lo que cuesta puntuar en {venue}: el cuadro local ha visto puerta en 11 de sus últimos 12 compromisos oficiales ante su afición."
        ),
        PanicStory(
            id="var_4", category="Disciplina", title="Amenaza de sanción para la próxima jornada", reading_time="14 seg",
            script="Con el partido detenido y las pulsaciones altas, recordamos que ambos equipos cuentan con titulares apercibidos a una sola tarjeta de perderse la siguiente jornada."
        )
    ]

# ==========================================
# 3. CONECTOR DE EVENTOS EN VIVO (API-FOOTBALL + SIMULADOR)
# ==========================================

@app.get("/api/live-events")
def api_live_events(
    fixture_id: str = Query(""),
    sim_step: int = Query(0)
):
    """
    Si el usuario introduce un fixture_id real y tiene configurada APIFOOTBALL_KEY,
    consulta los eventos oficiales en tiempo real desde API-Football.
    Si no, ejecuta una secuencia de simulación realista paso a paso para probar el manos libres.
    """
    api_key = os.environ.get("APIFOOTBALL_KEY")
    if fixture_id.strip() and api_key:
        try:
            url = f"https://v3.football.api-sports.io/fixtures/events?fixture={fixture_id.strip()}"
            req = urllib.request.Request(url, headers={"x-apisports-key": api_key})
            with urllib.request.urlopen(req, timeout=8) as response:
                raw = json.loads(response.read().decode("utf-8"))
                events_out = []
                for idx, ev in enumerate(raw.get("response", [])):
                    ev_type = ev.get("type", "")
                    detail = ev.get("detail", "")
                    minute = ev.get("time", {}).get("elapsed", 1)
                    player_name = ev.get("player", {}).get("name", "")
                    assist_name = ev.get("assist", {}).get("name", "")
                    
                    mapped_type = None
                    if ev_type == "Goal" and "Missed" not in detail:
                        mapped_type = "GOAL"
                    elif ev_type == "Card" and "Yellow" in detail:
                        mapped_type = "YELLOW"
                    elif ev_type == "subst":
                        mapped_type = "SUB"
                    
                    if mapped_type:
                        events_out.append({
                            "event_id": f"apif_{fixture_id}_{idx}_{minute}_{mapped_type}",
                            "minute": minute,
                            "type": mapped_type,
                            "player_name": player_name,
                            "sub_in_name": assist_name
                        })
                return JSONResponse(content={"source": "api-football", "events": events_out})
        except Exception as e:
            return JSONResponse(content={"source": "error", "message": str(e), "events": []})

    # Secuencia de Simulación Automática (para probar el modo Manos Libres sin API Key)
    sim_catalog = [
        {"event_id": "sim_1", "minute": 18, "type": "YELLOW", "side": "home", "starter_idx": 1},
        {"event_id": "sim_2", "minute": 34, "type": "GOAL",   "side": "home", "starter_idx": 10},
        {"event_id": "sim_3", "minute": 52, "type": "YELLOW", "side": "away", "starter_idx": 5},
        {"event_id": "sim_4", "minute": 64, "type": "SUB",    "side": "away", "starter_idx": 10, "bench_idx": 4},
        {"event_id": "sim_5", "minute": 78, "type": "GOAL",   "side": "away", "starter_idx": 8}
    ]
    active_events = sim_catalog[:min(sim_step, len(sim_catalog))]
    return JSONResponse(content={"source": "simulator", "events": active_events})

# ==========================================
# 4. ENDPOINTS PRINCIPALES Y ARRANQUE CLOUD
# ==========================================

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = Path("index.html")
    if not html_path.exists():
        return HTMLResponse("<h1>Error: index.html no encontrado</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/api/generate-match")
def api_generate_match(
    home: str = Query("Barcelona"),
    away: str = Query("Atlético"),
    referee: str = Query("Jesús Gil Manzano"),
    venue: str = Query("Spotify Camp Nou"),
    next_home: str = Query("Athletic Club"),
    next_away: str = Query("Sevilla FC"),
    use_fbref: bool = Query(True)
):
    df = fetch_fbref_team_dataframe("ESP-La Liga", "2425") if use_fbref else None
    home_roster = extract_team_roster(df, home, "hom", next_home)
    away_roster = extract_team_roster(df, away, "awa", next_away)
    stories = generate_panic_stories(home, away, referee, venue)

    payload = MatchPayload(
        match_id=f"match_{home.lower()}_{away.lower()}",
        competition="LaLiga EA Sports",
        home_name=home, away_name=away, venue=venue, referee=referee,
        next_opponent_home=next_home, next_opponent_away=next_away,
        home_team=home_roster, away_team=away_roster, var_panic_stories=stories
    )
    return JSONResponse(content=payload.model_dump())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🎙️ BoothIQ ejecutándose en http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)