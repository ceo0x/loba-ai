# Arena API - Guia completa para integrar un agente

Este documento explica como conectar un agente externo (cualquier lenguaje) al modo Arena de Loba AI.

Objetivo: que tu hermano pueda correr su bot por API y competir, **sin ver tus cartas**, en formato **partido a puntos** (default 100).

## 1) Conceptos clave

- `room_id`: identificador de sala.
- `player_token`: credencial privada de jugador dentro de esa sala.
- Cada request de juego usa `player_token` para autorizar y filtrar informacion.
- El estado del juego se consulta por `GET /arena/rooms/{room_id}/state`.
- Las jugadas se envian por `POST /arena/rooms/{room_id}/action`.
- La sala acumula score entre rondas hasta llegar a `target_points`.

## 2) Privacidad (muy importante)

El endpoint de estado aplica filtrado por token:

- Tu agente solo ve su propia mano en `players[mi_index].hand_full`.
- Las manos de otros jugadores llegan vacias (`[]`).
- Solo se entregan `valid_actions` si realmente es tu turno.

Si el agente no tiene turno:

- `valid_actions` viene vacio.
- Debe esperar y volver a consultar estado.

## 3) Flujo completo de partida (Arena)

### Paso A - Host crea sala

`POST /arena/rooms`

Body:

```json
{
  "host_model_name": "mi-bot",
  "room_name": "duelo-hermanos",
  "target_points": 100
}
```

Respuesta ejemplo:

```json
{
  "room": {
    "room_id": "abc123",
    "status": "waiting"
  },
  "host_player_token": "HOST_TOKEN"
}
```

### Paso B - Invitado hace join

`POST /arena/rooms/{room_id}/join`

Body:

```json
{
  "guest_name": "hermano-agent"
}
```

Respuesta ejemplo:

```json
{
  "room": {
    "room_id": "abc123",
    "status": "waiting"
  },
  "guest_player_token": "GUEST_TOKEN",
  "player_index": 1
}
```

### Paso C - Ambos marcan ready

`POST /arena/rooms/{room_id}/ready`

Body:

```json
{
  "player_token": "GUEST_TOKEN"
}
```

Cuando los dos estan ready, la sala pasa a `status: "in_game"` y se crea la primera ronda.

### Paso D - Loop de juego del agente

1. Leer estado (`/state`).
2. Si `room.match_finished == true`, terminar (el partido completo termino).
3. Si `valid_actions` esta vacio, dormir 200-500ms y repetir.
4. Elegir una accion valida.
5. Enviar accion (`/action`).
6. Repetir.

Regla de puntaje implementada:

- Al terminar una ronda, el ganador suma los puntos de cartas restantes del rival.
- Si nadie llega a `target_points`, comienza automaticamente la siguiente ronda.
- Gana el partido quien llegue primero al objetivo.

## 4) Endpoints Arena (referencia)

## `POST /arena/rooms`

Crea sala de arena.

Campos:

- `host_model_name` (string, requerido): modelo del host (debe estar registrado).
- `room_name` (string, opcional).
- `target_points` (int, opcional): objetivo del partido. Default `100`.

Errores comunes:

- `400 Unknown model '...'`

## `GET /arena/rooms/{room_id}`

Devuelve metadata de sala (sin estado de cartas).

## `POST /arena/rooms/{room_id}/join`

Hace join del invitado y entrega `guest_player_token`.

Body:

```json
{
  "guest_name": "nombre-visible"
}
```

Errores comunes:

- `400 Room already has a guest`
- `400 Match already started for this room`

## `POST /arena/rooms/{room_id}/ready`

Marca jugador como listo.

Body:

```json
{
  "player_token": "TOKEN"
}
```

Errores comunes:

- `403 Invalid player token`

## `GET /arena/rooms/{room_id}/state?player_token=...`

Devuelve estado de partida filtrado por token.

Campos utiles del response:

- `state.phase`: fase actual (`draw`, `meld`, `discard`)
- `state.current_player`: index del jugador con turno
- `state.players`: info de jugadores (mano propia visible, ajena no)
- `state.valid_actions`: acciones permitidas para este token (solo si tiene turno)
- `state.finished`: partida terminada o no
- `state.winner`: index ganador (cuando termina)
- `room.scores`: puntaje acumulado `[host, guest]`
- `room.target_points`: meta del partido
- `room.round_number`: ronda actual
- `room.match_finished`: si ya termino la serie completa
- `room.champion`: ganador final (`0` o `1`) cuando termina
- `room.last_round_result`: resumen de la ultima ronda cerrada

## `POST /arena/rooms/{room_id}/action`

Envia jugada.

Body:

```json
{
  "player_token": "TOKEN",
  "action": 0
}
```

Errores comunes:

- `400 Not your turn`
- `400 Invalid action`
- `403 Invalid player token`

## 5) Mapeo exacto de acciones

La accion es un entero. El significado depende de la fase.

Constante interna:

- `MAX_MELD_ACTIONS = 24`

## Fase `draw`

- `0` = robar del mazo (`Draw stock`)
- `1` = robar del descarte (`Draw discard`) si las reglas lo permiten

## Fase `meld`

- `2` = pasar sin bajar (`Skip meld`)
- `3..26` = bajar meld candidato
  - `3` corresponde al meld #0
  - `4` corresponde al meld #1
  - ...
  - maximo 24 candidatos por turno

En `valid_actions`, el backend ya devuelve label legible tipo:

- `Play meld 7H-8H-9H`

Recomendacion: elegir por label/heuristica y enviar el `id`.

## Fase `discard`

- base descarte = `3 + MAX_MELD_ACTIONS = 27`
- `27 + i` = descartar carta en posicion `i` de la mano actual

Ejemplo:

- accion `27` descarta la carta en indice de mano `0`
- accion `31` descarta la carta en indice de mano `4`

El estado tambien trae:

- `discard_options`: mapeo directo `{ action_id, hand_index, card }`

Esto facilita elegir descarte sin recalcular indices.

## 6) Como interpretar el estado

Campos tipicos en `state`:

- `phase`: fase actual (`draw` / `meld` / `discard`)
- `turn`: numero de turno
- `current_player`: jugador que actua ahora
- `stock_size`: cartas restantes en mazo
- `top_discard`: tope del descartel
- `discard_last_three`: ultimas 3 cartas del descarte
- `table_melds_grouped`: melds abiertos por jugador
- `players[*].cards_in_hand`: cantidad de cartas de cada jugador
- `players[*].hand_full`: cartas visibles (solo propias)
- `valid_actions`: lista de acciones legales para el jugador/token

Campos tipicos en `room`:

- `scores`: acumulado del partido
- `target_points`: objetivo para ganar
- `round_number`: ronda activa
- `last_round_result`: `{ round_number, winner, points_earned, scores }`
- `match_finished`: true cuando el partido termina
- `champion`: jugador ganador final

## 7) Estrategia minima recomendada para tu hermano

Implementar un loop robusto:

1. Poll estado cada 300ms.
2. Si no hay `valid_actions`, esperar (no forzar action).
3. Si hay acciones:
   - en `draw`, elegir `1` solo si su policy quiere descarte, sino `0`
   - en `meld`, priorizar acciones `>=3 && <27` si quiere abrir/bajar
   - en `discard`, elegir via `discard_options`
4. Enviar action.
5. Si recibe `400 Not your turn`, volver a poll estado.

## 8) Ejemplos curl rapidos

### Estado

```bash
curl "http://192.168.4.31:8000/arena/rooms/abc123/state?player_token=GUEST_TOKEN"
```

### Jugar accion

```bash
curl -X POST "http://192.168.4.31:8000/arena/rooms/abc123/action" \
  -H "Content-Type: application/json" \
  -d '{"player_token":"GUEST_TOKEN","action":0}'
```

## 9) Contrato para agentes externos

Para competir, el agente externo solo necesita:

- Base URL (ej: `http://192.168.4.31:8000`)
- `room_id`
- `player_token`

Y luego implementar:

- `get_state()`
- `choose_action(state.valid_actions, state.phase, state)`
- `play_action(action_id)`

Con eso ya puede jugar una partida completa en Arena.

## 10) Integración remota por WebSocket

Si querés jugar contra un servidor externo (protocolo remoto de tu hermano), revisar:

- `remote_api.md`
