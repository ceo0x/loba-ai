# Remote API - jugar contra protocolo WebSocket externo

Este documento describe el modo `/remote`, pensado para conectar tu backend local al servidor remoto de tu hermano y jugar como cliente bot.

## 1) Flujo rápido

1. Registrar modelo local en `/models/register`.
2. Crear sesión remota (`POST /remote/sessions`).
3. Iniciar sesión (`POST /remote/sessions/{id}/start`).
4. Consultar estado y eventos (`GET /remote/sessions/{id}`, `GET /remote/sessions/{id}/events`).
5. Detener sesión si hace falta (`POST /remote/sessions/{id}/stop`).

La UI está en:

- `GET /remote`

## 2) Crear sesión

`POST /remote/sessions`

Body:

```json
{
  "url": "ws://192.168.1.50:8765/",
  "name": "brother-bot",
  "token": "OPTIONAL_SECRET",
  "model_name": "mi-bot-local"
}
```

Notas:

- `model_name` debe existir en `/models`.
- Si no existe, la API responde `400`.

## 3) Iniciar / detener

Start:

```bash
curl -X POST http://127.0.0.1:8000/remote/sessions/<SESSION_ID>/start
```

Stop:

```bash
curl -X POST http://127.0.0.1:8000/remote/sessions/<SESSION_ID>/stop
```

## 4) Estado y eventos

Estado:

`GET /remote/sessions/{id}`

Campos relevantes:

- `status`: `created | connecting | running | stopping | stopped | finished | error`
- `connected`: si hay conexión WS activa
- `seat`: asiento asignado por el servidor remoto
- `actions_sent`, `observations_seen`
- `last_error`, `last_round_end`, `last_match_end`

Eventos:

`GET /remote/sessions/{id}/events`

Lista cronológica invertida de eventos del adaptador (`created`, `register_sent`, `registered`, `recv`, `action_sent`, `round_end`, `match_end`, `remote_error`, etc.).

## 5) Selección de acciones

El adaptador:

- recibe `obs.legal_actions` del protocolo remoto,
- intenta mapearlas al espacio de acciones local para consultar el modelo,
- elige una acción legal y la manda como `{"type":"act","action": ...}`.

Regla crítica:

- La acción enviada se toma de `legal_actions` (misma estructura JSON) para evitar errores de igualdad estructural.

## 6) Compatibilidad esperada

- Tu hermano mantiene el servidor WS (`register`, `observation`, `act`, `round_end`, `match_end`).
- Nuestro backend opera como cliente y no requiere cambiar su protocolo.

## 7) Ver también

- Arena local/tokenizado: `arena_api.md`
- Especificación remota de tu hermano: `REMOTE_PLAY_VS_BRO.md`
