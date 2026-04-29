const arenaState = {
  roomId: null,
  hostToken: null,
  matchState: null,
};

async function arenaApi(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

function arenaLog(msg) {
  const log = document.getElementById("arenaLog");
  const row = document.createElement("div");
  row.className = "log-item";
  row.textContent = `${new Date().toLocaleTimeString()} - ${msg}`;
  log.prepend(row);
}

function arenaRenderModels(items) {
  const holder = document.getElementById("arenaModelList");
  holder.innerHTML = "";
  items.forEach((m) => {
    const row = document.createElement("div");
    row.className = "model-row";
    row.innerHTML = `<span class="pill">${m.name} -> ${m.path}</span>`;
    holder.appendChild(row);
  });
}

async function arenaRefreshModels() {
  const data = await arenaApi("/models");
  arenaRenderModels(data.items || []);
}

function renderRoomMeta(room) {
  const out = document.getElementById("arenaHostRoomMeta");
  out.textContent = room
    ? `Room ${room.room_id} | estado: ${room.status} | ronda: ${room.round_number} | score host-guest: ${room.scores?.[0] ?? 0}-${room.scores?.[1] ?? 0} / ${room.target_points} | guest: ${room.guest_name || "esperando"}`
    : "";
}

function renderArenaState(matchState) {
  arenaState.matchState = matchState;
  const target = document.getElementById("arenaStateSummary");
  if (!matchState) {
    target.textContent = "Sin partida iniciada.";
    document.getElementById("arenaActions").innerHTML = "";
    return;
  }

  target.textContent = `Partida ${matchState.match_id} | turno ${matchState.turn} | current ${matchState.current_player} | finished: ${matchState.finished ? "si" : "no"}`;

  const actions = document.getElementById("arenaActions");
  actions.innerHTML = "";
  (matchState.valid_actions || []).forEach((a) => {
    const btn = document.createElement("button");
    btn.textContent = `${a.id}: ${a.label}`;
    btn.type = "button";
    btn.onclick = () => arenaPlayAction(a.id);
    actions.appendChild(btn);
  });
}

async function arenaPlayAction(actionId) {
  if (!arenaState.roomId || !arenaState.hostToken) return;
  try {
    const data = await arenaApi(`/arena/rooms/${arenaState.roomId}/action`, {
      method: "POST",
      body: JSON.stringify({ player_token: arenaState.hostToken, action: actionId }),
    });
    renderRoomMeta(data.room);
    renderArenaState(data.state);
    arenaLog(`Accion ejecutada: ${actionId}`);
      if (data.room?.last_round_result) {
        const rr = data.room.last_round_result;
        arenaLog(`Ronda ${rr.round_number}: gano P${rr.winner} (+${rr.points_earned}) | score ${rr.scores[0]}-${rr.scores[1]}`);
      }
      if (data.room?.match_finished) {
        arenaLog(`Partido finalizado. Campeon: P${data.room.champion}`);
      }
  } catch (err) {
    arenaLog(`Error accion: ${err.message}`);
  }
}

async function arenaRefreshState() {
  if (!arenaState.roomId || !arenaState.hostToken) return;
  try {
    const data = await arenaApi(
      `/arena/rooms/${arenaState.roomId}/state?player_token=${encodeURIComponent(arenaState.hostToken)}`
    );
    renderRoomMeta(data.room);
    renderArenaState(data.state);
    if (data.room?.last_round_result) {
      const rr = data.room.last_round_result;
      arenaLog(`Ronda ${rr.round_number}: gano P${rr.winner} (+${rr.points_earned}) | score ${rr.scores[0]}-${rr.scores[1]}`);
    }
    if (data.room?.match_finished) {
      arenaLog(`Partido finalizado. Campeon: P${data.room.champion}`);
    }
  } catch (err) {
    arenaLog(`Error refrescando estado: ${err.message}`);
  }
}

async function arenaInit() {
  try {
    await arenaApi("/health");
    const badge = document.getElementById("arenaHealthBadge");
    badge.textContent = "API: OK";
    badge.classList.add("ok");
  } catch {
    document.getElementById("arenaHealthBadge").textContent = "API: offline";
  }

  await arenaRefreshModels();

  document.getElementById("arenaRegisterModelBtn").onclick = async () => {
    const name = document.getElementById("arenaModelName").value.trim();
    const path = document.getElementById("arenaModelPath").value.trim();
    if (!name || !path) {
      arenaLog("Completa nombre y path.");
      return;
    }
    try {
      await arenaApi("/models/register", {
        method: "POST",
        body: JSON.stringify({ name, path }),
      });
      arenaLog(`Modelo ${name} registrado.`);
      await arenaRefreshModels();
      document.getElementById("arenaHostModelName").value = name;
    } catch (err) {
      arenaLog(`Error registrando modelo: ${err.message}`);
    }
  };

  document.getElementById("arenaCreateRoomBtn").onclick = async () => {
    const roomName = document.getElementById("arenaRoomName").value.trim();
    const hostModelName = document.getElementById("arenaHostModelName").value.trim();
    if (!hostModelName) {
      arenaLog("Debes indicar modelo host.");
      return;
    }
    try {
      const data = await arenaApi("/arena/rooms", {
        method: "POST",
        body: JSON.stringify({
          room_name: roomName || null,
          host_model_name: hostModelName,
        }),
      });
      arenaState.roomId = data.room.room_id;
      arenaState.hostToken = data.host_player_token;
      renderRoomMeta(data.room);
      arenaLog(`Sala creada: ${arenaState.roomId}`);
      arenaLog(`Token host: ${arenaState.hostToken}`);
      arenaLog("Comparte room_id con tu hermano para que haga join via API.");
    } catch (err) {
      arenaLog(`Error creando sala: ${err.message}`);
    }
  };

  document.getElementById("arenaHostReadyBtn").onclick = async () => {
    if (!arenaState.roomId || !arenaState.hostToken) {
      arenaLog("Primero crea sala.");
      return;
    }
    try {
      const data = await arenaApi(`/arena/rooms/${arenaState.roomId}/ready`, {
        method: "POST",
        body: JSON.stringify({ player_token: arenaState.hostToken }),
      });
      renderRoomMeta(data.room);
      renderArenaState(data.state || null);
      arenaLog("Host marcado como ready.");
      if (data.room?.target_points) {
        arenaLog(`Partido a ${data.room.target_points} puntos.`);
      }
    } catch (err) {
      arenaLog(`Error en ready: ${err.message}`);
    }
  };

  document.getElementById("arenaRefreshStateBtn").onclick = arenaRefreshState;
}

arenaInit().catch((err) => {
  console.error("[arena] Fatal startup error:", err);
});
