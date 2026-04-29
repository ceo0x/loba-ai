const remoteState = {
  sessionId: null,
  previousHand: [],
  lastActionSent: null,
  highlightDiscardUntil: 0,
  highlightDrawUntil: 0,
  highlightDiscardCardKey: null,
  highlightDrawCardKey: null,
  highlightDiscardTop: false,
};
const REMOTE_WS_URL_KEY = "loba_remote_ws_url";
const REMOTE_DEFAULT_WS_URL = "ws://192.168.4.20:8765";
const CARD_SPRITE_PATH = "/web/assets/cards/svg-cards.svg";

function randomSeed() {
  return Math.floor(Math.random() * 2147483647);
}

async function remoteApi(path, options = {}) {
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

function remoteLog(message) {
  const log = document.getElementById("remoteLog");
  const row = document.createElement("div");
  row.className = "log-item";
  row.textContent = `${new Date().toLocaleTimeString()} - ${message}`;
  log.prepend(row);
}

function getSavedWsUrl() {
  try {
    return localStorage.getItem(REMOTE_WS_URL_KEY) || REMOTE_DEFAULT_WS_URL;
  } catch {
    return REMOTE_DEFAULT_WS_URL;
  }
}

function saveWsUrl(url) {
  try {
    localStorage.setItem(REMOTE_WS_URL_KEY, url);
  } catch {
    /* ignore localStorage failures */
  }
}

function summarizeEvent(ev) {
  const data = ev?.data || {};
  const eventType = ev?.type || "event";
  if (eventType === "remote_error") {
    const payload = data.payload || {};
    return `${eventType} | code=${payload.code || "?"} | message=${payload.message || "?"}`;
  }
  if (eventType === "register_failed") {
    const payload = data.payload || {};
    return `${eventType} | ${JSON.stringify(payload)}`;
  }
  if (eventType === "registered") {
    const payload = data.payload || {};
    return `${eventType} | seat=${payload.seat ?? "?"} | name=${payload.name || "?"}`;
  }
  if (eventType === "action_sent") {
    const hand = Array.isArray(data.hand) ? data.hand : [];
    return `${eventType} | action=${JSON.stringify(data.action || {})}${data.meta ? ` | meta=${JSON.stringify(data.meta)}` : ""}${hand.length ? ` | hand=${JSON.stringify(hand)}` : ""}`;
  }
  if (eventType === "recv") {
    const payload = data.payload || {};
    return `${eventType} | type=${payload.type || "unknown"}`;
  }
  if (eventType === "round_end") {
    const payload = data.payload || {};
    return `${eventType} | winner=${payload.winner} | cumulative=${JSON.stringify(payload.cumulative || [])}`;
  }
  if (eventType === "match_end") {
    const payload = data.payload || {};
    return `${eventType} | winner=${payload.winner}`;
  }
  return `${eventType}${Object.keys(data).length ? ` | ${JSON.stringify(data)}` : ""}`;
}

function eventHasFallback(ev) {
  if (!ev || ev.type !== "action_sent") return false;
  return Boolean(ev.data?.meta?.fallback);
}

function renderSession(session) {
  const meta = document.getElementById("remoteSessionMeta");
  if (!session) {
    meta.textContent = "Sin sesión.";
    return;
  }
  const m = session.decision_metrics || {};
  meta.textContent = `Session ${session.session_id} | status=${session.status} | connected=${session.connected} | seat=${session.seat ?? "-"} | obs=${session.observations_seen} | acts=${session.actions_sent} | model=${m.model_decisions ?? 0} | fallback=${m.fallback_decisions ?? 0} | goOutAvail=${m.go_out_available_observations ?? 0} | goOutMissed=${m.go_out_missed ?? 0}`;
  renderRemoteTable(session.last_observation, session);
}

function renderMockStatus(payload) {
  const meta = document.getElementById("remoteMockMeta");
  const mock = payload?.mock;
  if (!meta || !mock) return;
  meta.textContent = `Mock running=${mock.running} | connected=${mock.connected} | ws=${payload.ws_url} | players=${mock.num_players} | seat=${mock.seat}`;
}

function remoteCardToLabel(card) {
  if (!card || typeof card !== "object") return "";
  if (card.joker) return "Joker";
  return `${card.rank || "?"}${card.suit || "?"}`;
}

function remoteCardKey(card) {
  if (!card || typeof card !== "object") return "";
  if (card.joker) return `J|${card.deck_id ?? "?"}`;
  return `${card.rank || "?"}|${card.suit || "?"}|${card.deck_id ?? "?"}`;
}

function cardLabelToSpriteId(label, hidden = false) {
  if (hidden || label === "hidden") return "back";
  if (!label) return "back";
  if (label === "Joker") return "joker_black";
  const suitCode = label.slice(-1);
  const rankCode = label.slice(0, -1);
  const suitMap = { C: "club", D: "diamond", H: "heart", S: "spade" };
  const rankMap = { A: "1", J: "jack", Q: "queen", K: "king" };
  const suit = suitMap[suitCode];
  if (!suit) return "back";
  const rank = rankMap[rankCode] || rankCode;
  return `${suit}_${rank}`;
}

function cardHTML(label, hidden = false, extraClass = "") {
  const spriteId = cardLabelToSpriteId(label, hidden);
  const title = hidden ? "Carta oculta" : (label || "Carta");
  return `
    <div class="playing-card ${hidden ? "back" : ""} ${extraClass}">
      <svg class="playing-card-svg" viewBox="0 0 169.075 244.64" role="img" aria-label="${title}">
        <use href="${CARD_SPRITE_PATH}#${spriteId}" xlink:href="${CARD_SPRITE_PATH}#${spriteId}"></use>
      </svg>
    </div>
  `;
}

function cardHTMLWithData(card, extraClass = "") {
  const label = remoteCardToLabel(card);
  const spriteId = cardLabelToSpriteId(label, false);
  const title = label || "Carta";
  const dataKey = remoteCardKey(card);
  const rank = card?.joker ? "" : (card?.rank ?? "");
  const suit = card?.joker ? "" : (card?.suit ?? "");
  const deckId = card?.deck_id ?? "";
  const joker = card?.joker ? "1" : "0";
  return `
    <div class="playing-card ${extraClass}" data-card-key="${dataKey}" data-rank="${rank}" data-suit="${suit}" data-deck-id="${deckId}" data-joker="${joker}">
      <svg class="playing-card-svg" viewBox="0 0 169.075 244.64" role="img" aria-label="${title}">
        <use href="${CARD_SPRITE_PATH}#${spriteId}" xlink:href="${CARD_SPRITE_PATH}#${spriteId}"></use>
      </svg>
    </div>
  `;
}

function updateHighlightsFromObservation(obs) {
  const now = Date.now();
  const hand = Array.isArray(obs?.hand) ? obs.hand : [];
  const currentHandKeys = hand.map(remoteCardKey);
  const previousHandKeys = remoteState.previousHand.map(remoteCardKey);

  if (remoteState.lastActionSent?.type === "Discard" || remoteState.lastActionSent?.type === "Cruzar") {
    const card = remoteState.lastActionSent.card;
    const key = remoteCardKey(card);
    remoteState.highlightDiscardUntil = now + 500;
    remoteState.highlightDiscardCardKey = key || null;
    remoteState.highlightDiscardTop = true;
  }

  if (remoteState.lastActionSent?.type === "DrawStock" || remoteState.lastActionSent?.type === "DrawDiscard") {
    const prevSet = new Set(previousHandKeys);
    const drawn = currentHandKeys.find((k) => !prevSet.has(k));
    remoteState.highlightDrawUntil = now + 500;
    remoteState.highlightDrawCardKey = drawn || null;
  }

  remoteState.previousHand = hand.slice();
  remoteState.lastActionSent = null;
}

function renderRemoteTable(obs, session) {
  const summary = document.getElementById("remoteStateSummary");
  const stockCount = document.getElementById("remoteStockCount");
  const stockVisual = document.getElementById("remoteStockVisual");
  const discardTop = document.getElementById("remoteDiscardTop");
  const players = document.getElementById("remotePlayersSummary");
  const melds = document.getElementById("remoteMelds");
  const ownHand = document.getElementById("remoteOwnHand");

  if (!obs) {
    summary.innerHTML = "<span>Sin observation aún.</span>";
    stockCount.textContent = "0";
    stockVisual.innerHTML = "";
    discardTop.innerHTML = "";
    players.innerHTML = "";
    melds.innerHTML = "<span class='muted'>Sin datos.</span>";
    ownHand.innerHTML = "";
    return;
  }
  updateHighlightsFromObservation(obs);
  const now = Date.now();

  const mySeat = Number(obs.seat ?? session?.seat ?? 0);
  const phase = obs.phase || "draw";
  const legalCount = Array.isArray(obs.legal_actions) ? obs.legal_actions.length : 0;
  const cumulative = Array.isArray(obs.cumulative_scores) ? obs.cumulative_scores : [];
  const reenganches = Array.isArray(obs.reenganches_used) ? obs.reenganches_used : [];
  const eliminated = Array.isArray(obs.eliminated) ? obs.eliminated : [];

  summary.innerHTML = `
    <span>Seat: <strong>${mySeat}</strong></span>
    <span>Jugadores: <strong>${obs.num_players ?? "?"}</strong></span>
    <span>Fase: <strong>${phase}</strong></span>
    <span>Acciones legales: <strong>${legalCount}</strong></span>
    <span>Pendiente descarte: <strong>${obs.pending_discard ? remoteCardToLabel(obs.pending_discard) : "no"}</strong></span>
  `;

  const stockSize = Number(obs.stock_size || 0);
  stockCount.textContent = String(stockSize);
  const stockCards = Math.min(5, Math.max(1, stockSize > 0 ? Math.ceil(stockSize / 12) : 0));
  stockVisual.innerHTML = stockSize > 0 ? new Array(stockCards).fill(null).map(() => cardHTML("hidden", true)).join("") : "<span class='muted'>Vacío</span>";

  const discardPulse = remoteState.highlightDiscardTop && now < remoteState.highlightDiscardUntil ? "remote-discard-flash" : "";
  discardTop.innerHTML = obs.discard_top ? cardHTML(remoteCardToLabel(obs.discard_top), false, `discard-top ${discardPulse}`) : "<span class='muted'>Vacío</span>";
  if (now >= remoteState.highlightDiscardUntil) {
    remoteState.highlightDiscardTop = false;
  }

  const handSizes = Array.isArray(obs.other_hand_sizes) ? obs.other_hand_sizes : [];
  players.innerHTML = "";
  for (let seat = 0; seat < Number(obs.num_players || handSizes.length || 0); seat += 1) {
    const block = document.createElement("div");
    block.className = `player ${seat === mySeat ? "current" : ""}`;
    const seatName = seat === mySeat ? `${session?.config?.name || "You"} (you)` : `Seat ${seat}`;
    block.innerHTML = `
      <strong>${seatName}</strong>
      <div class="muted">cartas: ${handSizes[seat] ?? "?"} | score: ${cumulative[seat] ?? "?"}</div>
      <div class="muted">reenganche: ${reenganches[seat] ?? 0} | eliminado: ${eliminated[seat] ? "sí" : "no"}</div>
      <div class="cards-row">
        ${new Array(Math.min(10, Number(handSizes[seat] || 0))).fill(null).map(() => cardHTML("hidden", true)).join("") || "<span class='muted'>sin cartas</span>"}
      </div>
    `;
    players.appendChild(block);
  }

  const myCards = Array.isArray(obs.hand) ? obs.hand : [];
  ownHand.innerHTML = myCards.length
    ? myCards.map((card, idx) => {
      const key = remoteCardKey(card);
      const classes = ["remote-hand-card"];
      if (remoteState.highlightDiscardCardKey && key === remoteState.highlightDiscardCardKey && now < remoteState.highlightDiscardUntil) {
        classes.push("remote-card-discarding");
      }
      if (remoteState.highlightDrawCardKey && key === remoteState.highlightDrawCardKey && now < remoteState.highlightDrawUntil) {
        classes.push("remote-card-drawn");
      }
      return `<div class="remote-hand-slot" style="--remote-hand-index:${idx}">${cardHTMLWithData(card, classes.join(" "))}</div>`;
    }).join("")
    : "<span class='muted'>Sin cartas visibles.</span>";

  const tableMelds = Array.isArray(obs.melds_on_table) ? obs.melds_on_table : [];
  if (!tableMelds.length) {
    melds.innerHTML = "<span class='muted'>No hay melds en mesa.</span>";
  } else {
    const groupedByOwner = new Map();
    tableMelds.forEach((m) => {
      const owner = Number(m.owner ?? -1);
      if (!groupedByOwner.has(owner)) groupedByOwner.set(owner, []);
      groupedByOwner.get(owner).push(m);
    });

    const ownerBlocks = Array.from(groupedByOwner.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([owner, ownerMelds]) => `
        <div class="meld-owner remote-meld-owner">
          <div class="meld-owner-title">Seat ${owner}</div>
          <div class="remote-meld-owner-list">
            ${ownerMelds.map((m) => `
              <div class="remote-meld-entry">
                <div class="muted">Meld #${m.meld_id} | ${m.kind}</div>
                <div class="meld-item">${(m.cards || []).map((c) => cardHTML(remoteCardToLabel(c))).join("")}</div>
              </div>
            `).join("")}
          </div>
        </div>
      `);

    melds.innerHTML = `<div class="remote-melds-row">${ownerBlocks.join("")}</div>`;
  }
}

async function refreshRemote() {
  if (!remoteState.sessionId) return;
  try {
    const data = await remoteApi(`/remote/sessions/${remoteState.sessionId}`);
    renderSession(data.session);
    const events = await remoteApi(`/remote/sessions/${remoteState.sessionId}/events`);
    const log = document.getElementById("remoteLog");
    log.innerHTML = "";
    const items = (events.items || []).slice(0, 120);
    const latestActionSent = items.find((ev) => ev.type === "action_sent");
    if (latestActionSent?.data?.action) {
      remoteState.lastActionSent = latestActionSent.data.action;
    }

    items.forEach((ev) => {
      const row = document.createElement("div");
      row.className = `log-item${eventHasFallback(ev) ? " log-item-fallback" : ""}`;
      row.textContent = `${new Date(ev.ts * 1000).toLocaleTimeString()} - ${summarizeEvent(ev)}`;
      if (eventHasFallback(ev)) {
        const badge = document.createElement("span");
        badge.className = "log-fallback-badge";
        badge.textContent = "FALLBACK";
        row.appendChild(document.createTextNode(" "));
        row.appendChild(badge);
      }
      log.appendChild(row);
    });
  } catch (err) {
    remoteLog(`Error refrescando: ${err.message}`);
  }
}

async function initRemote() {
  try {
    await remoteApi("/health");
    const badge = document.getElementById("remoteHealthBadge");
    badge.textContent = "API: OK";
    badge.classList.add("ok");
  } catch {
    document.getElementById("remoteHealthBadge").textContent = "API: offline";
  }
  const wsInput = document.getElementById("remoteWsUrl");
  const mockHostInput = document.getElementById("remoteMockHost");
  const mockPortInput = document.getElementById("remoteMockPort");
  const mockNameInput = document.getElementById("remoteMockName");
  const mockSeedInput = document.getElementById("remoteMockSeed");
  wsInput.value = getSavedWsUrl();
  if (mockHostInput) mockHostInput.value = "127.0.0.1";
  if (mockPortInput) mockPortInput.value = "8766";
  if (mockNameInput) mockNameInput.value = "brother-bot";
  if (mockSeedInput) mockSeedInput.value = String(randomSeed());
  wsInput.addEventListener("change", () => {
    const trimmed = wsInput.value.trim();
    if (trimmed) saveWsUrl(trimmed);
  });

  document.getElementById("remoteMockSeedRandomBtn").onclick = () => {
    if (mockSeedInput) mockSeedInput.value = String(randomSeed());
  };

  document.getElementById("remoteMockStartBtn").onclick = async () => {
    const host = mockHostInput?.value?.trim() || "127.0.0.1";
    const port = Number(mockPortInput?.value || "8766");
    const remoteName = mockNameInput?.value?.trim() || "brother-bot";
    const seedRaw = mockSeedInput?.value?.trim() || "";
    const parsedSeed = Number(seedRaw);
    const seed = Number.isInteger(parsedSeed) ? parsedSeed : randomSeed();
    if (mockSeedInput) mockSeedInput.value = String(seed);
    try {
      const data = await remoteApi("/remote/mock/start", {
        method: "POST",
        body: JSON.stringify({ host, port, num_players: 3, remote_name: remoteName, seed }),
      });
      renderMockStatus(data);
      wsInput.value = data.ws_url;
      saveWsUrl(data.ws_url);
      document.getElementById("remoteBotName").value = remoteName;
      remoteLog(`Mock iniciado en ${data.ws_url} (seed=${seed})`);
    } catch (err) {
      remoteLog(`Error start mock: ${err.message}`);
    }
  };

  document.getElementById("remoteMockStopBtn").onclick = async () => {
    try {
      const data = await remoteApi("/remote/mock/stop", { method: "POST" });
      renderMockStatus(data);
      remoteLog("Mock detenido.");
    } catch (err) {
      remoteLog(`Error stop mock: ${err.message}`);
    }
  };

  document.getElementById("remoteMockStatusBtn").onclick = async () => {
    try {
      const data = await remoteApi("/remote/mock/status");
      renderMockStatus(data);
      remoteLog(`Mock status: running=${data?.mock?.running}`);
    } catch (err) {
      remoteLog(`Error mock status: ${err.message}`);
    }
  };

  document.getElementById("remoteCreateBtn").onclick = async () => {
    const url = document.getElementById("remoteWsUrl").value.trim() || REMOTE_DEFAULT_WS_URL;
    const name = document.getElementById("remoteBotName").value.trim();
    const token = document.getElementById("remoteToken").value.trim();
    const modelName = document.getElementById("remoteModelName").value.trim();
    const remoteLikePolicy = Boolean(document.getElementById("remoteLikePolicyChk")?.checked);
    if (!url || !name) {
      remoteLog("Completa URL y name.");
      return;
    }
    saveWsUrl(url);
    const actionDelayRaw = document.getElementById("remoteActionDelay")?.value || "";
    const actionDelay = Math.max(0, Number(actionDelayRaw) || 0);
    try {
      const data = await remoteApi("/remote/sessions", {
        method: "POST",
        body: JSON.stringify({
          url,
          name,
          token: token || null,
          model_name: modelName || null,
          remote_like_policy: remoteLikePolicy,
          action_delay_seconds: actionDelay,
        }),
      });
      remoteState.sessionId = data.session.session_id;
      renderSession(data.session);
      remoteLog(`Sesión creada: ${remoteState.sessionId}${actionDelay ? ` (delay ${actionDelay}s)` : ""}`);
    } catch (err) {
      remoteLog(`Error creando sesión: ${err.message}`);
    }
  };

  document.getElementById("remoteRestartBtn").onclick = async () => {
    if (!remoteState.sessionId) {
      remoteLog("Primero crea una sesión.");
      return;
    }
    try {
      const data = await remoteApi(`/remote/sessions/${remoteState.sessionId}/restart`, { method: "POST" });
      const newId = data.session?.session_id;
      const oldId = data.previous_session_id;
      if (newId) {
        remoteState.sessionId = newId;
        renderSession(data.session);
        remoteLog(`Nueva partida: ${oldId} → ${newId}`);
      }
    } catch (err) {
      remoteLog(`Error restart: ${err.message}`);
    }
  };

  document.getElementById("remoteStartBtn").onclick = async () => {
    if (!remoteState.sessionId) {
      remoteLog("Primero crea sesión.");
      return;
    }
    try {
      const data = await remoteApi(`/remote/sessions/${remoteState.sessionId}/start`, { method: "POST" });
      renderSession(data.session);
      remoteLog("Sesión iniciada.");
    } catch (err) {
      remoteLog(`Error start: ${err.message}`);
    }
  };

  document.getElementById("remoteStopBtn").onclick = async () => {
    if (!remoteState.sessionId) return;
    try {
      const data = await remoteApi(`/remote/sessions/${remoteState.sessionId}/stop`, { method: "POST" });
      renderSession(data.session);
      remoteLog("Stop solicitado.");
    } catch (err) {
      remoteLog(`Error stop: ${err.message}`);
    }
  };

  document.getElementById("remoteRefreshBtn").onclick = refreshRemote;

  // Spectator panel: when checked, poll mock full-state and render all players' hands.
  const spectatorChk = document.getElementById("remoteSpectatorChk");
  const spectatorPanel = document.getElementById("remoteSpectatorPanel");
  spectatorChk.addEventListener("change", () => {
    spectatorPanel.style.display = spectatorChk.checked ? "" : "none";
    if (spectatorChk.checked) {
      refreshSpectator();
    }
  });

  window.setInterval(refreshRemote, 1500);
  window.setInterval(() => {
    if (spectatorChk.checked) {
      refreshSpectator();
    }
  }, 1500);
  try {
    const data = await remoteApi("/remote/mock/status");
    renderMockStatus(data);
  } catch {
    /* ignore mock status failures at startup */
  }
}

async function refreshSpectator() {
  const target = document.getElementById("remoteSpectatorContent");
  if (!target) return;
  try {
    const data = await remoteApi("/remote/mock/full-state");
    renderSpectator(data, target);
  } catch (err) {
    target.innerHTML = `<p class="muted">Mock no responde: ${err.message}</p>`;
  }
}

function renderSpectator(state, target) {
  if (!state || !state.running) {
    target.innerHTML = `<p class="muted">Mock server no está corriendo. Hacé "Start Mock".</p>`;
    return;
  }
  const players = Array.isArray(state.players) ? state.players : [];
  const remoteSeat = Number(state.remote_seat ?? 0);
  const currentPlayer = Number(state.current_player ?? 0);
  const phase = state.phase || "?";
  const winner = state.winner;
  const finished = Boolean(state.finished);
  const games = Number(state.games_played ?? 0);

  const headerBits = [
    `Partidas jugadas: <strong>${games}</strong>`,
    `Fase: <strong>${phase}</strong>`,
    `Turno actual: <strong>seat ${currentPlayer}</strong>`,
  ];
  if (finished) {
    headerBits.push(`Terminó — ganador: <strong>seat ${winner ?? "?"}</strong>`);
  }
  const headerHtml = `<p class="muted">${headerBits.join(" · ")}</p>`;

  const playerSections = players
    .map((p) => {
      const seat = Number(p.seat ?? 0);
      const isRemote = Boolean(p.is_remote);
      const isCurrent = seat === currentPlayer;
      const handCards = Array.isArray(p.hand) ? p.hand : [];
      const tag = isRemote ? "🤖 Bot" : `Oponente`;
      const opened = p.has_opened ? "✓ abrió" : "no abrió";
      const status = isCurrent ? '<span style="color:#0a0;">▶ jugando</span>' : "";
      const cardsHtml = handCards
        .map((c) => cardHTMLWithData(c, "remote-card-spectator"))
        .join("") || `<span class="muted">(vacía)</span>`;
      return `
        <div class="spectator-player" style="border:1px solid #333; padding:8px; margin-bottom:8px; border-radius:4px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <strong>${tag} — seat ${seat}</strong>
            <span class="muted">${handCards.length} cartas · ${opened} ${status}</span>
          </div>
          <div class="cards-row">${cardsHtml}</div>
        </div>
      `;
    })
    .join("");

  target.innerHTML = headerHtml + playerSections;
}

initRemote().catch((err) => {
  console.error("[remote] startup error:", err);
});
