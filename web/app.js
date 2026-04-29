const state = {
  matchId: null,
  data: null,
  revealBots: false,
  densityMode: "comfortable",
  playerConfigs: [],
  autoplayEnabled: false,
  /** Si no es null, el jugador humano se muestra y loguea con este nombre (ej. Juego rápido). */
  displayGuestName: null,
};

const MODELS_STORAGE_KEY = "loba_playground_models";
const LAST_MODEL_KEY = "loba_playground_last_model";
const DENSITY_MODE_KEY = "loba_playground_density_mode";
const quackAudio = new Audio("/assets/quack.mp3");
const CARD_SPRITE_PATH = "/web/assets/cards/svg-cards.svg";
const VALID_SPRITE_IDS = new Set([
  "back",
  "joker_black",
  ...["club", "diamond", "heart", "spade"].flatMap((suit) =>
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king"].map((rank) => `${suit}_${rank}`)
  ),
]);

const MAX_MELD_SLOTS = 24;
const DISCARD_ACTION_MIN = 3 + MAX_MELD_SLOTS;
const MIN_PLAYERS = 2;
const MAX_PLAYERS = 6;

const PHASE_LABELS = {
  draw: "ROBAR",
  meld: "BAJAR / PASAR",
  discard: "DESCARTAR",
};

const PHASE_HINTS = {
  draw: "Elegí robar del mazo o del descarte (si está permitido).",
  meld: "Bajá un juego válido o tocá pasar si no querés bajar.",
  discard: "Elegí una carta en tu mano, o arrastrala al recuadro de descarte.",
};

async function api(path, options = {}) {
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

function appendLog(msg) {
  const log = document.getElementById("log");
  const row = document.createElement("div");
  row.className = "log-item";
  row.textContent = `${new Date().toLocaleTimeString()} - ${msg}`;
  log.prepend(row);
}

function actionLabelFromState(actionId) {
  const current = state.data;
  if (!current || !Array.isArray(current.valid_actions)) return `Action ${actionId}`;
  return current.valid_actions.find((a) => a.id === actionId)?.label ?? `Action ${actionId}`;
}

function tryPlayQuack() {
  quackAudio.currentTime = 0;
  quackAudio.play().catch(() => {});
}

function tryPlayQuackTwice() {
  tryPlayQuack();
  window.setTimeout(() => {
    tryPlayQuack();
  }, 180);
}

function updateSetupPanelsVisibility() {
  const modelsPanel = document.getElementById("modelsSetupPanel");
  const matchPanel = document.getElementById("matchSetupPanel");
  const inMatchControls = document.getElementById("inMatchControls");
  const matchStarted = Boolean(state.matchId);
  const inProgress = matchStarted && !state.data?.finished;
  const shouldShow = !inProgress;
  if (modelsPanel) modelsPanel.style.display = shouldShow ? "" : "none";
  if (matchPanel) matchPanel.style.display = shouldShow ? "" : "none";
  if (inMatchControls) inMatchControls.style.display = inProgress ? "flex" : "none";
}

function syncControlCheckboxes() {
  const autoplay = document.getElementById("autoplayModeChk");
  const showBots = document.getElementById("showBotCardsChk");
  const showBotsInMatch = document.getElementById("showBotCardsChkInMatch");
  if (autoplay) autoplay.checked = state.autoplayEnabled;
  if (showBots) showBots.checked = state.revealBots;
  if (showBotsInMatch) showBotsInMatch.checked = state.revealBots;
}

function applyDensityMode() {
  const mode = state.densityMode === "compact" ? "compact" : "comfortable";
  document.body.dataset.density = mode;
}

function syncDensityControls() {
  const density = state.densityMode === "compact" ? "compact" : "comfortable";
  const setupSel = document.getElementById("densityModeSel");
  const inMatchSel = document.getElementById("densityModeSelInMatch");
  if (setupSel) setupSel.value = density;
  if (inMatchSel) inMatchSel.value = density;
  applyDensityMode();
}

function loadDensityMode() {
  try {
    const raw = localStorage.getItem(DENSITY_MODE_KEY);
    if (raw === "compact" || raw === "comfortable") return raw;
  } catch {
    /* ignore */
  }
  return "comfortable";
}

function saveDensityMode(mode) {
  try {
    localStorage.setItem(DENSITY_MODE_KEY, mode);
  } catch {
    /* ignore */
  }
}

function defaultPlayerConfig(index) {
  if (index === 0) return { kind: "human", model_name: "" };
  return { kind: "model", model_name: "" };
}

function getPlayerConfigsFromForm() {
  const holder = document.getElementById("playersConfig");
  if (!holder) return [];
  const blocks = holder.querySelectorAll("[data-player-config]");
  return Array.from(blocks).map((el) => {
    const kind = el.querySelector(".player-kind")?.value || "human";
    const model_name = (el.querySelector(".player-model")?.value || "").trim();
    return { kind, model_name };
  });
}

function renderPlayersConfig() {
  const holder = document.getElementById("playersConfig");
  if (!holder) return;
  if (!Array.isArray(state.playerConfigs) || !state.playerConfigs.length) {
    state.playerConfigs = [defaultPlayerConfig(0), defaultPlayerConfig(1)];
  }
  holder.innerHTML = state.playerConfigs.map((cfg, idx) => `
    <div data-player-config="${idx}">
      <label>Jugador ${idx + 1}</label>
      <select class="player-kind">
        <option value="human" ${cfg.kind === "human" ? "selected" : ""}>Humano</option>
        <option value="heuristic" ${cfg.kind === "heuristic" ? "selected" : ""}>Heuristico</option>
        <option value="random" ${cfg.kind === "random" ? "selected" : ""}>Random</option>
        <option value="model" ${cfg.kind === "model" ? "selected" : ""}>Modelo</option>
      </select>
      <input class="player-model" placeholder="nombre modelo (si aplica)" value="${cfg.model_name || ""}" />
    </div>
  `).join("");
}

function syncPlayerConfigsFromForm() {
  const cfgs = getPlayerConfigsFromForm();
  if (cfgs.length) state.playerConfigs = cfgs;
}

function buildWinnerText(stateSnapshot) {
  if (stateSnapshot.winner_detail) {
    const w = stateSnapshot.winner_detail;
    return `Jugador ${w.index} (${w.kind}${w.model_name ? `:${w.model_name}` : ""})`;
  }
  if (stateSnapshot.winner != null) return `Jugador ${stateSnapshot.winner}`;
  return "—";
}

function logPlayerLabel(player) {
  if (!player) return "unknown";
  if (player.kind === "human" && state.displayGuestName) return state.displayGuestName;
  return player.kind;
}

function appendEntriesLog(entries, snapshotState) {
  const players = snapshotState?.players || [];
  (entries || []).forEach((x) => {
    const pl = players[x.player];
    const kind = logPlayerLabel(pl);
    const isBotHidden = pl?.kind !== "human" && !state.revealBots;
    const details = [];
    if (x.drawn_card) {
      details.push(`robó ${isBotHidden ? "una carta" : x.drawn_card}`);
    }
    if (Array.isArray(x.meld_cards) && x.meld_cards.length) {
      details.push(`bajó ${isBotHidden ? "un juego" : x.meld_cards.join("-")}`);
    }
    if (x.discarded_card) {
      details.push(`descartó ${isBotHidden ? "una carta" : x.discarded_card}`);
    }
    const suffix = details.length ? ` (${details.join(" | ")})` : "";
    appendLog(`P${x.player} (${kind}) -> ${x.label}${suffix}`);
    if (pl?.kind === "model") {
      const loweredMeld = Number.isInteger(x.action) && x.action >= 3 && x.action < DISCARD_ACTION_MIN;
      if (loweredMeld) {
        tryPlayQuackTwice();
      } else {
        tryPlayQuack();
      }
    }
  });
}

function getSavedModels() {
  try {
    const raw = localStorage.getItem(MODELS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((m) => m && typeof m.name === "string" && typeof m.path === "string");
  } catch {
    return [];
  }
}

function saveModels(models) {
  localStorage.setItem(MODELS_STORAGE_KEY, JSON.stringify(models));
}

function upsertSavedModel(model) {
  const models = getSavedModels();
  const ix = models.findIndex((m) => m.name === model.name);
  if (ix >= 0) {
    models[ix] = model;
  } else {
    models.push(model);
  }
  saveModels(models);
}

function saveLastRegisteredModel(model) {
  try {
    localStorage.setItem(LAST_MODEL_KEY, JSON.stringify({ name: model.name, path: model.path }));
  } catch {
    /* ignore */
  }
}

function getLastRegisteredModel() {
  try {
    const raw = localStorage.getItem(LAST_MODEL_KEY);
    if (raw) {
      const o = JSON.parse(raw);
      if (o && typeof o.name === "string" && typeof o.path === "string") return o;
    }
  } catch {
    /* ignore */
  }
  const list = getSavedModels();
  if (list.length) return list[list.length - 1];
  return null;
}

function removeSavedModel(name) {
  const models = getSavedModels().filter((m) => m.name !== name);
  saveModels(models);
  const last = getLastRegisteredModel();
  if (last && last.name === name) {
    if (models.length) saveLastRegisteredModel(models[models.length - 1]);
    else localStorage.removeItem(LAST_MODEL_KEY);
  }
}

function formatPlayerRoleLine(p) {
  if (p.kind === "human" && state.displayGuestName) {
    return `${state.displayGuestName} (humano)`;
  }
  return `${p.kind}${p.model_name ? `:${p.model_name}` : ""}`;
}

function formatOwnerTitle(owner) {
  if (owner.kind === "human" && state.displayGuestName) {
    return `Jugador ${owner.player_index} (${state.displayGuestName} (humano))`;
  }
  return `Jugador ${owner.player_index} (${owner.kind}${owner.model_name ? `:${owner.model_name}` : ""})`;
}

async function syncSavedModelsToBackend() {
  const models = getSavedModels();
  for (const model of models) {
    try {
      await api("/models/register", {
        method: "POST",
        body: JSON.stringify(model),
      });
    } catch (err) {
      appendLog(`No se pudo registrar ${model.name}: ${err.message}`);
    }
  }
}

function cardLabelToSpriteId(label, hidden = false) {
  if (hidden || label === "hidden") return "back";
  if (!label) return "back";
  if (label === "Joker") return "joker_black";

  const suitCode = label.slice(-1);
  const rankCode = label.slice(0, -1);

  const suitMap = {
    C: "club",
    D: "diamond",
    H: "heart",
    S: "spade",
  };
  const rankMap = {
    A: "1",
    J: "jack",
    Q: "queen",
    K: "king",
  };

  const suit = suitMap[suitCode];
  if (!suit) return "back";
  const rank = rankMap[rankCode] || rankCode;
  return `${suit}_${rank}`;
}

function cardHTML(label, hidden = false, extraClass = "", dataAttrs = "") {
  const title = hidden ? "Carta oculta" : (label || "Carta");
  const spriteId = cardLabelToSpriteId(label, hidden);
  if (VALID_SPRITE_IDS.has(spriteId)) {
    return `
      <div class="playing-card ${hidden ? "back" : ""} ${extraClass}" ${dataAttrs}>
        <svg class="playing-card-svg" viewBox="0 0 169.075 244.64" role="img" aria-label="${title}">
          <use href="${CARD_SPRITE_PATH}#${spriteId}" xlink:href="${CARD_SPRITE_PATH}#${spriteId}"></use>
        </svg>
      </div>
    `;
  }

  const parsed = parseFaceLabel(label);
  return `
    <div class="playing-card playing-card-text ${parsed.colorClass} ${extraClass}" ${dataAttrs} aria-label="${title}">
      <span class="card-corner card-corner--tl">${parsed.rank}${parsed.suit}</span>
      <span class="card-center">${parsed.suit}</span>
      <span class="card-corner card-corner--br">${parsed.rank}${parsed.suit}</span>
    </div>
  `;
}

function parseFaceLabel(label) {
  if (!label || label === "Joker") {
    return { rank: "J", suit: "★", colorClass: "joker" };
  }
  const suitCode = label.slice(-1);
  const rankCode = label.slice(0, -1);
  const suitMap = { C: "♣", D: "♦", H: "♥", S: "♠" };
  const suit = suitMap[suitCode] || "♣";
  const colorClass = (suitCode === "D" || suitCode === "H") ? "red" : "black";
  return { rank: rankCode || "?", suit, colorClass };
}

function getRenderableHandSlots(player) {
  if (Array.isArray(player.hand_display) && player.hand_display.length) {
    return player.hand_display.map((slot) => ({
      label: slot.label,
      handIndex: slot.hand_index,
    }));
  }
  const labels = player.hand_full || player.hand || [];
  return labels.map((label, idx) => ({ label, handIndex: idx }));
}

function actionButtonClass(actionId) {
  if (actionId === 0 || actionId === 1) return "action-btn--draw";
  if (actionId === 2) return "action-btn--skip";
  if (actionId >= 3 && actionId < DISCARD_ACTION_MIN) return "action-btn--meld";
  if (actionId >= DISCARD_ACTION_MIN) return "action-btn--discard";
  return "";
}

function isPrimaryValidAction(actionId, phase, validIds) {
  if (!validIds.includes(actionId)) return false;
  if (phase === "draw") return actionId === 0 || actionId === 1;
  if (phase === "meld") {
    const meldIds = validIds.filter((id) => id >= 3 && id < DISCARD_ACTION_MIN);
    if (meldIds.length) return meldIds.includes(actionId);
    return actionId === 2;
  }
  if (phase === "discard") return actionId >= DISCARD_ACTION_MIN;
  return false;
}

function clearDiscardInteract() {
  if (typeof interact === "undefined") return;
  document.querySelectorAll("[data-interact-draggable]").forEach((el) => {
    try {
      interact(el).unset();
    } catch {
      /* ignore */
    }
    el.removeAttribute("data-interact-draggable");
  });
}

function bindDiscardDrag(validDiscardIds) {
  if (typeof interact === "undefined") return;
  clearDiscardInteract();
  const dropZone = document.getElementById("discardDropZone");
  if (!dropZone) return;

  document.querySelectorAll(".playing-card.clickable-discard").forEach((el) => {
    const raw = el.dataset.actionId;
    const actionId = raw != null ? parseInt(raw, 10) : NaN;
    if (Number.isNaN(actionId) || !validDiscardIds.includes(actionId)) return;

    el.setAttribute("data-interact-draggable", "1");
    interact(el).draggable({
      listeners: {
        start({ target }) {
          target.classList.add("is-dragging");
          target.setAttribute("data-drag-x", "0");
          target.setAttribute("data-drag-y", "0");
        },
        move({ target, dx, dy }) {
          const x = (parseFloat(target.getAttribute("data-drag-x")) || 0) + dx;
          const y = (parseFloat(target.getAttribute("data-drag-y")) || 0) + dy;
          target.style.transform = `translate(${x}px, ${y}px)`;
          target.setAttribute("data-drag-x", String(x));
          target.setAttribute("data-drag-y", String(y));
        },
        end({ target }) {
          target.classList.remove("is-dragging");
          const rect = target.getBoundingClientRect();
          const dz = dropZone.getBoundingClientRect();
          const cx = rect.left + rect.width / 2;
          const cy = rect.top + rect.height / 2;
          const inside = cx >= dz.left && cx <= dz.right && cy >= dz.top && cy <= dz.bottom;
          target.style.transform = "";
          target.removeAttribute("data-drag-x");
          target.removeAttribute("data-drag-y");
          const aid = parseInt(target.dataset.actionId, 10);
          if (inside && !Number.isNaN(aid)) {
            submitHumanAction(aid);
          }
        },
      },
    });
  });
}

function updatePhaseBanner(s, isHumanTurn) {
  const phaseKey = s.phase || "draw";
  document.body.dataset.phase = phaseKey;
  const humanDiscard = Boolean(isHumanTurn && phaseKey === "discard");
  document.body.dataset.humanDiscard = humanDiscard ? "true" : "false";

  const phaseText = document.getElementById("phaseBannerText");
  const phaseHint = document.getElementById("phaseBannerHint");
  if (phaseText) {
    phaseText.textContent = PHASE_LABELS[phaseKey] || phaseKey.toUpperCase();
  }
  if (phaseHint) {
    let hint = PHASE_HINTS[phaseKey] || "";
    if (!isHumanTurn && !s.finished) {
      hint = "Turno del rival o de la IA. Esperá o usá Autoplay.";
    } else if (s.finished) {
      hint = "Partida finalizada.";
    }
    phaseHint.textContent = hint;
  }

  const drop = document.getElementById("discardDropZone");
  if (drop) {
    drop.dataset.dropActive = humanDiscard ? "true" : "false";
  }
}

async function submitHumanAction(actionId) {
  const actor = state.data?.current_player;
  const pl = actor != null ? state.data?.players?.[actor] : null;
  const actorLabel = logPlayerLabel(pl || { kind: "human" });
  appendLog(`P${actor} (${actorLabel}) -> ${actionLabelFromState(actionId)}`);
  try {
    const resp = await api(`/matches/${state.matchId}/action`, {
      method: "POST",
      body: JSON.stringify({ action: actionId }),
    });
    renderState(resp.state);
    if (resp.human_log_entry) {
      appendEntriesLog([resp.human_log_entry], resp.state);
    }
    appendEntriesLog(resp.autoplay_log || [], resp.state);
  } catch (err) {
    appendLog(`Error accion: ${err.message}`);
  }
}

function renderState(payload) {
  if (!payload) return;
  const s = payload;
  state.data = s;
  if (typeof s.autoplay_enabled === "boolean") {
    state.autoplayEnabled = s.autoplay_enabled;
  }
  syncControlCheckboxes();
  updateSetupPanelsVisibility();

  clearDiscardInteract();

  document.getElementById("matchMeta").textContent = `Partida ${s.match_id} | turno #${s.turn} | jugador ${s.current_player}`;

  const summary = document.getElementById("stateSummary");
  const winnerText = buildWinnerText(s);
  summary.innerHTML = `
    <span>Stock: <strong>${s.stock_size}</strong></span>
    <span>Tope descarte: <strong>${s.top_discard ?? "—"}</strong></span>
    <span>Estado: <strong>${s.finished ? "finalizada" : "en curso"}</strong>${s.winner !== null ? ` | ganador <strong>${winnerText}</strong>` : ""}</span>
    <span>Ultima recompensa: <strong>${(s.last_reward ?? 0).toFixed(2)}</strong></span>
  `;

  const stockCount = document.getElementById("stockCount");
  if (stockCount) stockCount.textContent = String(s.stock_size ?? 0);

  const isHumanTurn = s.players[s.current_player]?.kind === "human" && !s.finished;
  updatePhaseBanner(s, isHumanTurn);

  const discardLastThree = document.getElementById("discardLastThree");
  const tail = s.discard_last_three || [];
  if (discardLastThree) {
    discardLastThree.innerHTML = tail.length
      ? tail.map((c, idx) => cardHTML(c, false, idx === tail.length - 1 ? "discard-top" : "")).join("")
      : "<span class='muted'>Vacío</span>";
  }

  const playersView = document.getElementById("playersView");
  playersView.innerHTML = "";

  const discardMap = new Map((s.discard_options || []).map((d) => [d.hand_index, d.action_id]));
  const validIds = (s.valid_actions || []).map((a) => a.id);

  s.players.forEach((p) => {
    const block = document.createElement("div");
    block.className = `player ${p.index === s.current_player ? "current" : ""}`;

    const isBot = p.kind !== "human";
    const shouldHideBot = isBot && !state.revealBots;
    const handSlots = shouldHideBot
      ? new Array(p.cards_in_hand).fill(null).map(() => ({ hidden: true, label: "hidden", handIndex: null }))
      : getRenderableHandSlots(p);

    const humanLargeClass = p.kind === "human" ? "human-large" : "";

    block.innerHTML = `
      <strong>Jugador ${p.index}</strong> (${formatPlayerRoleLine(p)})
      <div class="muted">cartas: ${p.cards_in_hand} | abrió: ${p.has_opened ? "sí" : "no"}</div>
      <div class="hand cards-row ${humanLargeClass}" data-player="${p.index}">
        ${handSlots.map((slot) => {
          if (slot.hidden) return cardHTML("hidden", true);
          const clickable = isHumanTurn
            && p.index === s.current_player
            && s.phase === "discard"
            && discardMap.has(slot.handIndex);
          const actionId = discardMap.get(slot.handIndex);
          const actionAttr = clickable && actionId != null ? `data-action-id="${actionId}"` : "";
          const attrs = actionAttr || "";
          return cardHTML(slot.label, false, clickable ? "clickable-discard" : "", attrs);
        }).join("") || "<span class='muted'>sin cartas</span>"}
      </div>
    `;
    playersView.appendChild(block);

    if (isHumanTurn && p.index === s.current_player && s.phase === "discard") {
      block.querySelectorAll(".playing-card.clickable-discard").forEach((node) => {
        const aid = node.dataset.actionId;
        if (aid != null) {
          node.addEventListener("click", () => submitHumanAction(parseInt(aid, 10)));
          node.title = "Click o arrastrá al descarte";
        }
      });
      const discardActionIds = validIds.filter((id) => id >= DISCARD_ACTION_MIN);
      bindDiscardDrag(discardActionIds);
    }
  });

  const tableMelds = document.getElementById("tableMelds");
  tableMelds.innerHTML = "";
  const grouped = s.table_melds_grouped || [];
  const hasAnyMeld = grouped.some((g) => Array.isArray(g.melds) && g.melds.length > 0);

  if (!hasAnyMeld) {
    tableMelds.innerHTML = "<span class='muted'>No hay melds en mesa.</span>";
  } else {
    grouped.forEach((owner) => {
      if (!owner.melds?.length) return;

      const section = document.createElement("div");
      section.className = "meld-owner";
      section.innerHTML = `<div class="meld-owner-title">${formatOwnerTitle(owner)}</div>`;

      owner.melds.forEach((meld) => {
        const item = document.createElement("div");
        item.className = "meld-item";
        item.innerHTML = meld.map((c) => cardHTML(c)).join("");
        section.appendChild(item);
      });

      tableMelds.appendChild(section);
    });
  }

  const actions = document.getElementById("actions");
  actions.innerHTML = "";
  s.valid_actions.forEach((a) => {
    const b = document.createElement("button");
    const typeClass = actionButtonClass(a.id);
    const primary = isPrimaryValidAction(a.id, s.phase, validIds);
    b.className = `action-btn ${typeClass}${primary ? " action-btn--primary-valid" : ""}`;
    b.type = "button";
    b.textContent = `${a.id}: ${a.label}`;
    b.disabled = !isHumanTurn;
    b.onclick = () => submitHumanAction(a.id);
    actions.appendChild(b);
  });
}

async function removeModel(name) {
  try {
    await api(`/models/${encodeURIComponent(name)}`, { method: "DELETE" });
  } catch (err) {
    appendLog(`No se pudo borrar en backend ${name}: ${err.message}`);
  }
  removeSavedModel(name);
  appendLog(`Modelo ${name} eliminado.`);
  await refreshModels();
}

async function refreshModels() {
  const data = await api("/models");
  const holder = document.getElementById("modelList");
  holder.innerHTML = "";
  data.items.forEach((m) => {
    const row = document.createElement("div");
    row.className = "model-row";

    const label = document.createElement("span");
    label.className = "pill";
    label.textContent = `${m.name} -> ${m.path}`;

    const removeBtn = document.createElement("button");
    removeBtn.className = "secondary small";
    removeBtn.textContent = "Quitar";
    removeBtn.onclick = () => removeModel(m.name);

    row.appendChild(label);
    row.appendChild(removeBtn);
    holder.appendChild(row);
  });
}

async function refreshState() {
  if (!state.matchId) return;
  const data = await api(`/matches/${state.matchId}`);
  renderState(data.state);
  if (state.data?.autoplay_enabled && !state.data?.finished) {
    const resp = await api(`/matches/${state.matchId}/autoplay`, { method: "POST" });
    renderState(resp.state);
    appendEntriesLog(resp.autoplay_log || [], resp.state);
  }
}

async function main() {
  const bindHandler = (id, prop, handler) => {
    const el = document.getElementById(id);
    if (!el) {
      console.warn(`[app] Missing DOM node #${id}; skipped ${prop} binding.`);
      return null;
    }
    el[prop] = handler;
    return el;
  };

  try {
    await api("/health");
    const health = document.getElementById("healthBadge");
    if (health) {
      health.textContent = "API: OK";
      health.classList.add("ok");
    }
  } catch {
    const health = document.getElementById("healthBadge");
    if (health) health.textContent = "API: offline";
  }

  document.body.dataset.phase = "draw";
  document.body.dataset.humanDiscard = "false";
  state.densityMode = loadDensityMode();
  applyDensityMode();
  state.playerConfigs = [defaultPlayerConfig(0), defaultPlayerConfig(1)];
  renderPlayersConfig();
  syncControlCheckboxes();
  syncDensityControls();
  updateSetupPanelsVisibility();

  await syncSavedModelsToBackend();
  await refreshModels();

  const onRevealBotsChange = (checked) => {
    state.revealBots = checked;
    syncControlCheckboxes();
    renderState(state.data);
  };
  bindHandler("showBotCardsChk", "onchange", (e) => onRevealBotsChange(Boolean(e.target.checked)));
  bindHandler("showBotCardsChkInMatch", "onchange", (e) => onRevealBotsChange(Boolean(e.target.checked)));

  const onDensityModeChange = (value) => {
    state.densityMode = value === "compact" ? "compact" : "comfortable";
    saveDensityMode(state.densityMode);
    syncDensityControls();
    renderState(state.data);
  };
  bindHandler("densityModeSel", "onchange", (e) => onDensityModeChange(e.target.value));
  bindHandler("densityModeSelInMatch", "onchange", (e) => onDensityModeChange(e.target.value));

  bindHandler("addPlayerBtn", "onclick", () => {
    syncPlayerConfigsFromForm();
    if (state.playerConfigs.length >= MAX_PLAYERS) return;
    state.playerConfigs.push(defaultPlayerConfig(state.playerConfigs.length));
    renderPlayersConfig();
  });

  bindHandler("removePlayerBtn", "onclick", () => {
    syncPlayerConfigsFromForm();
    if (state.playerConfigs.length <= MIN_PLAYERS) return;
    state.playerConfigs.pop();
    renderPlayersConfig();
  });

  bindHandler("autoplayModeChk", "onchange", (e) => {
    state.autoplayEnabled = Boolean(e.target.checked);
    syncControlCheckboxes();
  });

  bindHandler("registerModelBtn", "onclick", async () => {
    const name = document.getElementById("modelName").value.trim();
    const path = document.getElementById("modelPath").value.trim();
    if (!name || !path) {
      appendLog("Completa nombre y path del modelo.");
      return;
    }
    try {
      await api("/models/register", {
        method: "POST",
        body: JSON.stringify({ name, path }),
      });
      upsertSavedModel({ name, path });
      saveLastRegisteredModel({ name, path });
      appendLog(`Modelo ${name} registrado y guardado en localStorage.`);
      await refreshModels();
    } catch (err) {
      appendLog(`Error registrando modelo: ${err.message}`);
    }
  });

  async function startMatchFromForm({ quickGuest } = { quickGuest: false }) {
    if (quickGuest) {
      state.displayGuestName = "Invitado";
    } else {
      state.displayGuestName = null;
    }

    syncPlayerConfigsFromForm();
    const players = state.playerConfigs.map((p) => ({ kind: p.kind, model_name: p.model_name?.trim() || null }));

    if (quickGuest) {
      const last = getLastRegisteredModel();
      if (!last) {
        state.displayGuestName = null;
        appendLog("Juego rápido: no hay último modelo. Registrá un modelo primero.");
        return;
      }
      try {
        await api("/models/register", {
          method: "POST",
          body: JSON.stringify({ name: last.name, path: last.path }),
        });
      } catch (err) {
        state.displayGuestName = null;
        appendLog(`Juego rápido: no se pudo cargar el modelo (${err.message})`);
        return;
      }

      let hasModel = false;
      players.forEach((p) => {
        if (p.kind === "model") {
          hasModel = true;
          if (!p.model_name) p.model_name = last.name;
        }
      });

      if (!hasModel) {
        state.displayGuestName = null;
        appendLog("Juego rápido: elegí al menos un jugador tipo Modelo (vacío o con nombre).");
        return;
      }
    }

    try {
      const resp = await api("/matches", {
        method: "POST",
        body: JSON.stringify({
          players,
          autoplay_enabled: state.autoplayEnabled,
        }),
      });
      state.matchId = resp.match_id;
      renderState(resp.state);
      appendLog(quickGuest ? `Juego rápido (Invitado): ${state.matchId}` : `Partida creada: ${state.matchId}`);
      appendEntriesLog(resp.autoplay_log || [], resp.state);
    } catch (err) {
      state.displayGuestName = null;
      appendLog(`Error creando partida: ${err.message}`);
    }
  }

  bindHandler("createMatchBtn", "onclick", () => startMatchFromForm({ quickGuest: false }));

  bindHandler("quickGameBtn", "onclick", () => startMatchFromForm({ quickGuest: true }));

  bindHandler("refreshBtnInMatch", "onclick", refreshState);

  bindHandler("refreshBtn", "onclick", refreshState);
}

main().catch((err) => {
  console.error("[app] Fatal error during startup:", err);
});
