// Panneau « Journal des échanges » : affiche chaque requête envoyée au back et sa réponse.
import { $, escapeHtml } from "./utils.js";

const json = (data) => `<pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`;

export function logExchange({ method, url, body, status, response, durationMs, serverMs }) {
  const journal = $("journal");
  journal.querySelector(".vide")?.remove();
  const ok = typeof status === "number" && status < 400;
  const timing = serverMs ? `${Math.round(durationMs)} ms (back : ${serverMs} ms)` : `${Math.round(durationMs)} ms`;
  const entry = document.createElement("div");
  entry.className = "echange";
  entry.innerHTML =
    `<div class="ligne"><span class="methode">${method}</span><code>${url}</code>` +
    `<span class="code-http ${ok ? "ok" : "ko"}">${status}</span><span>${timing}</span></div>` +
    (body ? `<div class="sens">→ envoyé par le front</div>${json(body)}` : "") +
    `<div class="sens">← répondu par le back</div>${json(response)}`;
  journal.prepend(entry);
}

export function clearJournal() {
  $("journal").innerHTML = "";
}
