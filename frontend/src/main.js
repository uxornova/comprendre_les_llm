// Point d'entrée du front : relie les boutons aux fonctionnalités et vérifie que le back répond.
import { api } from "./core/api.service.js";
import { clearJournal } from "./core/journal.js";
import { $, escapeHtml } from "./core/utils.js";
import { generate } from "./features/generate.js";
import { nextWord } from "./features/next-word.js";
import { tokens } from "./features/tokens.js";

const features = { tokens, nextWord, generate };
const buttons = document.querySelectorAll("[data-feature]");

function readForm() {
  return {
    text: $("texte").value.trim(),
    temperature: Number($("temperature").value),
    maxNewTokens: Number($("nb-tokens").value),
  };
}

buttons.forEach((button) => {
  button.addEventListener("click", async () => {
    const feature = features[button.dataset.feature];
    const form = readForm();
    if (!form.text) return;

    buttons.forEach((b) => (b.disabled = true));
    $("resultat").innerHTML = '<p class="vide">Le back calcule…</p>';
    try {
      $("resultat").innerHTML = feature.render(await feature.call(form));
    } catch (error) {
      $("resultat").innerHTML = `<p class="vide">Erreur : ${escapeHtml(error.message)}</p>`;
    } finally {
      buttons.forEach((b) => (b.disabled = false));
    }
  });
});

$("temperature").addEventListener("input", (e) => ($("temp-val").value = e.target.value));
$("vider").addEventListener("click", clearJournal);

api.health()
  .then((d) => {
    $("point").className = "point ok";
    $("statut").textContent =
      `Back connecté — ${d.model} (${d.parameters_millions} M paramètres, ${d.ram_used_gb} Go de RAM)`;
  })
  .catch(() => {
    $("point").className = "point ko";
    $("statut").textContent = "Back injoignable — lancez : make dev";
  });
