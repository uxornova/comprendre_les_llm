import { api } from "../core/api.service.js";
import { showSpaces } from "../core/utils.js";

export const nextWord = {
  call: ({ text }) => api.nextWord(text, 10),
  render: (d) => {
    const max = d.candidates[0].probability;
    return d.candidates.map((c) =>
      `<div class="barre"><code>${showSpaces(c.token)}</code>` +
      `<div class="fond-barre"><div class="plein" style="width:${(c.probability / max) * 100}%"></div></div>` +
      `<span>${(c.probability * 100).toFixed(1)} %</span></div>`
    ).join("");
  },
};
