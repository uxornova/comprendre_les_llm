import { api } from "../core/api.service.js";
import { showSpaces } from "../core/utils.js";

const COLORS = ["#dbe8f8", "#fbe1d6", "#d3f0e5", "#fbecc7", "#f8dde8"];

export const tokens = {
  call: ({ text }) => api.tokens(text),
  render: (d) =>
    `<p><b>${d.token_count} tokens</b> pour ${d.char_count} caractères</p><div>` +
    d.tokens.map((t, i) =>
      `<span class="token" style="background:${COLORS[i % COLORS.length]}" title="id ${d.ids[i]}">${showSpaces(t)}</span>`
    ).join("") + "</div>",
};
