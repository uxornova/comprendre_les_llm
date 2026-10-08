import { api } from "../core/api.service.js";
import { escapeHtml } from "../core/utils.js";

export const generate = {
  call: ({ text, temperature, maxNewTokens }) => api.generate(text, temperature, maxNewTokens),
  render: (d) => `<p class="generation"><b>${escapeHtml(d.prompt)}</b><mark>${escapeHtml(d.completion)}</mark></p>`,
};
