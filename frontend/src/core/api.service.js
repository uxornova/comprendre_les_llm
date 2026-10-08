// Seul point de contact avec le back (équivalent de core/services/api.service.ts dans Matcha).
// Toutes les requêtes passent ici, ce qui permet de les journaliser au même endroit.
import { logExchange } from "./journal.js";

async function request(method, url, body) {
  const t0 = performance.now();
  const options = { method, headers: { "Content-Type": "application/json" } };
  if (body) options.body = JSON.stringify(body);

  let response, data;
  try {
    response = await fetch(url, options);
    data = await response.json();
  } catch (error) {
    logExchange({ method, url, body, status: "—", response: { erreur: `Back injoignable : ${error.message}` },
                  durationMs: performance.now() - t0 });
    throw error;
  }

  logExchange({ method, url, body, status: response.status, response: data,
                durationMs: performance.now() - t0, serverMs: response.headers.get("X-Process-Time-Ms") });
  if (!response.ok) throw new Error(JSON.stringify(data.detail ?? data));
  return data;
}

export const api = {
  health: () => request("GET", "/api/health"),
  tokens: (text) => request("POST", "/api/llm/tokens", { text }),
  nextWord: (text, k = 10) => request("POST", "/api/llm/next-word", { text, k }),
  generate: (text, temperature, maxNewTokens) =>
    request("POST", "/api/llm/generate", { text, temperature, max_new_tokens: maxNewTokens }),
};
