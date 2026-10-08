export const $ = (id) => document.getElementById(id);

export const escapeHtml = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// Les espaces en début de token sont rendus visibles : « ␣chat »
export const showSpaces = (s) => escapeHtml(s).replace(/ /g, "␣");
