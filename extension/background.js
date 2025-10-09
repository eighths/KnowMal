const API_BASE = "http://localhost:8000";  //개발용
//const API_BASE = "https://knowmal.duckdns.org";  

let isContextValid = true;

chrome.runtime.onSuspend.addListener(() => {
  isContextValid = false;
  console.log("Service Worker suspended - context invalidated");
});

chrome.runtime.onStartup.addListener(() => {
  isContextValid = true;
  console.log("Service Worker started - context valid");
});

async function fetchJSON(url, opts = {}, timeoutMs = 20000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...opts, signal: ctrl.signal });
    const text = await res.text();
    let json;
    try { json = text ? JSON.parse(text) : {}; } catch { json = { raw: text }; }
    if (!res.ok) {
      const msg = json?.detail || json?.error || `HTTP ${res.status}`;
      throw new Error(msg);
    }
    return json;
  } finally {
    clearTimeout(t);
  }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!isContextValid) {
    sendResponse({ 
      ok: false, 
      error: "Extension context invalidated. Please refresh the page and try again." 
    });
    return true;
  }

  if (msg?.type !== "maloffice.fetch") return; 

  (async () => {
    try {
      if (!isContextValid) {
        throw new Error("Extension context invalidated during processing");
      }

      const { url, init } = msg;
      const resp = await fetch(url, {
        method: init?.method || "GET",
        headers: init?.headers || {},
        body: init?.body,
        redirect: "follow",
      });

      const text = await resp.text();
      let data = null;
      try { data = JSON.parse(text); } catch { /* JSON 아니면 null */ }

      sendResponse({
        ok: resp.ok,
        status: resp.status,
        headers: Object.fromEntries(resp.headers.entries()),
        text,
        json: data
      });
    } catch (e) {
      console.error("Background script error:", e);
      sendResponse({ 
        ok: false, 
        error: (e && e.message) || String(e),
        contextInvalid: e.message.includes("context invalidated")
      });
    }
  })();

  return true;
});
