const API_BASE = "http://localhost:8000";
const CALLBACK_PREFIX = `${API_BASE}/auth/google/callback`;

const getExtId = () => chrome.runtime.id;
const log = (...a) => console.log("[KnowMal][BG]", ...a);

try { self.__KM_OAUTH_READY__ = true; } catch (_) {}

async function proxyFetch({ url, method = "GET", headers = {}, body, json, credentials = "include" }) {
  const init = {
    method,
    headers: { ...(json ? { "Content-Type": "application/json" } : {}), ...headers },
    credentials,
    body: json ? JSON.stringify(json) : body,
  };
  if (!init.headers["Content-Type"]) delete init.headers["Content-Type"];
  const res = await fetch(url, init);
  const ct = res.headers.get("content-type") || "";
  let data = null;
  try { data = ct.includes("application/json") ? await res.json() : await res.text(); } catch {}
  return { ok: res.ok, status: res.status, headers: Object.fromEntries(res.headers.entries()), data };
}

function broadcastOAuthDone() {
  chrome.tabs.query({ url: "https://mail.google.com/*" }, (tabs) => {
    for (const t of tabs) chrome.tabs.sendMessage(t.id, { type: "KM_OAUTH_DONE" }).catch?.(() => {});
  });
}

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  const url = changeInfo.url || tab?.url || "";
  if (url.startsWith(CALLBACK_PREFIX)) {
    log("callback detected → close tab & notify CS");
    try { chrome.tabs.remove(tabId); } catch {}
    broadcastOAuthDone();
  }
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    try {
      if (!msg || !msg.type) return sendResponse({ ok: false, error: "no type" });

      if (msg.type === "KM_PING") return sendResponse({ ok: true, pong: Date.now() });

      if (msg.type === "KM_EXT_INFO")
        return sendResponse({ ok: true, ext_id: getExtId(), api_base: API_BASE });

      if (msg.type === "KM_FETCH" || msg.type === "maloffice.fetch") {
        const r = await proxyFetch(msg.payload || msg);
        return sendResponse({ ok: r.ok, status: r.status, headers: r.headers, json: r.data });
      }

      if (msg.type === "KM_OAUTH_STATUS" || msg.type === "KM_OAUTH_ENSURE") {
        return; 
      }

      return sendResponse({ ok: false, error: "unknown type " + msg.type });
    } catch (e) {
      return sendResponse({ ok: false, error: String(e?.message || e) });
    }
  })();
  return true;
});