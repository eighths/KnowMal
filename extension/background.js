const API_BASE = "https://knowmal.duckdns.org"; 

self.API_BASE = typeof API_BASE !== "undefined" ? API_BASE : self.API_BASE || "https://knowmal.duckdns.org";

let isContextValid = true;

chrome.runtime.onSuspend.addListener(() => {
  isContextValid = false;
  console.log("Service Worker suspended - context invalidated");
});

chrome.runtime.onStartup.addListener(() => {
  isContextValid = true;
  console.log("Service Worker started - context valid");
});

console.log("[KnowMal] background boot, API_BASE =", self.API_BASE);

(async () => {
  try {
    if (typeof importScripts === "function") {
      importScripts("background_gmail.js");
      console.log("[KnowMal] background_gmail.js loaded via importScripts");
    } else {
      await import("./background_gmail.js");
      console.log("[KnowMal] background_gmail.js loaded via dynamic import");
    }
  } catch (e) {
    console.error("[KnowMal] failed to load background_gmail.js:", e);
  }
})();

// Utility function for JSON fetching with timeout
async function fetchJSON(url, opts = {}, timeoutMs = 20000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...opts, signal: ctrl.signal });
    const text = await res.text();
    let json;
    try { json = text ? JSON.parse(text) : {}; } catch (e) { json = { raw: text }; }
    if (!res.ok) {
      const msg = json?.detail || json?.error || `HTTP ${res.status}`;
      throw new Error(msg);
    }
    return json;
  } finally {
    clearTimeout(t);
  }
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  // 동기 응답 처리
  if (msg?.type === "KM_PING") {
    sendResponse({ ok: true, pong: Date.now() });
    return true;
  }

  // 비동기 응답이 필요한 경우
  if (!isContextValid) {
    sendResponse({ 
      ok: false, 
      error: "Extension context invalidated. Please refresh the page and try again." 
    });
    return true;
  }

  if (msg?.type === "maloffice.fetch" && msg.url) {
    (async () => {
      try {
        if (!isContextValid) {
          throw new Error("Extension context invalidated during processing");
        }

        const r = await fetch(msg.url, msg.init || {});
        const ct = r.headers.get("content-type") || "";
        let payload = null;
        try {
          payload = ct.includes("application/json") ? await r.json() : await r.text();
        } catch (e) {
          payload = null;
        }
        sendResponse({
          ok: r.ok,
          status: r.status,
          headers: { "content-type": ct },
          json: ct.includes("application/json") ? payload : undefined,
          text: !ct.includes("application/json") ? payload : undefined,
        });
      } catch (e) {
        sendResponse({ ok: false, status: 0, error: e?.message || String(e) });
      }
    })();
    return true;
  }

  if (msg?.type === "maloffice.fetchBinary") {
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
          credentials: init?.credentials || "include",
          referrer: init?.referrer || undefined,
          referrerPolicy: init?.referrerPolicy || undefined,
        });

        const buf = await resp.arrayBuffer();
        sendResponse({
          ok: resp.ok,
          status: resp.status,
          headers: Object.fromEntries(resp.headers.entries()),
          buffer: buf
        });
      } catch (e) {
        console.error("Background script error:", e);
        sendResponse({ 
          ok: false, 
          error: (e && e.message) || String(e),
          contextInvalid: e.message && e.message.includes("context invalidated")
        });
      }
    })();
    return true;
  }
});

// Gmail OAuth functionality
(() => {
  const SCOPES = [
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/gmail.readonly",
  ];

  function bufToBase64Url(buf) {
    const bytes = new Uint8Array(buf);
    let bin = "";
    for (let i = 0; i < bytes.byteLength; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }
  async function sha256base64url(input) {
    const data = new TextEncoder().encode(input);
    const digest = await crypto.subtle.digest("SHA-256", data);
    return bufToBase64Url(digest);
  }
  function getRedirectUri() {
    return chrome.identity.getRedirectURL();
  }
  async function getClientId() {
    try {
      const r = await fetch(`${self.API_BASE}/auth/google/client`);
      if (r.ok) {
        const j = await r.json().catch(() => ({}));
        if (j?.client_id) return j.client_id;
      }
    } catch (e) {}
    throw new Error(
      "GOOGLE_CLIENT_ID 미설정: /auth/google/client 구현 or background_gmail.js에 Client ID 하드코딩 필요"
    );
  }

  async function kmStartOAuthFlow() {
    const clientId = await getClientId();
    const redirectUri = getRedirectUri();

    const codeVerifier = bufToBase64Url(crypto.getRandomValues(new Uint8Array(32)));
    const codeChallenge = await sha256base64url(codeVerifier);

    const authUrl = new URL("https://accounts.google.com/o/oauth2/v2/auth");
    authUrl.searchParams.set("client_id", clientId);
    authUrl.searchParams.set("redirect_uri", redirectUri);
    authUrl.searchParams.set("response_type", "code");
    authUrl.searchParams.set("scope", SCOPES.join(" "));
    authUrl.searchParams.set("access_type", "offline");
    authUrl.searchParams.set("prompt", "consent");
    authUrl.searchParams.set("code_challenge", codeChallenge);
    authUrl.searchParams.set("code_challenge_method", "S256");

    const responseUrl = await chrome.identity
      .launchWebAuthFlow({ url: authUrl.toString(), interactive: true })
      .catch((e) => {
        throw new Error(`launchWebAuthFlow 실패: ${e?.message || e}`);
      });

    const u = new URL(responseUrl);
    const err = u.searchParams.get("error");
    if (err) {
      const desc = u.searchParams.get("error_description") || "";
      throw new Error(`Google Auth 오류: ${err}${desc ? ` (${desc})` : ""}`);
    }
    const code = u.searchParams.get("code");
    if (!code) throw new Error("authorization code 없음 (redirect_uri_mismatch 가능)");

    const r = await fetch(`${self.API_BASE}/auth/google/exchange`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "omit",
      body: JSON.stringify({ code, code_verifier: codeVerifier, redirect_uri: redirectUri }),
    });
    if (!r.ok) {
      const body = await r.text().catch(() => "");
      throw new Error(`exchange 실패 ${r.status}: ${body}`);
    }
    const js = await r.json().catch(() => ({}));
    const apiToken = js?.api_token;
    if (!apiToken) throw new Error("exchange 성공했지만 api_token이 없습니다.");

    await chrome.storage.session.set({ km_api_token: apiToken });
    return apiToken;
  }

  async function kmGetApiToken() {
    const d = await chrome.storage.session.get("km_api_token");
    return d?.km_api_token || null;
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (self.__KM_OAUTH_READY__ === true) {
      return; 
    }

    if (msg?.type === "KM_OAUTH_START") {
      (async () => {
        try {
          const token = await kmStartOAuthFlow();
          sendResponse({ ok: true, token, fallback: true });
        } catch (e) {
          console.error("[KnowMal][OAuth][fallback] ERROR:", e);
          sendResponse({ ok: false, error: e?.message || String(e), fallback: true });
        }
      })();
      return true; 
    }

    if (msg?.type === "KM_GET_TOKEN") {
      (async () => {
        try {
          const token = await kmGetApiToken();
          sendResponse({ ok: true, token, fallback: true });
        } catch (e) {
          sendResponse({ ok: false, error: e?.message || String(e), fallback: true });
        }
      })();
      return true;
    }

    if (msg?.type === "KM_OAUTH_READY") {
      sendResponse({ ok: true, ready: false, mode: "fallback" });
    }
  });
})();

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  (async () => {
    try {
      if (msg?.type === "KM_OAUTH_ENSURE") {
        const extId = chrome.runtime.id;
        const email = (msg && typeof msg.email === "string" ? msg.email.trim() : "");

        try {
          let authorized = false;
          const attempts = 3;
          for (let i = 0; i < attempts; i++) {
            const qs = new URLSearchParams({ ext_id: extId });
            if (email) qs.set("email", email);
            const r = await fetch(`${self.API_BASE}/auth/google/status?${qs.toString()}`, { credentials: "omit" });
            console.log("[KnowMal] OAuth status check response:", r.status);
            if (r.ok) {
              const j = await r.json();
              console.log("[KnowMal] OAuth status check result:", j);
              authorized = !!j?.authorized;
              if (authorized) break;
              try { await chrome.storage.local.remove(["KM_OAUTH_READY"]); } catch (_) {}
            }
            await new Promise(r => setTimeout(r, 300));
          }

          if (authorized) {
            const toSet = email ? { KM_OAUTH_READY: true, KM_OAUTH_EMAIL: email } : { KM_OAUTH_READY: true };
            await chrome.storage.local.set(toSet);
            console.log("[KnowMal] OAuth status confirmed, cache saved");
            sendResponse({ ok: true, authorized: true, authed: true });
            return;
          } else {
            console.log("[KnowMal] OAuth not authorized after retries");
          }
        } catch (e) {
          console.log("[KnowMal] OAuth status check failed:", e);
        }

        let tabId = null;
        try {
          try {
            let finalAuthorized = false;
            const finalAttempts = 2;
            for (let i = 0; i < finalAttempts; i++) {
              const qs2 = new URLSearchParams({ ext_id: extId });
              if (email) qs2.set("email", email);
              const r2 = await fetch(`${self.API_BASE}/auth/google/status?${qs2.toString()}`, { credentials: "omit" });
              if (r2.ok) {
                const j2 = await r2.json();
                finalAuthorized = !!j2?.authorized;
                if (finalAuthorized) break;
              }
              await new Promise(r => setTimeout(r, 200));
            }
            if (finalAuthorized) {
              const toSet = email ? { KM_OAUTH_READY: true, KM_OAUTH_EMAIL: email } : { KM_OAUTH_READY: true };
              await chrome.storage.local.set(toSet);
              console.log("[KnowMal] Skip opening OAuth tab: already authorized on final check");
              sendResponse({ ok: true, authorized: true, authed: true });
              return;
            }
          } catch (_) { /* ignore and proceed to open */ }

          let created = null;
          try {
            created = await chrome.tabs.create({
              url: `${self.API_BASE}/auth/google/start?ext_id=${encodeURIComponent(extId)}`,
              active: true,
            });
          } catch (e1) {
            console.warn("[KnowMal] tabs.create failed, trying windows.create", e1);
            try {
              const win = await chrome.windows.create({
                url: `${self.API_BASE}/auth/google/start?ext_id=${encodeURIComponent(extId)}`,
                focused: true,
                type: "popup",
                width: 980,
                height: 740,
              });
              if (win && win.tabs && win.tabs.length > 0) created = win.tabs[0];
            } catch (e2) {
              console.error("[KnowMal] windows.create also failed", e2);
            }
          }
          if (created && created.id != null) tabId = created.id;
        } catch (_) { /* noop */ }

        const maxMs = 120_000;
        const start = Date.now();
        let authorized = false;
        
        while (Date.now() - start < maxMs) {
          await new Promise(r => setTimeout(r, 1000));
          try {
            const qs = new URLSearchParams({ ext_id: extId });
            if (email) qs.set("email", email);
            const r = await fetch(`${self.API_BASE}/auth/google/status?${qs.toString()}`, { credentials: "omit" });
            if (r.ok) {
              const j = await r.json();
              if (j?.authorized) {
                authorized = true;
                break;
              }
            }
          } catch (e) {
          }
        }
        
        if (authorized) {
          if (tabId != null) {
            try { await chrome.tabs.remove(tabId); } catch (_) {}
          }
          const toSet = email ? { KM_OAUTH_READY: true, KM_OAUTH_EMAIL: email } : { KM_OAUTH_READY: true };
          await chrome.storage.local.set(toSet);
          console.log("[KnowMal] OAuth completed, cache saved");
          sendResponse({ ok: true, authorized: true, authed: true });
        } else {
          try {
            if (tabId == null) {
              console.warn("[KnowMal] OAuth not authorized and no tab tracked; forcing open tab once more");
              const t = await chrome.tabs.create({
                url: `${self.API_BASE}/auth/google/start?ext_id=${encodeURIComponent(extId)}`,
                active: true,
              });
              tabId = t?.id ?? null;
            }
          } catch (_) { /* ignore */ }
          console.log("[KnowMal] OAuth failed or timeout, clearing cache");
          await chrome.storage.local.remove(["KM_OAUTH_READY", "KM_OAUTH_EMAIL"]);
          sendResponse({ ok: false, error: "OAuth 미완료 또는 시간초과" });
        }
        return;
      }

      if (msg?.type === "KM_OAUTH_ENSURE_FORCE") {
        const extId = chrome.runtime.id;
        let tabId = null;
        try {
          let created = null;
          try {
            created = await chrome.tabs.create({
              url: `${self.API_BASE}/auth/google/start?ext_id=${encodeURIComponent(extId)}`,
              active: true,
            });
          } catch (e1) {
            console.warn("[KnowMal] FORCE tabs.create failed, trying windows.create", e1);
            try {
              const win = await chrome.windows.create({
                url: `${self.API_BASE}/auth/google/start?ext_id=${encodeURIComponent(extId)}`,
                focused: true,
                type: "popup",
                width: 980,
                height: 740,
              });
              if (win && win.tabs && win.tabs.length > 0) created = win.tabs[0];
            } catch (e2) {
              console.error("[KnowMal] FORCE windows.create also failed", e2);
            }
          }
          if (created && created.id != null) tabId = created.id;
        } catch (_) { /* noop */ }

        const maxMs = 120_000;
        const start = Date.now();
        let authorized = false;
        while (Date.now() - start < maxMs) {
          await new Promise(r => setTimeout(r, 1000));
          try {
            const r = await fetch(`${self.API_BASE}/auth/google/status?ext_id=${encodeURIComponent(extId)}`, { credentials: "omit" });
            if (r.ok) {
              const j = await r.json();
              if (j?.authorized) { authorized = true; break; }
            }
          } catch (e) {}
        }
        if (authorized) {
          if (tabId != null) { try { await chrome.tabs.remove(tabId); } catch (_) {} }
          await chrome.storage.local.set({ KM_OAUTH_READY: true });
          sendResponse({ ok: true, authorized: true, authed: true, forced: true });
        } else {
          await chrome.storage.local.remove(["KM_OAUTH_READY"]);
          sendResponse({ ok: false, error: "OAuth 미완료 또는 시간초과", forced: true });
        }
        return;
      }

      if (msg?.type === "KM_OAUTH_STATUS") {
        const extId = chrome.runtime.id;
        const email = (msg && typeof msg.email === "string" ? msg.email.trim() : "");
        try {
          const qs = new URLSearchParams({ ext_id: extId });
          if (email) qs.set("email", email);
          const r = await fetch(`${self.API_BASE}/auth/google/status?${qs.toString()}`, { credentials: "omit" });
          if (r.ok) {
            const j = await r.json();
            sendResponse({ ok: true, authorized: j?.authorized || false, authed: j?.authorized || false, email: j?.email });
          } else {
            sendResponse({ ok: true, authorized: false, authed: false });
          }
        } catch (e) {
          sendResponse({ ok: true, authorized: false, authed: false });
        }
        return;
      }

      if (msg?.type === "KM_FETCH") {
        try {
          const { url, init } = msg;
          console.log("[KnowMal] KM_FETCH request:", { url, init });
          
          const r = await fetch(url, init);
          console.log("[KnowMal] KM_FETCH response status:", r.status);
          
          const headers = {};
          r.headers.forEach((v, k) => (headers[k] = v));
          
          let json = null;
          let text = null;
          
          try {
            const ct = r.headers.get("content-type") || "";
            if (ct.includes("application/json")) {
              json = await r.json();
            } else {
              text = await r.text();
            }
          } catch (e) {
            text = await r.text();
          }
          
          console.log("[KnowMal] KM_FETCH final response:", { 
            ok: r.ok, 
            status: r.status, 
            json,
            text: text?.substring(0, 200) + "..."
          });
          
          sendResponse({ 
            ok: r.ok, 
            status: r.status, 
            headers, 
            json,
            text,
            body: text
          });
        } catch (e) {
          console.log("[KnowMal] KM_FETCH error:", e);
          sendResponse({ ok: false, error: String(e) });
        }
        return;
      }

      if (msg?.type === "KM_DOWNLOAD_FILE") {
        try {
          const { message_id, filename, account_email } = msg;
          console.log("[KnowMal] KM_DOWNLOAD_FILE request:", { message_id, filename, account_email });
          
          const extId = chrome.runtime.id;
          const downloadUrl = `${self.API_BASE}/gmail/download/${encodeURIComponent(message_id)}?filename=${encodeURIComponent(filename)}`;
          
          const headers = {
            "X-KM-Ext-Id": extId,
            "X-KM-Account-Email": account_email || undefined
          };
          
          const r = await fetch(downloadUrl, {
            method: 'GET',
            headers: headers
          });
          
          console.log("[KnowMal] KM_DOWNLOAD_FILE response status:", r.status);
          
          if (!r.ok) {
            const errorText = await r.text().catch(() => "");
            throw new Error(`Download failed: ${r.status} ${errorText}`);
          }

          const blob = await r.blob();
          const url = URL.createObjectURL(blob);
          
          console.log("[KnowMal] KM_DOWNLOAD_FILE completed:", filename);
          
          sendResponse({ 
            ok: true, 
            download_url: url,
            filename: filename
          });
        } catch (e) {
          console.log("[KnowMal] KM_DOWNLOAD_FILE error:", e);
          sendResponse({ ok: false, error: String(e) });
        }
        return;
      }
    } catch (e) {
      console.error("Background script error:", e);
      sendResponse({ 
        ok: false, 
        error: (e && e.message) || String(e),
        contextInvalid: e.message && e.message.includes("context invalidated")
      });
    }
  })();
  return true; 
});