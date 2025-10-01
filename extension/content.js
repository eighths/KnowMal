(() => {
  const API_BASE = "https://knowmal.duckdns.org";
  const OFFICE_RE = /\.(docx?|xlsx?|pptx?)$/i;

  const STYLE_ID = "maloffice-style-v2";
  function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
:root{
  --mo-bg: #ffffff;
  --mo-card: #fff;
  --mo-muted:#6b7280;
  --mo-border:#e5e7eb;
  --mo-primary:#3b82f6; /* 버튼 파랑 */
  --mo-primary-600:#2563eb;
  --mo-shadow: 0 10px 30px rgba(0,0,0,.15);
  --mo-radius: 16px;
}
#mo-overlay{
  position:fixed; inset:0; z-index:2147483647; display:none;
  background: rgba(0,0,0,.36); backdrop-filter: blur(2px);
  align-items:center; justify-content:center;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}
.mo-card{
  width:min(520px, 92vw); background:var(--mo-card);
  border:1px solid var(--mo-border); border-radius: var(--mo-radius);
  box-shadow: var(--mo-shadow);
}
.mo-head{
  display:flex; align-items:flex-start; gap:12px;
  padding:18px 18px 0 18px;
}
.mo-avatar{
  width:34px; height:34px; border-radius:10px;
  background: linear-gradient(135deg,#8bb0ff,#3b82f6 70%);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.6);
}
.mo-titlebox{flex:1}
.mo-title{
  font-weight: 800; font-size: 18px; letter-spacing:-.3px;
}
.mo-sub{ color:var(--mo-muted); font-size:13px; margin-top:2px; }

.mo-section{
  margin:16px 18px; padding:16px; border:1px dashed var(--mo-border);
  border-radius:12px; background:#fafafa;
}
.mo-section h4{ margin:0 0 10px; font-size:14px; font-weight:700; }

.mo-statusline{
  display:flex; align-items:center; gap:10px; margin:12px 18px 8px;
}
.mo-badge{
  display:inline-flex; align-items:center; gap:6px;
  border-radius:999px; padding:6px 12px; font-size:13px;
  background:#f3f4f6; border:1px solid var(--mo-border); color:#111;
}
.mo-badge.dot::before{
  content:""; width:8px; height:8px; border-radius:999px; background:#9ca3af;
}
.mo-badge.ok .dot::before{ background:#22c55e;}
.mo-badge.warn .dot::before{ background:#f59e0b;}
.mo-badge.err .dot::before{ background:#ef4444;}

.mo-msg{ font-size:13px; color:#374151; }

.mo-progress{
  height:8px; margin:8px 18px 0; background:#eef2ff; border-radius:999px; overflow:hidden;
}
.mo-progress > i{ display:block; height:100%; width:0%; background:linear-gradient(90deg,#93c5fd,#3b82f6); transition: width .35s ease; }

.mo-actions{
  display:flex; gap:12px; justify-content:flex-end; padding:16px 18px 18px;
}
.mo-btn{
  padding:10px 16px; border-radius:12px; font-weight:700; font-size:14px;
  border:1px solid var(--mo-border); background:#fff; cursor:pointer;
}
.mo-btn:hover{ filter: brightness(.98); }
.mo-btn.primary{
  background: var(--mo-primary); border-color: var(--mo-primary);
  color:#fff;
}
.mo-btn.primary:hover{ background: var(--mo-primary-600); }
    `;
    document.head.appendChild(s);
  }

  let $ov, $prog, $badge, $msg, $btnOpen, $btnClose;

  function ensureOverlay() {
    if ($ov) return;
    injectStyles();
    $ov = document.createElement("div");
    $ov.id = "mo-overlay";
    $ov.innerHTML = `
      <div class="mo-card" role="dialog" aria-modal="true">
        <div class="mo-head">
          <div class="mo-avatar"></div>
          <div class="mo-titlebox">
            <div class="mo-title">KnowMal</div>
            <div class="mo-sub">정적 분석 기반 피처 → AI 예측 · 보고서 링크 생성</div>
          </div>
        </div>

        <div class="mo-statusline">
          <span class="mo-badge dot" id="mo-badge"><span class="dot"></span>대기</span>
          <span class="mo-msg" id="mo-msg">준비 중…</span>
        </div>
        <div class="mo-progress"><i id="mo-prog"></i></div>

        <div class="mo-actions">
          <button class="mo-btn" id="mo-close">닫기</button>
          <button class="mo-btn primary" id="mo-open" disabled>리포트 확인</button>
        </div>
      </div>
    `;
    document.body.appendChild($ov);

    $prog = $ov.querySelector("#mo-prog");
    $badge = $ov.querySelector("#mo-badge");
    $msg = $ov.querySelector("#mo-msg");
    $btnOpen = $ov.querySelector("#mo-open");
    $btnClose = $ov.querySelector("#mo-close");
    $btnClose.onclick = () => ($ov.style.display = "none");
  }

  function showOverlay() {
    ensureOverlay();
    $ov.style.display = "flex";
    setBadge("대기", "dot");
    setMsg("서버에서 검사 준비 중…");
    setProgress(8);
    $btnOpen.disabled = true;
    $btnOpen.onclick = null;
  }
  function setMsg(t){ $msg.textContent = t; }
  function setProgress(pct){ $prog.style.width = `${Math.max(0, Math.min(100, pct))}%`; }
  function setBadge(text, kind){
    $badge.className = `mo-badge ${kind}`;
    $badge.innerHTML = `<span class="dot"></span>${text}`;
  }
  function setDone(reportUrl){
    setBadge("완료", "ok");
    setMsg("분석이 완료되었습니다. 리포트로 이동하세요.");
    setProgress(100);
    $btnOpen.disabled = false;
    $btnOpen.onclick = () => window.open(reportUrl, "_blank", "noopener,noreferrer");
  }
  function setError(text){
    setBadge("오류", "err");
    setMsg(text || "분석 실패");
    setProgress(100);
    $btnOpen.disabled = true;
  }

  function bgFetch(path, init = {}) {
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage(
        { type: "maloffice.fetch", url: `${API_BASE}${path}`, init },
        (resp) => {
          if (chrome.runtime.lastError) return reject(new Error(chrome.runtime.lastError.message));
          if (!resp || !resp.ok) return reject(new Error(resp?.json?.detail || resp?.error || `HTTP ${resp?.status}`));
          resolve(resp.json ?? resp.text);
        }
      );
    });
  }

  const getCookies = () => document.cookie || "";
  const getPageUrl = () => location.href;

  function isOfficeLink(aEl){
    try{
      const u = new URL(aEl.href);
      const name = decodeURIComponent(u.pathname.split("/").pop() || "").toLowerCase();
      return OFFICE_RE.test(name) || OFFICE_RE.test(aEl.textContent || "");
    }catch{ return false; }
  }

  function guessFilename(aEl){
    const fig = aEl.closest("figure.fileblock");
    if (fig){
      const nameEl = fig.querySelector(".filename .name");
      if (nameEl?.textContent) return nameEl.textContent.trim();
    }
    try{
      const u = new URL(aEl.href);
      return decodeURIComponent(u.pathname.split("/").pop() || "document.bin");
    }catch{ return "document.bin"; }
  }

  async function handleClick(e){
    const a = e.target.closest?.("a");
    if (!a || !a.href) return;
    if (!isOfficeLink(a)) return;

    e.preventDefault(); e.stopPropagation();

    showOverlay();
    setBadge("진행", "dot");
    setMsg("원격 파일 메타 확인…");
    setProgress(25);

    const payload = {
      url: a.href,
      filename: guessFilename(a),
      page_url: getPageUrl(),
      cookies: getCookies()
    };

    try{
      const r1 = await bgFetch("/tistory/fetch_url", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload)
      });
      if (!r1?.ok || !r1.id) throw new Error(r1?.detail || "fetch_url 실패");

      setMsg("공유 링크 생성…");
      setProgress(70);

      const r2 = await bgFetch(`/share/create?file_id=${encodeURIComponent(r1.id)}`, { method: "POST" });
      if (!r2?.ok || !r2.report_url) throw new Error(r2?.detail || "share 실패");

      setDone(r2.report_url);
    }catch(err){
      setError(err.message);
    }
  }

  window.addEventListener("click", handleClick, true);
  window.addEventListener("auxclick", handleClick, true);
})();
