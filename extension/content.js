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
  --mo-bg: #f8fafc;
  --mo-card: #ffffff;
  --mo-muted: #64748b;
  --mo-text: #0f172a;
  --mo-border: #e2e8f0;
  --mo-primary: #3b82f6;
  --mo-primary-600: #2563eb;
  --mo-shadow: 0 4px 20px rgba(0,0,0,.08);
  --mo-radius: 16px;
}
#mo-overlay{
  position:fixed; inset:0; z-index:2147483647; display:none;
  background: rgba(0,0,0,.36); backdrop-filter: blur(2px);
  align-items:center; justify-content:center;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, "Noto Sans KR", Arial;
}
.mo-card{
  width:min(520px, 92vw); background:var(--mo-card);
  border:1px solid var(--mo-border); border-radius: var(--mo-radius);
  box-shadow: var(--mo-shadow);
}
.mo-head{
  display:flex; align-items:center; gap:12px;
  padding:20px 20px 0 20px;
}
.mo-avatar{
  width:32px; height:32px; border-radius:10px;
  background: radial-gradient(circle at 30% 30%, #93c5fd, #2563eb);
  box-shadow: 0 4px 16px rgba(37,99,235,.3);
}
.mo-titlebox{flex:1}
.mo-title{
  font-weight: 700; font-size: 16px; color: var(--mo-text);
  margin: 0;
}
.mo-sub{ 
  color: var(--mo-muted); 
  font-size: 13px; 
  margin-top: 4px; 
  line-height: 1.4;
}

.mo-statusline{
  display:flex; align-items:center; gap:12px; 
  margin: 20px 20px 0 20px;
  flex-direction: row-reverse;
}
.mo-badge{
  display:inline-flex; align-items:center; gap:6px;
  border-radius:999px; padding:6px 10px; font-size:11px;
  background:#f3f4f6; border:1px solid var(--mo-border); 
  color:#374151; font-weight: 800;
}
.mo-badge.dot::before{
  content:""; width:8px; height:8px; border-radius:999px; background:#9ca3af;
}
.mo-badge.ok .dot::before{ background:#16a34a;}
.mo-badge.warn .dot::before{ background:#3b82f6;}
.mo-badge.err .dot::before{ background:#dc2626;}

.mo-msg{ 
  font-size:13px; 
  color: var(--mo-text); 
  flex: 1;
  min-width: 140px;
}

.mo-progress{
  height:10px; margin: 0 20px 0 0; 
  background:#f3f4f6; border-radius:10px; 
  overflow:hidden; border:1px solid #e5e7eb;
  box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
}
.mo-progress > i{ 
  display:block; height:100%; width:0%; 
  background:linear-gradient(90deg,#34d399,#06b6d4); 
  border-radius:10px; transition: width .3s ease; 
}

.mo-actions{
  display:flex; gap:16px; justify-content:center; 
  padding: 20px 20px 20px 20px;
}
.mo-btn{
  flex: 1;
  padding:12px 0; border-radius:10px; font-weight:600; font-size:14px;
  border:none; cursor:pointer; transition: all 0.2s ease;
}
.mo-btn:hover:not(:disabled) {
  transform: translateY(-1px);
}
.mo-btn:disabled {
  background: #f3f4f6;
  color: #9ca3af;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}
.mo-btn.primary{
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
  color:#fff;
  box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
}
.mo-btn.primary:hover:not(:disabled) {
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}
.mo-btn.secondary{
  background: transparent;
  color: #3b82f6;
  border: 2px solid #3b82f6;
  box-shadow: none;
}
.mo-btn.secondary:hover:not(:disabled) {
  background: #eff6ff;
  color: #2563eb;
  border: 2px solid #3b82f6;
  box-shadow: 0 2px 6px rgba(59, 130, 246, 0.2);
}
.mo-btn.secondary:active:not(:disabled) {
  background: #dbeafe;
  color: #1e40af;
  border: 2px solid #2563eb;
  transform: translateY(0);
  box-shadow: 0 1px 3px rgba(37, 100, 235, 0.3);
}
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
            <div class="mo-sub">문서 실행 전 악성 행위 예측 서비스</div>
          </div>
        </div>

        <div class="mo-statusline">
          <span class="mo-badge dot" id="mo-badge"><span class="dot"></span>대기</span>
          <div class="mo-progress"><i id="mo-prog"></i></div>
        </div>
        <div class="mo-msg" id="mo-msg" style="margin: 8px 20px 0 20px;">준비 중…</div>

        <div class="mo-actions">
          <button class="mo-btn secondary" id="mo-close">닫기</button>
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
    console.log("리포트 URL:", reportUrl);
    $btnOpen.onclick = () => {
      console.log("리포트 확인 버튼 클릭됨, URL:", reportUrl);
      window.open(reportUrl, "_blank", "noopener,noreferrer");
    };
  }
  function setError(text){
    setBadge("오류", "err");
    
    if (text && text.includes("Extension context invalidated")) {
      setMsg("확장 프로그램이 일시적으로 비활성화되었습니다. 페이지를 새로고침하고 다시 시도해주세요.");
    } else {
      setMsg(text || "분석 실패");
    }
    
    setProgress(100);
    $btnOpen.disabled = true;
    $btnOpen.textContent = "다시 시도";
    $btnOpen.onclick = () => {
      window.location.reload();
    };
  }

  function bgFetch(path, init = {}) {
    return new Promise((resolve, reject) => {
      if (!chrome.runtime?.id) {
        reject(new Error("Extension context invalidated. Please refresh the page and try again."));
        return;
      }

      chrome.runtime.sendMessage(
        { type: "maloffice.fetch", url: `${API_BASE}${path}`, init },
        (resp) => {
          if (chrome.runtime.lastError) {
            const errorMsg = chrome.runtime.lastError.message;
            if (errorMsg.includes("context invalidated") || errorMsg.includes("Extension context invalidated")) {
              reject(new Error("Extension context invalidated. Please refresh the page and try again."));
            } else {
              reject(new Error(errorMsg));
            }
            return;
          }
          
          if (!resp) {
            reject(new Error("No response from background script"));
            return;
          }

          if (resp.contextInvalid) {
            reject(new Error("Extension context invalidated. Please refresh the page and try again."));
            return;
          }

          if (!resp.ok) {
            reject(new Error(resp?.json?.detail || resp?.error || `HTTP ${resp?.status}`));
            return;
          }
          
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
