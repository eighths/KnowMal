(() => {
  // const API_BASE = "https://knowmal.duckdns.org";
  const API_BASE = "http://localhost:8000"; //개발용
  const OFFICE_RE = /\.(docx?|xlsx?|pptx?)$/i;
  const EXTRA_FILE_RE = /\.(zip|7z|rar|alz|egg|tar|gz|bz2|xz|pdf|hwp|hwpx|txt|rtf|json|ps1|js|vbs|wsf|jar|apk|ipa|exe|dll|msi|bat|cmd|lnk|scr|iso|img|bin)$/i;

  const STYLE_ID = "maloffice-style";
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
  width:min(720px, 96vw); background:var(--mo-card);
  border:1px solid var(--mo-border); border-radius: var(--mo-radius);
  box-shadow: var(--mo-shadow);
  min-height: 320px;
}
.mo-head{
  display:flex; align-items:center; gap:14px;
  padding:24px 24px 0 24px;
}
.mo-logo{ width:32px; height:32px; border-radius:10px; object-fit:contain; box-shadow: 0 4px 16px rgba(0,0,0,.1); border:1px solid #cbd5e1; background:#fff }
.mo-titlebox{flex:1}
.mo-title{
  font-weight: 700; font-size: 18px; color: var(--mo-text);
  margin: 0;
}
.mo-sub{ 
  color: var(--mo-muted); 
  font-size: 14px; 
  margin-top: 4px; 
  line-height: 1.4;
}

.mo-steps{ display:flex; align-items:center; gap:22px; margin: 32px 24px 0 24px; }
.mo-step{ display:flex; align-items:center; gap:14px; font-size:16px; color:#9ca3af; }
.mo-step i{ width:16px; height:16px; border-radius:999px; background:#e5e7eb; display:inline-block; box-shadow: inset 0 0 0 1px #cbd5e1 }
#mo-step-wait.active{ color:#374151 !important; }
#mo-step-wait.active i{ background:#4b5563 !important; }
#mo-step-run.active{ color:#1e40af !important; }
#mo-step-run.active i{ background:#3b82f6 !important; }
#mo-step-done.active{ color:#15803d !important; }
#mo-step-done.active i{ background:#16a34a !important; }

.mo-badge{
  display:inline-flex; align-items:center; gap:6px;
  border-radius:999px; padding:6px 10px; font-size:11px;
  background:#f3f4f6; border:1px solid var(--mo-border); 
  color:#374151; font-weight: 800;
}
.mo-badge.dot::before{
  content:""; width:8px; height:8px; border-radius:999px; background:#3b82f6;
}
.mo-badge.ok::before{ background:#16a34a;}
.mo-badge.warn::before{ background:#3b82f6;}
.mo-badge.err::before{ background:#dc2626;}
.mo-badge.dot{ background:#eff6ff; border-color:#bfdbfe; color:#1e40af; }
.mo-badge.ok { background:#ecfdf5; border-color:#a7f3d0; color:#065f46; }
.mo-badge.err{ background:#fef2f2; border-color:#fecaca; color:#991b1b; }

.mo-msg{ 
  font-size:14px; 
  color: #1e40af;
  margin: 20px 24px 0 24px;
  padding: 16px 0;
  background: rgba(239, 246, 255, 0.8);
  border: 2px dashed #bfdbfe;
  border-radius: 12px;
  text-align: center;
  font-weight: 500;
  line-height: 1.4;
}
.mo-msg.processing{ 
  color: #1e40af;
  background: rgba(239, 246, 255, 0.8);
  border-color: #bfdbfe;
}
.mo-msg.completed{ 
  color: #065f46;
  background: rgba(236, 253, 245, 0.8);
  border-color: #a7f3d0;
}
.mo-msg.error{ 
  color: #991b1b;
  background: rgba(254, 242, 242, 0.8);
  border-color: #fecaca;
}

#mo-overlay .mo-result::after, #mo-overlay .mo-result::before{ content: none !important; display: none !important; }
#mo-overlay .mo-badge::before{ content:"" !important; }


.mo-actions{
  display:flex; gap:16px; justify-content:center; 
  padding: 32px 24px 28px 24px;
}
.mo-actions .mo-btn:first-child{ order: -1; }
.mo-btn{
  flex: 1;
  padding:14px 0; border-radius:10px; font-weight:600; font-size:15px;
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
.mo-btn.download{
  background: #e5e7eb;
  color: #111827;
  border: 2px solid #e5e7eb;
}
.mo-btn.download:hover:not(:disabled){
  background: #e2e8f0;
  border-color: #e2e8f0;
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

.mo-result{ 
  margin: 24px 24px 0 24px; 
  padding: 20px; 
  border-radius: 12px; 
  border: 2px dashed; 
  display: none; 
  text-align: center;
}
.mo-result.safe{ 
  border-color: #34d399; 
  background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%); 
}
.mo-result.danger{ 
  border-color: #f87171; 
  background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); 
}
.mo-result .mo-result-content{
  display: flex;
  align-items: flex-start;
  gap: 16px;
  text-align: left;
}
.mo-result .mo-siren-icon{
  font-size: 32px;
  flex-shrink: 0;
  margin-top: 4px;
}
.mo-result .mo-text-block{
  flex: 1;
  text-align: center;
}
.mo-result .mo-main-title{ 
  font-size: 18px; 
  font-weight: 800; 
  margin: 0 0 8px 0; 
  color: #dc2626;
}
.mo-result .mo-sub-title{ 
  font-size: 15px; 
  font-weight: 600; 
  margin: 0 0 12px 0; 
  color: #991b1b;
}
.mo-result .mo-desc-line{ 
  font-size: 13px; 
  color: #7f1d1d; 
  margin: 0 0 4px 0; 
  line-height: 1.4;
}
.mo-result.safe .mo-main-title{ color: #16a34a; }
.mo-result.safe .mo-sub-title{ color: #15803d; }
.mo-result.safe .mo-desc-line{ color: #166534; }

#mo-step-done.active.status-safe{ color:#15803d !important; }
#mo-step-done.active.status-safe i{ background:#16a34a !important; }
#mo-step-done.active.status-danger{ color:#b91c1c !important; }
#mo-step-done.active.status-danger i{ background:#dc2626 !important; }
    `;
    document.head.appendChild(s);
  }

  let $ov, $prog, $badge, $msg, $btnOpen, $btnClose, $btnDownload, $result;
  let lastDownloadUrl = "";

  function ensureOverlay() {
    if ($ov) return;
    injectStyles();
    $ov = document.createElement("div");
    $ov.id = "mo-overlay";
    const logoUrl = (typeof chrome !== "undefined" && chrome.runtime?.getURL) ? chrome.runtime.getURL("images/KnowMal_logo.png") : "";
    $ov.innerHTML = `
      <div class="mo-card" role="dialog" aria-modal="true">
        <div class="mo-head">
          <img class="mo-logo" src="${logoUrl}" alt="KnowMal" />
          <div class="mo-titlebox">
            <div class="mo-title">KnowMal</div>
            <div class="mo-sub">문서 실행 전 악성 행위 예측 서비스</div>
          </div>
        </div>

        <div class="mo-msg" id="mo-msg">파일의 악성 행위를 검사 중입니다.</div>

        <div id="mo-result" class="mo-result" aria-live="polite"></div>
        <div class="mo-steps" aria-hidden="true">
          <span class="mo-step" id="mo-step-wait"><i></i>대기</span>
          <span class="mo-step" id="mo-step-run"><i></i>진행</span>
          <span class="mo-step" id="mo-step-done"><i></i>완료</span>
        </div>

        <div class="mo-actions">
          <button class="mo-btn primary" id="mo-open" disabled>상세 결과 확인</button>
          <button class="mo-btn download" id="mo-download" disabled>파일 다운로드</button>
          <button class="mo-btn secondary" id="mo-close">닫기</button>
        </div>
      </div>
    `;
    document.body.appendChild($ov);

    $prog = null; 
    $badge = null;
    $msg = $ov.querySelector("#mo-msg");
    $btnOpen = $ov.querySelector("#mo-open");
    $btnClose = $ov.querySelector("#mo-close");
    $btnDownload = $ov.querySelector("#mo-download");
    $result = $ov.querySelector("#mo-result");
    $btnClose.onclick = () => {
      $ov.style.display = "none";
      resetModalState();
    };
  }

  function showOverlay() {
    ensureOverlay();
    $ov.style.display = "flex";
    setMsg("파일의 악성 행위를 검사 중입니다.");
    setProgress(25);
    setStep("진행");
    $btnOpen.disabled = true;
    $btnOpen.onclick = null;
    if ($btnDownload){ $btnDownload.disabled = false; }
  }
  function setMsg(t, type = "processing"){ 
    $msg.textContent = t; 
    $msg.className = `mo-msg ${type}`;
  }
  function setProgress(pct){ 
  }
  function setBadge(text, kind){
    if (!$badge) return; 
    $badge.className = `mo-badge ${kind}`;
    $badge.textContent = text;
  }
  function setResult(kind){
    if (!$result) {
      return;
    }
    if (!kind){ 
      $result.style.display = "none"; 
      $result.className = "mo-result"; 
      $result.innerHTML = ""; 
      return; 
    }
    const isSafe = kind === "safe";
    $result.style.display = "block";
    $result.className = "mo-result " + (isSafe ? "safe" : "danger");
    if (isSafe){
      $result.innerHTML = `
        <div class="mo-result-content">
          <div class="mo-siren-icon">✅</div>
          <div class="mo-text-block">
            <div class="mo-main-title">정상: 안전한 파일</div>
            <div class="mo-sub-title">악성코드를 포함하지 않는 파일입니다.</div>
          </div>
        </div>`;
    } else {
      $result.innerHTML = `
        <div class="mo-result-content">
          <div class="mo-siren-icon">🚨</div>
          <div class="mo-text-block">
            <div class="mo-main-title">위험: 파일을 열지 마세요</div>
            <div class="mo-sub-title">악성코드가 탐지되었습니다</div>
            <div class="mo-desc-line">자세한 검사 결과를 보기 위해선</div>
            <div class="mo-desc-line">아래 상세 결과 확인 버튼을 클릭하세요</div>
          </div>
        </div>`;
    }
  }
  function setStep(state){
    const steps = [
      ["대기", document.getElementById("mo-step-wait")],
      ["진행", document.getElementById("mo-step-run")],
      ["완료", document.getElementById("mo-step-done")]
    ];
    for (const [name, el] of steps){
      if (!el) continue;
      el.classList.remove("active","done");
    }
    for (const [name, el] of steps){
      if (!el) continue;
      if (name === state){ el.classList.add("active"); }
    }
  }
  function setDone(reportUrl, status){
    console.log("[KnowMal] setDone 호출됨 - status:", status);
    if (status === "safe") {
      console.log("[KnowMal] 정상 파일로 판정");
    } else if (status === "danger") {
      console.log("[KnowMal] 악성 파일로 판정");
    } else {
      console.log("[KnowMal] 상태 불명");
    }
    $msg.style.display = "none";
    setProgress(100);
    setStep("완료");
    try{
      const doneEl = document.getElementById("mo-step-done");
      if (doneEl){
        doneEl.classList.remove("status-safe","status-danger");
        if (status === "safe") doneEl.classList.add("status-safe");
        if (status === "danger") doneEl.classList.add("status-danger");
      }
    }catch{}
    $btnOpen.disabled = false;
    console.log("리포트 URL:", reportUrl);
    $btnOpen.onclick = () => {
      console.log("리포트 확인 버튼 클릭됨, URL:", reportUrl);
      window.open(reportUrl, "_blank", "noopener,noreferrer");
    };
    if (status){ 
      console.log("[KnowMal] setDone에서 setResult 호출 - status:", status);
      setResult(status); 
    }
    try{
      const u = new URL(reportUrl);
      const urlStatus = u.searchParams.get("status"); // safe | danger
      if (urlStatus === "safe" || urlStatus === "danger"){ 
        console.log("[KnowMal] URL에서 추출한 상태로 setResult 호출 - urlStatus:", urlStatus);
        setResult(urlStatus);
        const doneEl = document.getElementById("mo-step-done");
        if (doneEl){
          doneEl.classList.remove("status-safe","status-danger");
          doneEl.classList.add(urlStatus === "safe" ? "status-safe" : "status-danger");
        }
      }
    }catch{}
    
    if (!status) {
      console.log("[KnowMal] 상태 불명, 기본적으로 정상으로 처리");
      setResult("safe");
    }

    if ($btnDownload){
      $btnDownload.onclick = () => {
        try{
          const urlToOpen = lastDownloadUrl;
          if (urlToOpen){ window.open(urlToOpen, "_blank", "noopener,noreferrer"); }
        }catch{}
      };
      $btnDownload.disabled = false;
    }
  }
  function resetModalState(){
    if ($result) {
      $result.style.display = "none";
      $result.className = "mo-result";
      $result.innerHTML = "";
    }
    if ($msg) {
      $msg.style.display = "block";
      setMsg("파일의 악성 행위를 검사 중입니다.", "processing");
    }
    setProgress(0);
    setStep("대기");
    if ($btnOpen) {
      $btnOpen.disabled = true;
      $btnOpen.textContent = "상세 결과 확인";
      $btnOpen.onclick = null;
    }
    try{
      const doneEl = document.getElementById("mo-step-done");
      if (doneEl){
        doneEl.classList.remove("status-safe","status-danger");
      }
    }catch{}
  }

  function setError(text){
    
    if (text && text.includes("Extension context invalidated")) {
      setMsg("확장 프로그램이 일시적으로 비활성화되었습니다. 페이지를 새로고침하고 다시 시도해주세요.", "error");
    } else {
      setMsg(text || "분석 실패", "error");
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

  function bgFetchBinary(url, init = {}){
    return new Promise((resolve, reject) => {
      if (!chrome.runtime?.id){ reject(new Error("Extension context invalidated.")); return; }
      chrome.runtime.sendMessage({ type: "maloffice.fetchBinary", url, init }, (resp)=>{
        if (chrome.runtime.lastError){ reject(new Error(chrome.runtime.lastError.message)); return; }
        if (!resp){ reject(new Error("No response from background script")); return; }
        resolve(resp);
      });
    });
  }

  const getCookies = () => document.cookie || "";
  const getPageUrl = () => location.href;

  function isOfficeLink(aEl){
    try{
      const u = new URL(aEl.href);
      const name = decodeURIComponent(u.pathname.split("/").pop() || "").toLowerCase();
      const host = (u.hostname || "").toLowerCase();
      const looksLikeTarget = OFFICE_RE.test(name) || OFFICE_RE.test(aEl.textContent || "");
      const looksLikeFile = looksLikeTarget || EXTRA_FILE_RE.test(name);
      const isTistoryHost = /(^|\.)tistory\.(com|io)$/.test(host) || /kakaocdn\.(net|com)$/.test(host);
      const pathHint = /attach|attachment|file|download/gi.test(u.pathname);
      return looksLikeFile || (isTistoryHost && pathHint);
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

  function findDownloadAnchorOrUrl(target){
    const a1 = target.closest?.("a");
    if (a1?.href) return { href: a1.href, anchor: a1 };
    const fig = target.closest?.("figure.fileblock, .fileblock");
    if (fig){
      const a2 = fig.querySelector("a[href]");
      if (a2?.href) return { href: a2.href, anchor: a2 };
      const elWithData = fig.querySelector("[data-href],[data-url]");
      const h = elWithData?.getAttribute?.("data-href") || elWithData?.getAttribute?.("data-url");
      if (h) return { href: h };
    }
    const el = target.closest?.("[data-href],[data-url]") || target;
    const dataHref = el?.getAttribute?.("data-href") || el?.getAttribute?.("data-url");
    if (dataHref) return { href: dataHref };
    return null;
  }

  function cancelEvent(e){
    try{ e.preventDefault(); }catch{}
    try{ e.stopImmediatePropagation?.(); }catch{}
    try{ e.stopPropagation(); }catch{}
  }

  async function handleClick(e){
    const found = findDownloadAnchorOrUrl(e.target);
    if (!found || !found.href) return;
    const fakeA = { href: found.href, textContent: found.anchor?.textContent || "" };
    if (!isOfficeLink(fakeA)) return;

    try { console.debug("[KnowMal] final download href:", found.href); } catch {}

    cancelEvent(e);
    lastDownloadUrl = found.href;

    showOverlay();

    const payload = {
      url: found.href,
      filename: guessFilename(found.anchor || { href: found.href, closest: () => null, textContent: "" }),
      page_url: getPageUrl(),
      cookies: getCookies(),
      user_agent: navigator.userAgent
    };

    try{
      const r1 = await bgFetch("/tistory/fetch_url", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload)
      });
      if (!r1?.ok) { throw new Error(r1?.detail || "fetch_url 실패"); }
      if (!r1?.ok || !r1.id) throw new Error(r1?.detail || "fetch_url 실패");

      setMsg("파일의 악성 행위를 검사 중입니다.");
      setProgress(70);

      const r2 = await bgFetch(`/share/create?file_id=${encodeURIComponent(r1.id)}`, { method: "POST" });
      if (!r2?.ok || !r2.report_url) throw new Error(r2?.detail || "share 실패");
      
      let norm = "";
      if (r2.status) {
        console.log("[KnowMal] 응답에서 상태 추출:", r2.status);
        const statusLower = String(r2.status).toLowerCase();
        norm = 
          /(safe|정상|clean|benign)/.test(statusLower) ? "safe" :
          /(mal|malicious|virus|trojan|infected|danger|위험)/.test(statusLower) ? "danger" : "";
      }
      
      if (!norm) {
        try{
          const u = new URL(r2.report_url);
          const urlStatus = u.searchParams.get("status");
          if (urlStatus) {
            console.log("[KnowMal] URL에서 상태 추출:", urlStatus);
            const urlStatusLower = urlStatus.toLowerCase();
            norm = 
              /(safe|정상|clean|benign)/.test(urlStatusLower) ? "safe" :
              /(mal|malicious|virus|trojan|infected|danger|위험)/.test(urlStatusLower) ? "danger" : "";
          }
        }catch{}
      }
      
      console.log("[KnowMal] 상태 감지 - 최종 결과:", norm);
      setDone(r2.report_url, norm);
    }catch(err){
      try{
        setMsg("파일의 악성 행위를 검사 중입니다.");
        setProgress(40);

        const bin = await bgFetchBinary(found.href, {
          credentials: "include",
          referrer: getPageUrl(),
          referrerPolicy: "strict-origin-when-cross-origin",
          headers: { "Referer": getPageUrl() }
        });
        if (!bin?.ok){ throw new Error(`원본 다운로드 실패: HTTP ${bin?.status}`); }
        const ab = bin.buffer;
        const fd = new FormData();
        const fileBlob = new Blob([ab]);
        fd.append("file", fileBlob, guessFilename(found.anchor || { href: found.href, closest: ()=>null, textContent: "" }));

        const up = await fetch(`${API_BASE}/scan/upload`, { method: "POST", body: fd });
        if (!up.ok){ throw new Error(`업로드 실패: HTTP ${up.status}`); }
        const upJson = await up.json();
        if (!upJson?.ok || !upJson?.id){ throw new Error("업로드 응답 오류"); }

        setMsg("파일의 악성 행위를 검사 중입니다.");
        setProgress(70);
        const sh = await fetch(`${API_BASE}/share/create?file_id=${encodeURIComponent(upJson.id)}`, { method: "POST" });
        if (!sh.ok){ throw new Error(`공유 실패: HTTP ${sh.status}`); }
        const shJson = await sh.json();
        if (!shJson?.ok || !shJson?.report_url){ throw new Error("공유 응답 오류"); }
        
        let norm2 = "";
        if (shJson.status) {
          const statusLower = String(shJson.status).toLowerCase();
          norm2 = 
            /(safe|정상|clean|benign)/.test(statusLower) ? "safe" :
            /(mal|malicious|virus|trojan|infected|danger|위험)/.test(statusLower) ? "danger" : "";
        }
        
        if (!norm2) {
          try{
            const u = new URL(shJson.report_url);
            const urlStatus = u.searchParams.get("status");
            if (urlStatus) {
              const urlStatusLower = urlStatus.toLowerCase();
              norm2 = 
                /(safe|정상|clean|benign)/.test(urlStatusLower) ? "safe" :
                /(mal|malicious|virus|trojan|infected|danger|위험)/.test(urlStatusLower) ? "danger" : "";
            }
          }catch{}
        }
        
        setDone(shJson.report_url, norm2);
      }catch(e2){
        setError(e2.message || err.message);
      }
    }
  }

  const captureOpts = { capture: true, passive: false };
  const bubbleOpts = { capture: false, passive: false };
  window.addEventListener("click", handleClick, bubbleOpts);
  window.addEventListener("auxclick", handleClick, bubbleOpts);
})();
