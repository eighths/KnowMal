(() => {
  const API_BASE = "http://localhost:8000";
  const OFFICE_RE = /\.(docx?|xlsx?|pptx?|html?|json)$/i;
  const STYLE_ID = "maloffice-style-v2";

  function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
:root{
  --mo-bg:#f8fafc;--mo-card:#fff;--mo-muted:#64748b;--mo-text:#0f172a;
  --mo-border:#e2e8f0;--mo-shadow:0 4px 20px rgba(0,0,0,.08);--mo-radius:16px;
}
#mo-overlay{position:fixed;inset:0;z-index:2147483647;display:none;
  background:rgba(0,0,0,.36);backdrop-filter:blur(2px);
  align-items:center;justify-content:center;
  font-family:system-ui,-apple-system,Segoe UI,Roboto,"Noto Sans KR",Arial;}
.mo-card{width:min(520px,92vw);background:var(--mo-card);
  border:1px solid var(--mo-border);border-radius:var(--mo-radius);box-shadow:var(--mo-shadow);}
.mo-head{display:flex;align-items:center;gap:12px;padding:20px 20px 0 20px;}
.mo-avatar{width:32px;height:32px;border-radius:10px;
  background:radial-gradient(circle at 30% 30%,#93c5fd,#2563eb);
  box-shadow:0 4px 16px rgba(37,99,235,.3);}
.mo-titlebox{flex:1}.mo-title{font-weight:700;font-size:16px;color:#0f172a;margin:0;}
.mo-sub{color:var(--mo-muted);font-size:13px;margin-top:4px;line-height:1.4;}
.mo-statusline{display:flex;align-items:center;gap:12px;margin:20px 20px 0 20px;flex-direction:row-reverse;}
.mo-badge{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:6px 10px;
  font-size:11px;background:#f3f4f6;border:1px solid var(--mo-border);color:#374151;font-weight:800;}
.mo-badge.dot::before{content:"";width:8px;height:8px;border-radius:999px;background:#9ca3af;margin-right:6px;}
.mo-badge.ok::before{background:#16a34a;}
.mo-badge.err::before{background:#dc2626;}
.mo-msg{font-size:13px;color:#0f172a;margin:8px 20px 0 20px;}
.mo-progress{height:10px;margin:0 20px 0 0;background:#f3f4f6;border-radius:10px;overflow:hidden;
  border:1px solid #e5e7eb;box-shadow:inset 0 1px 3px rgba(0,0,0,.1);}
.mo-progress>i{display:block;height:100%;width:0%;
  background:linear-gradient(90deg,#34d399,#06b6d4);border-radius:10px;transition:width .3s ease;}
.mo-actions{display:flex;gap:16px;justify-content:center;padding:20px;}
.mo-btn{flex:1;padding:12px 0;border-radius:10px;font-weight:600;font-size:14px;border:none;cursor:pointer;transition:all .2s;}
.mo-btn:disabled{background:#f3f4f6;color:#9ca3af;cursor:not-allowed;}
.mo-btn.primary{background:linear-gradient(135deg,#3b82f6 0%,#1d4ed8 100%);color:#fff;box-shadow:0 2px 8px rgba(59,130,246,.3);}
.mo-btn.primary:hover:not(:disabled){box-shadow:0 4px 12px rgba(59,130,246,.4);}
.mo-btn.secondary{background:transparent;color:#3b82f6;border:2px solid #3b82f6;}
.mo-btn.secondary:hover:not(:disabled){background:#eff6ff;color:#2563eb;border:2px solid #3b82f6;box-shadow:0 2px 6px rgba(59,130,246,.2);}
`;
    document.head.appendChild(s);
  }

  let $ov,$prog,$badge,$msg,$btnOpen;
  function ensureOverlay(){
    if($ov) return;
    injectStyles();
    $ov=document.createElement("div");
    $ov.id="mo-overlay";
    $ov.innerHTML=`
    <div class="mo-card" role="dialog" aria-modal="true">
      <div class="mo-head">
        <div class="mo-avatar"></div>
        <div class="mo-titlebox">
          <div class="mo-title">KnowMal</div>
          <div class="mo-sub">문서 실행 전 악성 행위 예측 서비스</div>
        </div>
      </div>
      <div class="mo-statusline">
        <span class="mo-badge dot" id="mo-badge">대기</span>
        <div class="mo-progress"><i id="mo-prog"></i></div>
      </div>
      <div class="mo-msg" id="mo-msg">준비 중…</div>
      <div class="mo-actions">
        <button class="mo-btn secondary" id="mo-close">창 닫기</button>
        <button class="mo-btn primary" id="mo-open" disabled>리포트 확인</button>
      </div>
    </div>`;
    document.body.appendChild($ov);
    $prog=$ov.querySelector("#mo-prog");
    $badge=$ov.querySelector("#mo-badge");
    $msg=$ov.querySelector("#mo-msg");
    $btnOpen=$ov.querySelector("#mo-open");
    $ov.querySelector("#mo-close").onclick=()=>($ov.style.display="none");
  }
  function showOverlay(){ ensureOverlay(); $ov.style.display="flex"; badge("진행"); msg("서버 준비 중…"); prog(8); disableOpen(); }
  const msg = (t)=>($msg.textContent=t);
  const prog = (p)=>($prog.style.width=`${Math.max(0,Math.min(100,p))}%`);
  const badge = (t,ok=false,err=false)=>{ $badge.textContent=t; $badge.className=`mo-badge ${ok?"ok":err?"err":"dot"}`; };
  const enableOpen = (url)=>{ 
    console.log("[KnowMal] enableOpen called with URL:", url);
    $btnOpen.disabled=false; 
    $btnOpen.onclick=()=>{
      console.log("[KnowMal] Opening URL:", url);
      window.open(url,"_blank","noopener,noreferrer");
    }; 
  };
  const disableOpen = ()=>{ $btnOpen.disabled=true; $btnOpen.onclick=null; };

  function bgSend(type, payload={}){
    return new Promise((resolve,reject)=>{
      if (typeof chrome === 'undefined' || !chrome.runtime) {
        reject(new Error("Chrome extension context not available"));
        return;
      }
      chrome.runtime.sendMessage({type, ...payload}, (resp)=>{
        if(chrome.runtime.lastError) return reject(new Error(chrome.runtime.lastError.message));
        resolve(resp);
      });
    });
  }
  function bgFetch(path, init={}){
    return new Promise((resolve,reject)=>{
      console.log("[KnowMal] bgFetch called with:", { path, init });
      if (typeof chrome === 'undefined' || !chrome.runtime) {
        reject(new Error("Chrome extension context not available"));
        return;
      }
      chrome.runtime.sendMessage({type:"KM_FETCH", url:`${API_BASE}${path}`, init}, (resp)=>{
        console.log("[KnowMal] bgFetch response:", resp);
        if(chrome.runtime.lastError) return reject(new Error(chrome.runtime.lastError.message));
        if(!resp?.ok) return reject(new Error(resp?.json?.detail || resp?.json?.error || `HTTP ${resp?.status}`));
        resolve(resp.json ?? resp.text);
      });
    });
  }

  function parseDownloadUrlAttr(node){
    const raw=node.getAttribute?.("download_url"); if(!raw) return null;
    const parts=raw.split(":"); const url=parts.slice(2).join(":"); const filename=parts[1]||"";

    let message_id = "";
    try {
      const u = new URL(url);
      message_id = u.searchParams.get("th") || "";
      console.log("[KnowMal] Found message_id from download_url:", message_id);
    } catch (e) {
      console.log("[KnowMal] Failed to parse download_url:", e);
    }
    
    return { url, filename, message_id };
  }
  function extractFromHref(a){
    try{
      const u=new URL(a.href, location.origin);
      const looks = u.hostname.includes("mail.google.com") && (u.search.includes("attid=")||u.search.includes("view=att")||u.search.includes("disp=safe"));
      if(!looks) return null;

      const message_id = u.searchParams.get("th") || "";
      console.log("[KnowMal] Found message_id from href:", message_id);
      
      const label=(a.getAttribute("aria-label")||a.textContent||"").trim();
      const m=label.match(/[^\/\\]+?\.(docx?|xlsx?|pptx?)/i);
      const filename=m?m[0]:"document.bin";
      return { url:a.href, filename, message_id };
    }catch(e){ return null; }
  }
  function findAttachment(target){
    let n=target;
    for(let i=0;i<8 && n;i++){ 
      const info=parseDownloadUrlAttr(n); 
      if(info && OFFICE_RE.test(info.filename)) {
        console.log("[KnowMal] Found attachment via download_url:", info);
        return info; 
      }
      n=n.parentElement; 
    }
    const a=target.closest?.("a");
    if(a){
      const info2=extractFromHref(a); 
      if(info2 && OFFICE_RE.test(info2.filename)) {
        console.log("[KnowMal] Found attachment via href:", info2);
        return info2;
      }
      const txt=(a.textContent||a.getAttribute("aria-label")||"").trim().toLowerCase();
      if(OFFICE_RE.test(txt)) {
        let message_id = "";
        try {
          const u = new URL(a.href, location.origin);
          message_id = u.searchParams.get("th") || "";
        } catch (e) {
        }
        const result = { url:a.href, filename: txt.match(OFFICE_RE)?.[0] || "document.bin", message_id };
        console.log("[KnowMal] Found attachment via text match:", result);
        return result;
      }
    }
    return null;
  }
  const filenameOnly=(n)=>{ try{ const u=new URL(n, location.origin); return decodeURIComponent(u.pathname.split("/").pop()||n);}catch(e){ return n; } };
  function parseIds(href){
    try{
      const u=new URL(href, location.origin);
      return {
        thread_id: u.searchParams.get("th") || u.searchParams.get("threadId") || null,
        permmsgid: u.searchParams.get("permmsgid") || null,
        message_id: u.searchParams.get("msgid") || null
      };
    }catch(e){ return {thread_id:null,permmsgid:null,message_id:null}; }
  }

  function getFallbackMessageId(){
    try{
      const el = document.querySelector('[data-legacy-message-id], [data-message-id]');
      const id1 = el?.getAttribute('data-legacy-message-id') || el?.getAttribute('data-message-id');
      if (id1) return id1;
      const u = new URL(location.href);
      const id2 = u.searchParams.get('msgid') || u.searchParams.get('th') || u.searchParams.get('message_id');
      if (id2) return id2;
    }catch(_){}
    return "";
  }

  function getActiveGmailEmail(){
    try{
      const imgAlt = document.querySelector('a[aria-label^="Google 계정:"] img')?.alt
        || document.querySelector('a[aria-label*="Google Account:"] img')?.alt
        || "";
      const m1 = imgAlt && imgAlt.match(/\(([^)]+)\)\s*$/);
      if (m1 && m1[1]) return m1[1].trim();

      const al = document.querySelector('a[aria-label^="Google 계정:"]')?.getAttribute('aria-label')
        || document.querySelector('a[aria-label*="Google Account:"]')?.getAttribute('aria-label')
        || "";
      const m2 = al && al.match(/\(([^)]+)\)\s*$/);
      if (m2 && m2[1]) return m2[1].trim();

      const title = document.querySelector('a[title*="@"]')?.getAttribute('title') || "";
      const m3 = title && title.match(/([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})/i);
      if (m3 && m3[1]) return m3[1].trim();

      const btnTxt = document.querySelector('a[aria-label*="@"]')?.getAttribute('aria-label') || "";
      const m4 = btnTxt && btnTxt.match(/([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})/i);
      if (m4 && m4[1]) return m4[1].trim();
    }catch(_){}
    return "";
  }

  function waitOAuthDoneEvent(timeoutMs=180000){
    return new Promise((resolve)=>{
      if (typeof chrome === 'undefined' || !chrome.runtime) {
        resolve(false);
        return;
      }
      let done=false;
      const timer=setTimeout(()=>{ if(done) return; done=true; chrome.runtime.onMessage.removeListener(onMsg); resolve(false); }, timeoutMs);
      function onMsg(m){
        if(m?.type==="KM_OAUTH_DONE" && !done){
          done=true; clearTimeout(timer); chrome.runtime.onMessage.removeListener(onMsg); resolve(true);
        }
      }
      chrome.runtime.onMessage.addListener(onMsg);
    });
  }
  async function ensureOAuth() {
    try {
      const pageEmail = getActiveGmailEmail();
      const res = await bgSend("KM_OAUTH_STATUS", { email: pageEmail });
      if(res?.ok && res?.authed) {
        console.log("[KnowMal] OAuth already authenticated in DB");
        try{
          const gmailEmail = pageEmail;
          const serverEmail = res?.email || "";
          if (gmailEmail && serverEmail && gmailEmail.toLowerCase() !== serverEmail.toLowerCase()){
            console.warn("[KnowMal] Gmail page account != server account", { gmailEmail, serverEmail });
            await chrome.storage.local.remove(["KM_OAUTH_READY"]);
            const f = await bgSend("KM_OAUTH_ENSURE_FORCE");
            if(!(f?.ok && (f?.authed || f?.authorized))) throw new Error("계정 전환 재인증 실패");
          }
        }catch(e){ console.log("[KnowMal] account compare skipped:", e); }
        await chrome.storage.local.set({ KM_OAUTH_READY: true });
        return true;
      }
      try { await chrome.storage.local.remove(["KM_OAUTH_READY"]); } catch(_) {}
    } catch (e) {
      console.log("[KnowMal] DB OAuth check failed:", e);
    }

    console.log("[KnowMal] No OAuth in DB, starting OAuth flow");
    const res = await bgSend("KM_OAUTH_ENSURE", { email: getActiveGmailEmail() });
    if(!res?.ok) throw new Error(res?.error || "OAuth 시작 실패");
    if(res.authed || res.authorized) {
      console.log("[KnowMal] OAuth completed");
      return true;
    }
    
    msg("Google 동의 창에서 인증을 완료하세요…"); prog(35);

    const race = await Promise.race([
      (async()=>{ 
        const deadline=Date.now()+180000;
        while(Date.now()<deadline){
          await new Promise(r=>setTimeout(r,1000));
          const s=await bgSend("KM_OAUTH_STATUS", { email: getActiveGmailEmail() });
          if(s?.ok && s?.authed) return true;
        }
        return false;
      })(),
      waitOAuthDoneEvent(180000)
    ]);
    if(!race) throw new Error("OAuth 시간 초과");
    await new Promise(r=>setTimeout(r, 300));
    return true;
  }

  async function forceReauthIfNeeded(error){
    const msgText = String(error?.message || error || "");
    if (/401|not linked|unauthorized/i.test(msgText)){
      console.warn("[KnowMal] Detected unauthorized, forcing OAuth");
      try{
        await chrome.storage.local.remove(["KM_OAUTH_READY"]);
        const res = await bgSend("KM_OAUTH_ENSURE_FORCE");
        if(res?.ok && (res.authed || res.authorized)) return true;
      }catch(_){}
    }
    return false;
  }

  async function runScan(att){
    msg("Gmail 첨부 파일 분석 요청…"); prog(60);
    const ids=parseIds(att.url);

    const message_id = att.message_id || ids.message_id || ids.thread_id || getFallbackMessageId();
    console.log("[KnowMal] Using message_id:", message_id, "from att:", att.message_id, "from parseIds:", ids.message_id);
    if (!message_id) {
      console.warn("[KnowMal] message_id fallback failed; scan may return 400. Continuing.");
    }

    const payload={
      message_id:message_id,
      filename:filenameOnly(att.filename)
    };
    console.log("[KnowMal] Sending scan request with payload:", payload);
    
    let r = null;

    try {
      const response = await fetch(`${API_BASE}/gmail/scan`, {
        method:"POST",
        headers:{ 
          "Content-Type":"application/json", 
          "X-KM-Ext-Id": (typeof chrome !== 'undefined' && chrome.runtime) ? chrome.runtime.id : "unknown",
          "X-KM-Account-Email": getActiveGmailEmail() || undefined
        },
        body: JSON.stringify(payload)
      });
      console.log("[KnowMal] Direct fetch response status:", response.status);
      const r_json = await response.json();
      console.log("[KnowMal] Direct fetch response:", r_json);
      
      if (response.ok) {
        let reportUrl = r_json.report_url || r_json.reportUrl || r_json.url;
        if (reportUrl) {
          if (reportUrl.includes('https://localhost')) {
            reportUrl = reportUrl.replace('https://localhost', 'http://localhost');
          }
          console.log("[KnowMal] Using report URL:", reportUrl);
          msg("검사 완료"); prog(100); badge("완료", true); enableOpen(reportUrl);
          return;
        }
      }
      throw new Error(`HTTP ${response.status}: ${r_json.detail || r_json.error || 'Unknown error'}`);
    } catch (e) {
      console.log("[KnowMal] Direct fetch failed, trying bgFetch:", e);
      const reauthed = await forceReauthIfNeeded(e);
      if (reauthed){
        const response2 = await fetch(`${API_BASE}/gmail/scan`, {
          method:"POST",
          headers:{ 
            "Content-Type":"application/json", 
            "X-KM-Ext-Id": (typeof chrome !== 'undefined' && chrome.runtime) ? chrome.runtime.id : "unknown",
            "X-KM-Account-Email": getActiveGmailEmail() || undefined
          },
          body: JSON.stringify(payload)
        });
        if (response2.ok){
          const r2 = await response2.json();
          let reportUrl2 = r2.report_url || r2.reportUrl || r2.url;
          if (reportUrl2){
            if (reportUrl2.includes('https://localhost')) {
              reportUrl2 = reportUrl2.replace('https://localhost', 'http://localhost');
            }
            msg("검사 완료"); prog(100); badge("완료", true); enableOpen(reportUrl2);
            return;
          }
        }
      }

      r = await bgFetch("/gmail/scan", {
        method:"POST",
        headers:{ "Content-Type":"application/json", "X-KM-Ext-Id": (typeof chrome !== 'undefined' && chrome.runtime) ? chrome.runtime.id : "unknown", "X-KM-Account-Email": getActiveGmailEmail() || undefined },
        body: JSON.stringify(payload)
      });
      console.log("[KnowMal] bgFetch response:", r);
    }

    const reportUrl = r.report_url || r.reportUrl || r.url || (r.data && (r.data.report_url||r.data.url));
    if(!reportUrl){
      if(r.task_id){
        msg("분석 중…"); prog(80);
        const deadline=Date.now()+120000;
        while(Date.now()<deadline){
          await new Promise(r=>setTimeout(r,1200));
          const s=await bgFetch(`/gmail/scan/status?task_id=${encodeURIComponent(r.task_id)}`, { method:"GET" });
          if((s.status==="done"||s.ok) && (s.report_url||s.url)) return s.report_url||s.url;
        }
      }
      throw new Error("리포트 URL 미수신");
    }
    return reportUrl;
  }

  async function startFlow(att){
    showOverlay(); badge("진행"); msg("Google OAuth 상태 확인…"); prog(20); disableOpen();
    try{
      const pageEmail = getActiveGmailEmail();
      let serverEmail = "";
      try {
        const st = await bgSend("KM_OAUTH_STATUS", { email: pageEmail });
        serverEmail = st?.email || "";
        if (pageEmail && serverEmail && pageEmail.toLowerCase() !== serverEmail.toLowerCase()){
          console.warn("[KnowMal] startFlow: pageEmail != serverEmail → force OAuth", { pageEmail, serverEmail });
          await chrome.storage.local.remove(["KM_OAUTH_READY"]);
          const f = await bgSend("KM_OAUTH_ENSURE_FORCE");
          if(!(f?.ok && (f?.authed || f?.authorized))) throw new Error("계정 전환 재인증 실패");
        }
      } catch(_){ /* 일반 ensure로 진행 */ }

      await ensureOAuth();
      msg("인증 확인 완료. 분석을 시작합니다…"); prog(45);
      await runScan(att);
    }catch(e){
      console.warn("[KnowMal] flow error", e);
      badge("오류", false, true); msg(e?.message || String(e)); prog(100); disableOpen();
    }
  }

  function maybeHandle(target, ev){
    const info=findAttachment(target);
    if(!info) return false;
    ev?.preventDefault?.(); ev?.stopPropagation?.(); ev?.stopImmediatePropagation?.();
    startFlow(info);
    return true;
  }
  const opts={capture:true,passive:false};
  window.addEventListener("pointerdown",(e)=>maybeHandle(e.target,e),opts);
  window.addEventListener("mousedown",(e)=>maybeHandle(e.target,e),opts);
  window.addEventListener("click",(e)=>maybeHandle(e.target,e),opts);
  window.addEventListener("auxclick",(e)=>maybeHandle(e.target,e),opts);

  console.log("[KnowMal] content_gmail ready");
})();