(() => {
  const API_BASE = "http://localhost:8000";
  const OFFICE_RE = /\.(docx?|xlsx?|pptx?|html?|json|zip|rar|7z|tar|gz|bz2|pdf|hwp|exe|dll|msi|bat|cmd|lnk|scr|iso|img|bin)$/i;
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

.mo-actions{
  display:flex; gap:16px; justify-content:center; 
  padding: 32px 24px 28px 24px;
}
.mo-btn{
  flex:1; padding:12px 0; border-radius:10px; font-weight:600; font-size:14px; 
  border:none; cursor:pointer; transition:all .2s;
}
.mo-btn:disabled{
  background:#f3f4f6; color:#9ca3af; cursor:not-allowed;
}
.mo-btn.primary{
  background:linear-gradient(135deg,#3b82f6 0%,#1d4ed8 100%); color:#fff; 
  box-shadow:0 2px 8px rgba(59,130,246,.3);
}
.mo-btn.primary:hover:not(:disabled){
  box-shadow:0 4px 12px rgba(59,130,246,.4);
}
.mo-btn.secondary{
  background:transparent; color:#3b82f6; border:2px solid #3b82f6;
}
.mo-btn.secondary:hover:not(:disabled) {
  background: #eff6ff;
  color: #2563eb;
  border: 2px solid #3b82f6;
  box-shadow: 0 2px 6px rgba(59,130,246,.2);
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

  let $ov,$msg,$btnOpen,$btnClose,$btnDownload,$result;
  let currentAttachment = null; // 현재 첨부 파일 정보 저장
  function ensureOverlay(){
    if($ov) return;
    injectStyles();
    $ov=document.createElement("div");
    $ov.id="mo-overlay";
    const logoUrl = chrome.runtime.getURL("images/KnowMal_logo.png");
    $ov.innerHTML=`
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
    </div>`;
    document.body.appendChild($ov);
    $msg=$ov.querySelector("#mo-msg");
    $btnOpen=$ov.querySelector("#mo-open");
    $btnClose=$ov.querySelector("#mo-close");
    $btnDownload=$ov.querySelector("#mo-download");
    $result=$ov.querySelector("#mo-result");
    $btnClose.onclick=()=>{
      $ov.style.display="none";
      resetModalState();
    };
  }
  function showOverlay(){ 
    ensureOverlay(); 
    $ov.style.display="flex"; 
    resetModalState(); // 모든 상태 초기화
    setStep("진행"); 
    setMsg("파일의 악성 행위를 검사 중입니다."); 
  }
  
  function setMsg(text, className = "processing"){
    $msg.textContent = text;
    $msg.className = `mo-msg ${className}`;
  }
  
  function setStep(state){
    const steps = [
      ["대기", document.getElementById("mo-step-wait")],
      ["진행", document.getElementById("mo-step-run")],
      ["완료", document.getElementById("mo-step-done")]
    ];
    for (const [name, el] of steps){
      if (!el) continue;
      el.classList.toggle("active", name === state);
    }
  }
  
  function setResult(kind){
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
            <div class="mo-desc-line">상세 결과 확인 버튼을 클릭하세요.</div>
          </div>
        </div>`;
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
    setStep("대기");
    if ($btnOpen) {
      $btnOpen.disabled = true;
      $btnOpen.textContent = "상세 결과 확인";
      $btnOpen.onclick = null;
    }
    disableDownload();
    currentAttachment = null;
    try{
      const doneEl = document.getElementById("mo-step-done");
      if (doneEl){
        doneEl.classList.remove("status-safe","status-danger");
      }
    }catch(_){}
  }

  const enableOpen = (url)=>{ 
    console.log("[KnowMal] enableOpen called with URL:", url);
    $btnOpen.disabled=false; 
    $btnOpen.onclick=()=>{
      console.log("[KnowMal] Opening URL:", url);
      window.open(url,"_blank","noopener,noreferrer");
    }; 
  };
  const disableOpen = ()=>{ $btnOpen.disabled=true; $btnOpen.onclick=null; };
  
  const disableDownload = ()=>{ 
    if($btnDownload) {
      $btnDownload.disabled=true; 
      $btnDownload.onclick=null; 
    }
  };
  
  const enableDownload = (attachment)=>{ 
    if($btnDownload && attachment) {
      $btnDownload.disabled=false; 
      $btnDownload.onclick=()=>{
        console.log("[KnowMal] Downloading file:", attachment.filename);
        downloadGmailAttachment(attachment);
      }; 
    }
  };
  
  async function downloadGmailAttachment(attachment) {
    try {
      console.log("[KnowMal] Starting download for:", attachment);
      
      const messageId = attachment.message_id;
      const filename = attachment.filename;
      
      if (!messageId) {
        console.error("[KnowMal] No message_id available");
        return;
      }
      
      const downloadResult = await bgSend("KM_DOWNLOAD_FILE", {
        message_id: messageId,
        filename: filename,
        account_email: getActiveGmailEmail()
      });
      
      if (!downloadResult?.ok) {
        throw new Error(downloadResult?.error || "다운로드 실패");
      }
      
      if (!downloadResult.download_url) {
        throw new Error("다운로드 URL을 받지 못했습니다");
      }
      
      console.log("[KnowMal] Fetching from URL:", downloadResult.download_url);
      const response = await fetch(downloadResult.download_url, {
        method: 'GET',
        headers: downloadResult.headers || {}
      });
      
      console.log("[KnowMal] Fetch response:", {
        status: response.status,
        statusText: response.statusText,
        headers: Object.fromEntries(response.headers.entries())
      });
      
      if (!response.ok) {
        throw new Error(`다운로드 실패: ${response.status} ${response.statusText}`);
      }
      
      let finalFilename = filename;
      const contentDisposition = response.headers.get('Content-Disposition');
      if (contentDisposition) {
        console.log("[KnowMal] Content-Disposition header:", contentDisposition);
        const utf8Match = contentDisposition.match(/filename\*=UTF-8''([^;]+)/);
        if (utf8Match) {
          try {
            finalFilename = decodeURIComponent(utf8Match[1]);
            console.log("[KnowMal] Extracted UTF-8 filename:", finalFilename);
          } catch (e) {
            console.warn("[KnowMal] Failed to decode UTF-8 filename:", e);
          }
        }
      }
      
      console.log("[KnowMal] Creating array buffer from response...");
      const arrayBuffer = await response.arrayBuffer();
      console.log("[KnowMal] ArrayBuffer created:", {
        size: arrayBuffer.byteLength
      });
      
      const blob = new Blob([arrayBuffer], { type: response.headers.get('content-type') || 'application/octet-stream' });
      console.log("[KnowMal] Blob created:", {
        size: blob.size,
        type: blob.type
      });
      
      const url = URL.createObjectURL(blob);
      console.log("[KnowMal] Blob URL created:", url);
      
      const downloadLink = document.createElement('a');
      downloadLink.href = url;
      downloadLink.download = finalFilename;
      downloadLink.target = '_blank';
      downloadLink.style.display = 'none';
      
      console.log("[KnowMal] Download link created:", {
        href: url,
        download: finalFilename,
        originalFilename: attachment.filename,
        extractedFromHeader: finalFilename !== filename,
        blobSize: blob.size,
        blobType: blob.type
      });
      
      console.log("[KnowMal] Download link attributes:", {
        href: downloadLink.href,
        download: downloadLink.download,
        target: downloadLink.target
      });
      
      if (downloadLink.download !== finalFilename) {
        console.warn("[KnowMal] Download attribute mismatch:", {
          expected: finalFilename,
          actual: downloadLink.download
        });
        downloadLink.download = finalFilename;
      }
      
      document.body.appendChild(downloadLink);
      console.log("[KnowMal] Clicking download link...");
      
      downloadLink.addEventListener('click', (e) => {
        console.log("[KnowMal] Download link clicked:", {
          href: e.target.href,
          download: e.target.download
        });
      });
      
      try {
        if (typeof chrome !== 'undefined' && chrome.downloads) {
          console.log("[KnowMal] Trying Chrome downloads API...");
          
          const reader = new FileReader();
          reader.onload = async function() {
            const base64Data = reader.result.split(',')[1];
            const dataUrl = `data:${blob.type};base64,${base64Data}`;
            
            try {
              await chrome.downloads.download({
                url: dataUrl,
                filename: finalFilename,
                saveAs: false
              });
              console.log("[KnowMal] Chrome downloads API success:", finalFilename);
            } catch (e) {
              console.warn("[KnowMal] Chrome downloads API failed, falling back to link click:", e);
              downloadLink.click();
            }
          };
          reader.readAsDataURL(blob);
        } else {
          console.log("[KnowMal] Chrome downloads API not available, using link click");
          downloadLink.click();
        }
      } catch (e) {
        console.warn("[KnowMal] Chrome downloads API error, using link click:", e);
        downloadLink.click();
      }
      
      document.body.removeChild(downloadLink);
      
      setTimeout(() => {
        try {
          URL.revokeObjectURL(url);
          console.log("[KnowMal] Blob URL revoked:", url);
        } catch (e) {
          console.warn("[KnowMal] URL revoke failed:", e);
        }
      }, 5000);
      
      console.log("[KnowMal] Download completed for:", filename);
    } catch (error) {
      console.error("[KnowMal] Download failed:", error);
      alert(`파일 다운로드에 실패했습니다: ${error.message}`);
    }
  }

  function setDone(reportUrl, attachment, status = "safe"){
    console.log("[KnowMal] setDone 호출됨 - status:", status);
    $msg.style.display = "none";
    setStep("완료");
    try{
      const doneEl = document.getElementById("mo-step-done");
      if (doneEl){
        doneEl.classList.remove("status-safe","status-danger");
        if (status === "safe") doneEl.classList.add("status-safe");
        else if (status === "danger") doneEl.classList.add("status-danger");
      }
    }catch(_){}
    
    currentAttachment = {
      ...attachment,
      message_id: attachment.message_id || getFallbackMessageId()
    };
    
    setResult(status);
    enableOpen(reportUrl);
    enableDownload(currentAttachment);
  }

  function bgSend(type, payload={}){
    return new Promise((resolve,reject)=>{
      if (typeof chrome === 'undefined' || !chrome.runtime) {
        reject(new Error("Chrome extension context not available"));
        return;
      }
      
      if (type === "KM_OAUTH_STATUS" || type === "KM_OAUTH_ENSURE" || type === "KM_OAUTH_ENSURE_FORCE") {
        resolve({ ok: true, authed: true, authorized: true });
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
      const m=label.match(/[^\/\\]+?\.(docx?|xlsx?|pptx?|zip|rar|7z|tar|gz|bz2|pdf|hwp|exe|dll|msi|bat|cmd|lnk|scr|iso|img|bin)/i);
      const filename=m?m[0]:"document.bin";
      return { url:a.href, filename, message_id };
    }catch(e){ return null; }
  }
  function findAttachment(target){
    console.log("[KnowMal] findAttachment called on:", target);
    let n=target;
    for(let i=0;i<8 && n;i++){ 
      console.log("[KnowMal] Checking element:", n, "iteration:", i);
      const info=parseDownloadUrlAttr(n); 
      console.log("[KnowMal] parseDownloadUrlAttr result:", info);
      if(info && OFFICE_RE.test(info.filename)) {
        console.log("[KnowMal] Found attachment via download_url:", info);
        return info; 
      }
      n=n.parentElement; 
    }
    const a=target.closest?.("a");
    console.log("[KnowMal] Found closest link:", a);
    if(a){
      console.log("[KnowMal] Link href:", a.href);
      console.log("[KnowMal] Link text:", a.textContent);
      console.log("[KnowMal] Link aria-label:", a.getAttribute("aria-label"));
      
      const info2=extractFromHref(a); 
      console.log("[KnowMal] extractFromHref result:", info2);
      if(info2 && OFFICE_RE.test(info2.filename)) {
        console.log("[KnowMal] Found attachment via href:", info2);
        return info2;
      }
      const txt=(a.textContent||a.getAttribute("aria-label")||"").trim();
      const txtLower = txt.toLowerCase();
      if(OFFICE_RE.test(txtLower)) {
        let message_id = "";
        try {
          const u = new URL(a.href, location.origin);
          message_id = u.searchParams.get("th") || "";
        } catch (e) {
        }
        const match = txt.match(OFFICE_RE);
        const result = { url:a.href, filename: match ? match[0] : "document.bin", message_id };
        console.log("[KnowMal] Found attachment via text match:", result);
        return result;
      }
    }
    
    const attachmentArea = target.closest('[data-attachment-id], [data-attachment-name]');
    if (attachmentArea) {
      const attachmentId = attachmentArea.getAttribute('data-attachment-id');
      const attachmentName = attachmentArea.getAttribute('data-attachment-name');
      if (attachmentId && attachmentName) {
        const result = {
          url: target.href || target.closest('a')?.href || '',
          filename: attachmentName,
          message_id: getFallbackMessageId()
        };
        console.log("[KnowMal] Found attachment via data attributes:", result);
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
    
    setMsg("Google 동의 창에서 인증을 완료하세요…"); setStep("진행");

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
    setMsg("Gmail 첨부 파일 분석 요청…"); setStep("진행");
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
        let status = r_json.status || "safe";
        if (reportUrl) {
          if (reportUrl.includes('https://localhost')) {
            reportUrl = reportUrl.replace('https://localhost', 'http://localhost');
          }
          console.log("[KnowMal] Using report URL:", reportUrl, "Status:", status);
          // 첨부파일 정보에 message_id 추가
          const attachmentWithId = {
            ...att,
            message_id: message_id
          };
          setDone(reportUrl, attachmentWithId, status);
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
          let status2 = r2.status || "safe";
          if (reportUrl2){
            if (reportUrl2.includes('https://localhost')) {
              reportUrl2 = reportUrl2.replace('https://localhost', 'http://localhost');
            }
            const attachmentWithId2 = {
              ...att,
              message_id: message_id
            };
            setDone(reportUrl2, attachmentWithId2, status2);
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
        setMsg("분석 중…"); setStep("진행");
        const deadline=Date.now()+120000;
        while(Date.now()<deadline){
          await new Promise(r=>setTimeout(r,1200));
          const s=await bgFetch(`/gmail/scan/status?task_id=${encodeURIComponent(r.task_id)}`, { method:"GET" });
          if((s.status==="done"||s.ok) && (s.report_url||s.url)) {
            const finalUrl = s.report_url||s.url;
            const attachmentWithId = {
              ...att,
              message_id: message_id
            };
            setDone(finalUrl, attachmentWithId, "safe");
            return finalUrl;
          }
        }
      }
      throw new Error("리포트 URL 미수신");
    }
    
    const attachmentWithId = {
      ...att,
      message_id: message_id
    };
    setDone(reportUrl, attachmentWithId, "safe");
    return reportUrl;
  }

  async function startFlow(att){
    showOverlay(); setStep("진행"); setMsg("Google OAuth 상태 확인…"); disableOpen();
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
      setMsg("인증 확인 완료. 분석을 시작합니다…"); setStep("진행");
      await runScan(att);
    }catch(e){
      console.warn("[KnowMal] flow error", e);
      setStep("완료"); setMsg(e?.message || String(e)); disableOpen();
    }
  }

  function maybeHandle(target, ev){
    console.log("[KnowMal] maybeHandle called on:", target);
    console.log("[KnowMal] Target text:", target.textContent || target.getAttribute("aria-label") || "");
    console.log("[KnowMal] Target href:", target.href || target.closest("a")?.href || "");
    
    const info=findAttachment(target);
    console.log("[KnowMal] findAttachment result:", info);
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