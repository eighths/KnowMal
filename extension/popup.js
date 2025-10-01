const $ = (id) => document.getElementById(id);

const drop       = $("drop");
const fileInput  = $("fileInput");
const btnUpload  = $("btnUpload");
const btnReport  = $("btnReport");
const statusPill = $("statusPill");
const prog       = $("prog");

let selectedFile = null;
let lastFileId   = null;
let lastReportUrl = null;

function setPill(type, text) {
  statusPill.className = "pill " + (type || "");
  const dot = statusPill.querySelector(".dot");
  if (dot) {
    if (type === "ok")   dot.style.background = "#16a34a";
    else if (type === "warn") dot.style.background = "#f59e0b";
    else if (type === "err")  dot.style.background = "#dc2626";
    else                      dot.style.background = "#9ca3af";
  }
  
  const old = statusPill.childNodes[statusPill.childNodes.length - 1];
  const textNode = document.createTextNode(text);
  if (old && old.nodeType === 3) statusPill.replaceChild(textNode, old);
  else statusPill.appendChild(textNode);
}

function setBusy(b) {
  btnUpload.disabled = b || !selectedFile;
  btnReport.disabled = b || !lastReportUrl;
}

function setProgress(p) {
  prog.style.width = (Math.max(0, Math.min(100, p)) + "%");
}

["dragenter","dragover"].forEach(ev => drop.addEventListener(ev, (e)=>{
  e.preventDefault(); drop.classList.add("drag");
}));
["dragleave","drop"].forEach(ev => drop.addEventListener(ev, (e)=>{
  e.preventDefault(); drop.classList.remove("drag");
}));
drop.addEventListener("drop", (e)=>{
  const f = e.dataTransfer.files?.[0];
  if (!f) return;
  selectedFile = f;
  fileInput.files = e.dataTransfer.files;
  setPill("warn","파일 선택됨");
  setBusy(false);
});

fileInput.addEventListener("change", ()=>{
  selectedFile = fileInput.files?.[0] || null;
  setPill(selectedFile ? "warn" : "", selectedFile ? "파일 선택됨" : "대기");
  setBusy(false);
});

btnUpload.addEventListener("click", async ()=>{
  if (!selectedFile) return;

  try {
    setBusy(true);
    setPill("warn","업로드 중…");
    setProgress(10);

    const fd = new FormData();
    fd.append("file", selectedFile, selectedFile.name);

    const upRes = await fetch(`${API_BASE}/scan/upload`, { method: "POST", body: fd });
    if (!upRes.ok) {
      const t = await upRes.text().catch(()=> String(upRes.status));
      throw new Error(`업로드 실패: ${t}`);
    }
    const upJson = await upRes.json();
    if (!upJson.ok || !upJson.id) {
      throw new Error(`업로드 실패: ${JSON.stringify(upJson)}`);
    }
    lastFileId = upJson.id;
    setProgress(60);

    const shRes = await fetch(`${API_BASE}/share/create?file_id=${encodeURIComponent(lastFileId)}`, {
      method: "POST"
    });
    if (!shRes.ok) {
      const t = await shRes.text().catch(()=> String(shRes.status));
      throw new Error(`공유 링크 생성 실패: ${t}`);
    }
    const shJson = await shRes.json();
    if (!shJson.ok || !shJson.report_url) {
      throw new Error(`공유 링크 생성 실패: ${JSON.stringify(shJson)}`);
    }
    lastReportUrl = shJson.report_url;

    setProgress(100);
    setPill("ok","완료");
  } catch (e) {
    console.error(e);
    setPill("err", String(e));
  } finally {
    setBusy(false);
  }
});

btnReport.addEventListener("click", ()=>{
  if (!lastReportUrl) {
    setPill("warn","먼저 업로드하세요");
    return;
  }
  window.open(lastReportUrl, "_blank");
});
