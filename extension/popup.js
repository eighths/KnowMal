const $ = (id) => document.getElementById(id);

const drop       = $("drop");
const dropText   = $("dropText");
const fileInfo   = $("fileInfo");
const fileInput  = $("fileInput");
const fileSelectBtn = $("fileSelectBtn");
const fileStatus = $("fileStatus");
const btnUpload  = $("btnUpload");
const btnReport  = $("btnReport");
const statusPill = $("statusPill");
const prog       = $("prog");

const vtDetections = $("vtDetections");
const vtTotal = $("vtTotal");
const vtDetectionsText = $("vtDetectionsText");

const overallConclusion = $("overallConclusion");
const detailedResults = $("detailedResults");

let selectedFile = null;
let lastFileId   = null;
let lastReportUrl = null;
let lastSha256   = null;

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

function updateFileDisplay(file) {
  if (file) {
    const drop = document.getElementById('drop');
    if (drop) {
      drop.style.display = "flex";
    }
    
    dropText.style.display = "none";
    fileInfo.style.display = "flex";
    
    const overallConclusion = document.querySelector('.warning-box');
    const detailedResults = document.querySelector('.detailed-results');
    if (overallConclusion) {
      overallConclusion.style.display = "none";
    }
    if (detailedResults) {
      detailedResults.style.display = "none";
    }
    
    const fileName = fileInfo.querySelector(".file-name");
    const fileSize = fileInfo.querySelector(".file-size");
    
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    
    const fileInputContainer = document.querySelector('.file-input-container');
    if (fileInputContainer) {
      fileInputContainer.style.display = "none";
    }
  } else {
    const drop = document.getElementById('drop');
    if (drop) {
      drop.style.display = "flex";
    }
    dropText.style.display = "none";
    fileInfo.style.display = "none";
    const fileInputContainer = document.querySelector('.file-input-container');
    if (fileInputContainer) {
      fileInputContainer.style.display = "flex";
    }
    fileSelectBtn.style.display = "flex";
    fileStatus.style.display = "block";
    fileStatus.textContent = "드래그하거나 클릭하세요";
    fileStatus.style.color = "#6b7280";
  }
}

function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

function setProgress(p) {
  prog.style.width = (Math.max(0, Math.min(100, p)) + "%");
}

function updateVirusTotalResult(detections, total) {
  if (vtDetections && vtTotal && vtDetectionsText) {
    vtDetections.textContent = detections || 0;
    vtTotal.textContent = total || 0;
    vtDetectionsText.textContent = detections || 0;
    
    updateOverallConclusion(detections, total);
  }
}

function updateOverallConclusion(detections, total) {
  const overallConclusion = document.querySelector('.warning-box');
  if (!overallConclusion) return;
  
  const percentage = total > 0 ? (detections / total) * 100 : 0;
  const warningIcon = overallConclusion.querySelector('.warning-icon');
  const warningTitle = overallConclusion.querySelector('.warning-title');
  const warningSubtitle = overallConclusion.querySelector('.warning-subtitle');
  
  if (percentage >= 50) {
    warningIcon.textContent = '🚨';
    warningTitle.textContent = '위험: 파일을 열지 마세요';
    warningSubtitle.textContent = '악성코드가 탐지되었습니다';
  } else if (percentage >= 10) {
    warningIcon.textContent = '⚠️';
    warningTitle.textContent = '주의: 신중하게 검토 필요';
    warningSubtitle.textContent = '악성코드 가능성이 있습니다';
  } else {
    warningIcon.textContent = '✅';
    warningTitle.textContent = '정상: 이상 없음';
    warningSubtitle.textContent = '알려진 악성코드가 없습니다';
  }
  
  overallConclusion.style.display = 'flex';
  const detailedResults = document.querySelector('.detailed-results');
  if (detailedResults) {
    detailedResults.style.display = 'none';
  }
}

function toggleDetails() {
  const detailsContent = document.querySelector('.details-content');
  const toggleIcon = document.querySelector('.toggle-icon');
  
  if (detailsContent.classList.contains('expanded')) {
    detailsContent.classList.remove('expanded');
    toggleIcon.textContent = '▼';
  } else {
    detailsContent.classList.add('expanded');
    toggleIcon.textContent = '▲';
  }
}

window.toggleDetails = toggleDetails;

function showReportPreview() {
  const drop = document.getElementById('drop');
  if (drop) {
    drop.style.display = "none";
  }
  
  fileInfo.style.display = "none";
  
  const overallConclusion = document.querySelector('.warning-box');
  if (overallConclusion) {
    overallConclusion.style.display = "flex";
  }
  
  const detailedResults = document.querySelector('.detailed-results');
  if (detailedResults) {
    detailedResults.style.display = "none";
  }
  
  const statusContainer = document.querySelector('.status-container');
  if (statusContainer) {
    statusContainer.style.display = "none";
  }
  
}

async function loadVirusTotalResult() {
  if (!lastSha256) {
    console.log("SHA256이 없어서 VirusTotal 결과를 로드할 수 없습니다.");
    return;
  }
  
  try {
    console.log(`VirusTotal 결과 로드 시도: ${lastSha256}`);
    
    const reportRes = await fetch(`${API_BASE}/scan/report/${lastSha256}`);
    console.log(`리포트 조회 응답: ${reportRes.status}`);
    
    if (!reportRes.ok) {
      console.error(`리포트 조회 실패: ${reportRes.status} ${reportRes.statusText}`);
      updateVirusTotalResult(0, 0);
      return;
    }
    
    const reportData = await reportRes.json();
    console.log("리포트 데이터:", reportData);
    
    if (reportData.virustotal && reportData.virustotal.available) {
      const malicious = reportData.virustotal.scan_summary?.malicious || 0;
      const total = reportData.virustotal.scan_summary?.total || 0;
      
      console.log(`VirusTotal 결과: ${malicious}/${total}`);
      updateVirusTotalResult(malicious, total);
    } else {
      console.log("VirusTotal 결과가 없거나 사용할 수 없음:", reportData.virustotal);
      updateVirusTotalResult(0, 0);
    }
  } catch (e) {
    console.error("VirusTotal 결과 로드 실패:", e);
    updateVirusTotalResult(0, 0);
  }
}

function resetToInitialState() {
  const drop = document.getElementById('drop');
  if (drop) {
    drop.style.display = "flex";
  }
  
  const overallConclusion = document.querySelector('.warning-box');
  if (overallConclusion) {
    overallConclusion.style.display = "none";
  }
  const detailedResults = document.querySelector('.detailed-results');
  if (detailedResults) {
    detailedResults.style.display = "none";
  }
  
  const detailsContent = document.querySelector('.details-content');
  const toggleIcon = document.querySelector('.toggle-icon');
  if (detailsContent) {
    detailsContent.classList.remove('expanded');
  }
  if (toggleIcon) {
    toggleIcon.textContent = '▼';
  }
  
  selectedFile = null;
  lastFileId = null;
  lastReportUrl = null;
  lastSha256 = null;
  fileInput.value = "";
  
  fileInfo.style.display = "none";
  const fileInputContainer = document.querySelector('.file-input-container');
  if (fileInputContainer) {
    fileInputContainer.style.display = "flex";
  }
  fileSelectBtn.style.display = "flex";
  fileStatus.style.display = "block";
  fileStatus.textContent = "드래그하거나 클릭하세요";
  fileStatus.style.color = "#6b7280";
  dropText.style.display = "none";
  
  const statusContainer = document.querySelector('.status-container');
  if (statusContainer) {
    statusContainer.style.display = "flex";
  }
  
  setPill("", "대기");
  setProgress(0);
  
  btnUpload.textContent = "서버로 전송"; 
  btnUpload.disabled = true; 
  btnReport.disabled = true;
  
  if (vtDetections && vtTotal && vtDetectionsText) {
    vtDetections.textContent = "-";
    vtTotal.textContent = "-";
    vtDetectionsText.textContent = "-";
  }
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
  updateFileDisplay(selectedFile);
  setPill("warn","파일 선택됨");
  btnUpload.disabled = false;
  setBusy(false);
});

drop.addEventListener("click", ()=>{
  fileInput.click();
});

fileSelectBtn.addEventListener("click", (e)=>{
  e.stopPropagation();
  fileInput.click();
});

fileInput.addEventListener("change", ()=>{
  selectedFile = fileInput.files?.[0] || null;
  updateFileDisplay(selectedFile);
  setPill(selectedFile ? "warn" : "", selectedFile ? "파일 선택됨" : "대기");
  btnUpload.disabled = !selectedFile;
  setBusy(false);
});

btnReport.addEventListener("click", ()=>{
  if (!lastReportUrl) {
    setPill("warn","먼저 업로드하세요");
    return;
  }
  window.open(lastReportUrl, "_blank");
});

btnUpload.addEventListener("click", async ()=>{
  if (btnUpload.textContent === "다른 파일 검사") {
    resetToInitialState();
  } else {
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
      lastSha256 = upJson.sha256;
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
      
      btnReport.disabled = false;
      
      btnUpload.textContent = "다른 파일 검사";
      btnUpload.disabled = false;
      
      showReportPreview();
      
      await loadVirusTotalResult();
    } catch (e) {
      console.error(e);
      setPill("err", String(e));
    } finally {
      setBusy(false);
    }
  }
});
