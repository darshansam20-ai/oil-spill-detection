/**
 * AEGIS-SAR Maritime Intelligence & Oil Spill Detection System
 * Production Frontend Application Logic
 */

(function () {
    "use strict";

    // Resolve backend URL from environment config, localStorage, or Render production default
    const DEFAULT_RENDER_URL = "https://oil-spill-detection-5k8z.onrender.com";
    const envConfigUrl = (typeof window !== "undefined" && window.__AEGIS_CONFIG__ && window.__AEGIS_CONFIG__.ML_SERVICE_URL && !window.__AEGIS_CONFIG__.ML_SERVICE_URL.startsWith("%"))
        ? window.__AEGIS_CONFIG__.ML_SERVICE_URL.trim()
        : "";

    // Application State
    const state = {
        selectedFile: null,
        backendUrl: localStorage.getItem("aegis_backend_url") || envConfigUrl || DEFAULT_RENDER_URL,
        currentResult: null,
        isProcessing: false,
        timerInterval: null,
        startTime: null,
    };

    // DOM Elements Cache
    const el = {
        // Status & Settings
        statusDot: document.getElementById("status-dot"),
        statusText: document.getElementById("service-status-text"),
        statusPill: document.getElementById("service-status-pill"),
        btnOpenSettings: document.getElementById("btn-open-settings"),
        settingsModal: document.getElementById("settings-modal"),
        btnCloseSettings: document.getElementById("btn-close-settings"),
        inputBackendUrl: document.getElementById("input-backend-url"),
        btnTestConnection: document.getElementById("btn-test-connection"),
        btnSaveSettings: document.getElementById("btn-save-settings"),
        connectionTestResult: document.getElementById("connection-test-result"),
        testResultText: document.getElementById("test-result-text"),

        // Form & Upload
        predictionForm: document.getElementById("prediction-form"),
        dropzone: document.getElementById("dropzone"),
        fileInput: document.getElementById("file-input"),
        dropzonePrompt: document.getElementById("dropzone-prompt"),
        selectedFileBar: document.getElementById("selected-file-bar"),
        selectedFileName: document.getElementById("selected-file-name"),
        selectedFileSize: document.getElementById("selected-file-size"),
        btnRemoveFile: document.getElementById("btn-remove-file"),
        btnSubmit: document.getElementById("btn-submit"),
        btnLoadSample: document.getElementById("btn-load-sample"),

        // Metadata Inputs & Sliders
        inputDate: document.getElementById("input-date"),
        inputTime: document.getElementById("input-time"),
        inputAoi: document.getElementById("input-aoi"),
        inputLat: document.getElementById("input-lat"),
        inputLon: document.getElementById("input-lon"),
        sliderThreshold: document.getElementById("slider-threshold"),
        valThreshold: document.getElementById("val-threshold"),
        sliderRadius: document.getElementById("slider-radius"),
        valRadius: document.getElementById("val-radius"),

        // Tracker & Error
        processingTracker: document.getElementById("processing-tracker"),
        trackerStepTitle: document.getElementById("tracker-current-step-title"),
        trackerTimer: document.getElementById("tracker-timer"),
        progressBarFill: document.getElementById("progress-bar-fill"),
        step1: document.getElementById("step-1"),
        step2: document.getElementById("step-2"),
        step3: document.getElementById("step-3"),
        step4: document.getElementById("step-4"),
        errorAlert: document.getElementById("error-alert"),
        errorTitle: document.getElementById("error-title"),
        errorMessage: document.getElementById("error-message"),
        btnCloseError: document.getElementById("btn-close-error"),

        // Results Container
        placeholderState: document.getElementById("placeholder-state"),
        activeResultsContainer: document.getElementById("active-results-container"),

        // KPIs
        kpiSpillsCount: document.getElementById("kpi-spills-count"),
        kpiSpillArea: document.getElementById("kpi-spill-area"),
        kpiAreaUnit: document.getElementById("kpi-area-unit"),
        kpiEpicenter: document.getElementById("kpi-epicenter"),
        kpiVesselsCount: document.getElementById("kpi-vessels-count"),
        kpiTopSuspect: document.getElementById("kpi-top-suspect"),
        kpiSuspectDistance: document.getElementById("kpi-suspect-distance"),

        // Viewers
        annotatedSarImg: document.getElementById("annotated-sar-img"),
        sarDimensionsBadge: document.getElementById("sar-dimensions-badge"),
        aisMapIframe: document.getElementById("ais-map-iframe"),
        btnExpandSar: document.getElementById("btn-expand-sar"),
        btnOpenMapTab: document.getElementById("btn-open-map-tab"),
        imageModal: document.getElementById("image-modal"),
        modalFullImg: document.getElementById("modal-full-img"),
        btnCloseImageModal: document.getElementById("btn-close-image-modal"),

        // Table
        tableVesselsCount: document.getElementById("table-vessels-count"),
        vesselsTableBody: document.getElementById("vessels-table-body"),

        // Export
        btnExportJson: document.getElementById("btn-export-json"),
        btnExportCsv: document.getElementById("btn-export-csv"),
        btnExportPng: document.getElementById("btn-export-png"),
    };

    // =========================================================================
    // Initialization & Event Listeners
    // =========================================================================
    function init() {
        setupEventListeners();
        checkBackendHealth();
    }

    function setupEventListeners() {
        // Slider value update
        el.sliderThreshold.addEventListener("input", (e) => {
            el.valThreshold.textContent = parseFloat(e.target.value).toFixed(2);
        });
        el.sliderRadius.addEventListener("input", (e) => {
            el.valRadius.textContent = `${e.target.value} km`;
        });

        // Dropzone & File Input
        el.dropzone.addEventListener("click", () => el.fileInput.click());
        el.dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            el.dropzone.classList.add("drag-over");
        });
        el.dropzone.addEventListener("dragleave", () => el.dropzone.classList.remove("drag-over"));
        el.dropzone.addEventListener("drop", handleFileDrop);
        el.fileInput.addEventListener("change", handleFileSelect);
        el.btnRemoveFile.addEventListener("click", (e) => {
            e.stopPropagation();
            clearSelectedFile();
        });

        // Demo sample loader
        el.btnLoadSample.addEventListener("click", loadSampleMetadata);

        // Form Submission
        el.predictionForm.addEventListener("submit", handleFormSubmit);

        // Settings Modal
        el.btnOpenSettings.addEventListener("click", openSettingsModal);
        el.btnCloseSettings.addEventListener("click", closeSettingsModal);
        el.btnSaveSettings.addEventListener("click", saveSettings);
        el.btnTestConnection.addEventListener("click", testBackendConnection);

        // Image & Map Actions
        el.btnExpandSar.addEventListener("click", openImageModal);
        el.annotatedSarImg.addEventListener("click", openImageModal);
        el.btnCloseImageModal.addEventListener("click", () => el.imageModal.style.display = "none");
        el.btnOpenMapTab.addEventListener("click", openMapInNewTab);

        // Alert Close
        el.btnCloseError.addEventListener("click", hideError);

        // Export Actions
        el.btnExportJson.addEventListener("click", exportJsonReport);
        el.btnExportCsv.addEventListener("click", exportVesselsCsv);
        el.btnExportPng.addEventListener("click", exportAnnotatedPng);
    }

    // =========================================================================
    // File Handling
    // =========================================================================
    function handleFileDrop(e) {
        e.preventDefault();
        el.dropzone.classList.remove("drag-over");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            processSelectedFile(e.dataTransfer.files[0]);
        }
    }

    function handleFileSelect(e) {
        if (e.target.files && e.target.files.length > 0) {
            processSelectedFile(e.target.files[0]);
        }
    }

    function processSelectedFile(file) {
        const allowedExts = [".tif", ".tiff", ".png", ".jpg", ".jpeg"];
        const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
        
        if (!allowedExts.includes(ext)) {
            showError("Invalid File Type", `Selected file "${file.name}" is not a supported format. Please upload a .tif, .tiff, .png, or .jpg SAR file.`);
            return;
        }

        const maxBytes = 100 * 1024 * 1024; // 100MB
        if (file.size > maxBytes) {
            showError("File Too Large", `File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds 100 MB limit.`);
            return;
        }

        state.selectedFile = file;
        el.selectedFileName.textContent = file.name;
        el.selectedFileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
        
        el.dropzonePrompt.style.display = "none";
        el.selectedFileBar.style.display = "flex";
        hideError();
    }

    function clearSelectedFile() {
        state.selectedFile = null;
        el.fileInput.value = "";
        el.dropzonePrompt.style.display = "flex";
        el.selectedFileBar.style.display = "none";
    }

    function loadSampleMetadata() {
        el.inputDate.value = "2018-12-19";
        el.inputTime.value = "06:15:22 UTC";
        el.inputAoi.value = "-89.50, 28.20, -88.70, 28.90";
        el.inputLat.value = "28.2218";
        el.inputLon.value = "-89.4873";
        el.sliderThreshold.value = "0.50";
        el.valThreshold.textContent = "0.50";
        el.sliderRadius.value = "20";
        el.valRadius.textContent = "20 km";
        
        const accordion = document.getElementById("metadata-accordion");
        if (accordion) accordion.open = true;
    }

    // =========================================================================
    // Backend Connectivity & Health Check
    // =========================================================================
    function getApiBaseUrl() {
        let base = state.backendUrl ? state.backendUrl.trim() : "";
        if (base.endsWith("/")) base = base.slice(0, -1);
        return base;
    }

    async function checkBackendHealth() {
        const base = getApiBaseUrl();
        const healthUrl = `${base}/health`;

        el.statusText.textContent = "CHECKING...";
        el.statusDot.className = "status-indicator-dot";

        try {
            const resp = await fetch(healthUrl, { method: "GET" });
            if (resp.ok) {
                const data = await resp.json();
                el.statusDot.className = "status-indicator-dot online";
                el.statusText.textContent = `ONLINE (${data.device || "CPU"})`;
            } else {
                throw new Error(`HTTP ${resp.status}`);
            }
        } catch (err) {
            el.statusDot.className = "status-indicator-dot offline";
            el.statusText.textContent = "OFFLINE";
        }
    }

    function openSettingsModal() {
        el.inputBackendUrl.value = state.backendUrl;
        el.connectionTestResult.style.display = "none";
        el.settingsModal.style.display = "flex";
    }

    function closeSettingsModal() {
        el.settingsModal.style.display = "none";
    }

    function saveSettings() {
        const url = el.inputBackendUrl.value.trim();
        state.backendUrl = url;
        localStorage.setItem("aegis_backend_url", url);
        closeSettingsModal();
        checkBackendHealth();
    }

    async function testBackendConnection() {
        let target = el.inputBackendUrl.value.trim();
        if (target.endsWith("/")) target = target.slice(0, -1);
        const healthUrl = target ? `${target}/health` : "/health";

        el.connectionTestResult.style.display = "block";
        el.connectionTestResult.className = "connection-test-result";
        el.testResultText.textContent = `Connecting to ${healthUrl}...`;

        try {
            const res = await fetch(healthUrl);
            if (res.ok) {
                const data = await res.json();
                el.connectionTestResult.className = "connection-test-result success";
                el.testResultText.textContent = `✓ Connected! Model: ${data.model_version || "OK"}, Device: ${data.device || "CPU"}`;
            } else {
                throw new Error(`Server returned HTTP ${res.status}`);
            }
        } catch (e) {
            el.connectionTestResult.className = "connection-test-result error";
            el.testResultText.textContent = `✕ Connection Failed: ${e.message}`;
        }
    }

    // =========================================================================
    // Prediction Pipeline Execution
    // =========================================================================
    async function handleFormSubmit(e) {
        e.preventDefault();
        if (state.isProcessing) return;

        if (!state.selectedFile) {
            showError("No SAR Image Selected", "Please choose or drop a Sentinel-1 SAR image (.tif, .png, .jpg) before starting detection.");
            return;
        }

        hideError();
        startProcessingAnimation();

        const formData = new FormData();
        formData.append("image", state.selectedFile);

        if (el.inputDate.value) formData.append("date", el.inputDate.value);
        if (el.inputTime.value) formData.append("time", el.inputTime.value);
        if (el.inputAoi.value) formData.append("aoi", el.inputAoi.value);
        if (el.inputLat.value) formData.append("lat", el.inputLat.value);
        if (el.inputLon.value) formData.append("lon", el.inputLon.value);
        formData.append("threshold", el.sliderThreshold.value);
        formData.append("search_radius_km", el.sliderRadius.value);

        const base = getApiBaseUrl();
        const predictUrl = `${base}/predict`;

        try {
            const resp = await fetch(predictUrl, {
                method: "POST",
                body: formData,
            });

            if (!resp.ok) {
                let errDetail = `HTTP ${resp.status} ${resp.statusText}`;
                try {
                    const errJson = await resp.json();
                    if (errJson.detail) errDetail = errJson.detail;
                    else if (errJson.message) errDetail = errJson.message;
                } catch (_) {}
                throw new Error(errDetail);
            }

            const result = await resp.json();
            stopProcessingAnimation(true);
            renderResults(result);

        } catch (err) {
            stopProcessingAnimation(false);
            showError("Inference Execution Failed", err.message || "An unexpected error occurred while communicating with the ML inference service.");
        }
    }

    // =========================================================================
    // Progress Animation & Stepper
    // =========================================================================
    function startProcessingAnimation() {
        state.isProcessing = true;
        state.startTime = Date.now();
        el.btnSubmit.disabled = true;
        el.btnSubmit.querySelector(".btn-text").textContent = "Executing Detection Pipeline...";

        el.processingTracker.style.display = "flex";
        el.progressBarFill.style.width = "10%";
        el.trackerStepTitle.textContent = "Stage 1/4: Uploading & Preprocessing SAR Image...";

        resetStepper();
        el.step1.className = "stepper-step active";

        let elapsedSeconds = 0;
        el.timerInterval = setInterval(() => {
            elapsedSeconds = ((Date.now() - state.startTime) / 1000).toFixed(1);
            el.trackerTimer.textContent = `Elapsed: ${elapsedSeconds}s`;

            const sec = parseFloat(elapsedSeconds);
            if (sec > 2.0 && sec <= 6.0) {
                el.progressBarFill.style.width = "38%";
                el.trackerStepTitle.textContent = "Stage 2/4: Neural Segmentation (ConvNeXt-Tiny + U-Net)...";
                el.step1.className = "stepper-step done";
                el.step2.className = "stepper-step active";
            } else if (sec > 6.0 && sec <= 9.0) {
                el.progressBarFill.style.width = "65%";
                el.trackerStepTitle.textContent = "Stage 3/4: Adapter Layer (Epicenter & ISO Normalization)...";
                el.step2.className = "stepper-step done";
                el.step3.className = "stepper-step active";
            } else if (sec > 9.0) {
                el.progressBarFill.style.width = "85%";
                el.trackerStepTitle.textContent = "Stage 4/4: AIS Proximity & Vessel Trajectory Radar...";
                el.step3.className = "stepper-step done";
                el.step4.className = "stepper-step active";
            }
        }, 100);
    }

    function stopProcessingAnimation(success = true) {
        state.isProcessing = false;
        clearInterval(state.timerInterval);
        el.btnSubmit.disabled = false;
        el.btnSubmit.querySelector(".btn-text").textContent = "Run Dual-Model Detection Pipeline";

        if (success) {
            el.progressBarFill.style.width = "100%";
            el.trackerStepTitle.textContent = "Pipeline Complete! Intelligence Generated.";
            el.step1.className = "stepper-step done";
            el.step2.className = "stepper-step done";
            el.step3.className = "stepper-step done";
            el.step4.className = "stepper-step done";
            setTimeout(() => {
                el.processingTracker.style.display = "none";
            }, 1200);
        } else {
            el.processingTracker.style.display = "none";
        }
    }

    function resetStepper() {
        el.step1.className = "stepper-step";
        el.step2.className = "stepper-step";
        el.step3.className = "stepper-step";
        el.step4.className = "stepper-step";
    }

    // =========================================================================
    // Results Rendering
    // =========================================================================
    function renderResults(res) {
        state.currentResult = res;

        // Hide placeholder and show active results
        el.placeholderState.style.display = "none";
        el.activeResultsContainer.style.display = "flex";

        const det = res.oil_spill_detection || {};
        const adapter = res.adapter_payload || {};
        const ais = res.ais_vessel_correlation || {};
        const artifacts = res.artifacts || {};

        // 1. KPIs
        el.kpiSpillsCount.textContent = det.spills_detected || 0;
        if (det.total_area_km2 !== null && det.total_area_km2 !== undefined) {
            el.kpiSpillArea.textContent = det.total_area_km2.toFixed(2);
            el.kpiAreaUnit.textContent = "sq km";
        } else {
            el.kpiSpillArea.textContent = (det.total_spill_pixels || 0).toLocaleString();
            el.kpiAreaUnit.textContent = "pixels";
        }

        const lat = adapter.spill_latitude !== undefined ? adapter.spill_latitude.toFixed(4) : "N/A";
        const lon = adapter.spill_longitude !== undefined ? adapter.spill_longitude.toFixed(4) : "N/A";
        el.kpiEpicenter.textContent = `${lat}, ${lon}`;

        el.kpiVesselsCount.textContent = ais.total_vessels_detected || 0;

        if (ais.primary_suspect) {
            el.kpiTopSuspect.textContent = ais.primary_suspect.ship_name || "Unknown";
            el.kpiSuspectDistance.textContent = `${ais.primary_suspect.minimum_distance_km.toFixed(1)} km from spill`;
        } else {
            el.kpiTopSuspect.textContent = "No Suspects";
            el.kpiSuspectDistance.textContent = "within radius";
        }

        // 2. Annotated SAR Image
        if (artifacts.annotated_image_data_uri) {
            el.annotatedSarImg.src = artifacts.annotated_image_data_uri;
            el.modalFullImg.src = artifacts.annotated_image_data_uri;
            if (det.image_dimensions) {
                el.sarDimensionsBadge.textContent = `${det.image_dimensions[1]} x ${det.image_dimensions[0]} px`;
            }
        }

        // 3. Interactive AIS Map
        if (artifacts.interactive_map_html) {
            el.aisMapIframe.srcdoc = artifacts.interactive_map_html;
        }

        // 4. Ranked Vessels Table
        const vessels = ais.ranked_vessels || [];
        el.tableVesselsCount.textContent = `${vessels.length} vessel${vessels.length === 1 ? "" : "s"} correlated`;
        el.vesselsTableBody.innerHTML = "";

        if (vessels.length === 0) {
            el.vesselsTableBody.innerHTML = `
                <tr>
                    <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 24px;">
                        No AIS vessels recorded within ${ais.search_radius_km || 20} km of the spill epicenter during this window.
                    </td>
                </tr>
            `;
        } else {
            vessels.forEach((v, idx) => {
                const tr = document.createElement("tr");
                if (idx === 0) tr.className = "top-suspect-row";

                let riskBadge = '<span class="risk-chip low">NORMAL</span>';
                if (idx === 0 || v.minimum_distance_km < 5.0) {
                    riskBadge = '<span class="risk-chip high">PRIORITY SUSPECT</span>';
                } else if (v.minimum_distance_km < 15.0) {
                    riskBadge = '<span class="risk-chip medium">PROXIMATE</span>';
                }

                tr.innerHTML = `
                    <td><span class="rank-badge ${idx === 0 ? "top-1" : ""}">${v.rank || idx + 1}</span></td>
                    <td><strong>${escapeHtml(v.ship_name || "UNKNOWN")}</strong></td>
                    <td><code>${v.mmsi || "N/A"}</code></td>
                    <td>${escapeHtml(v.vessel_type || "OTHER")}</td>
                    <td><strong>${v.minimum_distance_km ? v.minimum_distance_km.toFixed(2) : "N/A"} km</strong></td>
                    <td>${v.historical_position_count || 1} pts</td>
                    <td>${riskBadge}</td>
                `;
                el.vesselsTableBody.appendChild(tr);
            });
        }
    }

    // =========================================================================
    // Modals & Viewers
    // =========================================================================
    function openImageModal() {
        if (el.annotatedSarImg.src) {
            el.imageModal.style.display = "flex";
        }
    }

    function openMapInNewTab() {
        if (state.currentResult && state.currentResult.artifacts && state.currentResult.artifacts.interactive_map_html) {
            const blob = new Blob([state.currentResult.artifacts.interactive_map_html], { type: "text/html" });
            const url = URL.createObjectURL(blob);
            window.open(url, "_blank");
        }
    }

    // =========================================================================
    // Export Handlers
    // =========================================================================
    function exportJsonReport() {
        if (!state.currentResult) return;
        const jsonStr = JSON.stringify(state.currentResult, null, 2);
        downloadBlob(jsonStr, `incident_report_${state.currentResult.incident_id || "aegis"}.json`, "application/json");
    }

    function exportVesselsCsv() {
        if (!state.currentResult || !state.currentResult.ais_vessel_correlation) return;
        const vessels = state.currentResult.ais_vessel_correlation.ranked_vessels || [];
        if (vessels.length === 0) {
            alert("No vessels to export.");
            return;
        }

        const headers = ["Rank", "Ship_Name", "MMSI", "Vessel_Type", "Min_Distance_KM", "Track_Points"];
        const rows = vessels.map(v => [
            v.rank || 0,
            `"${(v.ship_name || "").replace(/"/g, '""')}"`,
            v.mmsi || "",
            `"${(v.vessel_type || "").replace(/"/g, '""')}"`,
            v.minimum_distance_km || 0,
            v.historical_position_count || 0
        ]);

        const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
        downloadBlob(csvContent, `vessel_rankings_${state.currentResult.incident_id || "aegis"}.csv`, "text/csv");
    }

    function exportAnnotatedPng() {
        if (!state.currentResult || !state.currentResult.artifacts || !state.currentResult.artifacts.annotated_image_data_uri) {
            alert("Annotated image not available.");
            return;
        }
        const a = document.createElement("a");
        a.href = state.currentResult.artifacts.annotated_image_data_uri;
        a.download = `sar_oil_spill_detection_${state.currentResult.incident_id || "aegis"}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    function downloadBlob(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // =========================================================================
    // Utilities
    // =========================================================================
    function showError(title, msg) {
        el.errorTitle.textContent = title;
        el.errorMessage.textContent = msg;
        el.errorAlert.style.display = "flex";
    }

    function hideError() {
        el.errorAlert.style.display = "none";
    }

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Start App
    init();
})();
