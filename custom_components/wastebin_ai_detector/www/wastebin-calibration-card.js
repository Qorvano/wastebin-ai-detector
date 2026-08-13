/* Wastebin AI Detector - calibration card.
 *
 * A thin UI over the integration's services: everything this card does
 * (capture, draw samples, label, set the region) can also be done from
 * Developer Tools. Rectangles are drawn on the live camera view in
 * FULL-IMAGE relative coordinates - the exact frame the calibration
 * store anchors its evidence in.
 *
 * Card config:
 *   type: custom:wastebin-calibration-card
 *   camera: camera.kamera_hinterhof_hd_stream   (required)
 *   bins:                                       (required)
 *     - id: gelbe_tonne
 *       name: Gelbe Tonne
 *   entities:                                   (optional, overlay)
 *     - binary_sensor.kamera_hinterhof_hd_stream_gelbe_tonne
 *   entry_id: <config entry id>                 (optional, single entry auto)
 */

const TEXTS = {
  en: {
    capture: "Capture snapshot",
    captured: "Captured: ",
    view: "View",
    roi: "Draw region",
    sample: "Draw sample",
    label: "Label",
    apply_roi: "Apply as region",
    clear: "Discard",
    present: "present",
    absent: "absent",
    unset: "-",
    save_labels: "Save labels",
    need_capture: "Capture a snapshot first - samples and labels attach to an archived file.",
    draw_first: "Draw a rectangle first.",
    saved_sample: "Sample saved for ",
    roi_set: "Region updated; relearn runs in the background.",
    labels_saved: "Labels saved. Relearn: ",
    overlay_hint: "Boxes show what the detector currently matches.",
    error: "Error: ",
  },
  de: {
    capture: "Schnappschuss aufnehmen",
    captured: "Aufgenommen: ",
    view: "Ansehen",
    roi: "Bereich zeichnen",
    sample: "Sample zeichnen",
    label: "Beschriften",
    apply_roi: "Als Bereich übernehmen",
    clear: "Verwerfen",
    present: "anwesend",
    absent: "abwesend",
    unset: "-",
    save_labels: "Beschriftung speichern",
    need_capture: "Bitte nehmen Sie zuerst einen Schnappschuss auf - Samples und Beschriftungen gehören zu einer archivierten Datei.",
    draw_first: "Bitte zeichnen Sie zuerst ein Rechteck.",
    saved_sample: "Sample gespeichert für ",
    roi_set: "Bereich aktualisiert; das Neu-Lernen läuft im Hintergrund.",
    labels_saved: "Beschriftung gespeichert. Neu-Lernen: ",
    overlay_hint: "Rahmen zeigen, was der Detektor aktuell erkennt.",
    error: "Fehler: ",
  },
};

/* Rectangles smaller than this fraction of the frame are accidental
 * click jitter, not a drawn box: at a 4K frame this is still < 8 px,
 * far below any lid or region a user would intentionally mark. */
const MIN_DRAW_FRAC = 0.002;
/* Coordinate resolution sent to the services: 1e-4 of the frame is
 * sub-pixel for any camera up to 10000 px wide, so rounding here can
 * never move a rectangle by a visible amount. */
const COORD_DECIMALS = 4;

class WastebinCalibrationCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._mode = "view";
    this._drawn = null; // {x, y, w, h} image-relative
    this._dragStart = null;
    this._filename = null;
    this._labels = {}; // bin id -> "present" | "absent"
    this._sampleBin = null;
    this._status = "";
    this._imgCounter = 0;
  }

  setConfig(config) {
    if (!config.camera) throw new Error("camera is required");
    if (!Array.isArray(config.bins) || !config.bins.length) {
      throw new Error("bins is required: a list of {id, name}");
    }
    this._config = config;
    this._sampleBin = config.bins[0].id;
    this._render();
  }

  set hass(hass) {
    this._hass = hass;
    this._t = TEXTS[(hass.language || "en").startsWith("de") ? "de" : "en"];
    if (!this._built) this._render();
    this._updateImage();
    this._updateOverlay();
  }

  getCardSize() {
    return 6;
  }

  _svc(service, data, wantResponse = false) {
    const payload = { ...data };
    if (this._config.entry_id) payload.entry_id = this._config.entry_id;
    return this._hass.callService(
      "wastebin_ai_detector", service, payload, undefined, true, wantResponse
    );
  }

  _setStatus(text) {
    this._status = text;
    const el = this.shadowRoot.getElementById("status");
    if (el) el.textContent = text;
  }

  async _capture() {
    try {
      const result = await this._svc("capture_snapshot", {}, true);
      this._filename = result?.response?.filename || null;
      this._labels = {};
      this._drawn = null;
      this._paintDrawn();
      this._setStatus(this._t.captured + (this._filename || "?"));
      await this._showArchivedFrame();
      this._renderLabelRow();
    } catch (err) {
      this._setStatus(this._t.error + (err.message || err));
    }
  }

  async _applyRoi() {
    if (!this._drawn) return this._setStatus(this._t.draw_first);
    try {
      await this._svc("set_roi", {
        roi_x: this._round(this._drawn.x),
        roi_y: this._round(this._drawn.y),
        roi_w: this._round(this._drawn.w),
        roi_h: this._round(this._drawn.h),
      });
      this._drawn = null;
      this._paintDrawn();
      this._setStatus(this._t.roi_set);
    } catch (err) {
      this._setStatus(this._t.error + (err.message || err));
    }
  }

  async _saveSample() {
    if (!this._filename) return this._setStatus(this._t.need_capture);
    if (!this._drawn) return this._setStatus(this._t.draw_first);
    try {
      await this._svc("add_sample", {
        filename: this._filename,
        bin: this._sampleBin,
        rect: [
          this._round(this._drawn.x),
          this._round(this._drawn.y),
          this._round(this._drawn.w),
          this._round(this._drawn.h),
        ],
        space: "image",
      });
      const bin = this._config.bins.find((b) => b.id === this._sampleBin);
      this._setStatus(this._t.saved_sample + (bin ? bin.name : this._sampleBin));
      this._drawn = null;
      this._paintDrawn();
    } catch (err) {
      this._setStatus(this._t.error + (err.message || err));
    }
  }

  async _saveLabels() {
    if (!this._filename) return this._setStatus(this._t.need_capture);
    const present = [];
    const absent = [];
    for (const [binId, state] of Object.entries(this._labels)) {
      if (state === "present") present.push(binId);
      if (state === "absent") absent.push(binId);
    }
    try {
      const result = await this._svc(
        "label_image",
        { filename: this._filename, present, absent },
        true
      );
      const relearn = result?.response?.relearn || "?";
      const warnings = result?.response?.warnings || [];
      this._setStatus(
        this._t.labels_saved + relearn +
        (warnings.length ? " (" + warnings.length + " warnings)" : "")
      );
    } catch (err) {
      this._setStatus(this._t.error + (err.message || err));
    }
  }

  async _showArchivedFrame() {
    /* Samples and labels attach to the archived file; the display must
     * match it. With a known entry_id the archived frame itself is
     * resolved through the media source; without one, the frame that
     * is already on screen (fetched at capture time) stays - never a
     * NEWER live fetch. */
    if (!this._config.entry_id || !this._filename) return;
    try {
      const resolved = await this._hass.callWS({
        type: "media_source/resolve_media",
        media_content_id:
          "media-source://media_source/local/wastebin_ai_detector/" +
          this._config.entry_id + "/" + this._filename,
      });
      if (resolved && resolved.url) {
        this.shadowRoot.getElementById("cam").src = resolved.url;
      }
    } catch (err) {
      /* media dir not exposed as media source: keep the current frame */
    }
  }

  _round(v) {
    const f = 10 ** COORD_DECIMALS;
    return Math.round(v * f) / f;
  }

  // -- drawing ---------------------------------------------------------

  _pointerPos(ev) {
    const rect = this.shadowRoot.getElementById("stage").getBoundingClientRect();
    return {
      x: Math.min(Math.max((ev.clientX - rect.left) / rect.width, 0), 1),
      y: Math.min(Math.max((ev.clientY - rect.top) / rect.height, 0), 1),
    };
  }

  _onDown(ev) {
    if (this._mode !== "roi" && this._mode !== "sample") return;
    ev.preventDefault();
    const stage = this.shadowRoot.getElementById("stage");
    if (stage.setPointerCapture) stage.setPointerCapture(ev.pointerId);
    this._dragStart = this._pointerPos(ev);
    this._drawn = null;
  }

  _onMove(ev) {
    if (!this._dragStart) return;
    ev.preventDefault();
    const cur = this._pointerPos(ev);
    this._drawn = {
      x: Math.min(this._dragStart.x, cur.x),
      y: Math.min(this._dragStart.y, cur.y),
      w: Math.abs(cur.x - this._dragStart.x),
      h: Math.abs(cur.y - this._dragStart.y),
    };
    this._paintDrawn();
  }

  _onUp() {
    this._dragStart = null;
    if (this._drawn && (this._drawn.w < MIN_DRAW_FRAC || this._drawn.h < MIN_DRAW_FRAC)) {
      this._drawn = null;
      this._paintDrawn();
    }
  }

  _paintDrawn() {
    const box = this.shadowRoot.getElementById("drawn");
    if (!box) return;
    if (!this._drawn) {
      box.style.display = "none";
      return;
    }
    box.style.display = "block";
    box.style.left = this._drawn.x * 100 + "%";
    box.style.top = this._drawn.y * 100 + "%";
    box.style.width = this._drawn.w * 100 + "%";
    box.style.height = this._drawn.h * 100 + "%";
    box.className = this._mode === "roi" ? "rect roi" : "rect sample";
  }

  // -- rendering -------------------------------------------------------

  _updateImage() {
    const img = this.shadowRoot.getElementById("cam");
    if (!img || !this._hass) return;
    const state = this._hass.states[this._config.camera];
    if (!state) return;
    const pic = state.attributes.entity_picture;
    if (pic) img.src = pic + "&card=" + this._imgCounter;
  }

  _updateOverlay() {
    const layer = this.shadowRoot.getElementById("overlay");
    if (!layer || !this._hass) return;
    if (this._mode !== "view" || !this._config.entities) {
      layer.replaceChildren();
      this._overlaySignature = null;
      return;
    }
    const signature = JSON.stringify(
      this._config.entities.map((entityId) => {
        const state = this._hass.states[entityId];
        return state ? [entityId, state.state, state.attributes.bbox] : null;
      })
    );
    if (signature === this._overlaySignature) return;
    this._overlaySignature = signature;
    const boxes = [];
    for (const entityId of this._config.entities) {
      const state = this._hass.states[entityId];
      if (!state || !state.attributes.bbox) continue;
      const [x, y, w, h] = state.attributes.bbox;
      const div = document.createElement("div");
      div.className = "rect detected" + (state.state === "on" ? " on" : "");
      div.style.left = x * 100 + "%";
      div.style.top = y * 100 + "%";
      div.style.width = w * 100 + "%";
      div.style.height = h * 100 + "%";
      const tag = document.createElement("span");
      tag.textContent = state.attributes.friendly_name || entityId;
      div.appendChild(tag);
      boxes.push(div);
    }
    layer.replaceChildren(...boxes);
  }

  _setMode(mode) {
    this._mode = mode;
    this._drawn = null;
    for (const btn of this.shadowRoot.querySelectorAll("[data-mode]")) {
      btn.classList.toggle("active", btn.dataset.mode === mode);
    }
    this.shadowRoot.getElementById("roi-actions").style.display =
      mode === "roi" ? "flex" : "none";
    this.shadowRoot.getElementById("sample-actions").style.display =
      mode === "sample" ? "flex" : "none";
    this.shadowRoot.getElementById("label-actions").style.display =
      mode === "label" ? "flex" : "none";
    this.shadowRoot.getElementById("stage").style.cursor =
      mode === "roi" || mode === "sample" ? "crosshair" : "default";
    this._paintDrawn();
    this._updateOverlay();
  }

  _cycleLabel(binId) {
    const order = [undefined, "present", "absent"];
    const cur = this._labels[binId];
    const next = order[(order.indexOf(cur) + 1) % order.length];
    if (next === undefined) delete this._labels[binId];
    else this._labels[binId] = next;
    this._renderLabelRow();
  }

  _renderLabelRow() {
    const row = this.shadowRoot.getElementById("label-bins");
    if (!row) return;
    row.replaceChildren(
      ...this._config.bins.map((bin) => {
        const btn = document.createElement("button");
        const state = this._labels[bin.id];
        btn.textContent =
          bin.name + ": " +
          (state === "present"
            ? this._t.present
            : state === "absent"
              ? this._t.absent
              : this._t.unset);
        btn.className =
          "chip" +
          (state === "present" ? " present" : state === "absent" ? " absent" : "");
        btn.onclick = () => this._cycleLabel(bin.id);
        return btn;
      })
    );
  }

  _render() {
    if (!this._config || !this._t) return;
    this._built = true;
    const t = this._t;
    this.shadowRoot.innerHTML = `
      <style>
        :host { display: block; }
        ha-card { padding: 12px; }
        .toolbar, .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
        button {
          background: var(--secondary-background-color);
          color: var(--primary-text-color);
          border: 1px solid var(--divider-color);
          border-radius: 6px; padding: 6px 10px; cursor: pointer; font: inherit;
        }
        button.active { background: var(--primary-color); color: var(--text-primary-color, #fff); }
        select {
          background: var(--secondary-background-color);
          color: var(--primary-text-color);
          border: 1px solid var(--divider-color);
          border-radius: 6px; padding: 6px 10px; font: inherit;
        }
        button.chip.present { background: var(--success-color, #0a0); color: #fff; }
        button.chip.absent { background: var(--error-color, #a00); color: #fff; }
        #stage { position: relative; user-select: none; touch-action: none; }
        #cam { display: block; width: 100%; border-radius: 6px; }
        .rect { position: absolute; box-sizing: border-box; pointer-events: none; }
        .rect.roi { border: 2px dashed var(--primary-color); background: rgba(3,169,244,.15); }
        .rect.sample { border: 2px solid var(--accent-color); background: rgba(255,152,0,.2); }
        .rect.detected { border: 2px solid var(--error-color, #a00); }
        .rect.detected.on { border-color: var(--success-color, #0a0); }
        .rect.detected span {
          position: absolute; top: -1.4em; left: 0; font-size: 11px;
          background: var(--card-background-color); padding: 0 4px; border-radius: 3px;
          white-space: nowrap;
        }
        #overlay { position: absolute; inset: 0; pointer-events: none; }
        #drawn { display: none; }
        #status { margin-top: 8px; font-size: 13px; color: var(--secondary-text-color); min-height: 1.2em; }
      </style>
      <ha-card>
        <div class="toolbar">
          <button id="capture">${t.capture}</button>
          <button data-mode="view" class="active">${t.view}</button>
          <button data-mode="roi">${t.roi}</button>
          <button data-mode="sample">${t.sample}</button>
          <button data-mode="label">${t.label}</button>
        </div>
        <div id="stage">
          <img id="cam" alt="camera" />
          <div id="overlay"></div>
          <div id="drawn" class="rect"></div>
        </div>
        <div class="actions" id="roi-actions" style="display:none">
          <button id="apply-roi">${t.apply_roi}</button>
          <button id="clear-roi">${t.clear}</button>
        </div>
        <div class="actions" id="sample-actions" style="display:none">
          <select id="sample-bin"></select>
          <button id="save-sample">OK</button>
          <button id="clear-sample">${t.clear}</button>
        </div>
        <div class="actions" id="label-actions" style="display:none">
          <span id="label-bins" class="actions"></span>
          <button id="save-labels">${t.save_labels}</button>
        </div>
        <div id="status"></div>
      </ha-card>
    `;
    this.shadowRoot.getElementById("capture").onclick = () => this._capture();
    for (const btn of this.shadowRoot.querySelectorAll("[data-mode]")) {
      btn.onclick = () => this._setMode(btn.dataset.mode);
    }
    this.shadowRoot.getElementById("apply-roi").onclick = () => this._applyRoi();
    this.shadowRoot.getElementById("save-sample").onclick = () => this._saveSample();
    this.shadowRoot.getElementById("save-labels").onclick = () => this._saveLabels();
    const clear = () => {
      this._drawn = null;
      this._paintDrawn();
    };
    this.shadowRoot.getElementById("clear-roi").onclick = clear;
    this.shadowRoot.getElementById("clear-sample").onclick = clear;
    const binSelect = this.shadowRoot.getElementById("sample-bin");
    for (const bin of this._config.bins) {
      const option = document.createElement("option");
      option.value = bin.id;
      option.textContent = bin.name;  /* config text: never innerHTML */
      binSelect.appendChild(option);
    }
    binSelect.value = this._sampleBin;
    binSelect.onchange = () => (this._sampleBin = binSelect.value);
    const stage = this.shadowRoot.getElementById("stage");
    stage.onpointerdown = (ev) => this._onDown(ev);
    stage.onpointermove = (ev) => this._onMove(ev);
    stage.onpointerup = () => this._onUp();
    stage.onpointercancel = () => this._onUp();
    this._renderLabelRow();
    this._updateImage();
    this._setMode(this._mode);
    this._setStatus(this._status);
  }
}

customElements.define("wastebin-calibration-card", WastebinCalibrationCard);
window.customCards = window.customCards || [];
window.customCards.push({
  type: "wastebin-calibration-card",
  name: "Wastebin Calibration Card",
  description:
    "Draw the region, lid samples and labels for the Wastebin AI Detector directly on the camera image.",
});
