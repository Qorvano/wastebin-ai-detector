/* Wastebin AI Detector - calibration card.
 *
 * A thin UI over the integration's services: everything this card does
 * (capture, draw the region contour, draw samples, label, set the
 * region) can also be done from Developer Tools. All coordinates are
 * FULL-IMAGE relative - the frame the calibration store anchors its
 * evidence in. The region preview uses SVG fill-rule "evenodd", the
 * exact rule the core rasterizes with: what you see is what is
 * computed.
 *
 * Card config:
 *   type: custom:wastebin-calibration-card
 *   camera: camera.kamera_hinterhof_hd_stream   (required)
 *   bins:                                       (required)
 *     - id: gelbe_tonne
 *       name: Gelbe Tonne
 *   status_entity: sensor.xyz_status            (optional: region prefill)
 *   entities:                                   (optional, overlay)
 *     - binary_sensor.kamera_hinterhof_hd_stream_gelbe_tonne
 *   entry_id: <config entry id>                 (optional, single entry auto)
 */

const TEXTS = {
  en: {
    capture: "Capture snapshot",
    captured: "Captured: ",
    view: "View",
    region: "Draw region",
    sample: "Draw sample",
    label: "Label",
    apply_region: "Apply region",
    undo: "Undo point",
    clear: "Discard",
    present: "present",
    absent: "absent",
    unset: "-",
    save_labels: "Save labels",
    need_capture: "Capture a snapshot first - samples and labels attach to an archived file.",
    draw_first: "Draw a rectangle first.",
    need_closed: "Close the contour first (tap the first point).",
    saved_sample: "Sample saved for ",
    sample_outside: "Warning: the sample lies (partly) outside the region - its present-label will not count until the region covers it.",
    region_set: "Region updated; relearn runs in the background.",
    multi_ring: "This region has several contours; applying will replace all of them with the drawn one.",
    labels_saved: "Labels saved. Relearn: ",
    region_hint: "Tap to add points around every spot where bins can ever stand; tap the first point to close. Drag points to adjust.",
    error: "Error: ",
  },
  de: {
    capture: "Schnappschuss aufnehmen",
    captured: "Aufgenommen: ",
    view: "Ansehen",
    region: "Region zeichnen",
    sample: "Sample zeichnen",
    label: "Beschriften",
    apply_region: "Region übernehmen",
    undo: "Punkt zurück",
    clear: "Verwerfen",
    present: "anwesend",
    absent: "abwesend",
    unset: "-",
    save_labels: "Beschriftung speichern",
    need_capture: "Bitte nehmen Sie zuerst einen Schnappschuss auf - Samples und Beschriftungen gehören zu einer archivierten Datei.",
    draw_first: "Bitte zeichnen Sie zuerst ein Rechteck.",
    need_closed: "Bitte schließen Sie zuerst die Kontur (ersten Punkt antippen).",
    saved_sample: "Sample gespeichert für ",
    sample_outside: "Hinweis: Das Sample liegt (teilweise) außerhalb der Region - sein Anwesend-Label zählt erst, wenn die Region es abdeckt.",
    region_set: "Region aktualisiert; das Neu-Lernen läuft im Hintergrund.",
    multi_ring: "Diese Region hat mehrere Konturen; Übernehmen ersetzt sie durch die gezeichnete.",
    labels_saved: "Beschriftung gespeichert. Neu-Lernen: ",
    region_hint: "Tippen setzt Punkte um alle Stellplätze, an denen je Tonnen stehen können; Tippen auf den ersten Punkt schließt. Punkte lassen sich ziehen.",
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
/* Hit radius for grabbing/closing on a vertex, in CSS pixels: the
 * platform convention for comfortable touch targets (Material/HIG use
 * 24-48 px targets; 14 px radius = 28 px diameter is the small end of
 * that range so neighboring vertices stay individually grabbable). */
const VERTEX_HIT_RADIUS_PX = 14;

class WastebinCalibrationCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._mode = "view";
    this._drawn = null; // sample rect {x,y,w,h} image-relative
    this._dragStart = null;
    this._filename = null;
    this._labels = {};
    this._sampleBin = null;
    this._status = "";
    this._imgCounter = 0;
    this._polygon = []; // [[x,y], ...] image-relative, being edited
    this._polygonClosed = false;
    this._dragVertex = null; // index of vertex being dragged
    this._prefilled = false;
    this._overlaySignature = null;
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
    this._maybePrefillRegion();
  }

  getCardSize() {
    return 6;
  }

  _region() {
    /* The configured region, from the status sensor's attribute; a
     * just-applied region overrides until the sensor reflects it. */
    let region = null;
    if (this._config.status_entity && this._hass) {
      const state = this._hass.states[this._config.status_entity];
      region = state ? state.attributes.region || null : null;
    }
    if (this._regionOverride) {
      if (
        region &&
        JSON.stringify(region.polygons) ===
          JSON.stringify(this._regionOverride.polygons)
      ) {
        this._regionOverride = null; /* sensor caught up */
      } else {
        return this._regionOverride;
      }
    }
    return region;
  }

  _maybePrefillRegion() {
    if (this._prefilled || this._polygon.length) return;
    const region = this._region();
    if (!region || !region.polygons || !region.polygons.length) return;
    this._polygon = region.polygons[0].map(([x, y]) => [x, y]);
    this._polygonClosed = true;
    this._prefilled = true;
    this._multiRing = region.polygons.length > 1;
    if (this._multiRing) this._setStatus(this._t.multi_ring);
    this._paintRegion();
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

  _round(v) {
    const f = 10 ** COORD_DECIMALS;
    return Math.round(v * f) / f;
  }

  // -- actions ---------------------------------------------------------

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

  async _applyRegion() {
    if (!this._polygonClosed || this._polygon.length < 3) {
      return this._setStatus(this._t.need_closed);
    }
    if (this._multiRing && !this._confirmReplace) {
      /* Replacing several stored contours with the one drawn ring is
       * destructive intent: surface it at the moment of the action and
       * require a second press. */
      this._confirmReplace = true;
      return this._setStatus(this._t.multi_ring);
    }
    try {
      await this._svc("set_roi", {
        polygons: [
          this._polygon.map(([x, y]) => [this._round(x), this._round(y)]),
        ],
      });
      this._prefilled = true;
      this._multiRing = false;
      this._confirmReplace = false;
      /* The status sensor lags the entry reload: until its region
       * attribute matches, outside-sample checks use this override. */
      this._regionOverride = {
        polygons: [this._polygon.map(([x, y]) => [x, y])],
      };
      this._setStatus(this._t.region_set);
    } catch (err) {
      this._setStatus(this._t.error + (err.message || err));
    }
  }

  _pointInRegion(x, y) {
    /* Even-odd ray casting, the same rule the core rasterizes with. */
    const region = this._region();
    const rings =
      region && region.polygons && region.polygons.length
        ? region.polygons
        : null;
    if (!rings) {
      const b = region ? region.bbox : null;
      if (!b) return true; // region unknown: no warning
      return x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h;
    }
    let inside = false;
    for (const ring of rings) {
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const [xi, yi] = ring[i];
        const [xj, yj] = ring[j];
        if (
          yi > y !== yj > y &&
          x < ((xj - xi) * (y - yi)) / (yj - yi) + xi
        ) {
          inside = !inside;
        }
      }
    }
    return inside;
  }

  async _saveSample() {
    if (!this._filename) return this._setStatus(this._t.need_capture);
    if (!this._drawn) return this._setStatus(this._t.draw_first);
    const r = this._drawn;
    try {
      await this._svc("add_sample", {
        filename: this._filename,
        bin: this._sampleBin,
        rect: [
          this._round(r.x),
          this._round(r.y),
          this._round(r.w),
          this._round(r.h),
        ],
        space: "image",
      });
      const bin = this._config.bins.find((b) => b.id === this._sampleBin);
      /* 9-point probe (corners, edge midpoints, center): still an
       * approximation for exotic concavities, but catches rects that
       * span holes or bridge a concave mouth, which corner-only
       * checks miss. The authoritative veto stays in learning_view. */
      const probes = [];
      for (const fx of [0, 0.5, 1]) {
        for (const fy of [0, 0.5, 1]) {
          probes.push([r.x + fx * r.w, r.y + fy * r.h]);
        }
      }
      const outside = probes.some(([cx, cy]) => !this._pointInRegion(cx, cy));
      this._setStatus(
        this._t.saved_sample +
          (bin ? bin.name : this._sampleBin) +
          (outside ? " - " + this._t.sample_outside : "")
      );
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

  // -- pointer handling ------------------------------------------------

  _pointerPos(ev) {
    const rect = this.shadowRoot.getElementById("stage").getBoundingClientRect();
    return {
      x: Math.min(Math.max((ev.clientX - rect.left) / rect.width, 0), 1),
      y: Math.min(Math.max((ev.clientY - rect.top) / rect.height, 0), 1),
    };
  }

  _vertexAt(pos) {
    const rect = this.shadowRoot.getElementById("stage").getBoundingClientRect();
    for (let i = 0; i < this._polygon.length; i++) {
      const [vx, vy] = this._polygon[i];
      const dx = (vx - pos.x) * rect.width;
      const dy = (vy - pos.y) * rect.height;
      if (Math.hypot(dx, dy) <= VERTEX_HIT_RADIUS_PX) return i;
    }
    return -1;
  }

  _onDown(ev) {
    if (this._mode === "sample") {
      ev.preventDefault();
      const stage = this.shadowRoot.getElementById("stage");
      if (stage.setPointerCapture) stage.setPointerCapture(ev.pointerId);
      this._dragStart = this._pointerPos(ev);
      this._drawn = null;
      return;
    }
    if (this._mode !== "region") return;
    ev.preventDefault();
    const stage = this.shadowRoot.getElementById("stage");
    if (stage.setPointerCapture) stage.setPointerCapture(ev.pointerId);
    const pos = this._pointerPos(ev);
    const hit = this._vertexAt(pos);
    if (hit >= 0) {
      if (
        !this._polygonClosed &&
        hit === 0 &&
        this._polygon.length >= 3
      ) {
        this._polygonClosed = true;
        this._paintRegion();
        return;
      }
      this._dragVertex = hit;
      return;
    }
    if (!this._polygonClosed) {
      this._polygon.push([pos.x, pos.y]);
      this._paintRegion();
    }
  }

  _onMove(ev) {
    if (this._mode === "sample" && this._dragStart) {
      ev.preventDefault();
      const cur = this._pointerPos(ev);
      this._drawn = {
        x: Math.min(this._dragStart.x, cur.x),
        y: Math.min(this._dragStart.y, cur.y),
        w: Math.abs(cur.x - this._dragStart.x),
        h: Math.abs(cur.y - this._dragStart.y),
      };
      this._paintDrawn();
      return;
    }
    if (this._mode === "region" && this._dragVertex !== null) {
      ev.preventDefault();
      const pos = this._pointerPos(ev);
      this._polygon[this._dragVertex] = [pos.x, pos.y];
      this._paintRegion();
    }
  }

  _onUp() {
    if (this._dragStart) {
      this._dragStart = null;
      if (
        this._drawn &&
        (this._drawn.w < MIN_DRAW_FRAC || this._drawn.h < MIN_DRAW_FRAC)
      ) {
        this._drawn = null;
        this._paintDrawn();
      }
    }
    this._dragVertex = null;
  }

  _undoVertex() {
    if (this._polygonClosed) {
      this._polygonClosed = false;
    } else {
      this._polygon.pop();
    }
    this._paintRegion();
  }

  _clearRegion() {
    this._polygon = [];
    this._polygonClosed = false;
    this._paintRegion();
  }

  // -- painting --------------------------------------------------------

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
    box.className = "rect sample";
  }

  _paintRegion() {
    const svg = this.shadowRoot.getElementById("region-svg");
    if (!svg) return;
    const show = this._mode === "region" || this._polygonClosed;
    svg.style.display = show && this._polygon.length ? "block" : "none";
    if (!show || !this._polygon.length) return;
    const pts = this._polygon
      .map(([x, y]) => `${(x * 100).toFixed(3)},${(y * 100).toFixed(3)}`)
      .join(" ");
    const ring = this._polygon
      .map(
        ([x, y], i) =>
          `${i ? "L" : "M"}${(x * 100).toFixed(3)} ${(y * 100).toFixed(3)}`
      )
      .join("");
    const dim = this.shadowRoot.getElementById("region-dim");
    if (this._polygonClosed) {
      /* Outside dimming: frame rect + ring with fill-rule evenodd -
       * exactly the interior the core computes. */
      dim.setAttribute("d", `M0 0H100V100H0Z ${ring}Z`);
      dim.style.display = "block";
    } else {
      dim.style.display = "none";
    }
    const line = this.shadowRoot.getElementById("region-line");
    line.setAttribute(
      "points",
      this._polygonClosed
        ? pts + " " + pts.split(" ")[0]
        : pts
    );
    const markers = this.shadowRoot.getElementById("region-points");
    const stageRect = this.shadowRoot
      .getElementById("stage")
      .getBoundingClientRect();
    /* Marker size mirrors the pointer hit radius (VERTEX_HIT_RADIUS_PX
     * in CSS pixels), converted per axis into viewBox units so the dot
     * stays circular on any stage aspect. */
    const rx = stageRect.width
      ? ((VERTEX_HIT_RADIUS_PX / 2) / stageRect.width) * 100
      : 1.1;
    const ry = stageRect.height
      ? ((VERTEX_HIT_RADIUS_PX / 2) / stageRect.height) * 100
      : 1.1;
    markers.replaceChildren(
      ...this._polygon.map(([x, y], i) => {
        const c = document.createElementNS(
          "http://www.w3.org/2000/svg", "ellipse"
        );
        c.setAttribute("cx", (x * 100).toFixed(3));
        c.setAttribute("cy", (y * 100).toFixed(3));
        const grow = i === 0 && !this._polygonClosed ? 1.4 : 1.0;
        c.setAttribute("rx", (rx * grow).toFixed(3));
        c.setAttribute("ry", (ry * grow).toFixed(3));
        c.setAttribute("class", i === 0 ? "vertex first" : "vertex");
        return c;
      })
    );
  }

  _updateImage() {
    const img = this.shadowRoot.getElementById("cam");
    if (!img || !this._hass) return;
    const state = this._hass.states[this._config.camera];
    if (!state) return;
    const pic = state.attributes.entity_picture;
    if (!pic) return;
    /* Sample/label modes keep the frame the captured file corresponds
     * to; view/region modes follow the live proxy (the URL only
     * changes when HA rotates the access token, so this does not
     * hammer the camera). */
    const frozen =
      this._filename && (this._mode === "sample" || this._mode === "label");
    const url = pic + "&card=" + this._imgCounter;
    if (!img.src || (!frozen && img.src !== url && !img.src.startsWith("/api/media"))) {
      img.src = url;
    }
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
    this.shadowRoot.getElementById("region-actions").style.display =
      mode === "region" ? "flex" : "none";
    this.shadowRoot.getElementById("sample-actions").style.display =
      mode === "sample" ? "flex" : "none";
    this.shadowRoot.getElementById("label-actions").style.display =
      mode === "label" ? "flex" : "none";
    this.shadowRoot.getElementById("stage").style.cursor =
      mode === "region" || mode === "sample" ? "crosshair" : "default";
    if (mode === "region") this._setStatus(this._t.region_hint);
    this._paintDrawn();
    this._paintRegion();
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
        .toolbar, .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; align-items: center; }
        button, select {
          background: var(--secondary-background-color);
          color: var(--primary-text-color);
          border: 1px solid var(--divider-color);
          border-radius: 6px; padding: 6px 10px; cursor: pointer; font: inherit;
        }
        button.active { background: var(--primary-color); color: var(--text-primary-color, #fff); }
        button.chip.present { background: var(--success-color, #0a0); color: #fff; }
        button.chip.absent { background: var(--error-color, #a00); color: #fff; }
        #stage { position: relative; user-select: none; touch-action: none; }
        #cam { display: block; width: 100%; border-radius: 6px; }
        .rect { position: absolute; box-sizing: border-box; pointer-events: none; }
        .rect.sample { border: 2px solid var(--accent-color); background: rgba(255,152,0,.2); }
        .rect.detected { border: 2px solid var(--error-color, #a00); }
        .rect.detected.on { border-color: var(--success-color, #0a0); }
        .rect.detected span {
          position: absolute; top: -1.4em; left: 0; font-size: 11px;
          background: var(--card-background-color); padding: 0 4px; border-radius: 3px;
          white-space: nowrap;
        }
        #overlay { position: absolute; inset: 0; pointer-events: none; }
        #region-svg {
          position: absolute; inset: 0; width: 100%; height: 100%;
          pointer-events: none; display: none;
        }
        #region-dim { fill: rgba(0,0,0,.45); fill-rule: evenodd; }
        #region-line {
          fill: none; stroke: var(--primary-color);
          vector-effect: non-scaling-stroke; stroke-width: 2px;
        }
        .vertex { fill: var(--primary-color); stroke: #fff; stroke-width: .3; }
        .vertex.first { fill: var(--accent-color); }
        #drawn { display: none; }
        #status { margin-top: 8px; font-size: 13px; color: var(--secondary-text-color); min-height: 1.2em; }
      </style>
      <ha-card>
        <div class="toolbar">
          <button id="capture">${t.capture}</button>
          <button data-mode="view" class="active">${t.view}</button>
          <button data-mode="region">${t.region}</button>
          <button data-mode="sample">${t.sample}</button>
          <button data-mode="label">${t.label}</button>
        </div>
        <div id="stage">
          <img id="cam" alt="camera" />
          <div id="overlay"></div>
          <svg id="region-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
            <path id="region-dim" d="" />
            <polyline id="region-line" points="" />
            <g id="region-points"></g>
          </svg>
          <div id="drawn" class="rect"></div>
        </div>
        <div class="actions" id="region-actions" style="display:none">
          <button id="apply-region">${t.apply_region}</button>
          <button id="undo-vertex">${t.undo}</button>
          <button id="clear-region">${t.clear}</button>
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
    this.shadowRoot.getElementById("apply-region").onclick = () =>
      this._applyRegion();
    this.shadowRoot.getElementById("undo-vertex").onclick = () =>
      this._undoVertex();
    this.shadowRoot.getElementById("clear-region").onclick = () =>
      this._clearRegion();
    this.shadowRoot.getElementById("save-sample").onclick = () =>
      this._saveSample();
    this.shadowRoot.getElementById("save-labels").onclick = () =>
      this._saveLabels();
    this.shadowRoot.getElementById("clear-sample").onclick = () => {
      this._drawn = null;
      this._paintDrawn();
    };
    const binSelect = this.shadowRoot.getElementById("sample-bin");
    for (const bin of this._config.bins) {
      const option = document.createElement("option");
      option.value = bin.id;
      option.textContent = bin.name; /* config text: never innerHTML */
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
    this._paintRegion();
    this._setStatus(this._status);
  }
}

customElements.define("wastebin-calibration-card", WastebinCalibrationCard);
window.customCards = window.customCards || [];
window.customCards.push({
  type: "wastebin-calibration-card",
  name: "Wastebin Calibration Card",
  description:
    "Draw the region contour, lid samples and labels for the Wastebin AI Detector directly on the camera image.",
});
