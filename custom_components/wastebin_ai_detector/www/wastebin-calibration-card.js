/* Wastebin AI Detector - calibration card.
 *
 * A thin UI over the integration's services: everything this card does
 * (capture, draw the region contour, mark bins, declare presence, set
 * the region) can also be done from Developer Tools. Marking a bin is
 * ONE statement - "THIS is that bin, and it is here": the card saves
 * the color sample and the present label together (add_sample +
 * label_image, which merges per bin). The presence tab covers the two
 * remaining cases: a bin that is here without a usable lid mark, and
 * a bin that is away. All coordinates are FULL-IMAGE relative - the
 * frame the calibration store anchors its evidence in. The region
 * preview uses SVG fill-rule "evenodd", the exact rule the core
 * rasterizes with: what you see is what is computed.
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
    sample: "Mark bins",
    label: "Presence",
    apply_region: "Apply region",
    undo: "Undo point",
    clear: "Discard",
    present: "here",
    absent: "away",
    unset: "-",
    save_labels: "Save presence",
    start_run: "Start learning run",
    stop_run: "End learning run",
    run_hint: "Declare what is standing there right now, then start. The system captures and learns on its own until you end the run - leave the bins exactly as declared.",
    run_active: "Learning run active: ",
    run_paused: "PAUSED, nothing is being collected: ",
    run_started: "Learning run started. The system now captures on its own.",
    run_stopped: "Learning run ended. Collected frames: ",
    run_locked: "A learning run is active - end it to capture or mark by hand.",
    declare_first: "Declare at least one bin as here or away first.",
    run_needs_status: "Add status_entity to the card config to run learning runs from here.",
    need_capture: "Capture a snapshot first - marks and presence attach to an archived file.",
    draw_first: "Mark at least two points on the lid first.",
    need_closed: "Close the contour first (tap the first point).",
    marked_pre: "Marked ",
    marked_post: " - color sample and “here” saved.",
    draw_next: "Drag the next rectangle.",
    all_marked: "All bins marked.",
    pick_bin: "Which bin is that? Tap it.",
    mark_hint: "Tap the four corners of a lid (a little inside the rim) and its centre, then tap the bin it belongs to. Corner patches capture the light gradient across the lid, not just its average.",
    points_left: " more point(s) recommended.",
    presence_hint: "For bins without a mark: tap to set here/away, then save. Away shots are the valuable negative examples.",
    nothing_set: "Nothing set - tap the bin buttons first.",
    sample_outside: "Warning: a marked point lies outside the region - the detector can only measure this lid once the region covers it.",
    region_set: "Region updated; relearn runs in the background.",
    multi_ring: "This region has several contours; applying will replace all of them with the drawn one.",
    labels_saved: "Presence saved. Relearn: ",
    region_hint: "Tap to add points around every spot where bins can ever stand; tap the first point to close. Drag points to adjust.",
    error: "Error: ",
  },
  de: {
    capture: "Schnappschuss aufnehmen",
    captured: "Aufgenommen: ",
    view: "Ansehen",
    region: "Region zeichnen",
    sample: "Tonnen markieren",
    label: "Anwesenheit",
    apply_region: "Region übernehmen",
    undo: "Punkt zurück",
    clear: "Verwerfen",
    present: "da",
    absent: "weg",
    unset: "-",
    save_labels: "Anwesenheit speichern",
    start_run: "Lernlauf starten",
    stop_run: "Lernlauf beenden",
    run_hint: "Erklären Sie, was gerade dasteht, und starten Sie dann. Das System nimmt selbstständig auf und lernt daraus, bis Sie den Lauf beenden - lassen Sie die Tonnen bitte genau so stehen.",
    run_active: "Lernlauf läuft: ",
    run_paused: "PAUSIERT, es wird nichts gesammelt: ",
    run_started: "Lernlauf gestartet. Das System nimmt jetzt selbst auf.",
    run_stopped: "Lernlauf beendet. Gesammelte Aufnahmen: ",
    run_locked: "Ein Lernlauf läuft - bitte beenden Sie ihn, um selbst aufzunehmen oder zu markieren.",
    declare_first: "Bitte erklären Sie zuerst mindestens eine Tonne als da oder weg.",
    run_needs_status: "Bitte ergänzen Sie status_entity in der Karten-Konfiguration, um Lernläufe hier zu steuern.",
    need_capture: "Bitte nehmen Sie zuerst einen Schnappschuss auf - Markierungen und Anwesenheit gehören zu einer archivierten Datei.",
    draw_first: "Bitte markieren Sie zuerst mindestens zwei Punkte auf dem Deckel.",
    need_closed: "Bitte schließen Sie zuerst die Kontur (ersten Punkt antippen).",
    marked_pre: "",
    marked_post: " markiert - Farb-Sample und „da“ gespeichert.",
    draw_next: "Ziehen Sie das nächste Rechteck.",
    all_marked: "Alle Tonnen markiert.",
    pick_bin: "Welche Tonne ist das? Bitte antippen.",
    mark_hint: "Tippen Sie die vier Ecken eines Deckels an (etwas innerhalb des Randes) und einmal die Mitte, dann die zugehörige Tonne. Die Eckproben erfassen den Lichtverlauf über den Deckel, nicht nur seinen Mittelwert.",
    points_left: " weitere Punkte empfohlen.",
    presence_hint: "Für Tonnen ohne Markierung: Tippen Sie den Button an (da/weg) und speichern Sie. Weggestellte Tonnen liefern die wertvollen Abwesend-Beispiele.",
    nothing_set: "Keine Angabe gesetzt - bitte tippen Sie zuerst die Tonnen-Buttons an.",
    sample_outside: "Hinweis: Ein markierter Punkt liegt außerhalb der Region - diesen Deckel kann der Detektor erst messen, wenn die Region ihn abdeckt.",
    region_set: "Region aktualisiert; das Neu-Lernen läuft im Hintergrund.",
    multi_ring: "Diese Region hat mehrere Konturen; Übernehmen ersetzt sie durch die gezeichnete.",
    labels_saved: "Anwesenheit gespeichert. Neu-Lernen: ",
    region_hint: "Tippen setzt Punkte um alle Stellplätze, an denen je Tonnen stehen können; Tippen auf den ersten Punkt schließt. Punkte lassen sich ziehen.",
    error: "Fehler: ",
  },
};

/* One lid is marked by its four corners plus its centre: five points
 * spread over the surface, so the pooled sample carries the light
 * GRADIENT across the lid (sunlit corner vs shaded corner) instead of
 * one rectangle's average. Not a tuning value - it is the geometry of
 * a quadrilateral lid, and fewer or more points still work (the patch
 * size derives from the points themselves). */
const LID_POINTS_RECOMMENDED = 5;
/* Coordinate resolution sent to the services: 1e-4 of the frame is
 * sub-pixel for any camera up to 10000 px wide, so rounding here can
 * never move a rectangle by a visible amount. */
const COORD_DECIMALS = 4;
/* Hit radius for grabbing/closing on a vertex, in CSS pixels: the
 * platform convention for comfortable touch targets (Material/HIG use
 * 24-48 px targets; 14 px radius = 28 px diameter is the small end of
 * that range so neighboring vertices stay individually grabbable). */
const VERTEX_HIT_RADIUS_PX = 14;

/* Session state that must survive ELEMENT RECREATION: Home Assistant
 * rebuilds dashboard cards on all kinds of updates, and every mark
 * triggers a relearn that changes entity states. A rebuilt element
 * would silently drop the capture, the marks and the current mode -
 * the user then has to press the mode button again before they can
 * draw, which is exactly the friction this store removes. Keyed by the
 * card's target so two cards on one dashboard stay independent. */
const SESSIONS = new Map();

class WastebinCalibrationCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._mode = "view";
    this._points = []; // [[x, y], ...] lid points, image-relative
    this._locked = false; // a declared learning run is active
    this._saving = false;
    this._dragStart = null;
    this._filename = null;
    this._labels = {}; // presence chips (mirror of what save would send)
    this._savedLabels = {}; // presence last persisted for _filename
    this._marks = {}; // binId -> rect, marks saved on _filename this session
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
    this._sessionKey = (config.entry_id || "") + "|" + config.camera;
    const saved = SESSIONS.get(this._sessionKey);
    if (saved) {
      this._mode = saved.mode;
      this._points = saved.points || [];
      this._filename = saved.filename;
      this._marks = saved.marks;
      this._labels = saved.labels;
      this._savedLabels = saved.savedLabels;
    }
    this._render();
  }

  _persist() {
    if (!this._sessionKey) return;
    SESSIONS.set(this._sessionKey, {
      mode: this._mode,
      points: this._points,
      filename: this._filename,
      marks: this._marks,
      labels: this._labels,
      savedLabels: this._savedLabels,
    });
  }

  set hass(hass) {
    this._hass = hass;
    this._t = TEXTS[(hass.language || "en").startsWith("de") ? "de" : "en"];
    if (!this._built) this._render();
    if (
      this._filename &&
      !this._frameShown &&
      (this._mode === "sample" || this._mode === "label")
    ) {
      /* Restored session (or a rebuilt element): put the archived frame
       * the marks belong to back on screen, once. */
      this._frameShown = true;
      this._showArchivedFrame();
    }
    this._updateImage();
    this._updateOverlay();
    this._renderRunRow();
    this._maybePrefillRegion();
  }

  getCardSize() {
    return 6;
  }

  _session() {
    /* The running learning run, read from the status sensor: the card
     * never keeps its own idea of whether a run is active. */
    if (!this._config.status_entity || !this._hass) return null;
    const state = this._hass.states[this._config.status_entity];
    if (!state) return null;
    const auto = state.attributes.auto_sampling || null;
    return auto && auto.declaration ? auto : null;
  }

  async _startRun() {
    const present = [];
    const absent = [];
    for (const [binId, value] of Object.entries(this._labels)) {
      if (value === "present") present.push(binId);
      if (value === "absent") absent.push(binId);
    }
    if (!present.length && !absent.length) {
      // Nothing declared yet: take the user to the row where that
      // happens instead of only complaining about it.
      this._setMode("label");
      return this._setStatus(this._t.declare_first);
    }
    try {
      await this._svc("start_learning", { present, absent }, true);
      this._setStatus(this._t.run_started);
      this._renderRunRow();
    } catch (err) {
      this._setStatus(this._t.error + (err.message || err));
    }
  }

  async _stopRun() {
    try {
      const result = await this._svc("stop_learning", {}, true);
      this._setStatus(
        this._t.run_stopped + (result?.response?.collected ?? "?")
      );
      this._renderRunRow();
    } catch (err) {
      this._setStatus(this._t.error + (err.message || err));
    }
  }

  _renderRunRow() {
    const row = this.shadowRoot.getElementById("run-actions");
    if (!row) return;
    const start = this.shadowRoot.getElementById("start-run");
    const stop = this.shadowRoot.getElementById("stop-run");
    const info = this.shadowRoot.getElementById("run-info");
    if (!this._config.status_entity) {
      /* The run state lives on the status sensor. Without it the card
       * could start a run it can neither show nor stop, so it offers
       * none and says why. */
      start.style.display = "none";
      stop.style.display = "none";
      info.textContent = this._t.run_needs_status;
      return;
    }
    const session = this._session();
    start.style.display = session ? "none" : "";
    stop.style.display = session ? "" : "none";
    if (session) {
      const declared = Object.entries(session.declaration)
        .map(([binId, state]) => {
          const bin = this._config.bins.find((b) => b.id === binId);
          return (
            (bin ? bin.name : binId) +
            " " +
            (state === "present" ? this._t.present : this._t.absent)
          );
        })
        .join(", ");
      /* The situation this run is collecting for, and how far it has
       * got. A paused run is the one state that must never hide: it
       * looks exactly like a working one otherwise. */
      const mine = (session.situations || []).find((s2) => {
        const present = Object.entries(session.declaration)
          .filter(([, state]) => state === "present")
          .map(([binId]) => binId)
          .sort();
        const absent = Object.entries(session.declaration)
          .filter(([, state]) => state === "absent")
          .map(([binId]) => binId)
          .sort();
        return (
          JSON.stringify([...(s2.present || [])].sort()) ===
            JSON.stringify(present) &&
          JSON.stringify([...(s2.absent || [])].sort()) ===
            JSON.stringify(absent)
        );
      });
      const kept = mine ? mine.retained : 0;
      info.textContent =
        this._t.run_active +
        declared +
        " (" +
        kept +
        "/" +
        (session.capacity_per_situation ?? "?") +
        ")" +
        (session.paused ? " - " + this._t.run_paused + session.paused : "");
      info.classList.toggle("warn", Boolean(session.paused));
    } else {
      info.textContent = "";
      info.classList.remove("warn");
    }
    /* While a run is active the yard must stay as declared, so manual
     * capturing and marking are out of reach until it ends. */
    this._locked = Boolean(session);
    for (const id of ["capture", "save-labels"]) {
      const el = this.shadowRoot.getElementById(id);
      if (el) el.disabled = this._locked;
    }
    this._renderMarkRow();
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
    if (this._session()) return this._setStatus(this._t.run_locked);
    try {
      const result = await this._svc("capture_snapshot", {}, true);
      this._filename = result?.response?.filename || null;
      this._frameShown = true;
      this._labels = {};
      this._savedLabels = {};
      this._marks = {};
      this._points = [];
      this._paintPoints();
      this._paintMarks();
      this._renderMarkRow();
      await this._showArchivedFrame();
      this._renderLabelRow();
      /* A fresh capture exists to be marked: enter mark mode right
       * away so the standard pass is capture -> drag -> tap bin ->
       * drag -> tap bin, with no mode button in between. */
      this._setMode("sample");
      this._setStatus(
        this._t.captured + (this._filename || "?") + " " + this._t.mark_hint
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

  async _saveSample(binId) {
    if (this._session()) return this._setStatus(this._t.run_locked);
    if (!this._filename) return this._setStatus(this._t.need_capture);
    if (this._points.length < 2) return this._setStatus(this._t.draw_first);
    this._saving = true; /* one statement at a time */
    this._renderMarkRow();
    try {
      await this._saveSampleInner(binId);
    } finally {
      this._saving = false;
      this._renderMarkRow();
    }
  }

  async _saveSampleInner(binId) {
    const patches = this._patches();
    const bin = this._config.bins.find((b) => b.id === binId);
    const name = bin ? bin.name : binId;
    /* Warn about the POINTS the user tapped, not about the corners of
     * the patches derived from them. Regions are drawn tightly around
     * the bins, so a patch on a lid corner routinely crosses the
     * contour by a pixel or two - warning about that fired on every
     * bin at once and meant nothing. A tap outside the region is a
     * real mistake and the only thing worth saying. The authoritative
     * veto stays in learning_view either way. */
    const outside = this._points.some(
      ([px, py]) => !this._pointInRegion(px, py)
    );
    for (const r of patches) {
      try {
        await this._svc("add_sample", {
          filename: this._filename,
          bin: binId,
          rect: [
            this._round(r.x),
            this._round(r.y),
            this._round(r.w),
            this._round(r.h),
          ],
          space: "image",
        });
      } catch (err) {
        return this._setStatus(this._t.error + (err.message || err));
      }
    }
    /* The store appends samples, it never replaces - so the overlay
     * keeps every saved patch too (marking the same bin again shows
     * both sets, matching what the server holds). */
    const saved = this._marks[binId] || (this._marks[binId] = []);
    saved.push(...patches);
    this._points = [];
    this._paintPoints();
    this._renderMarkRow();
    this._persist();
    /* One gesture, one statement: the marked lid is "THIS is that bin,
     * and it is here", so the present label is saved in the same step.
     * set_labels merges per bin - other bins are never clobbered. */
    let relearn = null;
    try {
      const result = await this._svc(
        "label_image",
        { filename: this._filename, present: [binId], absent: [] },
        true
      );
      relearn = result?.response?.relearn || "?";
      this._labels[binId] = "present";
      this._savedLabels[binId] = "present";
      this._renderLabelRow();
      this._persist();
    } catch (err) {
      /* The samples ARE stored, only the presence statement failed.
       * Show it as a PENDING "here" (chip and mark tag get the dirty
       * star), so "Save presence" completes the statement - marking
       * again would append a second set of patches instead. */
      this._labels[binId] = "present";
      this._renderLabelRow();
      return this._setStatus(
        this._t.error + (err.message || err) +
          (outside ? " " + this._t.sample_outside : "")
      );
    }
    /* Re-assert the mode: the card stays ready for the next lid
     * without any button in between, even if something rebuilt the
     * action row while the relearn was running. */
    this._setMode("sample");
    const missing = this._config.bins.filter((b) => !this._marks[b.id]);
    this._setStatus(
      this._t.marked_pre + name + this._t.marked_post +
        (relearn && relearn !== "ok" ? " (" + relearn + ")" : "") +
        (outside ? " " + this._t.sample_outside : "") +
        " " +
        (missing.length ? this._t.draw_next : this._t.all_marked)
    );
  }

  _frameAspect() {
    /* Width/height of the analysed frame in PIXELS: relative
     * coordinates are per-axis fractions, so distances must be
     * un-squashed before they can be compared geometrically. */
    const img = this.shadowRoot.getElementById("cam");
    if (img && img.naturalWidth && img.naturalHeight) {
      return img.naturalWidth / img.naturalHeight;
    }
    const rect = this.shadowRoot.getElementById("stage").getBoundingClientRect();
    return rect.height ? rect.width / rect.height : 1;
  }

  _patches() {
    /* One square patch per marked point, all the SAME size: half the
     * smallest distance between any two marked points. Sizing each
     * patch by its own nearest neighbour instead made corner patches
     * grow far larger than centre ones and pushed them over the lid
     * edge and out of the region - measured in the field on all three
     * bins at once. One uniform, conservative size derives from the
     * user's own geometry, keeps every patch clear of its neighbours,
     * and treats every marked spot as the equally important sample it
     * is. Squares are square in PIXELS (relative coordinates are
     * per-axis fractions) and are clamped into the frame.
     */
    const aspect = this._frameAspect();
    const scaled = this._points.map(([x, y]) => [x * aspect, y]);
    let closest = Infinity;
    for (let i = 0; i < scaled.length; i++) {
      for (let j = i + 1; j < scaled.length; j++) {
        const d = Math.hypot(
          scaled[i][0] - scaled[j][0],
          scaled[i][1] - scaled[j][1]
        );
        if (d < closest) closest = d;
      }
    }
    if (!Number.isFinite(closest)) return [];
    const r = closest / 2;
    const rx = r / aspect;
    return this._points.map(([x, y]) => {
      const x0 = Math.max(x - rx, 0);
      const y0 = Math.max(y - r, 0);
      const x1 = Math.min(x + rx, 1);
      const y1 = Math.min(y + r, 1);
      return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
    });
  }

  async _saveLabels() {
    if (this._session()) return this._setStatus(this._t.run_locked);
    if (!this._filename) return this._setStatus(this._t.need_capture);
    const present = [];
    const absent = [];
    for (const [binId, state] of Object.entries(this._labels)) {
      if (state === "present") present.push(binId);
      if (state === "absent") absent.push(binId);
    }
    if (!present.length && !absent.length) {
      return this._setStatus(this._t.nothing_set);
    }
    const button = this.shadowRoot.getElementById("save-labels");
    button.disabled = true; /* no double submit while the call runs */
    try {
      const result = await this._svc(
        "label_image",
        { filename: this._filename, present, absent },
        true
      );
      const relearn = result?.response?.relearn || "?";
      const warnings = result?.response?.warnings || [];
      this._savedLabels = { ...this._labels };
      this._renderLabelRow();
      this._persist();
      this._setStatus(
        this._t.labels_saved + relearn +
        (warnings.length ? " (" + warnings.length + " warnings)" : "")
      );
    } catch (err) {
      this._setStatus(this._t.error + (err.message || err));
    } finally {
      button.disabled = false;
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
      const pos = this._pointerPos(ev);
      this._points.push([pos.x, pos.y]);
      this._paintPoints();
      this._renderMarkRow();
      this._persist();
      const left = LID_POINTS_RECOMMENDED - this._points.length;
      this._setStatus(
        left > 0 ? left + this._t.points_left : this._t.pick_bin
      );
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
    if (this._mode === "region" && this._dragVertex !== null) {
      ev.preventDefault();
      const pos = this._pointerPos(ev);
      this._polygon[this._dragVertex] = [pos.x, pos.y];
      this._paintRegion();
    }
  }

  _onUp() {
    this._dragStart = null;
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

  _paintPoints() {
    /* The marked points and the patches derived from them: what the
     * user tapped and what will actually be sampled. */
    const layer = this.shadowRoot.getElementById("points");
    if (!layer) return;
    if (this._mode !== "sample" || !this._points.length) {
      layer.replaceChildren();
      return;
    }
    const boxes = [];
    if (this._points.length >= 2) {
      for (const r of this._patches()) {
        const div = document.createElement("div");
        div.className = "rect sample";
        div.style.left = r.x * 100 + "%";
        div.style.top = r.y * 100 + "%";
        div.style.width = r.w * 100 + "%";
        div.style.height = r.h * 100 + "%";
        boxes.push(div);
      }
    }
    for (const [x, y] of this._points) {
      const dot = document.createElement("div");
      dot.className = "lid-point";
      dot.style.left = x * 100 + "%";
      dot.style.top = y * 100 + "%";
      boxes.push(dot);
    }
    layer.replaceChildren(...boxes);
  }

  _paintMarks() {
    /* The marks saved on the current capture, labeled per bin: the
     * visible answer to "what have I already told the system about
     * this image?". Session-local by design - the card labels only
     * the snapshot it just captured. The tag text mirrors the CHIP
     * state (never a hardcoded "here"): if the presence is later
     * flipped or still pending, the mark says so too. */
    const layer = this.shadowRoot.getElementById("marks");
    if (!layer) return;
    const show =
      (this._mode === "sample" || this._mode === "label") && this._filename;
    if (!show) {
      layer.replaceChildren();
      return;
    }
    const boxes = [];
    for (const [binId, rects] of Object.entries(this._marks)) {
      const bin = this._config.bins.find((b) => b.id === binId);
      const state = this._labels[binId];
      const pending =
        (state || null) !== (this._savedLabels[binId] || null);
      const text =
        state === "present"
          ? this._t.present
          : state === "absent"
            ? this._t.absent
            : this._t.unset;
      rects.forEach((r, i) => {
        const div = document.createElement("div");
        div.className = "rect mark";
        div.style.left = r.x * 100 + "%";
        div.style.top = r.y * 100 + "%";
        div.style.width = r.w * 100 + "%";
        div.style.height = r.h * 100 + "%";
        if (i === rects.length - 1) {
          const tag = document.createElement("span");
          tag.textContent =
            (bin ? bin.name : binId) + ": " + text + (pending ? " *" : "");
          div.appendChild(tag);
        }
        boxes.push(div);
      });
    }
    layer.replaceChildren(...boxes);
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
    if (mode === "sample") {
      this._renderMarkRow();
      if (this._status !== this._t.pick_bin) {
        this._setStatus(this._t.mark_hint);
      }
    }
    if (mode === "label") {
      this._setStatus(
        this._session() ? this._t.run_locked : this._t.run_hint
      );
    }
    this._persist();
    this._paintPoints();
    this._paintMarks();
    this._paintRegion();
    this._updateOverlay();
  }

  _cycleLabel(binId) {
    /* A persisted presence cannot go back to "-": the store has no
     * unlabel operation, so "-" after a save would display a state
     * the server does not have. Saved bins toggle here/away only. */
    const order =
      this._savedLabels[binId] !== undefined
        ? ["present", "absent"]
        : [undefined, "present", "absent"];
    const cur = this._labels[binId];
    const next = order[(order.indexOf(cur) + 1) % order.length];
    if (next === undefined) delete this._labels[binId];
    else this._labels[binId] = next;
    this._renderLabelRow();
  }

  _renderMarkRow() {
    /* The bin choice IS the gesture's second half: drag a rectangle,
     * then tap the bin it shows. No dropdown to pre-select and no mode
     * button in between, so a full pass is drag/tap per bin. Buttons
     * stay disabled until a rectangle exists (and while a save runs),
     * which is also the affordance telling the user to draw first. */
    const row = this.shadowRoot.getElementById("mark-bins");
    if (!row) return;
    const ready =
      this._points.length >= 2 && !this._saving && !this._locked;
    row.replaceChildren(
      ...this._config.bins.map((bin) => {
        const btn = document.createElement("button");
        const marked = (this._marks[bin.id] || []).length;
        btn.textContent = bin.name + (marked ? " \u2713" : "");
        btn.className = "chip" + (marked ? " present" : "");
        btn.disabled = !ready;
        btn.onclick = () => this._saveSample(bin.id);
        return btn;
      })
    );
  }

  _renderLabelRow() {
    const row = this.shadowRoot.getElementById("label-bins");
    if (!row) return;
    let dirty = false;
    row.replaceChildren(
      ...this._config.bins.map((bin) => {
        const btn = document.createElement("button");
        const state = this._labels[bin.id];
        const pending =
          (state || null) !== (this._savedLabels[bin.id] || null);
        if (pending) dirty = true;
        btn.textContent =
          bin.name + ": " +
          (state === "present"
            ? this._t.present
            : state === "absent"
              ? this._t.absent
              : this._t.unset) +
          (pending ? " *" : "");
        btn.className =
          "chip" +
          (state === "present" ? " present" : state === "absent" ? " absent" : "");
        btn.onclick = () => this._cycleLabel(bin.id);
        return btn;
      })
    );
    const save = this.shadowRoot.getElementById("save-labels");
    if (save) save.classList.toggle("attention", dirty);
    this._paintMarks(); /* mark tags mirror the chip states */
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
        .rect.mark {
          border: 2px solid var(--success-color, #0a0);
          background: rgba(0, 160, 0, .12);
        }
        .rect.detected span, .rect.mark span {
          position: absolute; top: -1.4em; left: 0; font-size: 11px;
          background: var(--card-background-color); padding: 0 4px; border-radius: 3px;
          white-space: nowrap;
        }
        button.attention {
          border-color: var(--primary-color);
          box-shadow: 0 0 0 1px var(--primary-color) inset;
        }
        button:disabled { opacity: .5; cursor: wait; }
        #overlay, #marks, #points { position: absolute; inset: 0; pointer-events: none; }
        .lid-point {
          position: absolute; width: 10px; height: 10px; margin: -5px 0 0 -5px;
          border-radius: 50%; background: var(--accent-color);
          border: 2px solid #fff; box-sizing: border-box; pointer-events: none;
        }
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
        #status { margin-top: 8px; font-size: 13px; color: var(--secondary-text-color); min-height: 1.2em; }
        #run-info { font-size: 13px; color: var(--secondary-text-color); }
        #run-info.warn { color: var(--error-color, #a00); font-weight: 500; }
        #run-actions { margin-bottom: 4px; }
      </style>
      <ha-card>
        <div class="actions" id="run-actions">
          <button id="start-run">${t.start_run}</button>
          <button id="stop-run" style="display:none">${t.stop_run}</button>
          <span id="run-info"></span>
        </div>
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
          <div id="marks"></div>
          <div id="points"></div>
          <svg id="region-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
            <path id="region-dim" d="" />
            <polyline id="region-line" points="" />
            <g id="region-points"></g>
          </svg>
          </div>
        <div class="actions" id="region-actions" style="display:none">
          <button id="apply-region">${t.apply_region}</button>
          <button id="undo-vertex">${t.undo}</button>
          <button id="clear-region">${t.clear}</button>
        </div>
        <div class="actions" id="sample-actions" style="display:none">
          <span id="mark-bins" class="actions"></span>
          <button id="undo-point">${t.undo}</button>
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
    this.shadowRoot.getElementById("save-labels").onclick = () =>
      this._saveLabels();
    this.shadowRoot.getElementById("start-run").onclick = () =>
      this._startRun();
    this.shadowRoot.getElementById("stop-run").onclick = () =>
      this._stopRun();
    this.shadowRoot.getElementById("undo-point").onclick = () => {
      this._points.pop();
      this._paintPoints();
      this._renderMarkRow();
      this._persist();
    };
    this.shadowRoot.getElementById("clear-sample").onclick = () => {
      this._points = [];
      this._paintPoints();
      this._renderMarkRow();
      this._persist();
    };
    this._renderMarkRow();
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

/* The module can legitimately be evaluated twice in one page - the
 * integration registers it as an extra JS module while a user may also
 * have it as a dashboard resource, and the URL carries a version query
 * that changes on update. Defining the element twice throws and takes
 * the whole module down with it, so registration is idempotent. */
if (!customElements.get("wastebin-calibration-card")) {
  customElements.define("wastebin-calibration-card", WastebinCalibrationCard);
  window.customCards = window.customCards || [];
  window.customCards.push({
    type: "wastebin-calibration-card",
    name: "Wastebin Calibration Card",
    description:
      "Draw the region contour, mark bins and declare presence for the Wastebin AI Detector directly on the camera image.",
  });
}
