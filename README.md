# Wastebin AI Detector

**Camera-based waste-bin presence detection for Home Assistant. Fully
local, no cloud, no LLM, no extra hardware.**

> **Status: alpha (phase 1).** The detection core and the offline
> calibration CLI are functional and tested. The Home Assistant wiring
> (config flow, one `binary_sensor` per bin) is phase 2 and under
> construction. Installing the integration today sets up nothing yet.

## What it does

Point it at a camera that sees the spot where your waste bins live.
The detector tells you **which bins are currently there**, e.g. to
notify you on collection day that the yellow bin is still in the yard.

It works because bins are distinguishable by lid color: the detector
finds contiguous areas matching each bin's learned color inside a
region of interest you define once.

## How it works (and why there is no AI model)

There is no universal model that knows every yard, camera angle and
regional bin color scheme, and a model trained on one yard is useless
in the next. Instead, **each installation teaches the detector its own
bins**:

1. You define the camera, a region of interest (where bins can stand)
   and your bins (any number, any colors).
2. You mark each lid with five taps in a few snapshots (four corners
   plus the centre) and label a handful of images ("blue bin present /
   absent").
3. The detector learns everything else from that: per-bin hue bands
   (circular statistics), saturation/brightness floors, and the minimum
   blob area that separates "present" from "absent", computed
   discriminatively from your labeled images.

All thresholds live in a JSON *profile* that belongs to your
installation. The code itself contains no tuned numbers.

Detection is classic image analysis (HSV masks + connected components,
pure numpy) and takes milliseconds on a Raspberry Pi.

## Known limitations

- **Daylight only.** IR night frames are greyscale; the detector flags
  them (`grayscale_suspect`) instead of guessing. Gate your automation
  on the sun, keep the last daylight state at night.
- **Bins must differ in color.** Two same-colored bins cannot be told
  apart by a color detector. That is inherent, not a bug: since v0.6.0
  they additionally compete for every pixel, so both end up with a
  fragment of the shared color and the relearn reports how little each
  one keeps.
- **Grey and black lids cannot be color-calibrated.** A hue-based
  detector has nothing to lock onto on an achromatic surface (in
  Germany that is typically the black residual-waste bin). Practical
  workaround: stick a small marker of a distinct color onto the lid
  and calibrate on the marker. A learned achromatic mode is on the
  roadmap.
- **Calibrate across conditions.** Feed the learner snapshots from
  several days and weathers (morning shade, bright sun, rain) before
  trusting it; the more varied the calibration set, the more robust the
  learned thresholds.

## Calibration (phase 1, offline CLI)

```bash
python tools/calibrate.py init --store calib.json \
    --roi 0.30 0.25 0.45 0.65 --width 480 \
    --bin yellow="Yellow bin" --bin blue="Blue bin"

python tools/calibrate.py sample --store calib.json --image snap1.jpg \
    --bin yellow --rect 0.52 0.61 0.06 0.05 --space image

python tools/calibrate.py label --store calib.json --image snap1.jpg \
    --present yellow --present blue

python tools/calibrate.py learn --store calib.json --profile profile.json

python tools/calibrate.py detect --profile profile.json --image snap2.jpg --json
```

## Dynamic reconfiguration (v0.3.3)

Everything set up once can be changed later without losing a single
piece of training data - Settings > Devices & Services > Reconfigure:

- **Change camera** (with an honest "same field of view?" question) or
  mark the camera as **re-aimed** after it was bumped.
- **Edit the region of interest**: sample rectangles are stored
  relative to the full frame and survive; labels made under a smaller
  region stay usable when it grows.
- **Add, retire and reactivate bins**: retiring removes the sensor but
  keeps all evidence; a reactivated bin returns with its full history.
- Evidence is never deleted: `forget_image` only excludes (reversible
  via `restore_image`), `reconfirm_images` re-asserts labels after a
  reconfiguration, `mark_bin_appearance_changed` handles a municipality
  lid-color swap by putting old color evidence to sleep instead of
  erasing it.
- The calibration store is mirrored as `store.json` next to the
  archived snapshots on every save, so the archive folder stays
  self-contained (movable, CLI-usable, recoverable). Removing the
  integration exports the store there first.

Downgrade note: once a store was saved by v0.3.3 (schema v2), older
versions cannot read it; the original v1 store is kept as
`calibration_v1_backup.json` in the archive folder.

## Polygon region and shape plausibility (v0.5.0)

The region of interest can now be a free-form contour (any number of
points, drawn on the calibration card): draw it generously around
every spot where bins can EVER stand - hedges, gates and pavement
outside the contour can no longer produce false matches. Existing
rectangle setups keep working unchanged and can be refined into a
contour at any time; nothing is lost in either direction (the previous
store is kept as `calibration_v2_backup.json` in the archive folder;
older releases cannot read the new store format).

Detection additionally learns each lid's blob SHAPE (compactness and
aspect, with rotation-geometry safety margins) from your own
calibration images: a ragged hedge fringe or a tall sunlit streak of
lid color is rejected as geometrically implausible even inside the
region, so "no plausible blob" now honestly means absent.

Card config gains two keys:

```yaml
type: custom:wastebin-calibration-card
camera: camera.backyard
status_entity: sensor.backyard_status   # region prefill for the editor
bins:
  - id: gelbe_tonne
    name: Gelbe Tonne
```

In the card, "Draw region" replaces the rectangle mode: tap to add
points, tap the first point to close, drag points to adjust, then
apply - the relearn runs automatically and keeps all evidence.

## Learning runs: the system trains itself on a declared situation (v0.7.0)

Calibrating used to mean taking snapshots by hand and confirming each
one. Now you declare the situation once and let the system work:

1. Arrange the yard and mark each lid with five taps (once per bin).
2. In the presence row, declare what is standing there: **here** for
   the bins that are out, **away** for the ones that are not.
3. Press **Start learning run**.

There is no separate learning switch any more: a run is declared and
ended, and outside a run nothing is captured at all.

From then on the integration captures on its own at the configured
interval and records every usable frame as an observation of exactly
that declared situation. Manual capturing and marking are locked while
a run is active, because the declaration only holds as long as the yard
does: leave the bins as declared until you press **End learning run**.

**What a run may teach, and what it may not.** Your declaration says a
bin STANDS there. It does not say its lid is measurable in a given
frame, and those differ exactly when a shadow, a van or heavy rain
covers most of the lid. The presence threshold rests on the SMALLEST
positive area ever confirmed, so a truthful declaration plus one
covered lid would collapse it (measured on the real learner: a
305-fold drop, adopted silently). Frames from a run therefore teach:

- **colour**, always - that is what they are collected for;
- **negative evidence** when the bin was declared away, because "the
  yard is empty" is a complete statement that needs no judgement about
  visibility;
- **never** the smallest positive area, the lid shape bounds or the
  region-edge band. Those are extrema, and a human has to have looked
  at the frame to set them.

Declaring bins **away** is how you get negative evidence without
waiting for collection day: put them aside, declare them away, let a
run gather the "empty yard" across an afternoon of changing light.

What keeps an unattended run from slowly corrupting the models:

- **A bounded reservoir.** At most ceil(1 + 100/5) = 21 frames are kept
  per declared situation, and a new frame is only kept if it improves
  the spread over the measured light space. The number is derived from
  the percentile the colour floors are learned with, not chosen, and
  the dispersion rule doubles as the rate limit - leaving a run on for
  weeks is safe by construction.
- **Model-free gates only.** A frame is rejected when it is a night
  frame, overexposed or a broken keyframe (the very gates detection
  already uses), or when a marked patch is no longer ONE colour (a bin
  that moved). Never "does it match what I already know" - that would
  reject exactly the new light the run exists to capture.
- **A holdout test.** Every relearn with collected evidence learns
  twice, with and without it, and adopts the collected half only if the
  resulting profile still reproduces every image YOU labelled by hand.
  Otherwise the collected frames are set aside (never deleted), the run
  pauses and the relearn says why.
- **Anti-ratchet.** Collected frames never re-derive the light gates
  that admitted them, or the night protection would erode one step per
  relearn.

`discard_auto_evidence` / `restore_auto_evidence` take everything a run
collected out of or back into training in one call.

### Train everything (v0.7.5, off by default)

The safety rails above have a cost: under flat overcast light a pale
lid carries so little chroma that its hue scatters, the patch is filed
as incoherent and the frame is refused - so the washed-out condition
one most wants to learn is the one that never gets learned. The
integration option **Train everything** turns the three judgements
about light into observations:

- nothing about the colour of a marked patch is fatal any more: a patch
  without a coherent majority is learned, and so is one whose hue band
  ends up covering half the colour circle. Both refusals become
  warnings;
- the saturation and brightness floors come from every marked pixel
  rather than only the coherent ones (the hue band still comes from the
  coherent pixels wherever there are any, so colour identity stays as
  discriminative as the light allows);
- an overexposed frame is collected instead of refused;
- the holdout test still runs and still reports, but it no longer
  discards the evidence or pauses the run.

Two gates stay in place, because they are not about light: a broken
keyframe is corrupted data, and a greyscale night frame has no hue at
all - learning from it would drag every saturation floor to zero and
with it any ability to tell the bins apart.

Expect more background matches in exchange for detection that keeps
working in glare. The card marks a run collecting under this mode, and
a forced adoption is reported in the relearn response (`auto.forced`)
and as a warning.

## Bins exclude each other, and lids are marked by points (v0.6.0)

**Two bins can never occupy the same spot**, so detection now enforces
that physically:

- Every pixel belongs to at most ONE bin. When two learned color bands
  overlap (the learner warns about it), a contested pixel goes to the
  bin whose learned color it is closest to, in degrees. Ties go to
  nobody, which also makes the result independent of bin order.
- A blob detected INSIDE another bin's detected area cannot be a bin:
  a white sticker on a brown lid is a hole in the brown blob, and a
  candidate sitting in that hole is dropped. Only a bin whose own
  calibration separates and whose blob is at least the weakest lid ever
  confirmed for it may veto another - a bin that its own numbers call
  unreliable never erases someone else's evidence. Where an enclosure
  exists but no container qualifies, the bin is reported as contested
  and held instead of flipping.
- Resolution runs to a fixed point (a vetoed bin re-selects elsewhere,
  and that new blob is checked again), and thresholds are learned under
  exactly the same rules, so measurement and detection stay identical.

**Marking a lid takes five taps instead of a drawn rectangle**: the
four corners (a little inside the rim) and the centre. Each point
becomes its own sample patch, sized from the distance to its nearest
neighbour, so the pooled sample carries the light GRADIENT across the
lid (sunlit corner versus shaded corner) instead of one rectangle's
average. Fewer or more points work too; the patch size always derives
from the points themselves.

## Region-edge band and one-vote-per-image (v0.5.2)

Two field failures, two learned cures, no new knobs:

**Boundary slivers.** Background that the region contour cuts through
(a hedge leaf at the edge, a sunlit paving stone) can match a lid color
and, in harsh light where real lids shrink, exceed the learned
threshold. Detection now measures how deep each calibrated lid reaches
into the region and requires the same of any candidate that TOUCHES the
region boundary: a component confined to the boundary rim is background,
not a lid. Components fully inside the region are never filtered, so a
bin may still stand anywhere, and a lid that crosses the contour (tight
regions are normal) stays detected because it reaches deep. The band is
learned per bin from boundary-touching lids only, as a fraction of the
working grid, and stays inactive until two such observations exist - the
mechanism never invents separation. If absent-labeled frames show
boundary clutter reaching that depth, the relearn says so instead of
letting the overlap pass silently.

**One big sample outvoting several small ones.** Color models pooled
sample pixels, so a single large rectangle drawn on a washed-out lid
could outvote several small clean ones and blow up the learned hue band
(field: a bin's model collapsed and the sensor went unavailable). Every
sample IMAGE now carries one vote regardless of rectangle size, and the
mixture fit runs from two deterministic starts so a quantization spike
in a washed-out patch can no longer trap it.

## Calibration card (v0.4.0)

The integration ships a Lovelace card (registered automatically, no
resource setup needed). Add it to any dashboard:

```yaml
type: custom:wastebin-calibration-card
camera: camera.backyard
bins:
  - id: gelbe_tonne
    name: Gelbe Tonne
  - id: blaue_tonne
    name: Blaue Tonne
entities:            # optional: live detection overlay
  - binary_sensor.backyard_gelbe_tonne
  - binary_sensor.backyard_blaue_tonne
```

- **Capture snapshot** archives the current frame; marks and presence
  you create afterwards attach to it.
- **Draw region** tap points around every spot bins can ever stand and
  apply (runs through the same lossless path as the reconfigure
  dialog, via the `set_roi` service).
- **Mark bins** (v0.5.3) drag a rectangle over a lid, then tap the bin
  it shows. One gesture is one statement: "THIS is that bin, and it is
  here". The card saves the color sample and the present label
  together, shows the mark on the image and stays ready for the next
  rectangle; relearn runs automatically. A capture enters this mode by
  itself, so a full pass is capture, then drag/tap per bin - no mode
  button and no dropdown in between. The session (capture, marks,
  mode) survives dashboard rebuilds, which HA does on every state
  change a relearn causes.
- **Presence** covers the two remaining cases: a bin that is here
  without a usable lid mark, and a bin that is away (the valuable
  negative examples). Toggle and save; unsaved changes are starred.
- **View** overlays the currently detected blob boxes (the presence
  sensors also expose `bbox` and `centroid` attributes in full-image
  relative coordinates).

A failed background relearn (for example after a region change that
sets labels aside) now raises a Repairs issue instead of only a log
line; the next successful relearn clears it.

## Roadmap

- **Phase 2:** Home Assistant config flow, guided calibration, one
  `binary_sensor` per bin, learning mode with notification-based
  labeling, HACS release.
- **Phase 3:** blueprint combining bin presence with waste-collection
  calendar integrations ("bin day tomorrow, but the yellow bin is
  still in the yard").
- **Later:** learned achromatic mode for grey/black lids
  (saturation/value bands instead of a hue band, same percentile
  scheme).
- **Experiment:** data-derived augmentation for cold starts
  (brightness/gamma jitter spanning the range observed in the
  installation's own snapshot archive, never hand-picked factors), to
  widen the learned floors before days of real variance have been
  collected.

---

## Deutsch (Kurzfassung)

Kamerabasierte Mülltonnen-Erkennung für Home Assistant: komplett
lokal, ohne Cloud, ohne LLM, ohne Zusatzhardware. Jede Installation
lernt ihre eigenen Tonnen (Anzahl und Farben frei): einmal Bereich
festlegen, je Deckel ein kleines Rechteck ziehen, ein paar Bilder
labeln. Alle Schwellwerte werden daraus gelernt, im Code steht keine
einzige Stellschraube. Erkennung per HSV-Farbanalyse + Blob-Suche in
Millisekunden auf dem Raspberry Pi. Grenzen: nur bei Tageslicht
(IR-Nachtbilder werden erkannt und markiert), Tonnen müssen sich
farblich unterscheiden, und graue/schwarze Deckel sind farblich nicht
kalibrierbar (typisch: die Restmülltonne). Praktikabler Workaround:
ein kleiner farbiger Marker-Aufkleber auf dem Deckel, der
mitkalibriert wird. Status: Alpha, Phase 1 (Kern + CLI); die
HA-Anbindung (Config Flow, `binary_sensor` je Tonne) folgt in Phase 2.

## License

MIT, see [LICENSE](LICENSE).
