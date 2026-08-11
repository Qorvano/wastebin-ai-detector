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
2. You draw one small rectangle on each lid in a few snapshots and
   label a handful of images ("blue bin present / absent").
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
  apart by a color detector. That is inherent, not a bug.
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
