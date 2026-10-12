# AdVids — Synchronization Master Plan
**Version 1.3**
**Created: September 17, 2026**
**Last Updated: October 12, 2026**
**Status: Active — In Development**

---

## Change Log
| Version | Date | Change |
|---|---|---|
| 1.0 | Sep 17, 2026 | Initial plan created |
| 1.1 | Sep 18, 2026 | Added Product Principles. Algorithm fixes: exclude bar 1, fix downbeat scoring, strengthen climax weight, fix 2-photo cut logic |
| 1.2 | Sep 27, 2026 | Added Long-term Vision (AI training). Added Engineering Principles. Added Folder Structure. Added known bugs from visual analysis. Updated implementation order. |
| 1.3 | Oct 11, 2026 | Bug 1 fixed (accent_map.primary). Bug 2 fixed (duplicate secondary). Bug 3 partial fix (onset threshold filter). Git + GitHub setup complete. Folder structure reorganized. Syntax check rule added to Engineering Principles. Secondary points redesign planned (combined approach). analyze2.py rewritten in correct logical order (v1.2). |

---

## Product Vision
A synchronization algorithm that takes a pre-selected music track + 2-8 user photos and produces a perfectly synchronized short-form video ad. Simple to use, zero effort from user, professional quality output. Feels like operations. Delivers like art.

---

## Long-Term Vision — AI Music Generation

### The Flywheel

Phase 1 (Now):
Curated library + sync algorithm → working product

Phase 2 (V2):
Larger library + refined algorithm + user preference data

Phase 3 (V3):
Fine-tune MusicGen using our algorithm as reward signal

user preference data as additional training signal

Phase 4 (V4):
Fully custom model — precise millisecond-level timing control
“Generate energetic track, 15s, hits at 3.2s, 7.8s, 12.1s”


### The Competitive Moat
Our algorithm + user preference data = proprietary training dataset nobody else has.
Algorithm trains model → model improves product → product generates more training data.

### Technical Path
- Base model: Meta's MusicGen (open source, fine-tunable)
- Reward signal: our sync algorithm's quality score
- Training approach: RLHF
- Minimum dataset: ~10,000 tracks + human preference labels
- Realistic timeline: V3 after ~10,000 active users

---

## Current MVP Scope
- One pre-selected track only (energetic_01.mp3)
- 2-8 user photos
- Vertical format 9:16
- Prove sync works perfectly first
- No UI polish, no effects yet — just correct sync
- Then add visual quality on top

---

## What Is NOT In Scope Yet
- Music generation or AI selection
- UI/UX beyond basic function
- Payments and business model
- Multiple tracks
- Visual effects and filters (after sync is proven)
- Silent-first design (keep in mind, not priority)
- Mobile app

---

## Core Sync Principles
- Hard cuts only — no crossfades
- New photo appears ON the bar downbeat
- Effects happen within the held photo between cuts
- Visual pattern follows the audio peaks of the track
- Over-syncing is a failure mode — not every beat needs a cut
- First 3 seconds must hook immediately
- Relative scoring — everything scored relative to track's own range
- Subdivision points only generated where actual audio content exists
- Secondary points should align with real RMS/onset peaks between downbeats
- Cut timing to be adjusted for attack time offset (Bug 4 — pending)

---

## Reference Video Analysis
- Format: 464×848px vertical, 30fps, ~10 seconds
- Time signature: 2/4
- Pattern per bar:
  - Beat 1: NEW photo + 4 sixteenth note effects (B&W + graphic overlay)
  - Beat 2: SAME photo + cinematic filter, no overlay
- New photo appears ON beat 1 of next bar
- Only 4 photos for entire video
- Key lesson: read the rhythmic pattern of the track and follow it

---

## The 4-Layer Analysis System

### Layer 1 — Bar Structure
1. Time signature detection
2. Bar boundaries
3. Internal subdivisions — ONLY where actual audio content exists (onset >= threshold)
4. Peak strength at each subdivision
5. Rhythmic pattern within bars

### Layer 2 — Track Structure
- Energy curve across full track
- Climax point detection
- Energy rise and fall around climax
- Instrument separation and combination analysis

### Layer 3 — Accent Points
- PRIMARY: photo cut candidates (downbeats only)
- SECONDARY: effect candidates — to be placed at real RMS/onset peaks between downbeats (combined approach — pending)

### Layer 4 — Prioritized Accent Map
- Universal ranked list — purely musical
- Independent of photo count
- Generated once per track

---

## Instrument Hierarchy
**Tier 1 — Rhythmic Foundation:** Drums/percussion, Bass
**Tier 2 — Harmonic Structure:** Guitar, Piano, Keys
**Tier 3 — Melodic/Ornamental:** Trumpet, Strings, Vocals, Synth leads

---

## Relative Scoring System

### Accent Point Score Formula

Score =
onset_strength × 0.35
energy × 0.20
bar_position_weight × 0.25
climax_bonus × 0.20
(all normalized 0-100 relative to track’s own range)


### Scoring Rules (v1.2)
- Bar 1 excluded from all cut candidates
- Primary candidates = downbeats ONLY
- Downbeat weight = 100 only when onset > 40 OR near climax
- Plain weak-onset downbeat = max 40
- Climax proximity bonus = up to 50 points (active within 30% of track)
- Subdivision points only where onset >= MIN_ONSET_THRESHOLD (3.0)

---

## Secondary Points — Combined Approach (Planned)
Current approach (mathematical positions) to be replaced with:

Between each pair of downbeats:

Find mathematical beat positions (eighth/sixteenth)
Find actual RMS/onset peaks
Where both coincide (within attack time tolerance)
→ HIGH confidence secondary point
Where only beat position exists
→ MEDIUM confidence
Where only RMS peak exists
→ LOW confidence

This ensures effects are placed where music is actually doing something interesting — not at arbitrary mathematical positions.

---

## Known Bugs — Status

| Bug | Impact | Status |
|---|---|---|
| accent_map.primary saved wrong data | Visualizer | ✅ Fixed v1.2 |
| Secondary candidates duplicated primary | JSON data | ✅ Fixed v1.2 |
| Subdivisions generated regardless of audio | Effect quality | ✅ Partial fix v1.2 (onset threshold). Combined approach pending |
| Attack time offset (~115ms at climax) | Direct sync | ⏳ Pending — Options B/C to be tested |

### Attack Time Options (to be tested)
- Option A: Cut at bar boundary (15.25s) — mathematical
- Option B: Cut at RMS peak (~15.40s) — perceptual
- Option C: Cut between (~15.30-15.35s) — compromise

---

## Synchronization Plans (per photo count)

### Rules
- Always include: climax downbeat
- Always exclude: bar 1
- Phrase boundaries (every 4 bars) preferred
- Minimum musical gap between cuts
- Distribution follows musical energy
- No cuts in last 15% of track

### Plans (energetic_01)
| Photos | Cuts | Cut Points |
|---|---|---|
| 2 | 1 | 15.25s |
| 3 | 2 | 6.01, 15.25s |
| 4 | 3 | 6.01, 11.55, 15.25s |
| 5 | 4 | 2.32, 6.01, 11.55, 15.25s |
| 6 | 5 | 2.32, 6.01, 11.55, 15.25, 18.92s |
| 7 | 6 | 2.32, 6.01, 11.55, 13.4, 15.25, 17.08s |
| 8 | 7 | 2.32, 6.01, 11.55, 13.4, 15.25, 17.08, 18.92s |

---

## Technical Stack
| Tool | Version | Purpose |
|---|---|---|
| Python | 3.9 | Core language |
| Librosa | 0.11.0 | Onset strength, energy curve |
| Madmom | 0.16.1 | Bar detection, downbeat tracking |
| NumPy | 1.23.5 | Madmom compatibility |
| FFmpeg | latest | Video assembly |
| Git/GitHub | — | Version control (repo: grakovrs/project-A) |

---

## Development Environment

### Mac Mini (primary)
- All tools installed and working
- Project at: `~/Documents/advids/`
- GitHub: `https://github.com/grakovrs/project-A`

### Lenovo ThinkPad T490 (secondary)
- Windows 11 Pro
- All tools installed and working
- Project at: `C:\Users\P\Documents\advids\`

### Git Workflow

Before starting work: git pull
After finishing work: git add .
git commit -m “description”
git push
Never work on two machines simultaneously without pulling first


---

## Folder Structure

advids/
├── analysis/ ← generated JSON analysis files
├── output/ ← final video outputs (empty until assembler built)
├── photos/ ← test photos (8 photos)
├── scripts/ ← analyze2.py, verify_beats.py, visualize.html
├── temp/ ← temporary processing files (auto-cleared)
├── tests/ ← test scripts + test output videos
│ └── output/ ← test_2photos.mp4 through test_8photos.mp4
└── tracks/ ← 9 MP3 files (3 per style)


---

## Implementation Order
1. ~~Install Madmom~~ ✅
2. ~~Build 4-layer analysis script~~ ✅
3. ~~Build visual timeline analyzer~~ ✅
4. ~~Fix accent_map.primary save bug~~ ✅
5. ~~Fix duplicate secondary candidates~~ ✅
6. ~~Reorganize folder structure~~ ✅
7. ~~Set up Git + GitHub~~ ✅
8. ~~Onset threshold filter for subdivisions~~ ✅
9. ~~Rewrite analyze2.py in correct logical order~~ ✅
10. ~~Set up Lenovo ThinkPad T490~~ ✅
11. Bug 3 — Combined approach for secondary points
12. Bug 4 — Attack time offset (Options B/C)
13. Phrase boundary preference in sync plan
14. Pause-before-hit scoring bonus
15. Build video assembler on proven sync plans
16. Regular code audit

---

## Success Criteria
Cut lands within **50ms** of target perceptually.
Above 80ms feels slightly off. Above 150ms feels broken.

---

## Music Library
**MVP:** Pixabay (free, royalty-free)
**Test track:** energetic_01.mp3
**V2:** Epidemic Sound (~$15/mo)
**V3:** AI-generated via fine-tuned MusicGen

---

## Engineering Principles

**Security & Protection**
- All core algorithms are Trade Secrets from day one
- Code structured and access-controlled accordingly

**Code Quality**
- World-class professional precision
- Simple, clean, effective — no unnecessary complexity
- Hierarchical step numbering (1.1, 1.1.1) for complex scripts
- Every function does one thing well

**Syntax Check Rule**
After every manual file edit — before running anything:

python3 -m py_compile <filepath> && echo “Syntax OK”

Catches indentation errors, missing colons, unclosed brackets instantly.

**Performance**
- Light and effective — no redundant computation
- Temp files cleared after use (with approval)

**Safety**
- No files deleted without explicit owner approval
- All destructive operations require confirmation
- Data integrity maintained at all times

**Housekeeping**
- Folder structure maintained as defined
- Temp files in /temp only, test files in /tests only, scripts in /scripts only
- Regular audits mandatory

**Audit Schedule**
- After every major feature addition
- Before any version release
- Whenever a bug is discovered
- Covers: correctness, security, performance, folder structure

---

## Product Principles
- Target: self-employed and micro/small businesses
- Use case: instant professional video ads from phone photos
- Feels like operations. Delivers like art.
- First 3 seconds rule: most striking photo appears first
- 3 output videos per session — user picks favorite
- Trade Secrets: core technology protected from day one
- AI training pipeline: algorithm becomes reward signal for future model