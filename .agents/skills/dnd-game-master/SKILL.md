---
name: dnd-game-master
description: "AI Dungeon Master patterns adapted from Sstobo/Claude-Code-Game-Master: RAG-based source material ingestion, entity schemas (NPC/Location/Plot/Consequence), campaign state management, session memory, dice mechanics, and D&D 5e character creation. Use when building or extending AI GM systems, grounding game state in imported rulebook content, or designing persistent session memory for tabletop RPGs."
risk: low
source: local
date_added: "2026-03-24"
attribution: "Adapted from https://github.com/Sstobo/Claude-Code-Game-Master (CC BY-NC-SA 4.0, Sean Stobo)"
---

# DnD Game Master Patterns

Patterns and schemas from a production AI DM system. Use these when building
GM state management, session memory, or RAG-grounded gameplay on top of
extracted RPG mechanics.

---

## Use this skill when

- Designing campaign or session state persistence for an AI GM
- Building entity managers (NPC, Location, Plot, Consequence) in any RPG system
- Grounding AI GM responses in imported source material via RAG
- Implementing D&D 5e (or analogous) character creation stat calculations
- Adding dice roll mechanics with advantage/disadvantage
- Designing the world-state JSON schema for UrsaMU's AI GM bridge

## Do not use this skill when

- You only need chargen (use `/pdf-to-chargen` or `/ursamu-chargen`)
- The system has no persistent session state
- Working in a purely stateless context (single-shot Q&A, no campaign)

---

## RAG-based source material ingestion

The reference system vectorizes PDFs/DOCX/TXT into a per-campaign ChromaDB store,
then retrieves relevant passages at query time to ground GM responses.

### Extraction pipeline (5 steps)

```
Prepare → Extract → Merge → Review → Save
```

1. **Prepare** — validate file format, create campaign vectors directory
2. **Extract** — chunk text semantically, embed via `sentence-transformers`
3. **Merge** — deduplicate chunks against existing store, preserve metadata
4. **Review** — show extracted entity counts per category before committing
5. **Save** — persist ChromaDB store to `world-state/campaigns/<name>/vectors/`

### Semantic chunking categories

| Category | What it captures |
|----------|-----------------|
| `npcs` | Named characters, attitudes, dialogue examples |
| `locations` | Place descriptions, connections, hazards |
| `items` | Equipment, rarity, mechanics, attunement |
| `plot_hooks` | Objectives, rewards, consequences, level range |
| `monsters` | CR, tactics, treasure, conditions |
| `traps` | Trigger, effect, detection/disable DCs |
| `factions` | Leader, members, allies, enemies |

### RAG query pattern

```python
# Retrieve passages relevant to an entity or scene
passages = vector_store.query(
    query_text=entity_description,
    category_filter=category,       # optional: scope to one category
    n_results=5,
    distance_threshold=0.4,         # cosine distance; lower = stricter
)

# Apply retrieved context to enrich entity
enhanced = entity_enhancer.apply(entity, passages)
```

### Applying to pdf-to-chargen

The `langchain-rag` skill + this pattern solves the 150k char ceiling:
instead of feeding the full text to each extraction node, embed the rulebook
at import time and retrieve targeted passages per extraction category. The
`extractionPlan` from `analyze_structure` becomes the query text.

```typescript
// Conceptual — replace textContent slice with RAG retrieval
const passages = await vectorStore.query(extractionPlan.stats, { n: 20 });
const result   = await llm(EXTRACT_STATS_SYSTEM, passages.join("\n"));
```

---

## Entity schemas

Use these as the canonical shape for world-state entities. They map directly
to the extraction categories in the pdf-to-chargen pipeline.

### NPC

```json
{
  "name": "string",
  "description": "string",
  "attitude": "ally | enemy | neutral",
  "location_tags": ["string"],
  "stats": {
    "hp": 0, "max_hp": 0, "ac": 0, "xp": 0, "level": 1,
    "str": 10, "dex": 10, "con": 10, "int": 10, "wis": 10, "cha": 10
  },
  "conditions": [],
  "inventory": [],
  "dialogue_examples": ["string"],
  "source": "string",
  "is_party_member": false,
  "history": []
}
```

### Location

```json
{
  "name": "string",
  "description": "string",
  "connections": { "north": "string", "south": "string" },
  "features": ["string"],
  "inhabitants": ["npc_name"],
  "hazards": ["string"],
  "tags": ["string"]
}
```

### Plot / Quest

```json
{
  "title": "string",
  "type": "main | side | personal | world",
  "status": "active | completed | failed | dormant",
  "objectives": [{ "text": "string", "completed": false }],
  "stakeholders": ["npc_name"],
  "locations": ["location_name"],
  "rewards": "string",
  "level_range": [1, 5],
  "staleness_turns": 0
}
```

### Consequence (future event)

```json
{
  "description": "string",
  "trigger_condition": "string",
  "status": "active | resolved | cancelled",
  "created_session": 1,
  "resolution": null
}
```

### Campaign overview

```json
{
  "name": "string",
  "genre": "string",
  "current_location": "string",
  "campaign_date": "string",
  "campaign_time": "string",
  "session_count": 0,
  "active": true
}
```

---

## Session state management

### World-state directory layout

```
world-state/campaigns/<name>/
├── campaign-overview.json   — metadata, current location, date, session count
├── npcs.json                — all NPCs keyed by name
├── locations.json           — all locations keyed by name
├── plots.json               — quests keyed by title
├── consequences.json        — future events array
├── items.json               — equipment/treasures
├── facts.json               — categorized world info (world_building, session_events, etc.)
├── character.json           — active player character
├── session-log.md           — markdown session history
└── vectors/                 — ChromaDB vector store for RAG
```

### Atomic JSON write pattern

Always write via temp file → rename to prevent corruption:

```python
import json, os, tempfile

def atomic_write(path: str, data: dict) -> None:
    dir_ = os.path.dirname(path)
    with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        tmp = f.name
    os.replace(tmp, path)   # atomic on POSIX
```

### Session context aggregation

At the start of every GM turn, build a full context block:

```
Active campaign: <name>  |  Location: <current>  |  Session: <n>  |  Date: <in-game>
Party: <members with HP>
Active plots: <titles>
Pending consequences: <descriptions>
Recent facts: <last 5 session_events>
```

This block is prepended to every LLM call — equivalent to the UrsaMU
`formatCharacterContext()` field in `IGameSystem`.

---

## D&D 5e character creation formulas

### Ability score → modifier

```python
def modifier(score: int) -> int:
    return (score - 10) // 2
```

### HP calculation

```python
def starting_hp(hit_die: int, con_score: int, level: int = 1) -> int:
    con_mod = modifier(con_score)
    return hit_die + con_mod + (level - 1) * ((hit_die // 2 + 1) + con_mod)
```

### XP thresholds (levels 1–20)

```python
XP_THRESHOLDS = [
    0, 300, 900, 2700, 6500, 14000, 23000, 34000, 48000, 64000,
    85000, 100000, 120000, 140000, 165000, 195000, 225000, 265000,
    305000, 355000
]

def level_for_xp(xp: int) -> int:
    for lvl, threshold in reversed(list(enumerate(XP_THRESHOLDS, 1))):
        if xp >= threshold:
            return lvl
    return 1
```

### HP status thresholds

| Status | Condition |
|--------|-----------|
| `HEALTHY` | hp > max_hp * 0.5 |
| `BLOODIED` | hp <= max_hp * 0.25 |
| `UNCONSCIOUS` | hp <= 0 |

---

## Dice mechanics

### Notation parser

Standard RPG dice notation: `NdX+M`  (e.g. `3d6+2`, `1d20`, `2d20kh1`)

```python
import re, random

def roll(notation: str) -> dict:
    m = re.match(r"(\d+)d(\d+)([kK][hHlL]\d+)?([-+]\d+)?", notation.strip())
    n, sides = int(m[1]), int(m[2])
    rolls = [random.randint(1, sides) for _ in range(n)]

    keep = m[3]
    if keep:
        k = int(keep[2:])
        rolls = sorted(rolls, reverse=keep[1].lower() == "h")[:k]

    modifier = int(m[4]) if m[4] else 0
    total = sum(rolls) + modifier

    return {
        "notation": notation,
        "rolls": rolls,
        "modifier": modifier,
        "total": total,
        "natural_20": sides == 20 and n == 1 and rolls[0] == 20,
        "natural_1":  sides == 20 and n == 1 and rolls[0] == 1,
    }
```

### Advantage / disadvantage

```python
advantage    = "2d20kh1"   # keep highest
disadvantage = "2d20kl1"   # keep lowest
```

---

## Applying these patterns to UrsaMU AI GM bridge

The `IGameSystem.formatCharacterContext(sheet)` method should return a
compact version of the session context block above — no MUSH codes,
suitable for LLM injection:

```typescript
formatCharacterContext: (sheet) => [
  `System: ${gameSystem.name}`,
  `Character: ${sheet.name ?? "Unknown"}`,
  `Stats: ${gameSystem.stats.map(s => `${s}=${sheet[s] ?? 0}`).join(", ")}`,
  `Status: HP ${sheet.hp ?? "?"}/${sheet.maxHp ?? "?"}`,
].join("\n"),
```

The `coreRulesPrompt` field is the ≤150-word distillation of the same
source material this system would ingest via RAG — extracted by the
`extract_gm_context` node in pdf-to-chargen.

---

## References

| Resource | Why |
|----------|-----|
| https://github.com/Sstobo/Claude-Code-Game-Master | Original source (CC BY-NC-SA 4.0) |
| `langchain-rag` skill | RAG implementation for the vector store approach |
| `langgraph-persistence` skill | Checkpointer patterns analogous to session save/restore |
| `deep-agents-memory` skill | Long-term memory patterns for campaign state |
| `/pdf-to-chargen` skill | Extracts `coreRulesPrompt` and `moveThresholds` from source PDFs |
