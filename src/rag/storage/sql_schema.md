# The datasheet database (`wahadb_clean.sqlite`)

A SQLite copy of Wahapedia's Warhammer 40,000 data: units (datasheets), their models, weapons, keywords and
abilities, and stratagems, detachments and enhancements. Text encoding is already repaired. Last update of the
data: 2026-09-01.

## Rules for writing SQL

- Every column is `TEXT`, numbers included: compare and sort numbers with `CAST(t AS INTEGER)`. Some stat values are
  not numbers (`D3`, `D6`, `N/A`, `Melee`, `-`).
- There are no declared keys: join on the columns listed under Joins, with their exact case (`Id` vs `id`,
  `FactionId` vs `factionId`, `DatasheetId`).
- `=` compares text exactly and is case-sensitive; `LIKE` ignores the case of ASCII letters. Keywords are written
  like `Infantry`, `Battleline`, `Grenades`; stratagem names in capitals, like `GRENADE`.
- One `SELECT` (or `WITH ... SELECT`) per call. Select the columns you need and filter; results are cut after 50 rows.
- Booleans are the text `'true'` or `'false'`.

## Tables

| Table | One row is | Main columns |
|---|---|---|
| `factions` | a faction (25) | `Id` (`SM` = Space Marines, `CSM` = Chaos Space Marines, `TAU`, `TYR`, …), `name` |
| `datasheets` | a unit's datasheet (1 680) | `Id`, `name`, `factionId`, `sourceId`, `role`, `loadout`, `transport`, `legend` (lore text, not Warhammer Legends), `virtual`, `link` |
| `datasheets_models` | a model profile of a datasheet (a unit can have several, one per `line`) | `DatasheetId`, `line`, `name`, `m` (Move), `t` (Toughness), `sv` (Save), `invSv` (invulnerable save, `-` if none), `w` (Wounds), `ld` (Leadership), `oc` (Objective Control), `baseSize` |
| `datasheets_wargear` | a weapon profile of a datasheet | `DatasheetId`, `name`, `range` (inches, or `Melee`), `type` (`Ranged`/`Melee`), `a` (Attacks), `bsWs` (BS or WS: `3` means 3+), `s` (Strength), `ap` (`0`, `-1`, …), `d` (Damage), `description` (weapon abilities) |
| `datasheets_keywords` | a keyword of a datasheet | `DatasheetId`, `keyword`, `model` (the models it applies to; empty = the unit), `isFactionKeyword` |
| `datasheets_abilities` | an ability on a datasheet | `DatasheetId`, `line`, `abilityId`, `name`, `description`, `type` (`Core`, `Faction`, `Datasheet`, `Wargear`, …), `parameter` |
| `abilities` | a Core or Faction ability's text | `Id`, `name`, `factionId`, `description` |
| `datasheets_unit_composition` | a line of unit composition | `DatasheetId`, `line`, `description` |
| `datasheets_options` | a wargear option | `DatasheetId`, `line`, `description` |
| `datasheets_models_cost` | a points cost | `DatasheetId`, `line`, `description` (unit size), `cost` (points) |
| `datasheets_leader` | a leader-to-unit link | `LeaderId` (the leader's datasheet), `attachedId` (a unit it can lead) |
| `stratagems` | a stratagem | `id`, `name`, `FactionId`, `type`, `cpCost`, `turn`, `phase`, `detachment`, `detachmentId`, `description` (lines `WHEN:`, `TARGET:`, `EFFECT:`, `RESTRICTIONS:`) |
| `datasheets_stratagems` | a datasheet-to-stratagem link | `DatasheetId`, `stratagemId` |
| `detachments` | a detachment | `Id`, `factionId`, `name`, `type` |
| `detachment_abilities` | a detachment rule | `Id`, `detachmentId`, `name`, `description` |
| `enhancements` | an enhancement | `id`, `FactionId`, `name`, `cost`, `detachmentId`, `description` |
| `source` | a publication | `Id`, `name`, `type`, `edition`, `version`, `errataDate` |

## Joins

- `datasheets.Id` = `DatasheetId` of every `datasheets_*` table; `datasheets.factionId` = `factions.Id`;
  `datasheets.sourceId` = `source.Id`.
- `datasheets_abilities.abilityId` = `abilities.Id`: see the abilities trap below.
- `datasheets_stratagems.stratagemId` = `stratagems.id`; `stratagems.FactionId` = `factions.Id`.
- `detachments.Id` = `detachment_abilities.detachmentId` = `stratagems.detachmentId` = `enhancements.detachmentId`.
- `datasheets_leader.LeaderId` and `.attachedId` = `datasheets.Id`.

## Traps

- **Unit names are not unique** (1 680 datasheets, 1 291 names): the same name can exist in several factions or
  sources. Filter by `factionId` when it matters, and say which one you used.
- **A broad name is a category, not a unit.** "A Space Marine" is not a datasheet name: `name LIKE '%Space Marine%'`
  finds only odd units. Use the faction (`factionId = 'SM'`) and keywords from `datasheets_keywords` (`Battleline`,
  `Infantry`, `Vehicle`, …), then give the typical value and the notable exceptions (example 2).
- **A characteristic belongs to a model profile, not to a unit.** A datasheet can have profiles of different values
  (a sergeant and a heavier veteran, for example): keep each profile's name in a category result, so that a unit
  that is an exception for one profile is not given that value as a whole (example 2).
- **Warhammer Legends and Forge World.** A datasheet's source tells where it comes from: `source.name` ends with
  `(Warhammer Legends)` for Legends units (retired from standard play) and with `(Forge World)` for Forge World
  units. Exclude Legends unless the question asks for them: `source.name NOT LIKE '%(Warhammer Legends)%'`. Do not
  use `datasheets.legend` for this: it is lore text.
- **Editions are per record.** `source.edition` is `'11'`, `'10'`, or `'0'` (no edition recorded: Legends and Forge
  World packs). Two datasheets have no source (`sourceId` empty). `stratagems`, `detachments`, `enhancements` and
  `abilities` have no source column: their edition is not recorded. The Core Rules rulebook is 10th edition.
- **Abilities are in two places.** Core and Faction abilities have an `abilityId` and their text in `abilities`; the
  others (`Datasheet`, `Wargear`, …) have an empty `abilityId` and their text in `datasheets_abilities.description`.
  `abilities` repeats each ability once per faction (91 rows, 71 `Id`s), so join it through
  `(SELECT DISTINCT Id, name, description FROM abilities)`, never on `Id` alone (example 4).
- **Core stratagems** have an empty `FactionId` and come in two forms. Rows of `type` `Core Stratagem` are linked to
  datasheets in `datasheets_stratagems`; rows whose `type` reads like the rulebook's labels (`Core – Wargear
  Stratagem`, `Core – Strategic Ploy Stratagem`, …) are linked to no datasheet. A name can have both forms
  (`COMMAND RE-ROLL`) or only one (`GRENADE` exists only as `Core – Wargear Stratagem`), and their texts can differ;
  their source is not recorded. Which units can use a core stratagem is written in its `TARGET:` line (for
  `GRENADE`, a unit with the `Grenades` keyword). Other rows with an empty `FactionId` belong to other game modes
  (`Boarding Actions`, `Challenger`) or describe core actions (`Movement Ability`, `Shooting Ability`).
- **Game modes of detachments.** `detachments.type` is empty for the standard game and `Boarding Actions` for that
  game mode (59 of 327 detachments); a faction's stratagems include those of its Boarding Actions detachments (for
  Space Marines, `Boarding Strike`). For a question about the standard game, leave those out. A stratagem whose
  `detachmentId` matches no row of `detachments` (six Aeldari `Army Rules` stratagems) has an unknown game mode,
  not the standard one.
- **Rows can repeat.** `datasheets_keywords` holds 16 358 rows but 12 172 different ones, and a datasheet's points
  rows can appear twice, between header rows such as `YOUR UNIT COSTS` with an empty `cost`. Use `DISTINCT` when you
  list or count.
- **Model profiles.** A datasheet can have several rows in `datasheets_models` (one per kind of model, by `line`);
  keep all profiles of a unit together when they differ. Filter units by their keywords or an `EXISTS` condition,
  then return all profiles of those units; filtering model rows to one characteristic hides mixed-profile units.
- Some `datasheets_abilities.type` values are leftovers of the export in Russian (for example
  `Special (правая колонка)`); treat them as other datasheet abilities. The datasheet `Example Wargear` is a
  placeholder (`virtual = 'true'`).

## Examples

All examples run as written; replace the names and IDs.

1. A unit's model profiles, with its faction and source:

```sql
SELECT d.name, f.name AS faction, m.name AS model, m.m, m.t, m.sv, m.invSv, m.w, m.ld, m.oc,
       s.name AS source, s.edition
FROM datasheets d
JOIN factions f ON f.Id = d.factionId
JOIN datasheets_models m ON m.DatasheetId = d.Id
LEFT JOIN source s ON s.Id = d.sourceId
WHERE d.name = 'Intercessor Squad' AND d.factionId = 'SM'
ORDER BY CAST(m.line AS INTEGER);
```

2. A category: Space Marine Battleline units, Legends excluded. Use one query for representative units across
   the observed Toughness values. It ranks units with mixed profiles first, chooses at most two representatives
   per value, then returns every profile of each chosen unit. This is a sample even if `truncated` is false.
   Keep `source` and `edition` as the publication column names:

```sql
WITH category_profiles AS (
  SELECT DISTINCT d.Id, d.name, s.name AS source, s.edition,
                  m.line, m.name AS model, m.t
  FROM datasheets d
  JOIN datasheets_models m ON m.DatasheetId = d.Id
  LEFT JOIN source s ON s.Id = d.sourceId
  WHERE d.factionId = 'SM'
    AND d.Id IN (SELECT DatasheetId FROM datasheets_keywords WHERE keyword = 'Battleline')
    AND (s.name IS NULL OR s.name NOT LIKE '%(Warhammer Legends)%')
),
unit_variants AS (
  SELECT Id, COUNT(DISTINCT t) AS toughness_values
  FROM category_profiles
  GROUP BY Id
),
ranked AS (
  SELECT p.Id, p.name, p.source, p.edition, p.t,
         ROW_NUMBER() OVER (
           PARTITION BY p.t
           ORDER BY v.toughness_values DESC, p.name, p.Id
         ) AS representative
  FROM (
    SELECT DISTINCT Id, name, source, edition, t FROM category_profiles
  ) p
  JOIN unit_variants v ON v.Id = p.Id
),
selected AS (
  SELECT DISTINCT Id, name, source, edition
  FROM ranked
  WHERE representative <= 2
)
SELECT s.Id, s.name, s.source, s.edition,
       GROUP_CONCAT(p.model || ': T' || p.t, '; ') AS profiles
FROM selected s
JOIN category_profiles p ON p.Id = s.Id
GROUP BY s.Id, s.name, s.source, s.edition
ORDER BY s.name, s.Id;
```

For an overall distribution, query the complete category separately, grouping by publication and edition.
`model_profiles` counts distinct datasheet/model-line profiles, not individual miniatures; `datasheets` counts
distinct units having that Toughness. A mixed-profile unit can appear in multiple Toughness groups, so do not
sum these unit counts to obtain a total. No LIMIT is applied before this aggregation. A truncated aggregate
result is still incomplete; narrow its publication/category scope before making overall claims:

```sql
WITH profiles AS (
  SELECT DISTINCT d.Id, m.line, m.t, s.Id AS source_id, s.name AS source, s.edition
  FROM datasheets d
  JOIN datasheets_models m ON m.DatasheetId = d.Id
  LEFT JOIN source s ON s.Id = d.sourceId
  WHERE d.factionId = 'SM'
    AND d.Id IN (SELECT DatasheetId FROM datasheets_keywords WHERE keyword = 'Battleline')
    AND (s.name IS NULL OR s.name NOT LIKE '%(Warhammer Legends)%')
)
SELECT source_id, source, edition, t, COUNT(*) AS model_profiles, COUNT(DISTINCT Id) AS datasheets
FROM profiles
GROUP BY source_id, source, edition, t
ORDER BY source, edition, CAST(t AS INTEGER), t;
```

3. A unit's weapons:

```sql
SELECT w.name, w.range, w.type, w.a, w.bsWs, w.s, w.ap, w.d, w.description
FROM datasheets_wargear w
WHERE w.DatasheetId = '000001157'  -- Intercessor Squad (SM)
ORDER BY CAST(w.line AS INTEGER), CAST(w.lineInWargear AS INTEGER);
```

4. A unit's abilities, from both places, one row per ability:

```sql
SELECT da.type,
       COALESCE(a.name, da.name)               AS ability_name,
       COALESCE(a.description, da.description) AS ability_text
FROM datasheets_abilities da
LEFT JOIN (SELECT DISTINCT Id, name, description FROM abilities) a ON a.Id = da.abilityId
WHERE da.DatasheetId = '000000070'  -- Tactical Squad (SM)
ORDER BY CAST(da.line AS INTEGER);
```

5. A core stratagem's text, in every form, and the units that have the keyword its target needs:

```sql
SELECT name, type, cpCost, turn, phase, description
FROM stratagems
WHERE name = 'GRENADE' AND FactionId = '';
```

```sql
SELECT DISTINCT d.name
FROM datasheets d
JOIN datasheets_keywords k ON k.DatasheetId = d.Id AND k.keyword = 'Grenades'
WHERE d.factionId = 'SM'
ORDER BY d.name;
```

6. The stratagems of a unit's own faction, with their detachment and game mode (`game_mode`: empty = the standard
   game, `Boarding Actions`, or `NULL` when the detachment is not in `detachments`):

```sql
SELECT s.name, s.FactionId AS faction, s.type, s.detachment, dt.type AS game_mode, s.cpCost, s.phase
FROM datasheets d
JOIN datasheets_stratagems ds ON ds.DatasheetId = d.Id
JOIN stratagems s ON s.id = ds.stratagemId AND s.FactionId = d.factionId
LEFT JOIN detachments dt ON dt.Id = s.detachmentId
WHERE d.Id = '000001157'  -- Intercessor Squad (SM)
ORDER BY dt.type, s.detachment, s.name;
```

7. A unit's keywords:

```sql
SELECT DISTINCT keyword, model, isFactionKeyword
FROM datasheets_keywords
WHERE DatasheetId = '000000070'  -- Tactical Squad (SM)
ORDER BY isFactionKeyword DESC, keyword;
```

8. A faction's detachments, with their rules:

```sql
SELECT dt.name AS detachment, da.name AS rule, da.description
FROM detachments dt
LEFT JOIN detachment_abilities da ON da.detachmentId = dt.Id
WHERE dt.factionId = 'SM'
ORDER BY dt.name;
```

9. The units a leader can lead:

```sql
SELECT attached.name
FROM datasheets leader
JOIN datasheets_leader l ON l.LeaderId = leader.Id
JOIN datasheets attached ON attached.Id = l.attachedId
WHERE leader.Id = '000000073'  -- Captain (SM)
ORDER BY attached.name;
```

10. A unit's points:

```sql
SELECT DISTINCT description AS unit_size, cost
FROM datasheets_models_cost
WHERE DatasheetId = '000001157' AND cost != ''  -- Intercessor Squad (SM)
ORDER BY CAST(cost AS INTEGER);
```
