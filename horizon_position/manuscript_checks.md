# Manuscript checks — things to fix in the RASTI tex when the blocks land

**Temporary.** This file exists so findings about the manuscript survive
between sessions while the blocks in `paper_text.tex.in` are still being
written. Delete it once the final sweep is done and the fixes are in the paper
repo.

The live manuscript is `eigsep_instrument_rasti/rasti_template.tex` in
`~/Documents/research/papers/eigsep_instrument/`. **Not** `revised_rasti/` —
that is a latexdiff artefact full of `\DIFadd`/`\DIFdel` markup, and editing it
by mistake is easy because the filename is identical.

Line numbers below are as of 2026-09-01 and will drift as soon as a block is
inserted. Anchor on the quoted text, not the number.

---

## 1. Two sentences contradict block 4

Block 4 argues that an unmodelled 1 m vertical error leaves a residual the
foreground filter does *not* remove — 3.52 mK against a 1.16 mK median retained
signal, and past `N_HAND` the position error is the term limiting the residual.
Two existing sentences say the opposite and must change regardless of anything
else we do.

**`rasti_template.tex:398`**, in `subsec:box`, immediately before the lidar
sentence:

> As shown in Section~\ref{subsec:fwd_modelling}, the resulting structure is
> foreground-like and is suppressed by the same eigenmode filter applied to the
> foregrounds.

**`rasti_template.tex:662`**, the closing paragraph of `subsec:fwd_modelling`:

> Using six modes, the coupling is suppressed below 10\,mK in each case, showing
> that the residual structure caused by the horizon coupling lives in the same
> low-dimensional subspace as the unperturbed antenna temperature. While
> metre-scale shifts can produce Kelvin-level changes in antenna temperature,
> these changes are foreground-like and removed by the standard foreground
> filter.

Block 4 inserts after line 660, so 662 currently lands two paragraphs later and
withdraws what block 4 has just argued. **No block in the template replaces
662** — block 5 replaces line 638 only. That is a gap in the block set, not
something a later edit introduces.

To be fair to 662: "suppressed below 10 mK" is not false, it answers a weaker
question than block 4 asks. The clause that has to go is "removed by the
standard foreground filter".

## 2. `Section~\ref` capitalisation

Lowercase `section~\ref` appears 21 times; capitalised `Section~\ref` appears
exactly once — at line 398, the same sentence flagged above. Lowercase is the
house style here, so fix it when 398 is rewritten. Check any new block text for
the same slip before pasting.

## 3. British spelling and grammar

The manuscript should be British throughout and mostly is — `characterise`,
`normalised`, `minimise`, `polarisation`, `optimised`, `digitised` are all
already in the `-ise` form, and an earlier revision changed "Minimizing" to
"Minimising" in a section heading.

**Not yet swept.** A full pass is still to do, over both the manuscript and any
block text we paste in. Worth checking: `-ise`/`-isation` vs `-ize`, `-yse` vs
`-yze`, `-our`/`-re` endings, doubled consonants (`modelling`, `travelling`,
`labelling`), `artefact`, `towards` vs `toward`, `metre` vs `meter`, `grey`,
`aluminium`, `defence`.

Spotted in passing while looking at something else, so treat as a seed and not
as results: `analyzer` (×5, though 3 are inside a Keysight product URL and
should stay), `traveling` (×1), `artifact` (×2), and bare `toward` (×3) against
`towards` (×1) — the last is a consistency question as much as a spelling one.
Exclude LaTeX false positives when scanning: `\centering`, `xcolor`,
`\textcolor`, "Interferometers"/"parameters" for `meter`, and URLs.

---

## When this is done

Fold anything durable into `horizon_position/CLAUDE.md` — the framing
constraints there are the ones that must outlive the draft — and delete this
file.
