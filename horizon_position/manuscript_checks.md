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

**Not an error: "none the less" as three words.** That is the journal's house
style, confirmed by the author 2026-09-01. Do not "correct" it to
"nonetheless".

---
## 4. The Vivaldi sentence points at a label that does not exist

Block 1's Vivaldi sentence, as pasted, reads `section~\ref{subsec:deployments}`.
**There is no such label** — it compiles to `??`. The Deployments section is
`\label{sec:deployments}` (`:750`); the subsection that actually holds the fact,
`\subsection{Field Measurements}` (`:791`), is unlabelled.

Task A5 in the rasti repo's `docs/revision-round2-plan.md` renames that
subsection to `Deployment History`. Add `\label{subsec:deployment_history}` in
the same edit and point the Vivaldi sentence there.

**Use the date, not an ordinal.** The Vivaldi flew in **October 2024**. That is
the *second suspension* but **deployment 3** in the numbering
`revision-round2-plan.md` uses (1 = Oct 2023, 2 = Jul 2024, 3 = Oct 2024,
4 = Jul 2025). "The second EIGSEP suspension" and "deployment 2" would name
different events in the same paper. §6.5 already refers to deployments by date;
do the same here.

Related: the draft in `paper_text.tex.in` said the Vivaldi flew on the *first*
suspension. That was wrong — `:792` lists July 2024 (prototype platform),
October 2024 (Vivaldi), July 2025 (bowtie) — and has been corrected in the
template and in the author's text. Nothing to do in the manuscript beyond the
above; recorded so the wrong fact is not re-derived.

## 5. Khalichi et al. is now printed twice

The author's block 1 moves the Khalichi deferral out of the caveat paragraph and
into paragraph 2, where it reads "a broader eigenmode-based analysis of beam
chromaticity, foreground complexity, and overlap with global 21-cm signal
models". The identical phrase already stands at **`rasti_template.tex:354`**, in
`subsec:eig_antenna`:

> A detailed eigenmode-based analysis of beam chromaticity, foreground
> complexity, and overlap with global 21-cm signal models will be described in
> Khalichi et al. 2026 (in prep.).

`subsec:covariance` comes first, so `:354` is the one to shrink:

> The final geometry was selected based on the spectral smoothness of the
> antenna reflection coefficient and the frequency evolution of its radiation
> pattern over the EIGSEP operating band (section~\ref{subsec:covariance}),
> following the analysis of Khalichi et al. 2026 (in prep.).

**Verify:** the twelve-word phrase appears exactly once in the compiled paper.
`grep -c "eigenmode-based analysis of beam chromaticity"` should return 1.

## 6. Trough depths are sky brightness temperature — verify the sentence survived

Block 1's units paragraph mixes two reference planes: `MED`, `SEP0` and `SEP1`
are **antenna** temperature (the ensemble is attenuated by eta before
filtering), while the "deeper than 50\,mK" selection behind `NDEEP` is **sky**
brightness temperature — `signal_loss.ipynb` computes it as
`-T21_sky.min(axis=1)`, because trough depth describes the model, not the
observation. At eta ~ 0.45 a 50\,mK sky trough is ~22\,mK in antenna
temperature, so a reader who assumes one plane reconstructs the wrong selection.

An intermediate revision of the pasted text dropped "Trough depths are quoted as
sky brightness temperature." **Verify it is back**, either as its own sentence
or attached to the selection ("whose absorption troughs, in sky brightness
temperature, are deeper than 50\,mK"). This is three paragraphs after the
section names EDGES, MIST, REACH and SARAS, so the reader arrives with a 500\,mK
sky trough in mind.

## 7. `$~100\,\textrm{m}$` renders without the tilde

**`rasti_template.tex:240`**:

> EIGSEP achieves this electromagnetic isolation by using an antenna suspended
> $~100\,\textrm{m}$ above the ground in the canyon.

A `~` in math mode is a non-breaking space, not `\sim`. This prints "100 m" with
no tilde at all. Use `$\sim$100\,m`; `:835` already has it right as
`$\sim100\,\text{m}$`. Sweep for the same slip elsewhere:
`grep -n '\$~' rasti_template.tex`.

## 8. The delay/isolation argument is internally inconsistent

Three separate problems in the same passage, all fixed by adopting one
convention and stating it once. **(a) and (b) were fixed in the manuscript on
2026-09-01, along with the convention sentence below. (c) is open.**

**The convention, as adopted at `eq:delay`.** The factor of two is stated in
prose, not folded into the displayed equation: `\tau = \Delta\ell/c = 1/\Delta\nu`
is general and true, whereas `\tau = 2d/c` would make the round-trip case a law.
It is not one — for sky emission from direction $\hat{s}$ off a scatterer at
$\vec{r}$ the differential path is $d - \hat{s}\cdot\vec{r}$, which runs from 0
to $2d$. The `where` sentence now reads:

> where $\tau$ is the signal delay, $\Delta\ell$ is the differential path length
> between the direct and scattered rays, $c$ is the speed of light, and
> $\Delta\nu$ is the period of the resulting spectral ripple. For a structure at
> a distance $d$ from the antenna, $\Delta\ell\leq2d$, the bound being reached
> for a round trip between the antenna and the structure.

This is exact in both geometries and licences both conversions in the passage
without either needing its own explanation. The `\leq` is what leaves room for
grazing incidence without the text having to discuss it. Do not replace it with
an equality, and do not move the factor of two into the equation.

**(a) FIXED — the inequalities at `:239` were the wrong way round.**

> ... this isolation will suppress spectral ripples with periods of
> $\Delta\nu \leq 1.5\,\text{MHz}$, corresponding to $\tau \gtrsim670\,\text{ns}$.

A reflector at 100\,m gives a round-trip differential path of 200\,m, hence
$\tau \geq 670$\,ns and a ripple period $\Delta\nu \leq 1.5$\,MHz. Those fine
ripples are the ones that *survive*; what the isolation removes is the broad
ripples, $\Delta\nu \gtrsim 1.5$\,MHz, from delays $\tau \lesssim 670$\,ns. As
written the sentence claims to suppress exactly the harmless ones, and it
contradicts both its own figure caption and the "gap in delay space" the figure
shows. Suggested:

> ... this isolation will confine environmental reflections to delays
> $\tau \gtrsim 670\,\text{ns}$, suppressing spectral ripples with periods
> $\Delta\nu \gtrsim 1.5\,\text{MHz}$.

**(b) FIXED — the 30\,m and 100\,m numbers used different conventions.**

`:269` and the 670\,ns figure are round-trip: 100\,m -> 200\,m -> 670\,ns ->
1.5\,MHz. The scrutiny sentence is one-way: 10\,MHz -> 100\,ns -> 30\,m, which
is a *differential path*, so the corresponding distance is 15\,m. Fix by naming
the factor of two once, at equation~\ref{eq:delay} — $\Delta\ell$ is the
differential path length, at most $2d$ for a structure at distance $d$ — and
then quoting **$\sim15$\,m** in the scrutiny sentence. This also improves the
argument: the danger zone is 15\,m against a 100\,m requirement, a factor of
seven, where the current text implies three.

**(c) DIAGNOSED 2026-09-01 — 600\,ns in the caption vs 670\,ns in the text.**
`:269` reads "An antenna at 100\,m suppresses reflections below 600\,ns
(brown)". **The 600\,ns is correct and is not an artefact.** Do not "fix" it
to 670.

The edge is the round trip to the *nearest terrain*, which sits on the canyon
side-slope at ~90.5\,m slant range — not the ~101\,m of vertical drop beneath
the antenna. The antenna hangs over a local low point of the wash and the
ground rises ~18\,m within ~35\,m to the south-west, so the closest reflector
is 11 per cent nearer than the drop implies. Verified twice: from the
ray-traced distance map in the generating notebook (min slant range 91.0\,m ->
607\,ns) and by brute force against the 0.5\,m DEM in
`horizon_position/output/marjum_dem.npz` (90.49\,m at dE -24.5\,m,
dN -25.0\,m, 22.8 deg off nadir -> 604\,ns). The published curve sits below the
axis floor to 594\,ns then climbs five decades in five bins.

Ruled out: transform/window smearing (BH7 costs about one 5.0\,ns bin, not
70\,ns); a one-way delay or a refractive index (the code is
`dlys = 2 * rmag / c`, vacuum `c`); a wrong datum (height is measured to the
DEM post directly below the antenna, checked numerically).

**Two consequences.**

1. The caption says "100\,m" but the figure's own legend reads **"101\,m"**.
   The brown curve is 101.444\,m: the notebook requests
   `[1, 5, 10, 25, 50, 75, 100]` and snaps each to a coarse height grid, where
   1 and 5 both snap to 1.0, so six curves are drawn and the last is 101.444.
   Correcting the caption to 101\,m alone makes the arithmetic look worse
   (2 x 101.444/c = 677\,ns), so correct it together with point 2.
2. The figure and the body text answer **different geometric questions**, which
   is why they disagree and why neither is wrong. The body's 670\,ns is the
   round trip for the nominal 100\,m **conductor-free** requirement. The
   figure's floor is the slant range to **terrain**, which is not a conductor
   and was never covered by that requirement. At a true 100\,m suspension the
   nearest terrain is 89.2\,m -> 595\,ns. So make the caption state what sets
   its own edge — an antenna suspended 101\,m above the canyon floor sees no
   terrain closer than ~90\,m in slant range, hence ~600\,ns — and leave the
   body's 670\,ns as the conductor requirement. This does not disturb the
   `eq:delay` sentence: `\Delta\ell \leq 2d` is exact, because $d$ is the
   distance to the structure, i.e. the slant range, not the height.

**Provenance, and a reproducibility gap.** Physics:
`eigsep_terrain/notebooks/terrain_s11.ipynb`, cells 17--19. Plotting:
`papers/eigsep_instrument/notebooks/s11-plot.ipynb`, cell 5. Both intact. But
the geometry input `horizon_models_v000.npz` was written by a version of
`EIGSEP Horizons.ipynb` that is **not in git** — the `savez` is commented out in
every committed version, and both committed versions produce a different array
shape than the file on disk. The figure cannot currently be regenerated from
committed code alone. Worth fixing before the data-availability statement is
finalised (Task C4 in the rasti repo's `docs/revision-round2-plan.md`).

**What is deliberately not said.** The round-trip convention is exact only for a
signal transmitted from the antenna — the case Fig.~\ref{fig:reflections}
actually computes, and the text already hedges this. For sky emission the
differential path is $2d\cos\theta$, so near-grazing geometries give short
delays that 100\,m does not remove; in practice they subtend little solid angle
and the canyon horizon blocks most of them. This is why the passage says
**nominal requirement** and **suppress** rather than eliminate. Keep those two
words and do not open the argument — but expect a referee to raise it, and note
that the 2\,m platform requirement sits inside the danger zone by this same
arithmetic (2\,m -> 4\,m -> 13\,ns -> 75\,MHz) and is answered by the platform
being modelled in the electromagnetic simulations, not avoided.

## 9. Two numbers now live only in Overleaf

The author's units paragraph quotes open-sky fractions for all three beams:
0.36--0.55 (bowtie), 0.59--0.80 (Vivaldi), 0.35 (isotropic). All three are
correct against `output/beam_sims.npz` — computed as `1 - fgnd`, giving
0.3571--0.5489, 0.5875--0.7965, and 0.3467 at every frequency for the isotropic
beam (its standard deviation across the band is exactly zero, so the single
value is right, not a rounding).

But only the bowtie range comes from tokens (`ETALO`/`ETAHI`). The other two are
hand-typed, which is the failure mode that took out the rotation paragraph. Emit
`ETALO{key}`/`ETAHI{key}` per beam from the loop in `beam_comparison.ipynb` and
token the sentence. Note this also leaves `ETAVIV`/`ETABOW` (the band means,
0.70 and 0.45) unused, alongside `DYNRANGE`, `FGM1`, `RMS1` and `RMS4`; update
the header list in `paper_text.tex.in` when that happens.

---


## 10. Blocks 3--5, as folded into `subsec:fwd_modelling`

All substituted numbers were checked against the arrays on 2026-09-01 and are
correct: 5.2/0.9/9.5\,K, the 1.82 -> 3.52/2.04/1.81\,mK floors, the 1.16\,mK
median, 30/11/2.4\,mK at 10\,m, 1.84\,mK and "1 per cent", and the caption's
7/4/10 and 99 per cent/10th mode. The author's "approximately three times"
replaces the draft's "several times" and is right (3.52/1.16 = 3.03).

**Fixed by the author on 2026-09-01 — confirm they survived the final sweep:**

- "the floors ... **do** not impose" (was "does not impose")
- "the **dependence** of the antenna temperature **on** the magnitude" (was
  "dependency ... of")
- "These **measurements** will be folded **into**" (was "measurement ... in to")
- `section~\ref{subsec:box}` — the tilde was missing
- panel references: `Fig.~\ref{fig:horizon}(b)`, not the bare `b` the draft had
- the bridge sentence names the **bowtie** simulation of `subsec:covariance`,
  since that section used three beams

That bridge sentence is verified true: `position_sims.npz` and `beam_sims.npz`
agree on `t_start`, `n_times`, `t_ground` = 300\,K, `t_receiver` = 50\,K, site
coordinates, `sky_model` = gsm16 and `beam_lmax` = 128, and `run_sims.py:71`
calls `eigsim.load_beam()`, which `beams.py:3` documents as the stored MWSS
bowtie -- the same beam as the centre panel of Fig.~\ref{fig:singular_values}.

**Resolved by the rewrite:** the old closing paragraph at `:662` ("Using six
modes, the coupling is suppressed below 10\,mK ... removed by the standard
foreground filter") is gone. That was the gap in the block set flagged in
item 1 -- no block replaced it -- and the author closed it by hand.

**Still open:**

- **The word "requirement" contradicts itself across three paragraphs.**
  Paragraph 3 has "where the position requirements are the most stringent";
  paragraph 4 has the floors "do not impose a requirement on the absolute
  position"; the closer has "the stringent requirement on the vertical position
  knowledge". The third is defensible -- "knowledge" is prior width, which is
  the block's actual claim -- but three sentences after the disclaimer it reads
  as a contradiction. Paragraph 3 -> "where an unmodelled error costs the most";
  the closer -> "Because the vertical prior must be narrow".
- **The own-basis numbers were cut** (OWNLO10--OWNHI10 = 1.66--2.03\,mK at
  10\,m against 1.82\,mK unperturbed). Paragraph 1 keeps the qualitative
  version, "carry no knowledge of the displacement", which the block header says
  does most of the work. The numbers were the quantitative rebuttal to "a 1\,m
  position error is worse than the entire foreground sky". `signal_loss.ipynb`
  still asserts the claim holds (`_own_dev < 0.05`). One clause restores it if a
  referee pushes.
- **Item 1's other half is now urgent.** The closer points at
  `subsec:box`, and `rasti_template.tex:398` in that subsection says the
  structure "is foreground-like and is suppressed by the same eigenmode filter
  applied to the foregrounds" -- which block 4 refutes. The two passages now
  cross-reference each other while disagreeing.
**SETTLED 2026-09-01: block 4's closing paragraph and the caption's final
clause are both cut.** The paragraph carried the cosine-similarity and
single-mode-fragility argument -- the case that reading an excess off a
fixed-depth filter is unsafe -- and read as pedagogy rather than result. The
*positive* case for forward modelling survives in paragraph 4 (known
displacement is absorbed, unknown is marginalised, the sensitivities set the
prior width), and that is the case the section needs to make.

The caption's trailing clause went with it, because its only job was to supply
evidence for the dropped argument. The caption now ends:

> ... after 7 modes for the eastward displacement, 4 for the northward, and
> 10 for the vertical shift.

**Verify both cuts held, and that neither came back.** Do not restore one
without the other -- the clause with no argument, or the argument with no
figure evidence, is worse than either state.

Nothing is lost bibliographically: `2018Natur.564E..32H`,
`2019ApJ...874..153B` and `2019ApJ...880...26S` are all still cited at
`rasti_template.tex:161` (the introduction's systematic-artefact sentence) and
at `:236` (the ground-plane sentence in `subsec:covariance`).

One loose end this creates, harmless but worth knowing: `signal_loss.ipynb`
asserts that `CLEAR_U` and `SPIKE` still coincide, which existed to gate the
caption's "where ..." clause. The assert is now guarding no published sentence.
Keep it as a sanity check -- it is still a real property of the data -- but do
not puzzle over it, and do not delete it on the grounds that nothing uses it.

## 11. Sign symmetry: settled, do not re-add the clause

The draft said a horizontal shift leaves a response "neither proportional to the
displacement nor symmetric in its sign", and the vertical one "symmetric in its
sign". The author cut both. **The cut was right for the numbers as quoted**, and
this section exists so the clause is not restored on a hunch.

`run_sims.py` displaces each axis both ways (19 positions = 1 + 3 axes x 3
magnitudes x 2 signs) and `floors_for` in `signal_loss.ipynb` cell 20 averages
the two signs. Nominal-basis floors at N_HAND = 9, in mK, against an
unperturbed 1.816:

| axis | mag | -- | + | mean |
|---|---|---|---|---|
| East | 1 | 2.042 | 2.044 | 2.043 |
| North | 1 | 1.822 | 1.800 | 1.811 |
| Up | 1 | 3.521 | 3.520 | 3.521 |
| East | 10 | 10.227 | 12.175 | 11.201 |
| North | 10 | 2.247 | 2.558 | 2.402 |
| Up | 10 | 31.205 | 29.373 | 30.289 |

**At 1\,m -- the magnitude every headline number is quoted at -- all three axes
are symmetric to three significant figures.** The north row is at the
unperturbed floor either way, so its sign difference is noise. Asymmetry appears
only at 10\,m, and meaningfully only for East (10.2 vs 12.2, quoted as 11). At
that magnitude the sentence's job is an order-of-magnitude direction comparison
(30 vs 11 vs 2.4), which a 20 per cent spread does not disturb. If anyone wants
the averaging made explicit, four words at first use -- "averaged over the two
displacement signs" -- is enough; a clause about sign asymmetry is not.

Minor caveat on the surviving sentence, "the residual it leaves grows roughly in
proportion to the displacement": true for the floor from 1 to 10\,m
(3.52 -> 30, 8.5x), but the *excess over the unperturbed floor* grows 1.71 ->
28.5\,mK, i.e. 16.7x. At 0.1\,m the excess is masked by the 1.82\,mK
unperturbed floor entirely. The claim holds in the regime the ladder walks; it
is not a scaling law.

---

## When this is done

Fold anything durable into `horizon_position/CLAUDE.md` — the framing
constraints there are the ones that must outlive the draft — and delete this
file.
