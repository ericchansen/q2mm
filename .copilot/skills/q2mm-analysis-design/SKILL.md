---
name: q2mm-analysis-design
description: 'MANDATORY before writing any q2mm benchmark analysis, comparison, or validation document. Forces restating the user question verbatim, listing deliverable artifacts (table headers, figure captions), mapping each artifact back to the question, and flagging mismatches BEFORE writing the doc. Use when about to create or rewrite docs/benchmarks/*.md, when about to produce a comparison page, when summarizing benchmark results, or when packaging a PR with new results.'
license: MIT
allowed-tools: Read, Edit, Grep
---

# q2mm Analysis Design

A benchmark analysis page that doesn't answer the user's literal question is worse than no page at all — it ships a wrong-but-confident answer that has to be retracted. This skill exists because a previous agent shipped a 230-line page comparing `final_obj_score` when the user had asked about parameter values and R² vs published papers.

## Step 1 — Restate the user's question VERBATIM

Find the user message that triggered this work and quote it exactly. Do not paraphrase. Do not "interpret." If you can't find it, ask.

Then write what the user literally asked for as a single declarative sentence: *"The user wants ___."*

If the question has multiple parts, enumerate them: *"The user wants (a) X, (b) Y, (c) Z."*

## Step 2 — List every deliverable artifact

Enumerate what will exist when the doc is done. For each artifact, write:
- **Type**: table, figure, paragraph, code block
- **Content**: what columns/rows/axes/topic
- **Source**: where the data comes from (JSON path, paper DOI, computed from FFs, etc.)

Example:
- *Table 3.1*: per-system R² comparison. Columns: published-paper R², q2mm-from-published R², q2mm-from-QFUERZA R². Rows: bond length, bond angle, eigenvalue diagonal. Source: `q2mm-data/benchmarks/<system>/{convergence,from-qfuerza}/validation_results.json` for q2mm; published-paper PDFs for paper R².
- *Table 3.2*: per-parameter abs deviation. Columns: param-id, published value, QFUERZA-optimized value, abs deviation, % deviation, chemical motif. Source: parsed from `<system>_optimized.fld` files.
- *Paragraph 4.1*: physical-chemistry walkthrough of the 5 largest bond-length deviations in rh-enamide. Source: synthesis from Table 3.2 + chemistry knowledge.

If you cannot enumerate every artifact in ≤ 15 bullets, the scope is too broad — break it into a separate doc.

## Step 3 — Map each artifact to a piece of the user's question

For each numbered artifact in Step 2, write which part of the user's question (from Step 1) it addresses.

Format: *"Table 3.1 answers question part (a): are q2mm-optimized R² values close to published-paper R²?"*

If an artifact does NOT map to any part of the question, ask yourself:
- Is this scope creep? → drop the artifact
- Is it scaffolding the reader needs? → keep but mark as "background"
- Did I misunderstand the question? → revisit Step 1

If a question part has NO artifact mapped to it, this is the critical failure mode:
- Add the missing artifact, OR
- Explicitly note in the doc "We did not answer (X) because ___" and tell the user why

## Step 4 — Flag mismatches BEFORE writing

Before writing a single sentence of the doc:

- [ ] Every user-question part has at least one artifact answering it
- [ ] Every artifact maps to a user-question part (or is marked background)
- [ ] The success-pass criterion is stated explicitly: how does the reader know the answer is "yes" or "no"?
- [ ] Negative results are accommodated: if QFUERZA fails to recover the published params on system X, the doc still has a place for that finding (not just "successful systems only")

If any box is unchecked, stop and fix the design. Do not start writing.

## Step 5 — Now write the doc

Open with:
1. The user's question (paraphrased gracefully for the published audience)
2. The TL;DR answer with the actual headline number
3. A roadmap of what's in the doc

Then walk through the artifacts in the order you listed them in Step 2. Each section ends with a "what this means" paragraph that ties back to the user's question.

## Step 6 — Self-critique pass

Before declaring the doc done:
- Re-read the user's question (Step 1).
- Read your TL;DR.
- Does the TL;DR answer the question? If no, the doc is wrong even if every section is correct.
- Are there places where the doc says "we did X" but should say "the data showed Y"?
- Are there places where the doc editorializes ("a strong validation", "a remarkable result") without evidence?

Per AGENTS.md §2 rule 8: *"Every claim must be grounded in evidence. ... do not embellish, glorify, or editorialize."*

## Common mismatches to refuse

- User asks about parameters → doc compares objective scores (proxy, not the question)
- User asks about R² → doc compares improvement percentages (proxy, not the question)
- User asks about physical-chemistry interpretation → doc only has numbers (no chemistry)
- User asks "did we recover the published FF?" → doc shows only "did the optimizer converge to *a* minimum?" (different question)
- User asks for a comparison → doc shows results from only one condition (missing the comparison)

## Output

After Step 4, write the design as a checklist in the session plan or a markdown comment block. The doc author (you, in the next session) reads this before writing. Reviewers (the user) read this to confirm scope alignment.
