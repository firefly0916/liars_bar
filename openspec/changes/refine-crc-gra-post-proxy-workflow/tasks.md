## 1. Tracking And Specification

- [x] 1.1 Create OpenSpec proposal/design/tasks for the staged post-proxy refinement track
- [x] 1.2 Create a living modification log with status, risks, and experiment slots

## 2. Offline Audit, No Training Change

- [x] 2.1 Add tests for audit label precedence and legacy field interpretation
- [x] 2.2 Implement offline reasoning-action audit records and summary output
- [x] 2.3 Generate case-study markdown from existing backed-up results
- [x] 2.4 Validate locally without touching training scripts

## 3. Data Provenance And Split Audit

- [x] 3.1 Add a data-provenance report for proxy training, proxy validation, alignment construction, and formal evaluation sources
- [x] 3.2 Record whether post-proxy alignment data was generated from old logs or from a fresh policy run
- [x] 3.3 Decide whether a fresh post-proxy alignment collection is required before new training variants

## 4. Candidate Expansion Variant

- [x] 4.1 Add tests for concrete legal candidate enumeration
- [x] 4.2 Add tests for proxy top/bottom candidate selection and group-size bounds
- [x] 4.3 Implement isolated candidate-expansion variant
- [x] 4.4 Run local dry-run and server smoke training
- [x] 4.5 Launch controlled 200-game formal eval in a new run root

## 5. Optional Protocol Handling

- [ ] 5.1 Use audit output to decide whether protocol repair deserves a separate variant
- [ ] 5.2 If justified, add protocol-filter-only ablation
- [ ] 5.3 Evaluate with fixed 200-game formal eval

## 6. Optional Reasoning Regeneration

- [ ] 6.1 Decide whether action-conditioned reasoning regeneration is necessary after candidate expansion results
- [ ] 6.2 If justified, specify a separate OpenSpec change before implementation
