### Requirement: Isolated post-proxy refinement tracking

The workflow SHALL maintain a living modification log for post-proxy CRC-GRA refinements.

#### Scenario: New modification point

- **GIVEN** a proposed change to audit, filtering, candidate construction, protocol handling, or reasoning supervision
- **WHEN** the change is considered for implementation
- **THEN** the log SHALL record its motivation, affected code, expected impact, risk, verification plan, server run root, and status.

### Requirement: Offline audit before training changes

The workflow SHALL support an offline audit path that does not update policy parameters.

#### Scenario: Audit formal evaluation logs

- **GIVEN** formal evaluation logs and proxy audit outputs
- **WHEN** the offline auditor is run
- **THEN** it SHALL output structured audit records, aggregate summary, case-study markdown, and a scorecard.
- **AND** it SHALL distinguish legacy high-EVGap action/proxy disagreement from semantic reasoning-action mismatch.

### Requirement: Isolated candidate expansion variant

The workflow SHALL add candidate expansion only as an isolated training variant.

#### Scenario: Expanded candidates

- **GIVEN** one audited decision point with legal actions and proxy model access
- **WHEN** the candidate-expansion variant builds a group
- **THEN** it SHALL include the logged action and proxy-best action when available.
- **AND** it SHOULD include bounded additional concrete legal alternatives such as proxy second-best, proxy worst, legal challenge, truthful play, and bluff play when distinct.
- **AND** it SHALL write outputs to a new run root without modifying best-main artifacts.

### Requirement: Data provenance before new training variants

The workflow SHALL record data lineage before launching new post-proxy training variants.

#### Scenario: Audit data splits

- **GIVEN** a proposed new training variant after proxy distillation
- **WHEN** the variant is prepared for server execution
- **THEN** the workflow SHALL identify the run roots or seeds used for proxy training, proxy validation, alignment data construction, and formal evaluation.
- **AND** it SHALL state whether alignment data came from old logs or a fresh post-proxy policy run.
- **AND** it SHALL record any known overlap risk before interpreting results.
