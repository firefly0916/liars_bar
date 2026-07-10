## 1. OpenSpec

- [x] 1.1 Create proposal, design, and spec for the isolated online GRPO baseline

## 2. Tests

- [x] 2.1 Add tests for group-normalized advantages and reward independence
- [x] 2.2 Add tests for the GRPO trainer CLI contract
- [x] 2.3 Add tests for the GRPO runner isolation and 200-game formal default

## 3. Implementation

- [ ] 3.1 Add `scripts/train_online_grpo_baseline.py`
- [ ] 3.2 Add `scripts/run_online_grpo_baseline.sh`
- [ ] 3.3 Ensure the runner does not call or modify best-result scripts

## 4. Verification and Deployment

- [ ] 4.1 Run local unit and syntax validation
- [ ] 4.2 Deploy new scripts to AutoDL
- [ ] 4.3 Run an AutoDL online GRPO smoke test
- [ ] 4.4 Run 200-game formal evaluation for the online GRPO final adapter
