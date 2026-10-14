# Implementation status: 0.1.0 alpha

The numbered files in this folder remain the original proposal. The authoritative
implemented behavior is in `docs/api.md`, tests, and `src/neuraljaxwork`.

Implemented: PyTorch-style forward modules; Flax NNX state transforms; JAX autodiff;
explicit RNGs; dense/convolution/transpose convolution, normalization/pooling,
RNN/LSTM, attention; all requested primary architecture families; explicit training
steps and separate GAN optimizers; tiny smoke tests; data-loader cursor;
checkpoint resume; hierarchy JSON; run artifacts; optional small Task/Trainer;
seven runnable examples and notebook frontends.

Deliberate alpha differences:

- NNX discovers ordinary list/dict children. Raw-array attributes are not rejected
  by a custom registration layer. Use Parameter/Buffer for supported persistent state.
- NNX gradient states are used directly; ParameterSet selects an owner module.
  Parameter groups, freezing, provenance validation, and structural invalidation
  diagnostics are not yet implemented. Rebuild optimizers after structure changes.
- No universal transactional rollback guarantee around eager/transformed exceptions.
- The pure `apply` helper supports deterministic module calls; advanced stochastic
  pure calls use explicit NNX split/merge state rather than the proposed RNG overload.
- Architecture output is an inspectable hierarchy/config/alias description, not
  ModelSpec factories or an operation-level execution graph. Custom code remains required.
- State dictionaries omit alias metadata; full checkpoints validate the described
  graph and joint state paths. Exact resume is verified locally, not across hardware.
- The trainer supports loss/count/optional-correct reducers and manual steps.
  Automatic resume, scheduler/callback state, and arbitrary metric reducers remain
  explicit-loop responsibilities.
- No bidirectional recurrence, KV caches, arbitrary adaptive pool sizes, parameter
  containers beyond module containers, or loss-module wrappers yet. Functional
  losses cover the examples.
- Cats/dogs has a tested RGB architecture but no bundled dataset notebook yet.
  The implemented image showcases prioritize the requested MNIST CNN and
  Fashion-MNIST GAN. Full training/visual-quality studies remain separate runs.

These limits are documented rather than hidden behind placeholder implementations.
