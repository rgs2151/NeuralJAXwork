# Historical prototype

The 2023 uppercase `JAXsaw` package and its demo are preserved here for
historical reference. They are not installed, tested, or supported by the new
package. The original README describes planned features as well as implemented behavior.

Use `src/jaxsaw` and the repository's current README. The new API uses
JAX autodiff and Flax NNX state transformations; it does not accept old model or
checkpoint objects. Port architectures explicitly to `nn.Module.forward`.
