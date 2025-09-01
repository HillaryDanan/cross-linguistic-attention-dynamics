# Methodological Considerations

## Layer-Specific Processing
Following reviewer feedback, we acknowledge that transformer layers handle different types of processing:
- **Layers 1-3**: Primarily syntactic alignment
- **Layers 4-7**: Mixed processing (our hypothesis zone)
- **Layers 8-12**: Primarily semantic processing

The orthographic transparency effect may peak outside our predicted layers 4-7. Our analysis will track effects across all layers to identify the actual peak.

## Flesch-Kincaid Matching Caveat
While matching texts by Flesch-Kincaid readability ensures complexity equivalence, we must verify this doesn't inadvertently select for specific syntactic structures that could confound results. We'll run post-hoc analyses to check for syntactic biases in our matched pairs.

## Tokenization as Separate Factor
Subword tokenization differences between Spanish and English are tracked separately from attention geometry effects. This isolation is critical for attributing differences to orthographic transparency rather than tokenization artifacts.
