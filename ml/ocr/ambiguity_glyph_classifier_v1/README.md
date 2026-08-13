# Ambiguity glyph classifier V1

This project-owned Apache-2.0 model classifies four isolated source glyphs: `O`, `o`, `l`, and `I`.
It exists because the exact official recognizer collapsed these identities after the conservative spacing stage
had already passed every non-ambiguity selection case without altering recognized nonspace characters.

Training, validation, and public fixtures are fresh procedural Noto glyph crops. They contain no private image,
article image, Chandler pixel, graph position, semantic role, word list, or production-composition pixel. The
public archive is single-use and remains unopened until exact committed selection evidence authorizes it.

A component pass does not create a production manifest, promote a model store, approve packaging, or make the
release eligible. Composition must still prove that source groups and glyph order are correct end to end.
