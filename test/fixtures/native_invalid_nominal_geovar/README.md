Fresh Sonnet Lite 18.53 rejects this project during parsing because `SCD` is
not a scaled-dimension keyword. Its requested width equals the saved nominal
width. Before the repair, DiffMoM returned the unchanged project and bypassed
the invalid header. The original input, engine outputs and failed-run metadata
are retained exactly. `reproduction.toml` records the old reader/native result.

The regression checks rejection before geometry emission. This failed native
analysis has no response matrix and provides no electromagnetic accuracy claim.
