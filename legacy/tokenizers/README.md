# Earlier tokenizers

The four-special-token predecessor BPE files and old root metadata were removed
from HEAD because no current reader workflow or test uses them. Their exact bytes
remain at commit `e76cc03`, under this directory's former subpaths recorded in
[the migration CSV](../MIGRATIONS.csv). See the [history note](../README.md).
Use the [canonical seven-special-token release](../../tokenizer/README.md).
