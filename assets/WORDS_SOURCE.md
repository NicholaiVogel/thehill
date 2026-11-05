# Word list provenance

`words.txt` was built by `tools/build_words.py` on 2026-11-05:

1. Took wordfreq 3.1.1 `top_n_list("en", 80000)`, in frequency order.
   https://github.com/rspeer/wordfreq. Code: Apache-2.0. Data: CC BY-SA 4.0, which requires attribution and share-alike for redistributed word data.
2. Kept only `[a-z]+` words. Single letters other than "a" and "i" were dropped.
3. Kept words that the LibreOffice en_US Hunspell dictionary (SCOWL size 60, version 2020.12.07) accepts in lowercase, checked with spylls. This drops names and fragments.
   https://github.com/LibreOffice/dictionaries/tree/master/en. SCOWL license: permissive (MIT-like).
4. Dropped words on LDNOOBW's English list.
   https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words. License: CC BY 4.0.
5. Stopped at 20,000 words.

To rebuild, put `en_US.dic`, `en_US.aff` and the LDNOOBW `en` list (saved as `bad.txt`) in the working directory. Then:

```sh
uv venv && uv pip install wordfreq spylls
.venv/bin/python tools/build_words.py
```
