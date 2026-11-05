import re
from spylls.hunspell import Dictionary
from wordfreq import top_n_list

# Hunspell accepts lowercase abbreviations ("kg", "ltd", "xv"), which make poor typing words.
TWO_LETTER = set("to of in is it on be as at he by my or we an so me if up do no us go am oh hi ah uh um yo ow ye ox".split())
ROMAN = re.compile(r"m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})")
NOT_ROMAN = {"i", "mix"}

d = Dictionary.from_files("en_US")
bad = {w.strip().lower() for w in open("bad.txt") if w.strip()}
# Slurs and insults LDNOOBW misses.
bad |= set("""dyke dykes retard retards retarded gyp gypped gypsy gypsies homo homos cripple cripples
midget midgets redskin redskins slut sluts whore whores spaz spastic mongoloid tranny wetback
dickhead dickheads""".split())
# Names and abbreviations Hunspell lists in lowercase.
bad |= set("""john johns johnny jenny hwy seq approx viz prob exec execs pol pols ft km cm mm mph rpm
jun univ dept govt corp inc ref refs avg est def vol vols etc ave secs yrs mos""".split())
out = []
for w in top_n_list("en", 100000):
    if not re.fullmatch(r"[a-z]+", w) or w in bad:
        continue
    if len(w) == 1 and w not in ("a", "i"):
        continue
    if len(w) == 2 and w not in TWO_LETTER:
        continue
    if not re.search(r"[aeiouy]", w):
        continue
    if ROMAN.fullmatch(w) and w not in NOT_ROMAN:
        continue
    if not d.lookup(w):
        continue
    out.append(w)
    if len(out) == 20000:
        break
open("words.txt", "w").write("\n".join(out) + "\n")
print(len(out), out[:30], out[-10:])
