# check_tail_ckpt_sequences.py PORT TAG MODELID : tail-checkpoint regression sequences against a running server (follow-up, edited history, shorter prefix, shared tail, 9-way interleave). ONLY=name,.. runs listed requests once (oracle on a stock server). Compare md5s: stock-cold == new, repeat == cold.
import sys, json, urllib.request, hashlib
port, tag, mid = sys.argv[1], sys.argv[2], sys.argv[3]
def post(msgs, mt=48):
    body = json.dumps({"model": mid, "messages": msgs, "max_tokens": mt, "temperature": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=900))
    c = r["choices"][0]["message"]; tm = r["timings"]
    return [hashlib.md5(((c.get("content") or "") + "|" + (c.get("reasoning_content") or "")).encode()).hexdigest()[:8], tm["cache_n"], tm["prompt_n"], round(tm["prompt_ms"])]
W = ["alpha","river","stone","lamp","orbit","violin","cedar","harbor","quartz","meadow","ember","tundra"]
def fill(n, seed): return " ".join(f"{W[(i*7+seed)%12]}{(i*31+seed)%97}" for i in range(n))
def U(t): return {"role":"user","content":t}
def A(t): return {"role":"assistant","content":t}
S = {"role":"system","content":"Be terse. " + fill(150, 1)}
base = fill(500, 11)
seq = []
c1 = [S, U("Context: " + base + " Q1: name a river from the text.")]
seq.append(("c1", c1))
c2 = c1 + [A("river12"), U("Q2: and a stone?")]
seq.append(("c2_follow", c2))
c3 = c2 + [A("stone5"), U("Q3: and a lamp? Answer in two words.")]
seq.append(("c3_follow", c3))
# edited history: one word changed deep in the first turn, then same follow-up
base_e = base.replace("cedar", "cedarx", 3)
seq.append(("edit_deep", [S, U("Context: " + base_e + " Q1: name a river from the text."), A("river12"), U("Q2: and a stone?")]))
# edit of the assistant turn only
seq.append(("edit_asst", c1 + [A("it is river99"), U("Q2: and a stone?")]))
# shorter prefix: first ~300 words of the context only, same system
seq.append(("short_prefix", [S, U("Context: " + " ".join(base.split()[:300]) + " Q1: name a river from the text.")]))
# shared page k, different earlier pages: same big tail text after different heads
tailtxt = fill(400, 21)
for k, head in enumerate([fill(40, 31), fill(40, 32), fill(300, 33)]):
    seq.append((f"sharedtail{k}", [U(head + " " + tailtxt + " Q: say one word.")]))
# 9 interleaved conversations, then continue the first ones (eviction of the tail LRU)
convs = [[U("Topic %d: " % j + fill(450 + 13*j, 40 + j) + " Q1: say one word.")] for j in range(9)]
for j, c in enumerate(convs): seq.append((f"il{j}", c))
for j in range(3):
    seq.append((f"il{j}_cont", convs[j] + [A("word"), U("Q2: another word?")]))
import os
only = os.environ.get("ONLY")
out = {}
if only:                                  # oracle mode: fresh server, listed requests, first-time (cold) only
    for name, m in seq:
        if name in only.split(","):
            out[f"cold_{name}"] = post(m); print(tag, "cold", name, out[f"cold_{name}"], flush=True)
else:
    for name, m in seq:                   # cold then immediate repeat
        if name.startswith("il"): continue
        out[f"a_{name}"] = post(m); out[f"b_{name}"] = post(m)
        print(tag, name, out[f"a_{name}"], out[f"b_{name}"], flush=True)
    for name, m in seq:                   # 9 interleaved, then continue the first 3
        if name.startswith("il"):
            out[f"a_{name}"] = post(m); print(tag, name, out[f"a_{name}"], flush=True)
json.dump(out, open(f"res_{tag}_ident3.json", "w"))
