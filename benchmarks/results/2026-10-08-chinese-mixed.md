# 2026-10-08, Chinese mixed with English, by ear

One listener heard three sentences spoken by `ksay` with the `zh` extra (voice
`zf_xiaobei`), by `say -v Tingting` and by `say -v "Tingting (Enhanced)"`, on the Mac
described in the [main report](2026-10-08-apple-m2-max.md). It is one listener and three
sentences, not a measurement. Apple's licence does not allow publishing recordings of
its voices, so there is no audio here; `ksay`'s side is rebuilt with the commands in the
README.

| | Sentence |
| --- | --- |
| 1 | 今天我们来讨论一下 machine learning 的应用，请在 Settings 里打开 Wi-Fi，然后点击 OK。 |
| 2 | 这个 API 返回 JSON 格式的数据，你可以用 Python 调用 requests 库来解析。 |
| 3 | 你好，世界。今天天气很好，我们一起去公园散步吧。 |

## What the listener heard

- In `say`'s recordings the English words come from what is clearly a second voice,
  different from the one that speaks the Chinese. `ksay` stays in one voice, and the
  English words sound natural in it.
- `say` reads JSON as the letters J, S, O, N. `ksay` says the word.

## What a recognizer hears

`benchmarks/transcripts.py` (Whisper `small.en`, sampling off) on sentence 2:

| Recording | Transcript |
| --- | --- |
| `ksay`, `zf_xiaobei` | JAGA API found Hui JSON goshu de shou jue Ni kuei yong baizen diow yong request kulai jie jie |
| `say -v Tingting` | jager, a, p, i, fang hui, j, s, o, n, geshida shuju. nikei yong pithan, diao yong requests, hu lai jie xi. |
| `say -v "Tingting (Enhanced)"` | If you want to find out more about JSO and JSO, visit Python.com to find out more about JSO. |

For `Tingting` the recognizer confirms the letters: "a, p, i" and "j, s, o, n". For
`ksay` it writes the word JSON. The transcript of Tingting (Enhanced) is made up and
settles nothing about it.

The recognizer is not a judge of the rest. On sentences that are mostly Chinese it
invents text. Counting how many of the four English words of a sentence it catches put
Tingting (Enhanced) above `ksay` on sentence 1 (3 of 4 against 1 of 4), which is not
what the listener heard, presumably because it hears a Chinese voice's English less
well than a person does. Such counts were dropped as a measure.
