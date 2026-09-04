# GLM-5.3-Flash MLX runtime for M3 Ultra 512 GB

`zai-org/GLM-5.3-Flash`をApple M3 Ultra 512 GBで動かすための、text-only・single-node・decode-first runtimeです。OpenCodeなどから利用できるOpenAI互換APIを提供します。既定は公式tensor layoutとDirect NoPE cacheを使う経路です。exactなpacked decode MoE、correctness未合格のpacked grouped FP8 prefill、およびsingle-latent＋compact IndexPool cacheをそれぞれ実験的にopt-inできます。sparse DSA prefillは未実装です。

提供する主なendpointは次のとおりです。

- `POST /v1/chat/completions`（streaming、tools対応）
- `POST /v1/responses`（streaming対応）
- `GET /v1/models`
- `GET /health`
- `GET /v1/metrics`

演算backendはMLX/Metalです。GLM-5.3固有の34 KDA層、11 DSA/IndexPool層、4-stream mHC、288-expert MoEを実装した`mlx-vlm`を固定revisionで利用し、公式Transformersとの比較で判明した次の数値修正をruntime起動時に適用します。

- 全text FFNの`swiglu_limit=10`
- router logitsのFP32化
- MLA low-rank RMSNormを`1e-5`
- Indexer LayerNormを`1e-6`
- IndexPool最終出力を`-1` sentinelまたは`[0, Kv)`へsanitize
- decode/prefill attention gatherでIndexPool範囲を独立再検査
- mHC係数、KDA decay、router biasのFP32保持

## 1. セットアップ

Python 3.11以上と`uv`を使います。

```bash
uv sync
uv run glm53 inspect /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash
uv run glm53 attest /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash
```

GPUアクセスを制限するsandbox内ではMLXをimportできません。通常のmacOS Terminalから実行してください。

## 2. FP8 direct execution

公式checkpointを変換・repack・全BF16展開せず、そのまま使用します。FP8対象の各moduleは次のcanonical tensorを保持します。

```text
weight            uint8 E4M3 [out, in]
weight_scale_inv  FP32       [ceil(out/128), ceil(in/128)]
```

Metal kernelが128×128 block scaleを参照し、weightをregister/threadgroupへloadした時だけFP32へdecodeします。weight全体をBF16 bufferへ展開しません。MoE expertも288個をstackせず、公式checkpointの個別tensorを参照してrouter結果ごとにbucketします。

永続tensorではlayoutとstorage lifetimeを別契約として扱います。`row-major-contiguous`はownershipを意味しません。checkpointのread-only mmapはownerを保持する`borrowed-stable`、runtimeが作るfused projectionとpacked expert bankは`owned`です。再利用されるloader staging/scratchは`borrowed-ephemeral`であり、明示的に`materialize_owned()`しない限りresident境界を越えられません。

## 3. OpenAI互換serverを起動する

```bash
uv run glm53 serve \
  --model /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --host 127.0.0.1 \
  --port 8080
```

batch-1 decodeだけをpacked selected top-8 kernelへ移し、prefillではpacked bank上のDirect演算順を維持するexact backendは次のflagで有効化します。

```bash
uv run glm53 serve \
  --model /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --experimental-packed-decode-moe \
  --experimental-compact-nope-dsa-cache
```

`L=1`は連続bankを直接読むgate/up/downの3 kernel、`L>1`はDirectと同じexpert bucket、tiled-GEMM/GEMV、BF16 reduction順を使います。grouped kernelは呼びません。`--experimental-packed-grouped-moe`とは排他的で、どちらも指定しない既定backendはDirectのままです。

全42 routed MoE層を層単位で連続FP8 bankへ移行し、GPU sorted grouped prefillを使う場合だけ次を指定します。公式FP8＋FP32 scaleのままで、BF16 weight copyは作りません。

```bash
uv run glm53 serve \
  --model /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --experimental-packed-grouped-moe
```

このflagは既定offです。kernel単体の実測損益分岐は16 routesですが、17-token oracleの全logits hashを維持するためruntimeは256 routes未満をpacked Direct-compatible pathへfallbackします。したがって256-token prefillはgrouped、batch-1 decodeと短いpromptはbit-identicalなselected top-8経路です。

NoPE latentのK/V二重保持と全context分のpacked Indexer token historyを除去するproduction cacheは、次のflagで明示的に有効化します。

```bash
uv run glm53 serve \
  --model /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --experimental-compact-nope-dsa-cache
```

この経路は11 DSA層を`SingleNoPELatentCache + CompactIndexPoolCache`へ切り替えます。prefill 1〜256 tokenから直接compact poolを構築し、4,096-token generation headroomを256-token境界へ予約します。raw rollback state上限は監査済み`index_kpool`から`16 + index_kpool - 1`として導出され、公式checkpointでは19 tokenです。batch 1専用で、batch > 1はfail closedします。既定Direct cache、MoE backend、prompt/context admissionは変わりません。

通常運転では同時sequenceを1に固定し、interactive decodeを優先します。既定memory設定はwired 440 GB、MLX cache 32 GB、prefill chunk 2048 tokensです。

CPU expert bucket/full-KV DSA prefillを巨大promptへ誤って起動しないよう、次の2段階admissionを既定で適用します。

- prompt上限: 256 tokens（実機probe済み）
- prompt + generationの総context上限: 16,384 tokens
- requestのgeneration上限: 4,096 tokens

`--max-prompt-tokens`と`--max-context-tokens`は別々に変更できます。256より大きいpromptは、GPU expert bucketing/grouped GEMMとselected-KV DSAが入るまで実験設定として扱ってください。

`--max-tokens`はrequest省略時の既定値であると同時に、各requestのgeneration hard capです。既定では4,096を受理し、4,097以上をHTTP 400で拒否します。

serverは公式Hugging Face [revision `04c4e9e`](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/04c4e9e95c5da8862dced7e5056455116f83a7e0)に固定されています。起動時にはconfig、tokenizer、chat template、index等の既知SHA-256に加え、全62 weight shardを含むcheckpoint content digestを照合します。全payload attestationはM3 Ultra実測で約130–142秒です。

attestation後にtext target全体（実測319.706 GB、297.75 GiB）をunified memoryへmaterializeします。この処理はM3 Ultra実測で約30.6秒です。保存dtypeはFP8/BF16のままであり、BF16 model copyは作りません。一時的なsmoke testだけ常駐化を省く場合は`--no-warm-residency`を指定できます。content attestationは省略されません。

API keyを使う場合は`--api-key`または`GLM53_API_KEY`を指定します。

```bash
curl http://127.0.0.1:8080/health

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-5.3-flash",
    "messages": [{"role": "user", "content": "RustでLRU cacheを実装して"}],
    "temperature": 0,
    "max_tokens": 512,
    "stream": false
  }'
```

## 4. OpenCode

[examples/opencode.json](examples/opencode.json)をプロジェクトの`opencode.json`へコピーするか、既存設定へprovider部分を統合します。serverは安定したmodel alias `glm-5.3-flash`を受け付けます。OpenCodeの`limit.context`は総contextなので、exampleは安全なprompt 256 + output 4,096 = 4,352を広告します。prompt単独の上限はserver側でも強制します。

```bash
opencode --model glm53/glm-5.3-flash
```

API keyを有効にした場合は、exampleの`options.apiKey`を`"{env:GLM53_API_KEY}"`へ変更します。

## Prefix cache（opt-in）

既定ではAPCを無効化しています。通常のBF16 hybrid state/KV cache容量は512 GB構成で十分であり、まず単一sessionの正しさを優先するためです。明示的に検証する場合だけ有効化してください。

```bash
# Direct cacheのRAM APCを有効化
uv run glm53 serve --apc --apc-blocks 256

# compact NoPE cacheのRAM exact-snapshot APC
uv run glm53 serve --apc --apc-blocks 256 \
  --experimental-compact-nope-dsa-cache

# SSD tierも使う（experimental。attested content identityを使用）
uv run glm53 serve --apc --apc-blocks 512 \
  --apc-disk-path /Volumes/SDXC-512/glm53-apc \
  --experimental-disk-apc
```

disk namespaceはcheckpoint全shard、index、tokenizer/chat template、KV codec設定、固定mlx-vlm revision、v4 row-contiguous custom Metal kernel ABI、cache backend、NoPE DSA cache ABIから生成します。Directは`glm53-nope-dsa-v1`、compactはsingle latent、fixed-absolute-capacityのcompact IndexPool v4、kpool4/int64、rollback16/raw19、self-contained APEを明示する`glm53-nope-dsa-v4`です。さらにMoE backendを分離します。packed-decodeはrow-contiguous packed bank ABIとpacked selected decode ABIを含み、packed-groupedはそれらに加えてgrouped kernel ABIと256-route runtime thresholdを含みます。cache backendと3つのMoE backendの全組み合わせが別namespaceです。compact cacheのRAM APCは`state/meta_state` exact snapshotで16-token continuation parityを確認済みです。compact disk APCは未実装のため、`--apc-disk-path`との併用をweight load前にfail closedします。APC自体は既定offです。

`/v1/metrics`と`/health`でqueue、prefill/decode速度、APC状態を確認できます。production decodeは`nested-cache-eval-clear-v1` policyでrecurrent cacheを256 tokenごとにmaterializeします。環境変数の既存値はserver初期化時に上書きし、CLIで任意値は公開しません。`/v1/metrics`の`server.recurrent_state_materialization`にはconfigured interval、完了回数、前回境界からのdecode step、最終境界step、active/cache/peak memoryを記録します。Metal buffer object数は公開APIがないため報告しません。

## 現在の境界

- 対象はM3 Ultra 512 GB、batch 1、text target stackです。
- MTPはcorrectnessと追加weight trafficのgateが未完了のため既定offです。
- text-only runtimeではvision towerをloadしません。
- 1Mはmodel-native上限にすぎません。server既定はprompt 256、総context 16,384です。OpenCode exampleは安全な総context 4,352（prompt 256 + output 4,096）を広告します。
- 既定のbatch-1 decodeはtop-8 expertを3個のMetal kernelへ融合しています。packed-decode opt-inは連続bankを直接読み、4,096-token実測で10.87から12.54 tok/sへ向上しました。
- 既定prefillはCPU expert bucketです。packed-decodeも同じDirect semanticsを維持します。packed-grouped opt-inだけがGPU route sortとgrouped MMAを全42 MoE層へ適用しますが、full-model correctness未合格です。DSA full-KV SDPAは残り、prompt上限256も変更していません。
- 設計レポートの15 tok/s gateに対し、現在の常駐後実測は11.4 tok/sです。API/runtimeとして利用可能ですが、このperformance gateは未達です。

## M3 Ultra 512 GB実測

2026-08-28〜31、このリポジトリの公式checkpointで測定した値です。

| 項目 | 実測 |
|---|---:|
| strict text load | 2.04 s |
| canonical target tensor | 319.706 GB / 297.75 GiB |
| FP8 residency warmup | 30.57 s |
| warm decode token 2–3 | 11.39–11.48 tok/s |
| decode中active memory | 約319.86 GB |
| OpenAI chat completion | HTTP 200（実model、安定alias） |
| deterministic 256-token prefill | 17.49 s / 14.63 tok/s / peak 320.64 GB |
| greedy oracle | 固定prompt、16/128 tokens、各step全vocab logits hash |
| layer 3 packed expert feasibility | 6.752 GiB / pack 0.202 s / peak 327.02 GB / steady +4 bytes |
| layer 3 grouped FP8 MoE, 256 tokens | 111.50 → 22.03 ms / 5.06× / working peak +319.7 MB |
| full-model opt-in grouped prefill, 256 tokens | warm median 5.675 → 2.324 s / 2.442×（2 warmup＋5 samples） |
| full-model opt-in decode | 11.44 → 13.26 tok/s / 1.159× |
| full-model opt-in startup memory | peak 319.742 GB / steady 319.708 GB |
| packed-decode 4,096-token decode | 10.871 → 12.535 tok/s / 1.153× / exact token・logits・final cache state |
| packed-decode synthetic 2k / 256k | 12.437 / 12.002 tok/s / retention 0.965 / Direct比1.143×・1.137× |
| packed-decode 256-token prefill | 5.702 → 5.921 s / +3.85% / full-vocab byte-identical / grouped dispatch 0 |
| packed-decode startup | 178.76 s / peak 327.156 GB / `/health` HTTP 200 |
| fused packed gate+up+SwiGLU decode probe | layer 3/5 exact / MoE 1.199×・1.183× / 2k 1.097× / 4,096 token 13.728 tok/s（runtime gate未達） |
| residual packed decode MoE fusion probe | A/B/C/D exact / 13.811・13.967・13.998・14.133 tok/s / D 70.754 ms（15 tok/s gate未達） |
| compiled packed FFN / FP32 router screen | isolated A/B/C/D 14.158・14.407・14.168・14.358 tok/s / Bのみ14.4 screen通過 / 15 tok/s候補なし |
| full-model replayable Metal capture feasibility | 57分時点130 GiB・resource収集中 / standard decode解析から棄却 / 15分・32 GiB budget固定 |
| bounded Metal System Trace A/B | exact / GPU busy 63.961→63.895 ms/token / idle 7.334→6.044 ms/token / submission -54/16 token |
| stateless decode compile envelope sweep | A/B/C/D exact / 14.401・14.392・14.381・14.398 tok/s / B/C/Dすべてscreen棄却 |
| dynamic GPU idle attribution | idle再構成誤差0% / readback→次tokenがidleの96.5%・95.3% / compile差分の102.6%を帰属 |
| device-resident greedy chain N=1/2/4/8 | exact / 14.191・14.379・14.421・14.348 tok/s / N=4で飽和 / 15 tok/s未達 |
| device-resident greedy chain N=16 | 64-token screenが300秒budget超過 / partial trace削除 / correctness claimなし |
| async_eval readback scope | A-only 7.303 ms / A+B後のA read 12.322 ms / B残り0.236 ms / stream-wide同期 |
| functional stateful decode executable | Tier 0合格 / KDA trace 1回・state exact・host build -55.1% / step 7 gated RMSNorm非exactでTier 1棄却 |
| compiled KDA numerical barrier | A/C/E 64/64 exact / B/D 62/64 exact / 最初のbit差はcompiled sigmoid FP32 2 ULP / projection独立差なし |
| compiled KDA recurrent readout localization | layer 10/22/25/42を再現 / recurrence・state・tail exact / 最初の差はQ scale FP32 1 ULP |
| exact compiled KDA Q-scale final gate | runtime scalarで34/34層×64 step・公式16/128 exact / 14.632 tok/sで14.7 gate未達・MLX compile停止 |
| resident tensor ownership gate | reusable staging破損を再現・遮断 / 42 bank owned+row-major / 16/128 oracle exact / ready 43.38 s / peak 319.706 GB |
| cache lifecycle / retention policy | 4 class独立accounting / draft 4,096 rotations・target eviction 0 / active pin / RAM APC exact |
| materialization / cache-write ownership | compact・RAM APC・prefill→decodeでA/B/C exact / no-ownerでもvalue生成 / invalid destination atomic / 16/128 oracle exact |
| DSA pooled workspace geometry | 128K/Q256 32 MiB・256K/Q256 64 MiB / 1Mは64 rows×4 blocks / 88境界とtop-k/expand exact |
| hybrid semantic prefix snapshot contract | RAM-only owned snapshot / 1・255・256・257・1023・1024境界×64-step replay exact / transactional capture・restore / final resident 0 |
| KDA state index load/store guards | slot 0/1・sentinel -1 / 全34層 Direct/compact / invalid read/write/restore atomic / 16/128 oracle exact |
| layerwise KDA digest soak screen | 4,096 logical tokens / 21 checkpoints / 34層 A/B exact / rollback 1・8・16 / APC exact / drift 0 |
| layerwise KDA digest soak extended | 256,000 logical tokens / 1,000 materializations / 34層×1,005 checkpoints A/B exact / steady drift 369,900 bytes |
| cumulative state allocation churn screen | 4,096 logical tokens / 51.657 GB allocation / 124 APC ownership cycles / rollback・拒否操作 exact |
| layer-local packed MoE microcapture | layer 3/24/44 × 5 stages完走 / 1層active 7.277 GB / trace 3.17–3.23 GB / full model非resident |
| row-blocked vector KDA、4K/8K/16K | R=4勝者 / R=1比3.063×・2.977×・3.500× / current比1.638×・1.704×・1.999× |
| row-blocked KDA full-model、2K/4K | 46.008→45.954 s / 91.305→91.198 s / 各1.00118×（1.02× gate未達） |
| row-blocked KDA decode | 76.394→76.613 ms / +0.286% / logits・final state exact |
| cumulative hybrid-cache allocation、全4 arm | 各1M physical capacity完走 / cross-arm logits・KDA exact / active drift 311,300 bytes |
| layer 3 NoPE IndexPool, T=2049 | shape 1×1×2049×2051 / unused 2,102,274 / out-of-range 0 |
| kpool4 KV dtype separation, 256k | BF16 268.44 MB / FP8 token 135.27 MB / FP8 group64 142.61 MB / index・mask hash一致 |
| long-context first decode, 256k | Direct/compact/restore logits・DSA output一致 / leaf 112・167 / peak 331.46 GB |
| layer 3 grouped route amplification | local 0.00408 → final 0.18605 / route固定 0.02034（9.15×縮小） |
| layer 5 grouped route amplification | local 0.00426 → final 0.20199 / route固定 0.01830（11.04×縮小） |
| suffix sweep最速screening通過 c=29 | warm median 4.573 s / Direct比1.241× / L2 0.01958 / KL 3.37e-4 |
| Direct-order BM8 parity anchor | 10/10 stage・full logits・全router hash一致 / warm median 6.554 s / Direct比0.866× |
| layer 3 DSA steady decode, 2049 → 256k | 2.331 → 2.614 ms / retention 0.892 / selected幅2051固定 |
| layer 3 DSA pool rebuild, 2049 → 256k | 2.558 → 5.604 ms / retention 0.456 / 256k pool update 3.137 ms |
| all 11 DSA persistent steady, 2049 → 256k | 9.296 → 15.255 ms / retention 0.609 / token 2–16 median |
| all 11 DSA restored first token, 2049 → 256k | 13.891 → 39.704 ms / rebuild追加2.876 → 27.146 ms |
| NoPE single latent storage, 256k | 5,911,347,200 → 2,955,673,600 bytes / 2,955,673,600 bytes削減 |
| NoPE capacity boundary, 256k | dual step256 87.797 ms / single step256 87.595 ms / single preallocated 39.071 ms |
| all-DSA preallocated pool-row, 2049 → 256k | 9.326 → 9.377 ms / aggregate retention 0.994 |
| decomposed IndexPool update, 2049 → 256k | 2.927 → 8.121 ms / retention 0.360 / packed-token append 4.665 ms @256k |
| compact authoritative IndexPool, 256k | 1,483,609,600 → 208,460,549 bytes / 85.95%削減 / raw state 108,889 bytes |
| compact arbitrary rollback | target mod 0/1/2/3 / trim 1–16 / 最大5 pool row / 全11 DSA層byte-identical |
| compact all-DSA dependency chain, 2049 → 256k | 9.525 → 11.705 ms / retention 0.814 |
| production compact full-model decode, 2k → 256k | 11.005 → 10.633 tok/s / retention 0.966 |
| production compact 2k vs Direct | 11.005 vs 10.920 tok/s / 1.008× |
| production compact DSA total, 2k → 256k | 13.444 → 19.877 ms |
| production compact active/peak, 256k | 324.396 / 324.585 GB |
| Metal v4 contiguous ABI, raw → enforced | 11.302 → 11.213 tok/s / 0.796%回帰 / working peak +0 bytes |
| recurrent materialization 50 → 256 | warm median 92.044 → 92.163 ms / +0.129% / active drift 4.79 MB @256 |
| recurrent state 100k soak, interval 256 | 100,000完走 / retention 0.984 / active drift 79.28 MB（64 MiB gate未達） |
| recurrent state 100k fixed capacity | 100,000完走 / retention 0.985 / active drift 23.7 KB / authoritative drift 0 bytes |
| production Direct materialization 50 → 256 | 4,096 token完走 / 11.045 → 11.001 tok/s / 0.400%回帰 / 16 boundaries exact |

再測定コマンド:

```bash
uv run python scripts/bench_fp8.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --warm-residency --tokens 16

uv run python scripts/probe_packed_expert_bank.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --layer 3

uv run python scripts/probe_grouped_fp8_moe.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --layer 3 --warmups 2 --repeats 5

uv run python scripts/probe_packed_grouped_runtime.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --tokens 256

uv run python scripts/probe_packed_decode_runtime.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --output bench-results/m3ultra512-packed-decode-runtime-20260831.json

uv run python scripts/probe_fused_packed_gate_up_swiglu_decode.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --output bench-results/m3ultra512-fused-packed-gate-up-swiglu-decode-20260901.json

uv run python scripts/probe_residual_packed_decode_moe_fusion.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --output bench-results/m3ultra512-residual-packed-decode-moe-fusion-20260901.json

uv run python scripts/probe_compiled_packed_ffn_fp32_router.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --output bench-results/m3ultra512-compiled-packed-ffn-fp32-router-20260901.json

uv run python scripts/probe_row_blocked_vector_kda_prefill.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --output bench-results/m3ultra512-row-blocked-vector-kda-prefill-20260831.json

uv run python scripts/soak_cumulative_hybrid_allocation_1m.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --output bench-results/m3ultra512-cumulative-hybrid-allocation-1m-20260831.json

uv run python scripts/localize_grouped_fp8_divergence.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --tokens 256 --warmups 2 --repeats 5 \
  --output bench-results/m3ultra512-grouped-fp8-divergence-20260828.json

uv run python scripts/probe_nope_indexpool_safety.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --layer 3 \
  --output bench-results/m3ultra512-nope-indexpool-safety-20260828.json

uv run python scripts/probe_kpool4_kv_dtype_separation.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --layer 3 \
  --output bench-results/m3ultra512-kpool4-kv-dtype-separation-20260831.json

uv run python scripts/trace_grouped_fp8_route_amplification.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --tokens 256 \
  --output bench-results/m3ultra512-grouped-fp8-route-amplification-20260828.json

uv run python scripts/sweep_grouped_fp8_suffix.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --tokens 256 --warmups 2 --repeats 5 \
  --output bench-results/m3ultra512-grouped-fp8-suffix-sweep-20260829.json

uv run python scripts/probe_grouped_fp8_parity_ladder.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --tokens 256 --warmups 2 --repeats 5 \
  --output bench-results/m3ultra512-grouped-fp8-parity-ladder-20260829.json

uv run python scripts/probe_long_context_dsa_decode_frontier.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --layer 3 --warmups 2 --repeats 5 \
  --output bench-results/m3ultra512-long-context-dsa-decode-frontier-20260829.json

uv run python scripts/probe_persistent_all_dsa_session_frontier.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --warmup-steps 2 \
  --output bench-results/m3ultra512-persistent-all-dsa-session-frontier-20260829.json

uv run python scripts/probe_single_buffer_nope_latent_cache_frontier.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --warmup-steps 4 --measured-steps 16 \
  --output bench-results/m3ultra512-single-buffer-nope-latent-cache-frontier-20260829.json

uv run python scripts/probe_incremental_indexpool_update_copies.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --warmup-steps 4 --measured-steps 16 \
  --output bench-results/m3ultra512-incremental-indexpool-update-copies-20260829.json

uv run python scripts/probe_compact_authoritative_indexpool_state.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --warmup-steps 4 --measured-steps 16 \
  --output bench-results/m3ultra512-compact-indexpool-arbitrary-rollback-20260830.json

uv run python scripts/probe_compact_nope_dsa_runtime.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --output bench-results/m3ultra512-compact-nope-dsa-runtime-20260830.json

uv run python scripts/probe_metal_input_layout_abi.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --output bench-results/m3ultra512-metal-input-layout-abi-20260830.json

uv run python scripts/probe_recurrent_state_materialization_frontier.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --output bench-results/m3ultra512-recurrent-state-materialization-frontier-20260830.json

uv run python scripts/soak_recurrent_state_100k.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --output bench-results/m3ultra512-recurrent-state-100k-fixed-capacity-20260830.json

uv run python scripts/probe_bounded_recurrent_materialization.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --output bench-results/m3ultra512-bounded-recurrent-materialization-20260831.json
```

### Cumulative hybrid-cache allocation soak

Direct/packed-decode MoE × Direct/compact cacheの4 armを、Metal allocatorを共有しない別processで測定しました。fresh cacheへ1 tokenを実forwardし、実際に確保されたDSA physical capacityを累積して各arm 1M以上まで継続しています。Direct cacheは256-token粒度で3,907 cache・1,000,192 capacity、compact cacheは4,352-token粒度で230 cache・1,000,960 capacityでした。

0/100k/500k/1Mの固定prompt final logits、first-decode logits/token、全cache state、KDA/DSA digestは各armのbaselineとexact一致し、cross-armでもlogits/tokenとKDA stateが一致しました。KDA stateは全cycle・全armで147,619,840 bytes、Direct/compactのallocation granularityも同一です。state leafはDirect 112、compact 167で固定、NaN・positive OOB・Metal error・live cache残留は0でした。

全armのpost-clear active driftは311,300 bytes、cache driftは0、peakは最大320.006 GBです。fixed-prompt first-decode latency retentionはDirect/Direct 0.996、packed/Direct 0.983、Direct/compact 0.996、packed/compact 1.013で、全armが0.95 gateを通過しました。これによりpacked-decode＋Direct cacheはdefault昇格候補です。compact cacheはbatch 1制約があるためsingle-session opt-inを維持します。このsoakはprobe-onlyで、既定backend、cache ABI、APC identity、server admissionを変更していません。

### Packed expert bank feasibility

GPU MoE prefillの前提確認として、実checkpointのlayer 3だけを4個の連続bufferへin-memory packingしました。`gate_up_weight`と`down_weight`はuint8 E4M3、scaleはFP32のままで、BF16展開はありません。全288 expert × 6 tensor = 1,728 sliceが元tensorとbyte-identicalで、既存selected top-8出力もbit-identicalでした。

全model常駐状態のactive memoryは319.706 GB、pack中peakは327.023 GBでした。module参照の切替、旧expert解放、`mx.clear_cache()`後は319.706 GBへ戻り、baselineとの差は4 bytesです。したがって層単位移行で元mmap-backed tensorを解放でき、定常的なexpert bank二重化を避けられることを確認しました。これはfeasibility probeであり、serverとloaderの既定経路はまだ変更していません。

0.202秒は既にresidentなtensorから1層をpackする時間です。attestation、checkpoint load、全weight residencyを含むserver-ready時間とは分離して扱います。

### Opt-in packed decode MoE runtime

`--experimental-packed-decode-moe`は全42 routed MoE層を同じ連続bankへlayerwiseに移行します。batch-1 decodeはbankから選択top-8だけを読む3個のMetal kernelを使い、prefillはfull-bank addressingでDirectと同じGEMV/tiled-GEMM・weighted BF16 reduction順を再現します。expert IDはruntime descriptorなのでexpert数に比例したMetal variantを生成せず、forward中に小さなdescriptor arrayも割り当てません。grouped kernelはどのsequence長でも呼びません。

prompt 1/16/128/256のfull-vocab logits、4,096-token decodeの全tokenと指定step logits、最終KDA/DSA state、RAM APC continuation、256k synthetic-cache continuationはDirectとexact一致しました。4,096-token compact decodeは10.871から12.535 tok/sへ1.153×向上し、materialization 16回、active drift 2.05 MBです。synthetic-cache decodeは2kで12.437 tok/s、256kで12.002 tok/s、retention 0.965でした。256-token prefillは2 warmup＋5 samplesのmedianで5.702から5.921秒、回帰3.85%で5% gate内です。

全層の旧expertは解放され、steady weightはuint8 E4M3＋FP32 scaleの304.480 GB、BF16 weight展開はありません。install peakは327.156 GB、opt-in compact serverは178.76秒でreadyとなり`/health` HTTP 200を返しました。このbackendはexperimental opt-inのままで、既定Direct backend、prompt 256、総context 16,384、compact cache ABIは変更していません。

### Fused packed gate+up+SwiGLU decode probe

packed batch-1 decodeのgate投影とup投影を同じMetal dispatchで計算し、SwiGLUまでkernel内で完了するprobeを追加しました。down投影は既存packed kernelを使うため、routed expert部分は3 projection dispatchから2 dispatchになります。MLX v0.32.2のSigmoidと同じ`abs(x)`＋符号分岐の安定化式、BF16 projection store境界、BF16 activation演算木を再現しています。weightはpacked uint8 E4M3、scaleはFP32のままで、BF16 weight展開はありません。

公式checkpointのlayer 3/5ではrouter ID・score、gate、up、activated hidden、down、weighted route、routed output、shared加算後の最終MoE outputがすべてbyte-identicalでした。full-model synthetic 2kの全logits hash、4,096-tokenの全生成token、指定stepのfull-vocab logits hash、最終KDA/DSA stateも既存packed pathとexact一致し、materializationは両armとも16回です。

一方、selected routed MoE speedupはlayer 3/5で1.199×・1.183×となり、最小1.20× gateへ届きませんでした。full-model 2kは1.097×、4,096-token decodeは12.566から13.728 tok/s（1.093×）で、1.12×および14 tok/s gateも未達です。したがってこれはexact correctness anchorとして保存しますが、runtime、kernel ABI、server、APC identity、admissionには導入しません。

### Residual packed decode MoE fusion probe

d99fのexact gate+up+SwiGLU融合を全armの非production baselineとし、残るdown集約とshared expertを直交分離しました。Aは既存down集約＋既存shared、Bはcustom down集約、Cはshared gate+up+SwiGLU融合、DはB+Cです。down集約は、既存BF16 downを読むB1と、down kernel内でFP32 score乗算まで行うB2を先に比較しました。両方ともraw down、weighted FP32、reduced FP32、final BF16がexactで、代表層の合計medianが短いB1をfull-model armへ採用しました。

layer 3/5ではB1/B2の全集約境界、shared gate/up/activated hidden/down、A/B/C/Dの最終MoE出力がbyte-identicalです。2k full-model logits hash、4,096-tokenの全生成token、全evidence logits hash、最終KDA/DSA stateも全armでAとexact一致し、materializationは各16回、NaNとMetal errorは0でした。

4,096-token medianはA 72.407 ms / 13.811 tok/s、B 71.597 ms / 13.967 tok/s、C 71.438 ms / 13.998 tok/s、D 70.754 ms / 14.133 tok/sです。B1はA比1.011×、shared融合は1.014×、組合せDは1.023×でした。代表層のshared単体は0.361〜0.364 msから0.316〜0.319 msへの1.129〜1.149×であり、以前のrouted+shared差分から推定した0.12〜0.13 ms/layerはshared単独critical-path costを過大評価していました。Dは15 tok/sの66.667 msまで4.087 ms残すため、runtime、kernel ABI、server、APC identity、admissionへは導入しません。今後の性能probeではDをexactな非production baselineとして使います。

### Compiled packed FFN and resident FP32 router probe

aad32b1-DのMoE内部を固定し、A=eager FFN＋BF16 router、B=compiled FFN＋BF16 router、C=eager FFN＋resident FP32 router、D=compiled FFN＋resident FP32 routerを、それぞれMetal allocatorと`mx.compile` cacheを共有しない4つのprocessで測定しました。compiled armはmodel loadとD path確定後に42 sparse layerの`_ffn_c`をresetし、one-token warm forwardで全42層をcompileしてから2k synthetic-cache screenへ進みます。

layer 3/5のrouter raw logits、selected indices、routing scores、compiled MoE output、HC expand outputは全armでbyte-identicalです。2k full-vocab logits hashも4 processで一致し、NaNはありません。screenはA 14.158、B 14.407、C 14.168、D 14.358 tok/sでした。BはA比1.0175×で14.4候補保持ラインを通過しましたが、Cは1.0007×でrouter cast除去の実効利益がありません。Dも1.0141×に留まり、resident FP32 routerとの正の相乗効果は見られませんでした。

42層のcompiled warmupはB 75.3 ms、D 74.2 msで、compiled graphのsteady active増分は152,587,911 bytesです。resident routerは42層合計99,090,432 bytesのBF16を198,180,864 bytesのFP32へ置換し、active増分は91,977,436 bytesでした。15 tok/sを超えるarmがなかったため、設計どおり4,096-token、256k synthetic continuation、late/early retentionのqualificationは実行していません。Bは次のMetal captureで比較するprofiling候補として保持しますが、runtime、kernel ABI、server、APC identity、admissionへは導入せず、exact nonproduction baselineはaad32b1-Dのままです。

### Steady packed decode Metal capture

aad32b1-DをMoE内部のexact nonproduction baselineとして固定し、A=eager sparse FFN shell、B=`mx.compile(_ffn_block)` sparse FFN shellだけを比較するcapture scriptを追加しました。FP32 routerは含めません。各armは別processで公式checkpointをloadし、2049-token Direct synthetic cache、2 greedy warmupの後、実agent経路と同じfull-vocab logits、argmax、CPU readback、次token構築を含む8 tokenをcaptureします。`mx.clear_cache()`、256-token materialization、hash、memory probe、loggingはcapture区間外です。開始時の物理capacityは2304 tokenなので、2051→2059のcapture中にcache growthはありません。

`.gputrace`はrepo外の新規pathへだけ書け、既存pathとrepo内pathをfail closedで拒否します。Xcode bundleは相対path、file size、全file内容をcanonical SHA-256化し、repoにはidentity、full-vocab hash、post-cache state hash、capacity不変証拠、手動Dependencies解析欄だけを残します。A/B両方が別PIDで完走し、token、logits、cache stateがexact一致した場合だけartifactをcompleteにします。

この320B all-resident modelではGPUToolsが再生用Metal bufferをdownload/content-address化する固定費が大きく、Aの初回試行は57分時点で130 GiBに達しても最初のcapture evalを処理中でした。interactive turn内では停止し、再生不能なpartial bundleを削除しました。この結果をfull-model replayable `.gputrace`方式のnegative feasibility evidenceとし、A/B完走は行いません。capture processのwall latencyにはGPUTools overheadが入るため性能値には使えません。

同じ事故を防ぐため、full-model captureには外部supervisorを追加しました。既定budgetは15分、trace 32 GiB、空き64 GiBです。main processがblocking `mx.eval`内でもprocess groupを監視し、超過時はTERM/KILL、partial trace削除、`capture_complete=false`・`correctness_claim=false`の小さなnegative JSON保存を行います。今回の57分・130 GiB結果も`m3ultra512-full-model-gputrace-negative-evidence-20260901.json`へ固定しました。

```bash
MTL_CAPTURE_ENABLED=1 uv run python \
  scripts/capture_steady_packed_decode_critical_path.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --arm A \
  --trace /private/tmp/glm53-steady-packed-decode-A-20260901.gputrace

MTL_CAPTURE_ENABLED=1 uv run python \
  scripts/capture_steady_packed_decode_critical_path.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --arm B \
  --trace /private/tmp/glm53-steady-packed-decode-B-20260901.gputrace
```

### Bounded packed decode telemetry

whole-modelは再生用resourceを保存しないXcode `Metal System Trace`へ切り替えました。A=eager FFN shell、B=compiled FFN shellを別processでloadし、2049-token Direct synthetic cache＋2 warmupから16 tokenだけを8秒System Traceへ記録します。その後traceを停止して同じchildで272 tokenまで継続し、step 256のmaterialization、p50/p95、full-vocab hash、最終cache stateを取得します。DSA physical capacityは事前に2560 tokenへ予約し、capacity growthを測定から除外しています。

A/Bの全生成token、step 1/16/256/272 full-vocab hash、post-cache stateはexact一致し、capacityも不変でした。Aはp50 70.628 ms、Bは69.458 msです。16-token System TraceではAのGPU busy/idleが63.961/7.334 ms/token、Bが63.895/6.044 ms/tokenでした。GPU intervalは両方3104件で不変、command-buffer submissionは5952から5898へ54件減りました。したがって`mx.compile(_ffn_block)`の約1.29 ms/tokenの利益は算術kernel短縮ではなく、command submissionとGPU idle削減です。BのGPU busyは既に63.895 ms/tokenなので15 tok/sの66.667 msまで算術上の余白はありますが、idle 6.044 ms/tokenの約半分をさらに消す必要があります。

非再生traceはA 59,816,627 bytes、B 61,047,931 bytesで、full-model replayable captureの130 GiBより約2,200分の1です。trace本体はrepo外に置き、canonical SHA-256とXML export集計だけを`m3ultra512-packed-decode-bounded-telemetry-20260901.json`へ保存しています。

### Stateless decode compilation envelope sweep

現行compiled sparse FFNをAとし、cache-free tensor graphだけを広げるB/C/Dを別processで比較しました。Bはattention出力から現層FFN完了まで、Cは現層FFNから次層attentionのHC collapse＋input normまで、Dは両者を単一compiled callableへ含めます。KDA/DSA attention本体、cache object、logical offset、materialization counter、APC stateは全armでcompile外です。dense layer 0–2も既定eagerのままです。

2049-context Direct cache、2 warmup、16-token bounded System Trace、272-token continuationで、A/B/C/Dのp50は69.439/69.485/69.539/69.456 ms、14.401/14.392/14.381/14.398 tok/sでした。GPU busyは63.776/64.010/64.049/63.879 ms/token、idleは6.147/6.057/6.229/6.061 ms/tokenです。最大のcommand-buffer削減はDの1.438件/token、idle削減はBの0.090 ms/tokenで、0.75 ms・2件/token gateには全arm届きませんでした。

生成token、step 1/16/255/256/257/272のfull-vocab logits、最終cache stateは全arm exactです。Direct/compact cacheの固定3-token differentialも一致し、NaN/Metal errorは0、step 256 materialization、capacity、state leaf、idle時state不変、64 MiB drift、512 MiB working peakの各gateも合格しました。xctrace attach直後のlatencyは約2.9–3.1秒の観測overheadを含むためfirst-token evidenceに使わず、attach前のsynthetic restore first decode 76.4–77.1 msを別記録しています。MLXにはcompile-cache/retrace数の公開APIがないため回数は推定せず、固定shape signature `[1,1]`とfresh-cache warmup hash一致だけを保存します。

従ってB/C/Dはexact診断anchorとして保存し、E/F、4096-token、256k、prefill qualificationへは進めません。現行FFN compileより広いlayer-local envelopeでは残り2.74 ms/tokenを回収できないことが確定しました。そこで、次節では新しいtraceを取らずに、bounded telemetryのapplication/driver intervalを動的な境界へ帰属します。

### Dynamic GPU idle boundary attribution

新しいfull-model traceは取得せず、bounded telemetryの正本A/Bを再解析しました。`metal-gpu-intervals`のcommand-buffer IDを`metal-application-command-buffer-submissions`へjoinし、各gapを「前GPU interval終了から次commitまで」のapplication/runtime starvationと、「commit済みから次GPU開始まで」のdriver/dependencyへ分離しています。重複GPU intervalをmergeした再構成idleはA 117,345,521 ns、B 96,697,006 nsで、既存System Trace集計との差は両armとも0%です。

| dynamic boundary | A app ms/token | A driver ms/token | A total ms/token | B app ms/token | B driver ms/token | B total ms/token |
|---|---:|---:|---:|---:|---:|---:|
| argmax/readback → next-token submission | 6.323 | 0.758 | 7.080 | 5.048 | 0.709 | 5.757 |
| within-token unclassified | 0.000 | 0.222 | 0.222 | 0.000 | 0.258 | 0.258 |
| capture startup / queue fill | 0.000 | 0.032 | 0.032 | 0.000 | 0.029 | 0.029 |

16 tokenには15個の周期的な長いtoken境界があり、A/B idleの96.54%/95.25%を占めます。この境界はA→Bで1.3235 ms/token短縮し、全idle短縮1.2905 ms/tokenの102.6%を説明します。100%を超えるのは、Bの小さなwithin-token gapが約0.033 ms/token悪化したためです。短縮のうち約1.2745 ms/tokenはapplication starvation、約0.0489 ms/tokenはdriver/dependencyです。従ってcompiled FFNの利益はGPU算術の短縮ではなく、CPU readback後に次token graphがMetalへ到達するまでの待ち時間短縮です。

このSystem Traceにはdynamic command-buffer/frame IDはありますが、shader dispatch intervalは記録されていません。static metallib inventoryは実dispatchの証拠に使わず、KDA/DSA、attention→FFN、LM head→argmaxの内部境界は推測していません。明示的なwithin-token unclassified budgetはA 3.03%、B 4.27%で10% gate内です。次の性能候補は一般的なcompile envelope拡大ではなく、token IDをdevice-residentのまま次forwardへ渡し、streaming用CPU readbackを遅延・集約する独立probeです。runtime、server、APC、cache ABI、admissionは変更していません。

### Device-resident greedy token chain probe

前節の周期境界を介入実験で分解しました。exact packed residual-D＋compiled sparse FFNを固定し、argmax tensorをPython整数へ変換せず次のembedding入力へ渡すautoregressive chainをN=1/2/4/8で別process測定しました。CPU readbackはchain末尾のstack一回だけで、64 tokenあたり64/32/16/8回です。これはspeculative decodingではなく、各stepでtarget modelのexact greedy tokenを使う通常のautoregressive unrollです。

| chain | tok/s | wall ms/token | GPU busy | GPU idle | app starvation | driver/dependency | readbacks | stream silence p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.191 | 70.469 | 64.014 | 6.743 | 5.712 | 0.832 | 64 | 72.3 ms |
| 2 | 14.379 | 69.545 | 63.756 | 5.915 | 5.240 | 0.457 | 32 | 142.7 ms |
| 4 | 14.421 | 69.344 | 63.853 | 5.547 | 5.083 | 0.245 | 16 | 291.3 ms |
| 8 | 14.348 | 69.696 | 63.936 | 5.468 | 5.117 | 0.133 | 8 | 580.0 ms |

N=1–8の64 step全生成token、全step full-vocab logits、最終cache stateはexactです。Direct/compact differential、255/256/257 materialization境界、capacity、state leaf、NaN/Metal errorも合格しました。N=2/4は全stop位置についてRAM APC snapshotを不変のまま保持し、EOS、stop token、multi-token stop、cancel、generation/context cap後にrestore＋accepted prefix replayしたcacheがN=1 oracleとexact一致します。N=2のworking peakは61.4 MB、active driftは-1.69 MB、stream silence p95は142.7 msです。

長いapplication starvation gapの個数は64/32/16/8でreadback回数と一致しますが、総application starvationは5.712/5.240/5.083/5.117 ms/tokenとほぼ残りました。chainを広げるほど一つのgap中に次chunkのPython/MLX graph構築時間が集約されるため、driver/readback部分だけが減り、submission starvationの本体は消えません。従って前節の`argmax/readback → next-token submission`は「readback専用コスト」ではなく、readbackと次forward graph構築を合わせた動的境界だったと厳密化します。

N=16は64-token bounded screenが300秒を超えたためprocess groupを停止し、不完全な430 MB traceを削除しました。correctness/performance claimは行わず、full-model lazy dependency graphのresource frontierというnegative evidenceだけを保存しています。N=2が15 tok/s gateを通らなかったため4,096-token、256k continuation、RAM APC/server streaming qualificationには進めず、全chain armをruntimeへ昇格しません。次の候補はreadback遅延ではなく、約5.1 ms/token残るPython/MLX autoregressive graph constructionをC++/Metal側のstateful実行単位へ移すか、MLX command submissionを非同期に重ねる独立probeです。runtime、server、APC、cache ABI、admissionは変更していません。

### Asynchronous autoregressive submission gate

full-model lookaheadの前提となるMLX readback契約を、独立したBF16 4096×4096 GEMM A/Bと5-sample Metal System Traceで検証しました。Aだけを`async_eval`してreadbackするcontrolと、A、Bの順に`async_eval`してからA、Bをreadbackするpairを同じdefault Metal streamで比較しています。

host中央値はA-only wait 7.303 ms、A/B submit後のA readback wait 12.322 ms、直後のB remaining wait 0.236 msでした。A readbackはcontrolの1.687×を待ち、Bに残るのはcontrolの3.23%だけです。Traceには5組すべてでA/Bの2 GPU frameがあり、A-only GPU busy中央値6.069 msに対してA+Bは11.455 ms（1.888×）、B frameはA frameより中央値0.289 ms後に終了しています。従って`a.item()`はA固有eventだけでなく、同じstreamへ後から投入済みのB完了まで待つstream-wide同期です。

Tier 1がevent-scoped gateを満たさないため、full-model async_eval arm、in-flight cache version、stop rollback、performance測定は実行していません。この条件ではone-step lookaheadを先にsubmitしても現在tokenのreadbackが未来tokenまで待ち、通常stream cadenceを維持できません。MLX Python schedulingによるchain/lookahead探索はここで終了し、次工程は45-layer forward＋state update＋LM head＋argmaxを再利用可能なfunctional stateful decode executableへ固定できるかのfeasibility gateです。runtime、server、APC、cache ABI、admissionは変更していません。

### Functional stateful decode executable feasibility

full modelをcompileする前に、公式checkpointのlayer 0 KDAだけを該当shardから選択loadし、state schema、prefill→decode transition、fixed-signature compileを段階評価しました。Tier 0ではconv `[1,3,24576]` BF16とrecurrent `[1,64,128,128]` FP32を明示入力・出力にし、不正leaf数・conv shape・recurrent shapeの3 caseをstate不変のままfail closedしました。4-token prefill後のeager decodeとfunctional decodeはoutput、conv、recurrent stateがbyte-identicalです。

Tier 1ではlogical positionをint32 runtime tensorとして渡し、offset 0/1/255/256/2048の全点で同一compile signatureのまま正確に+1しました。Python trace counterは1回、64-stepのconv/recurrent stateは全step exact、strided/contiguous入力も一致し、warm host graph-build中央値はeager比で55.1%減りました。process-first compile callは1秒未満でしたが、MLXのpersistent compiler cacheは消去していないため完全cold値とは主張しません。active/peakは約0.28/0.35 GBで、full model、DSA、MoE bankはロードしていません。

しかしfull KDA outputはstep 7で初めてbyte identityを失いました。recurrence outputと次stateはexactですが、gated RMSNormの8,192要素中1要素が`2.384e-7`、最終projectionの4,096要素中1要素が`2.980e-8`異なります。gated normとrecurrenceを明示auxiliary outputにしても差は残りました。eager同士とcompiled同士はそれぞれ全64 step byte-identicalなので、Metal recurrenceの不定性ではなく`mx.compile`がeagerのgated-RMSNorm演算境界を変えたsystematic differenceです。

これは事前定義した「compile fusionで数値境界が変わる」停止条件です。誤差許容値を後から緩めずTier 1を棄却し、Tier 2 DSA、complete layer、full token→argmax executableはロード・実行していません。従ってMLX-native full stateful executableで15 tok/sを狙う経路はここで終了し、次候補は阻害境界だけをnative primitive化するか、MLX C++ extension/独自executorへ移す工程です。runtime、server、APC、cache ABI、admissionは変更していません。

### Compiled KDA numerical barrier localization

同じ公式layer 0 KDA recurrenceを64 step進め、そのpost-state出力だけをA=eager norm＋eager projection、B=compiled norm＋compiled projection、C=materialized eager norm＋compiled projection、D=materialized compiled norm＋eager projection、E=materialized eager norm＋eager projectionへ分けました。A/C/Eはnormとfinal projectionが64/64 byte-identicalです。B/Dだけが62/64 exactで、zero-based step 6から同じ2 stepで分岐しました。CがexactでDが分岐するため、final projectionは独立blockerではなく、compiled gated RMSNormの差を下流へ伝えているだけです。full compiled KDAのconv/recurrent stateは引き続き64 step exact、全compiled callableのPython trace countは1です。

gated RMSNormをinput FP32、square、mean reduction、rsqrt、normalize、weight、gate FP32、sigmoid、FP32 gate multiply、BF16 roundingへ分解しました。square/reduction/rsqrt/weightまでは全fixtureでexactです。最初のbit差はstep 0から`sigmoid_gate`にあり、compiled値はeagerから最大2 FP32 ULP、続くFP32 gate multiplyも最大2 ULP異なります。多くのstepではBF16 castで吸収されますが、zero-based step 6では8,192要素中1要素が1 BF16 ULP境界を越え、final projectionの4,096要素中1要素へ伝播しました。step 0/1/6/7/8/63についてreference/compiledの値、raw bit pattern、ULP距離をartifactへ保存しています。

従って観測blocker数は1で、次の候補はeagerと同じsigmoid順序を含むopaqueなexact gated RMSNorm primitiveです。primitive実装前の因果分離だけを行ったcommitであり、runtime、kernel ABI、server、APC、cache ABI、admissionは変更していません。

### Exact sigmoid gate Metal barrier probe

公式MLX v0.32.2のsigmoid演算順をcustom Metalへ移し、B=sigmoidだけをopaque化、C=gated RMSNorm全体を融合する2候補を比較しました。custom JITの既定`metal::exp`では正本と最大2 FP32 ULPずれましたが、同じsign-tail式を`metal::precise::exp`へ固定すると、layer 0の実gate 8,192要素と±0、subnormal、BF16境界近傍を含むsynthetic fixtureの全bitが`mx.sigmoid`と一致しました。B/Cともlayer 0の64-step output、conv/recurrent state、zero-based step 6/31、offset 0/1/255/256/2048、strided入力がexactです。Bのhost graph-build削減は57.0%、Cは59.0%、working peak増分は約4.8 MB / 0 bytesでした。

しかし全34 KDA層へ展開すると、B/Cとも30層だけが64/64 exactで、layer 10/22/25/42が同じstepで分岐しました。失敗4層の最初の差はgated RMSNormより前の`gated_delta_update` recurrent outputにあり、B/Cで同一です。各層の最終conv/recurrent stateは引き続きexactで、eager normをcompiled final projectionへ与えるanchorも全件exactでした。従ってlayer 0での「compiled sigmoidだけがblocker」という結論は全層へ一般化できず、sigmoid-only barrierとfused gated RMSNormはいずれも全KDA exactness gateで棄却します。代表layer 0/20/44のnonzero stateとsnapshot→restore/replayはexactですが、公式16/128-token oracleとfull-token性能はfail-closedで未実行です。runtime、kernel ABI、server、APC、cache ABI、admissionは変更していません。

### Compiled KDA recurrent readout localization

前節で分岐したlayer 10/22/25/42とexact対照layer 0/20/44を64 stepずつ再実行し、A=eager、B=full compiled、C=compiled prefixをmaterializeしてeager recurrence/tail、D=eager prefixをcompiled recurrence wrapperへ入力、E=eager recurrenceをmaterializeしてcompiled exact-sigmoid tailへ入力する5 armで因果を分離しました。失敗4層ではB/Cだけが同じstepで分岐し、D/Eは全64 step exactです。全armの最終conv/recurrent state hashも一致しました。従ってrecurrent Metal kernel、state transition、readout、exact-sigmoid tailは、入力がexactなら独立blockerではありません。compiled prefixをmaterializeするidentity barrierだけでも差は消えません。

Q経路をprojection、FP32 cast、square、sum、rsqrt、L2 normalize、head-dim scale、BF16 roundingへ分解すると、4層すべてで最初の差は`q_l2normalized * 128**-0.5`でした。L2 normalizeまではbyte-identicalですが、scale後の8,192 FP32要素の大部分が最大1 ULPずれ、そのうち1–2要素がBF16境界を越えてrecurrent token outputへ伝播します。最初の分岐はzero-based step 3（layer 10/22/25）とstep 1（layer 42）です。compiled callableは各1 trace、host graph-build削減は全対象で40% gateを維持しました。

従って次候補はrecurrent readout primitiveではなく、FP32定数bit `0x3db504f3`とeagerの乗算・BF16丸め順をopaqueに固定する最小Q-scale barrierです。このcommitは局在化だけを行い、新primitive、runtime、kernel ABI、server、APC、cache ABI、admissionは変更していません。公式oracleとfull-token性能は34層exact候補がないためfail closedで未実行です。

### Exact compiled KDA Q-scale final gate

Q normalization後のscaleをA=eager、B=compiled Python定数、C=bit固定FP32 runtime scalar入力、D=opaque Metal FP32 multiply、E=opaque Metal multiply＋BF16 roundの順で比較しました。基準値は式から再生成せずIEEE binary32 bit pattern `0x3db504f3`として渡します。既知failure layer 10/22/25/42とcontrol layer 0では、Bが保存済みの1 ULP差を再現する一方、C/D/EはいずれもQ-scale FP32、BF16 Q、recurrent output、全step state、gated norm、final projectionまで64/64 step byte-identicalです。最小候補Cを選んだため、新しいMetal primitiveやcommand bufferは不要です。

Cを全34 KDA層へ展開すると34/34層×64/64 stepがbyte-identicalで、各compiled callableのtraceは1回、既知5層のrepeatとstrided/contiguous入力もexactでした。第3のnumerical blockerはありません。host graph-build削減は最小56.8%、中央値58.4%でhard floor 40%を通過しましたが、preferred 60%には届きません。局所working peak増分は最大19.5 MBです。

この結果で公式gateを解除し、packed-decode full modelについてbaseline/candidate双方の16/128-token full-vocab oracleを実行しました。全token、全step logits hashが一致し、34 compiled layerは各1 traceです。さらにexact residual-D、compiled sparse FFN、packed decodeを同時に有効化した2049-context 64-step screenでは、baseline 14.367 tok/sに対してcandidateは14.632 tok/s（1.0185×）、全64 logitsと最終KDA/DSA cacheがexact、working peak増分39.6 MB、NaN 0でした。

ただし事前固定した14.7 tok/s gateへ0.46%届きません。gateを緩めず、bounded System Trace、4,096-token、RAM APC、compact NoPE、256k qualificationには進めません。数値的にはMLX compile方式が成立することを証明しましたが、性能経済性のhard stopによりproduction候補へ昇格せず、この方式の追加opaque barrier探索も終了します。runtime、kernel ABI、server、APC、cache ABI、admissionは変更していません。

### Resident tensor ownership boundaries

外部loaderの再利用staging bufferから得たviewを後段のfusionまで保持すると、次tensorのloadで既存weightが静かに上書きされ得ます。このfailure modeを直接再現するfixtureを追加し、`OWNED`、ownerをresident lifetimeまで保持する`BORROWED_STABLE`、次load後の内容を保証しない`BORROWED_EPHEMERAL`を明示しました。layoutは独立した契約で、row-contiguousなephemeral tensorもresident structureへ直接渡すとfail closedします。

q/kv projection風の同一staging viewでは未所有aliasが上書き後の値へ変わることを再現した一方、明示materializeしたq/kvとfused outputはsource全面上書き・owner解放後もbyte-exactでした。Packed Expert Bankでもuint8 FP8 weightとFP32 scaleを同じ方法で順次上書きし、全4 resident bufferとselected top-8 outputが不変です。bare arrayまたはephemeral leaseをbank constructorへ渡す経路は拒否します。

公式checkpointのpacked-decode実機gateでは42/42 bankの4 bufferすべてが`owned + row-major-contiguous`で、16/128-tokenの全step full-vocab oracleが一致しました。load＋residencyは43.38秒、startup peakは319,706,119,424 bytesで、既存340 GB gateと旧packed startup peak比64 MiB非回帰gateを通過しています。公式mmap parameterは配列自身をownerとして保持するstable borrow、derived q/k/v fusionとNoPE projection viewはowned materializationです。checkpoint attestationはsource integrity、このfixtureはload後のlifetime integrityを検証し、runtime ABI、backend policy、APC、admission、serverは変更していません。

### Cache state lifecycle and retention classes

将来のDFlashや長期agent sessionに先立ち、物理allocatorを分割せずlogical lifecycleとaccounting domainを固定しました。`TARGET_PREFIX`はDSA latent/IndexPool prefix identityに属する`LONG_REUSE`、`ACTIVE_RECURRENT`は現在requestにpinする`PINNED_REQUEST`、`SNAPSHOT_STATE`はAPC ownerだけが管理する`SNAPSHOT_OWNED`、`DRAFT_TRANSIENT`は自身のhard budget内で循環する`SHORT_BOUNDED`です。lifecycleとretentionは別enum・別fieldで明示し、dtype、layout、tensor shapeから推定しません。

小型policy simulatorではtarget 100 blocks相当を保持したままdraftを4,096回rotateし、draft residentは32、draft evictionは4,064、target evictionは0でした。target storage identityとdigestは不変です。target pressureとdraft pressureを同時に与えてもactive recurrent storage identity/digestは不変で、active entryはprefix・snapshot・draftの全LRUに入りません。外部pressureによるactive eviction、draft pressureによるtarget eviction、target pressureによるsnapshot/draft evictionはfail closedします。

Snapshot captureはcaller/active storageからowned copyを作り、restoreもsnapshotとは別のowned active storageを作ります。capture後にactive stateを更新し、target eviction、draft rotation、restore、再更新、再restoreを行ってもsnapshot digestと復元stateはbyte-exactでした。snapshot bytesはactive/prefix budgetへ課金されず、APC policyだけがevictできます。実runtimeのRAM APCでも16-step continuation logits、post-state、snapshot immutabilityが一致しています。

Prefix reuse identityにはmodel revision、checkpoint fingerprint、backend policy、attention cache ABI、KDA state ABI、IndexPool ABI、token digestを明示します。同一token列でも前6項目のどれかが異なれば全てmissとなり、accidental sharingはありません。class別にresident/peak bytes、allocation/eviction count、cumulative allocated bytesを記録し、anonymous allocationは0です。このcommitはpolicy/simulatorだけで、DFlash、物理pool、runtime cache implementation、APC namespace、ABI、admission、serverは変更していません。

### State materialization and cache-write ownership

forwardの一時値を計算する必要性と、その値をpersistent cacheへ書くownershipを別軸へ固定しました。`MaterializationRequest(require_value, cache_write_slot)`はdestinationをproducer起動前にpreflightし、slot `None`または`-1`はcache writeだけを抑止します。`require_value=True`ならownerなしでもproducerは必ず一度実行されます。`require_value=False`かつownerなしはallocation-free no-opであり、valueなしのwrite、slot `< -1`、capacity以上、bool/floatの暗黙castはstateへ触れる前に拒否します。clipやmoduloはありません。

actual compact NoPE cache、RAM APC restore、32-token prefillから最初のsparse decodeへのtransitionでA=materialize+write、B=materialize+no-write、C=no materialize+no-writeを比較しました。Aは既存production操作とvalue、cache、selected indexがbyte-exactです。BはAと同じtemporary valueを生成しながらcache offset、physical capacity、resident bytes、state digestを一切変更せず、Cはproducerを起動しません。invalid destinationもproducer call 0、authoritative state不変です。APC snapshotは全arm後もimmutableでした。

packed-decode＋compact-nope-dsaの実checkpointでも公式16/128-token全vocab oracleとRAM APC 16-step continuation、post-state、snapshot immutabilityが一致しました。この契約はtensor ownership、layout、cache lifecycleとは独立で、runtime cache implementation、ABI、APC namespace、backend、admission、serverを変更しません。semantic prefix snapshotはこの直交契約の上に構築します。

### Bounded DSA Indexer workspace geometry

DSA Indexerのkey dimensionをlogical contextではなく`pool_count = quotient + (remainder != 0)`、すなわち`ceil_div(context_tokens, index_kpool=4)`へ固定しました。FP32 pooled-logits面は`query_block_rows × pool_count × 4 bytes`で計画し、64 MiBを上限にquery軸だけを分割します。128K/Q256は32,768 pools・32 MiB・1 block、256K/Q256は65,536 pools・64 MiB・1 block、1M/Q256は262,144 pools・64 rows×4 blocksです。262,144→262,145 tokenでは65,536→65,537 poolsとなり、partial末尾poolをfloor divisionで落としません。

Q=1/63/64/65/127/128/255/256とcontext=1–5、2,047–2,049、4,095–4,097の88組合せでunsplitとrow-blockedを比較し、raw FP32 logits、top-k score、top-k pool index、expanded token index、valid/sentinel位置が全てbyte-exactでした。selected幅はcontextによらず2,051、sentinel以外の範囲外indexは0です。128K/256K operator qualificationも同じ結果で、処理後active-memory driftは0 bytesでした。8K/Q256のone-block fast pathは2 warmup＋5×64反復でreference比0.9977×です。

memory accountingはFP32 logits、exact argsort full-order、top-k score、IndexPool expansion、selected outputをtransientへ、pool keys/indices/validityをpersistent IndexPoolへ分離します。256Kではlogits面64 MiB、full-order index scratch 64 MiBです。Metal allocatorで測ったcandidate全体のworking peakは568.9 MBで、これはsynthetic score式、masked score、exact argsortのbackend working setも含むため64 MiB logits budgetとは別指標です。referenceとcandidateを同時保持するdifferential harness peakは1.133 GBでした。いずれも処理後residentは0へ戻り、256 MiBの`Q × logical_context` logitsを生成する経路やmaximum context基準のresident scratchはありません。

実checkpoint layer 3は`index_kpool=4`、`index_topk=2048`、head dim 128であることを確認し、packed-decode＋compact-nope-dsaの公式16/128-token full-vocab oracleとRAM APC continuationもexactです。このcommitはplanner/operator qualificationだけで、production DSA kernel、cache implementation、ABI、APC namespace、backend、server admissionを変更していません。

### Hybrid semantic prefix snapshot contract

RAM限定の`SemanticPrefixSnapshot`を、個別tensor bundleではなく「prefix直後の完全なsemantic boundary」へ固定しました。identityはcheckpoint revision/fingerprint、MoE/cache backend、attention cache ABI、KDA state ABI、IndexPool ABI、prefix token digestを含みます。boundaryはabsolute/logical token位置、256-token materialization epoch、rollback epoch、34 KDA層のactive slot、11 DSA層のlatent/KVとIndexPool logical extentを一体として保持します。v1ではabsolute positionとlogical prefix lengthの一致を要求し、forward完了・cache update完了・GPU materialize済みのquiescent pointだけをcapture対象とします。

captureはlive stateをmaterializeした後、全45層を別のMLX-owned storageへcloneし、schema・全state digest・component boundaryを再検査してからstoreへ一度だけpublishします。capture中にlive stateが変化した場合や途中componentで失敗した場合、snapshot/accountingは公開されません。restoreもidentityを最初に検証し、全replacementを別storageへ準備してからlive cache handleの参照を一度だけ交換するため、失敗時にKV/KDA/IndexPoolの一部だけが復元される経路はありません。snapshotは`SNAPSHOT_STATE`・`SNAPSHOT_OWNED`でprefix LRUから独立し、restoreしても消費されず、明示deleteまでimmutableです。disk serializationはv1の非対象です。

実checkpointのpacked-decode＋compact-nope-dsaでposition 1/255/256/257/1023/1024を測定し、各境界でuninterrupted、capture-only、capture→8-step別入力mutate→restore→64-step replay、同一snapshotの2回目restoreを比較しました。全64-step full-vocab logits、34層KDA state、11層DSA latent/KV、IndexPool、slot/index metadataがbyte-exactで、NaNは0です。6 snapshotの累積owned allocation 978,948,000 bytesは全て明示releaseされ、最終`SNAPSHOT_STATE` residentは0、anonymous allocationは0でした。公式16/128-token oracleもexactです。このcontract screenはserver API、disk APC、cache ABI、backend、admissionを変更しません。次工程は4K程度のmutate/restoreを反復するsemantic replay qualificationです。

### KDA state index load/store guards

KDAのconv state slot 0とrecurrent state slot 1について、`0 <= index < capacity`だけをアクセス可能とし、`-1`をunused/no-access sentinelへ固定しました。`index < -1`、`index >= capacity`、bool、float、NaN/Inf、暗黙castはstate tensorへ触れる前に拒否します。固定mlx-vlmの`ArraysCache`型とserialized state schemaは変えず、read/writeと`state` materialization/restore propertyへ同じ`KDAStateIndexContract`を適用するため、RAM/disk APCから復元された同型cacheにもguardが残ります。

read、write、materialization source、restore destinationの各境界で`-2`、capacity、capacity+1を注入し、conv/recurrent digest、state index metadata、decode/materialization counter、lifecycle accounting、APC-visible snapshotが完全不変であることを確認しました。`-1`はread/write/allocation/counter mutationなし、capacity-1は正常にread/writeできます。Python int、NumPy int32/int64、MLX int32/int64 scalarだけを受理し、clip、modulo、負index aliasはありません。

rollback 1/2/3/4/8/15/16はsnapshotへexact restoreし、17-token要求とmalformed destinationは全destinationのpreflight後にfail closedします。実checkpointではDirect/compact双方の34/34 KDA層が同じ2-slot guard下にあり、RAM APC 16-step continuationと公式16/128-token全vocab oracleがbyte-exactでした。cache ABI、APC namespace、backend、server admissionは変更していません。次のlong soakではこの契約上でlayer別KDA state digestを記録します。

### Layerwise KDA state digest soak

長時間state driftを最初の観測層へ局在化するため、34 KDA層それぞれのconv、recurrent、slot/index metadataを独立したraw-bit SHA-256へします。BF16はFP32へ変換せずuint16 viewをhashします。256-token production materializationごとに加え、token 0/1/255/257/4095を観測し、materialization前後のdigestも比較します。不一致時のartifactは最初のtoken/checkpoint、layer、state kind、座標、dtype、左右bit pattern、slot、直前digest、materialization count、cumulative allocation、lifecycle別resident bytesを保存して即停止します。

screenは同じteacher-forced token列をA=uninterrupted、B=RAM APC/rollback付きでlockstep実行しました。4,096 logical token、21 checkpoint、34層のconv/recurrent/index digest、全step logits、最終logitsが全てexactです。Bは1/8/16-token rollback→replayとtoken 2,048のRAM APC save/loadを含み、17-token rollbackはKDA/DSA双方でstate変更前に拒否しました。materializationは16回、state leafは167で固定、最初の256境界以降のauthoritative driftは0 bytes、active driftは7.96 MB、peakは320.505 GBです。

observerを同じstateへ反復してもtensor binding、counter、lifecycle accountingは不変で、surviving active allocationは0 bytesでした。256 cadenceのlayerwise診断を無効にしたdecode時間に対する償却overheadはA 0.546%、B 0.544%で1% gate内です。lifecycle accountingは実cache nbytesと全checkpointで一致し、cumulative allocationは2,002,196,730 bytes / 43,520 physical token slots、anonymous allocationは0です。このscreenは100k qualificationの代用ではありません。

100k qualificationは100,000 logical token（A/B合計200,050 model forward）を完走しました。390回のproduction materialization、396 checkpoint、34/34 KDA層のconv/recurrent/index digest、全step logits、最終logitsは全てexactで、first divergence、NaN、invalid index、Metal errorは0です。Bは24回のRAM APC save/load、rollback/replay 1/8/16 tokenを各2回、17-token fail-closedを含み、全eventでsnapshot/state/logitsがexactでした。

authoritative stateはfirst materialization以降1,354,772,633 bytes/armでdrift 0、state leafは167で固定、anonymous allocationは0です。active-memory boundednessは初期lazy residencyを含むtoken 1ではなく、最初のproduction materialization（token 256）以降で評価し、全393観測点の幅は357,456 bytes、peakは325.300 GBでした。observer overheadはA 0.476%、B 0.479%です。したがって100k qualificationは全gate合格で、256k extended soakへ進めます。

100kと256kは数時間のoperator-runなので、同じatomic artifact scriptをユーザー側で実行します。100kは合格済みです。process再開やdisk cache resumeは主張せず、中断時も最後の4,096-token milestoneと`complete=false`を残します。

256k extended artifactでは1000回のmaterializationと全event件数を明示gateにし、最初と最後の10,000-token steady windowからlate throughput retentionを記録します。lifecycle accountingはTARGET_PREFIX / ACTIVE_RECURRENT / SNAPSHOT_STATE / DRAFT_TRANSIENTごとにstart/end/deltaを保存し、resident/peakだけでなくcumulative allocated bytes/tokens、allocation/eviction countを最終summaryへ固定します。256-token進捗行にはelapsed、logical steps/s、推定残時間も出力します。

256k extended qualificationは256,000 logical token、A/B合計512,050 model forwardを13時間36分で完走しました。34/34 KDA層×1,005 checkpoint、全step full-vocab logits、1,000回のmaterialization前後、62回のRAM APC、rollback/replay 1/8/16 token各2回、17-token fail-closedの計69 eventは全てexactです。first divergence、NaN、invalid index、Metal errorは0で、state leafは167、authoritative driftは0 bytesでした。独立した100k qualificationと重なる395 checkpointのKDA digest系列も完全一致します。

steady active-memory driftは369,900 bytes、peakは333.166 GBでgate内です。最初/最後の10,000-token medianによるlate throughput retentionはA 0.9759、B 0.9761、observer overheadはA 0.465%、B 0.468%でした。lifecycle accountingはanonymous allocation 0のまま、35,363,328 cumulative physical token slotsと446,825,650,554 cumulative bytesを経験し、最終residentは6,475,734,066 bytesにboundedしています。うちSNAPSHOT_STATEは440,349,916,488 cumulative bytesをallocateしつつ最終resident 0 bytes、DRAFT_TRANSIENTは全指標0です。これによりtoken-countを軸にしたKDA state-safety qualificationは完了とし、次はlogical token数を抑えてallocation/APC/rollback churnを高密度化する試験へ移ります。

```bash
uv run python scripts/soak_layerwise_kda_state_digests.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --steps 100000 \
  --output bench-results/m3ultra512-layerwise-kda-state-digest-100k-20260902.json

# 100k accepted後のみ
uv run python scripts/soak_layerwise_kda_state_digests.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --steps 256000 \
  --output bench-results/m3ultra512-layerwise-kda-state-digest-256k-20260903.json
```

### Cumulative state allocation churn

token-countとは独立にallocator/lifetime pressureを増やすため、A=通常decodeとB=高密度RAM APC ownership churnを同じ短いteacher-forced token列で比較します。Bはsnapshot capture、owned restore、旧active release、SNAPSHOT_STATE→ACTIVE_RECURRENT/TARGET_PREFIX transfer、snapshot discardを繰り返し、rollback/replay 1/8/16と、wrong identity restore・invalid KDA index・BORROWED_EPHEMERALのresident昇格・rollback 17のfail-closed操作を交互に挿入します。

物理allocationとownership transferを混同しないよう、各lifecycleでallocated/released bytes、allocation/release/eviction count、transfer in/outを独立記録します。各観測点で`allocated + transfer_in - released - transfer_out == resident`を要求し、一時SNAPSHOT_STATEとDRAFT_TRANSIENTは各cycle後にbaselineへ戻します。失敗artifactはlogical tokenだけでなくoperation/allocation sequence、lifecycle、ownership、APC generation、rollback depth、resident before/after、最初のKDA layer/state差を保存します。

ここでいうtemporary baselineは絶対0ではなく、cycle開始時のresident量です。rollback source snapshotを複数token保持している間にAPC churnが重なっても、その既存snapshotは生存しなければなりません。最初のqualification attemptはtoken 1,022でこの正当な348,397,593-byte rollback snapshotをleakと誤認して安全停止しました。このnegative evidenceは`m3ultra512-state-cumulative-allocation-churn-qualification-failed-overlap-20260904.json`へ分離しています。判定をcycle前後同量へ修正し、10 GB developer smokeではtoken 184の重複caseを含む32 cycle、11,389,480,934 cumulative bytesで全gateを再確認しました。

実checkpointのscreenは4,096 logical token、A/B合計8,218 model forward、124 APC ownership cycle、4 rollback/replayを実行し、累積51,656,675,634 bytes / 1,122,816 physical token slotsへ到達しました。17 checkpointのfull-vocab logits、full cache、34層KDA digestは全てexactです。4種の拒否操作は各31回、計124回すべてstate/snapshot/accounting/binding変更前に拒否されました。materializationは16回、authoritative drift 0 bytes、steady active drift 230,015 bytes、final resident 400,439,346 bytes、peak 320.507 GB、anonymous allocationとNaN/Metal errorは0です。

qualificationはlogical tokenを16,384以下に保ったまま、256k state soakの基準446,825,650,554 cumulative bytes以上を要求します。約1時間のoperator-runになるためユーザー側で実行します。

実checkpointのqualificationは3,696.6秒で完走しました。16,384 logical token、A/B合計33,027 model forward、641 APC ownership cycle、32 rollback/replay、64 materializationを実行し、65 checkpointのfull-vocab logits、full cache、34層KDA digestは全てexactです。4種の拒否操作は計641回すべてstate/snapshot/accounting/binding変更前に拒否されました。累積469,639,955,364 bytes / 22,430,720 physical token slotsへ到達し、256k soak基準を超えています。authoritative driftは0 bytes、steady active driftは131,917 bytes、peakは321.272 GB、final lifecycle residentは696,795,186 bytesです。SNAPSHOT_STATEとDRAFT_TRANSIENTの最終residentは0、anonymous allocation、NaN、invalid access、Metal errorも0でした。これにより短いlogical sequenceへ高密度なownership/APC/rollback圧力を加えるallocation-churn qualificationは完了です。

```bash
uv run python scripts/stress_state_cumulative_allocation_churn.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --tier qualification \
  --output bench-results/m3ultra512-state-cumulative-allocation-churn-qualification-20260904.json
```

### Packed decode operator microcaptures

再生可能captureは代表layer 3/24/44へ縮小しました。`load_model()`のlazy mappingから対象MoE層だけをpack/materializeし、full model payloadをresident化しません。各層についてrouter、routed expert、shared expert、最終加算、full FFNの5 stageを別processでcaptureし、15/15 caseが32 GiB・15分budget内で完走しました。

各caseのbankは7,249,526,784 bytes、steady activeは7,277,060,232 bytes、peakは14,526,587,016 bytesです。traceは3.171–3.228 GB、15本合計47,912,152,961 bytesで、capture finalizeは0.859–1.870秒でした。全3層で`ffn-add`と`full-ffn`のoutput hashが一致します。static scanでは5個のGLM custom kernel labelを回収できますが、全stageのresident metallib inventoryが同じため実dispatch本数の証拠には使いません。kernel dispatch、intermediate allocation、stage間gapはrepo外`.gputrace`をXcode event viewで確認する契約です。

この結果から次の最適化対象はweight readそのものではなく、compiled範囲外に残るpure graph/command boundaryです。decoder layer全体のstateful compileには進まず、attention後処理、FFN HC/norm/residual、LM head/readbackを個別にbounded telemetryへ追加し、GPU idleを最も削れる境界だけを選びます。runtime、server、APC、cache ABI、admissionは変更していません。

### Row-blocked vector KDA prefill probe

GLM-5.3のvector gate KDAについて、Dv行を1/2/4/8行ずつ1 SIMD groupで処理するprobe専用Metal kernelを比較しました。q/k、vector-g、betaを行間で共有しながら、各行のrecurrence、`simd_sum`、BF16 output store、FP32 final stateは現行kernelと同じ順序です。公式shape `H=64, Dk=Dv=128`の128〜16,384 tokensで2 warmup＋5 samplesを測り、4K以上のR=1比geomean 3.172×となったR=4を勝者に選びました。R=4のR=1比は4K/8K/16Kで3.063×/2.977×/3.500×、現行kernel比は1.638×/1.704×/1.999×です。

zero/nonzero initial state、maskなし/tail/internal gap、token mod 0/1/2/3、interleaved strided inputは全Rでoutputとfinal stateがbyte-identicalでした。early/middle/late layer 0/20/44の128-tokenと全34 KDA層もoutput、conv state、recurrent stateがexact一致し、R=4を通した公式16/128-token greedy oracleも全tokenと全step full-vocab logits hashが一致しました。working peak増加は0 bytesです。

NoPE fixtureは全11 DSA層で`use_nope=True`、`qk_rope_head_dim=0`、`rotary_emb`不在をassertし、Direct/compactのprefill/decode中の`nn.RoPE`と`mx.fast.rope`呼出しは0でした。一方、IndexPoolのtoken position処理は維持され、2049-token synthetic stateの全selected index/output hashはDirect/compactで一致しています。「位置indexがない」のではなく、「位置埋め込み演算がない」という契約です。

ただしpacked-decode MoE＋Direct cacheのwhole-model prefillは、productionと同じ2048-token chunkで2Kが46.008→45.954秒、4Kが91.305→91.198秒、どちらも1.00118×に留まり、1.02× gateへ届きませんでした。decodeもexactですが76.394→76.613 ms（+0.286%）です。4K one-shotは既存FP8 projectionの1D gridが2^32 threadsへ達するため使わず、serverと同じ2×2048 chunkで測定しました。条件に従い8K/16K whole-modelは未実行です。

したがってR=4はexact KDA correctness/performance anchorとしてartifactへ保存しますが、runtime kernel、ABI、server、admission、既定backendへは導入しません。次はpacked-decode＋compact候補の累積allocation 1M soakであり、KDA単体最適化を15 tok/s経路とは扱いません。

### Sorted grouped FP8 MoE feasibility

layer 3のpacked bankを直接読むprefill専用Metal kernelを実装しました。route planはGPU上のargsort、histogram、prefix sumで構築し、expert境界に揃えた32-route descriptorごとに32×32×32 `simdgroup_matrix` GEMMを実行します。FP8 weightはthreadgroup tileへFP32 decodeするだけで、永続的なBF16 weight展開はありません。hot pathにはNumPy変換、`.item()`、明示的`mx.eval`を含みません。

32/64/128/256/512 tokenの全点でDirectFP8MoEとの`rtol=0.02, atol=0.02` parityに合格し、speedupは2.12× / 3.45× / 4.42× / 5.06× / 5.26×でした。256 tokenのmax/mean/RMS errorは0.01953 / 0.001077 / 0.001972、追加working peakは319.7 MBです。descriptorの実範囲が単一expertだけを含み、全routeを重複なく一度ずつ被覆することをprobeでassertしています。31/32/33-route bucket、zero-count expert、expert 0/287の境界testにも合格しました。

forced-groupedとpacked fallbackを8-route刻みで比較すると、8 routesは0.929×、16 routesは1.337×でした。したがって測定で挟まれた最小dispatch thresholdは16 routesで、それ未満は既存packed pathへfallbackします。要求系列の64/128/192/256 routesも1.64× / 1.50× / 1.74× / 2.35×です。公式checkpointの16-token oracleは全stepのlogits hashが不変で、batch-1 decodeはselected top-8経路を維持しています。

shared expertを独立phaseとして測ると、256 tokenで11.81 msでした。routed gate/upは6.67 ms、downは3.49 msで、shared expertがlayer-local grouped経路の最大phaseです。phaseごとの測定は各境界で同期するため単純加算せず、ボトルネック比較に使用します。

開発中のscalar grouped variantは256 tokenで0.48–0.71×に留まり棄却しました。性能向上はdispatch統合だけではなく、MMA GEMMによるweight reuseで得られています。このprobeもlayer 3限定で、serverとloaderの既定経路は変更していません。

### 全42 MoE層のopt-in統合

`--experimental-packed-grouped-moe`はlayer 3–44を順番にpack、materialize、module交換し、旧288 expertを解放してから次層へ進みます。全42層でpack直後からclear後に減る量とbank容量がともに7,249,526,784 bytesで一致しました。起動peakは319.742 GB、clear後steadyは319.708 GBで、model規模の二重化はありません。既定server backendとprompt上限256は変更していません。

同一processのDirect比較をcold first passとwarm kernelに分離して再測定しました。256-token warm medianはDirect 5.675秒、grouped 2.324秒（2.442×）で、各backendともwarmup 2回後の5 samplesは安定しています。first passも5.698秒と2.327秒でした。ただしgrouped側は同一process内でDirect実行後なので、共通する非MoE kernelのcompile cacheを再利用し得ます。decodeは11.44から13.26 tok/sへ向上しました。early/middle/lateのlayer 3/24/44は`rtol=0.02, atol=0.02` parityに合格しました。17-token prompt＋16 decodeの全step logits hashも一致しましたが、136 routesは256-route threshold未満なので、このoracleが証明するのはpacked fallbackとdecodeの正確性だけです。opt-in serverは全shard attestationから174.1秒でreadyとなり、`/health`はHTTP 200を返しました。

256-token最終logitsはargmax一致、top-10 overlap 9/10ですが、top-10のset/orderは不一致でrelative L2は0.206、max absolute errorは1.234375です。原因は丸め順差と推定していますが品質影響は未評価であり、full-model grouped correctness gateは未合格です。したがってopt-inのまま維持し、既定化やprompt上限引き上げには進みません。backend別APC namespaceも引き続き必須です。

層別localizationでは、全42層をpacked fallbackへ固定した256-token最終logitsがDirectとbyte-identicalでした。groupedを一層だけ有効にすると、early層の誤差が後段で大きく増幅され、relative L2の最大はlayer 5の0.202、argmax不一致はlayer 9で発生しました。late層ほど誤差は小さく、layer 44は0.00293です。累積系列はlayer 3で0.186へ跳ね、layer 5で最大0.229となった後、約0.18〜0.21を非単調に推移しました。特定一層の破損でも42層にわたる滑らかな蓄積でもなく、広いearly-layer sensitivityのパターンです。これは原因の局在化結果であり、品質受入ではありません。

layer 3/5を一層だけgroupedにしたpaired traceでは、target layer入力、attention、mHC、normalized router input、top-8 indices/scoresはbyte-identicalで、差はrouted MoE出力から始まりました。局所relative L2はlayer 3で0.00408、layer 5で0.00426です。次層のattention/mHCはこの差を連続的に伝播し、最初のrouter不一致はそれぞれlayer 4（67/2,048 slots変更）とlayer 6（52/2,048 slots変更）、最初の2倍超増幅も同じ層のrouted MoEで発生しました。T=256の全DSA層はIndexPool short-context bypassです。

自由routingの最終relative L2は0.18605 / 0.20199、KLは0.03363 / 0.03076、top-10 overlapはどちらも9/10でした。targetより後段へDirect基準のindicesとscoresを固定するとrelative L2は0.02034 / 0.01830へ9.15× / 11.04×縮小し、KLは0.000392 / 0.000316、top-10 setは10/10へ戻りました。両targetで同じ因果パターンであり、後続routerのexpert membership・順序とmixture weightを合わせた差が主要な増幅経路です。この実験だけではexpert切替とscore driftの寄与は分離していません。また67/2,048と52/2,048はslot位置の不一致であり、top-8集合が変わったtokenは7/256と4/256です。一方、route固定後も約0.02の連続誤差は残るため、full-model grouped correctness gateは未合格のままです。runtime policy、APC、admission、server既定経路は変更していません。

### Direct-prefix / grouped-suffix sweep

cutoff `c`をlayer 3〜`c-1`はpacked Direct fallback、`c`〜44はgrouped FP8と定義し、c=3〜45の全43点を同じ256-token fixtureとfresh cacheで測定しました。c=3は既存all-grouped logits hashと一致し、c=45はfinal logits、全router indices/scoresともDirectとbyte-identicalです。全cutoffでpacked bankのweightはuint8 E4M3 code、scaleはFP32のままです。

screening条件（argmax一致、top-10 set 10/10、relative L2 ≤0.02、KL ≤5e-4）はc=29とc=31〜45が通過しました。c=30はL2 0.02510で不通過となり、screening指標の非単調性も再確認しました。最速通過点c=29はL2 0.01958、KL 0.000337、warm median 4.573秒で、Direct 5.676秒に対して1.241×です。screening通過かつ1.5×へ届くcutoffは0件なのでsuffix policyはruntimeへ導入せず、APC identityも変更していません。次はgrouped kernelの加算順序、scale適用、tile reductionの数値差を縮める工程です。

router補助armでは、layer 3のfinal L2がfree 0.18605、Direct indices固定＋current score 0.01989、indices＋scores固定 0.02034でした。layer 5は0.20199、0.02032、0.01830です。Direct expert membership/slot orderの固定だけで9.35× / 9.94×縮小し、scoreもDirectへ固定した追加倍率は0.978× / 1.110×でした。したがって主要因は後続expert membership・順序側で、mixture-weight driftの寄与は小さくtarget間で符号も一貫しません。このscreenはrelease correctnessではなく、grouped full-model correctnessは未合格のままです。

### Direct-order grouped parity ladder

現在のBM32 `simdgroup_matrix`経路とDirect expert bucketの間に、probe専用のexpert-aligned BM8 kernelを追加しました。packed uint8 E4M3 weightとFP32 block scaleを直接読み、gate/upを別dispatch、projectionをBF16 storeします。Directの実際の演算木に合わせ、1-route expertはGEMV式`x * FP8 * scale`、複数route expertはtiled-GEMM式`decoded = FP8 * scale; x * decoded`を使います。top-8 reductionもexpert ID昇順で、各weighted contributionをBF16へ丸めてからBF16 scatter-addする順序を再現します。

layer 3/5、32/64/128/256/512 tokensの全10 caseでgate、up、SwiGLU、down、weighted route、Direct-style reduction、shared加算後の最終MoE出力がbyte-identicalでした。512-tokenのBM8単独working peakは最大366,691,302 bytes（349.7 MiB）です。診断時にDirect・BM32・BM8の全stage tensorを同時保持したladder総peak 885,851,194 bytesとは分離して記録しています。

全42層をBM8へ切り替えた256-token実行でも最終logits hashと全router indices/scores hashがDirectと一致し、16-token decode oracleも全token ID・全step logits hash一致です。一方、warm medianはDirect 5.673秒に対してBM8 6.554秒、speedupは0.866×で、1.5×性能gateには届きません。したがってBM8はbyte-identical correctness anchorとして保存し、runtime、server、APC identity、admissionには導入しません。grouped prefill最適化はここで停止し、長context DSAの測定を優先します。DFlash2型external drafterへ着手する場合だけ、8-token verification相当の64 routesをforced BM8で再評価します。既存BM32 grouped correctnessは未合格のままです。

### Long-context DSA decode frontier

公式checkpointの最初のDSA層であるlayer 3について、実weightとdeterministicなlatent/indexer cacheを使うS=1 operator probeを追加しました。2048/2049/8k/16k/32k/64k/128k/256kと、pool tailが0/1/2/3になる32768〜32771を測定しています。長prompt prefillやserver admissionの迂回実行は行わず、履歴cache stateを直接構築します。

2048はdense bypass、2049以降はIndexPoolです。steady incrementalはcomplete poolを再利用してpartial suffixだけを更新し、pool rebuildは`_pool`を外してsession restoreやbatch shape変更後を模擬します。各caseでcompile first-run、2 warmup、5 samplesを分離しました。phase同期を挟むpool update / score / argsort selection / expansion / latent gather / selected attentionの個別時間と、最終outputだけを同期するend-to-end時間は別々に記録しています。

steady end-to-end medianは2049の2.331 msから256kの2.614 msで、sparse-path retentionは0.892です。256kのphase medianはpool update 0.928 ms、score 0.486 ms、selection 0.378 ms、expand 0.485 ms、gather 0.302 ms、attention 0.637 msでした。selected attention幅は全sparse contextで2051に固定され、steady working peakは256kで165,526,102 bytesです。このlayer-local operatorでは、全pool score・full argsort・gather・attentionのいずれにも急激な長context劣化は見つかりませんでした。

一方、pool rebuildは2.558 msから5.604 msへ伸び、retentionは0.456、256kのpool updateは3.137 ms、working peakは577,117,289 bytesです。したがって今回選ぶ次候補は、complete pool stateをsession restore/APCのfirst-class stateとして扱うことです。このcommitではstate schemaやAPC ABIは変更せず、baseline測定と候補選定だけを固定します。

全contextでsteady/rebuildのindex・output hashが一致し、分解operatorと固定mlx-vlmのIndexer/SparseAttentionもbyte-identicalでした。全indexは`-1`または`[0, Kv)`、反復hash一致、NaNなし、pool-tail全境界合格、256k OOMなしです。この結果はlayer 3 operatorの特性であり、製品KPIのfull-model `decode_tps(256k) / decode_tps(2k) >= 0.8`や256kで15 tok/sを証明しません。

後続roadmapには、DSA prefill chunk 512/1024/2048/4096/8192とpool-score scratch、layer共有可能なpage table・row metadata、層別top-k再計算、idle/dummy forwardによるKDA/DSA state mutation禁止を残します。未実装のshared-row-planはcache ABIへ戻しません。DFlash2型external drafterはtarget MLA/KV stateを共有し、独自KV poolを原則持たず、`acceptance_by_position[0..k-1]`をfirst-class benchmarkにします。

### Persistent all-DSA session frontier

全11 DSA層（3, 7, 11, ..., 43）へ独立した実weight、latent cache、Indexer cacheを持たせ、2049 / 32k / 64k / 128k / 256kから16 tokenを連続decodeするoperator probeを追加しました。各stepではcache objectを作り直さず、resident armはtoken 1以前から`_pool`を保持し、restored armはtoken 1だけ`_pool=None`としてfull rebuild後のtoken 2–16で同じpool stateを再利用します。

residentの全11層aggregate token 2–16 medianは2049で9.296 ms、256kで15.255 ms、retentionは0.609です。0.8 decision gateに届かないため、full-model synthetic-cache frontierへはまだ進まず、次はall-DSA steady degradationの局所化です。256kのp95は38.271 msで、context開始時の物理cache容量が尽きるtoken 2では78.037 msまで上がりました。これはIndexPool境界ではなく、11層のlatent/indexer KVCacheが256-token単位で同時に容量拡張されるcopy spikeです。既存prefixは全拡張でbyte-identicalに保持されています。

restoredのfirst tokenは2049で13.891 ms、256kで39.704 ms、residentに対するpool rebuild追加は2.876 / 27.146 msです。token 2–16 medianは9.314 / 15.316 msまでresident相当に戻り、再構築は一回限りでした。256k pool payloadはBF16 184,560,640 bytes、int64 23,070,080 bytes、bool 720,940 bytes、計208,351,660 bytesです。7 GB/sという楽観的なdisk I/Oでも29.765 msが下限なので、128k/256kのscreeningではpool payload永続化より再構築が安価でした。ただし実I/O benchmarkではなく、disk APC ABIは変更していません。RAM APCもzero-copy保持経路が存在しないため候補化していません。

全160 step × 11層でindexは`-1`または`[0, Kv)`、unused slotはすべて`-1`、NaNなし、resident/restoredのindex・output hash一致、pool tail 0/1/2/3一致、cache object identity保持、idle measurementによるstate mutationなしです。256k residentの16-token memory driftは2,960,935,042 bytes、peakは13,814,772,322 bytesでした。これらはDSA operator集合の測定であり、KDA/MoE/lm_headを含むfull-model decode性能ではありません。

### Single-buffer NoPE latent cache frontier

全11 DSA層について、現行dual KVCache / dual preallocated / single latent step256 / single latent preallocatedの4 armを2049 / 128k / 256kで比較しました。warmupはpool-tailの4 shapeをすべて通す4 step、測定はpersistent cacheで16 stepです。非同期end-to-end sessionとは別にphase同期sessionを再生し、全arm・全step・全層で現行KVCacheとのindex/output hashと、manual phase decompositionとのhashがbyte-identicalであることを確認しています。この`ProbeNoPELatentCache`はprobe内だけにあり、runtimeやcache ABIには導入していません。

256kでdual storageは5,911,347,200 bytes、single storageは2,955,673,600 bytesで、2,955,673,600 bytes削減しました。step256 armの16-token memory driftはdual 2,958,169,240 bytesに対してsingle 5,202,062 bytes、working peakは3,465,020,985 bytesから569,959,264 bytesへ低下しています。latent copy bytesも5,905,580,032から2,952,790,016へ半減しました。steady medianはdual 15.438 ms、single 15.650 msで1.014×、5%非劣化gate内です。

ただしsingle-buffer単独では256k capacity-boundary latencyは87.797 msから87.595 msで、実質的なspike低減はありません。測定用の16-token headroomを持つsingle armではlatent copyが0、boundaryは39.071 ms、working peakは509,214,351 bytesです。容量は256-token境界へ切り上げられ、Indexer token/pool stateの拡張は別に残るため、runtime候補は「single NoPE latent buffer + admitted generation全体のhybrid cache capacity reservation」です。16-token固定のpolicyではありません。preallocated steadyもdual比1.009×で5%非劣化gate内です。

single preallocatedの2049→256k retentionは0.625で、0.8 gateには届きません。consumer境界で同期したphase medianでは、latent projection/append 0.690→0.735 ms、Indexer pool update 2.695→7.053 ms、pool score 1.005→2.222 ms、argsort/top-k 0.355→1.430 ms、pool expansion 1.502→1.845 ms、gather 0.975→1.184 ms、selected attention 4.444→4.516 msでした。最大の絶対増加はIndexer pool updateなので、full-model frontierへは進まず、次はこのphaseをtoken append、partial-pool再計算、complete-pool row copyへ分解します。

### Incremental IndexPool update copy decomposition

全11 DSA層のIndexer updateをcurrent-token projection、packed-token append、最大4-token partial-pool recomputation、complete-prefix carry、pool publicationの5区間へ分けました。2049 / 128k / 256kの各contextで4 warmup後に16 tokenを連続実行し、Indexer token cacheとsingle latent cacheには測定区間を越えるcapacityを予約しています。容量拡張は別armで測定し、steady統計へ混ぜていません。

`reference-concat`、`preallocated-pool-row`、`segmented-pool`を比較した結果、preallocated armは全48 case × 全11層でreferenceとのindex/output hashがbyte-identicalでした。all-DSA aggregate medianは9.326 msから9.377 msでretention 0.994です。しかしMLXのslice assignmentは物理的なin-place row更新ではなくfunctional scatterです。256kのphase同期値はprojection 0.545 ms、packed-token append 4.665 ms、partial recomputation 1.505 ms、pool-row scatter 1.372 ms、publication 0.034 msで、5区間合計は8.121 msです。packed-token bufferは約1,483.6 MB、pool bufferは約208.35 MBのprefixを新しいMLX valueへcarryするため、pool-update retentionは0.360に留まりました。

segmented armはimmutable complete prefixをcopyせず、pool carryは0.044 ms、complete-prefix copy bytesは0です。一方、prefixとtailを別matmulでscoreすると48 step case中14 caseでindex order hashがreferenceと一致せず、byte-identical gateを通過しませんでした。attention output hashまで変わったのはその一部ですが、runtime候補にはしません。したがってpreallocated armはaggregate性能を満たすがcopy-free条件で棄却、segmented armはcopy-freeだがexact parity条件で棄却です。次はpacked Indexer token appendをcopy-freeにする状態表現と、Direct-orderを保つsegmented scoreを独立に検証します。runtime、server、APC ABI、admissionは変更していません。pool indicesもint64のままです。

### Compact authoritative IndexPool state

全context分のpacked Indexer token historyを保持せず、contiguous pool buffer、最大3-tokenのactive tail、16-token rollbackを任意のpool位置で復元するjournalを残すprobeを追加しました。raw stateの上限は`16 + index_kpool - 1 = 19` tokenです。初期変換時だけpoolをbyte-preserving leaf bufferへmaterializeし、BF16はuint16経由でbitを維持してfull-history計算graphを切り離します。decode/append/score/trim hot pathにはNumPy変換、`.item()`、明示的`mx.eval`を含みません。scoreへ渡すlogical poolのshapeとrow orderはreferenceと同一です。未完成poolについては、exact score shapeを保つためinvalidなpreview rowをlogical末尾へ反映し、rollback先がpool境界でなければactive tailからそのrowをbyte-identicalに再計算します。staleなfuture rowはcapacity bufferに残ってもlogical sliceへ露出しません。

256kの全11層ではfull packed history 1,483,609,600 bytesに対し、pool 208,351,660 bytes、active tail＋rollback journal 108,889 bytes、合計208,460,549 bytesです。削減率は85.95%で80% gateを通過しました。single NoPE latent logical payloadと合わせた256k authoritative payload概算は3,161,419,525 bytesです。16-token decode後のactive-memory driftはindependent 797,650 bytes、dependency-chained 237,827 bytesで64 MiB gateを十分下回り、token数比例のraw-state増加はありません。

全3 context × 16 step × 11層でindex/output hashとpool keys/indices/validがfull-history oracleとbyte-identicalです。`mx.depends`で11層を直列化したarmも全hash一致し、2049→256k medianは9.525→11.705 ms、critical-path retentionは0.814でした。独立aggregateは8.634→9.426 msです。rollback target mod 0/1/2/3、trim 1/2/3/4/8/15/16、1〜5 pool row横断、capacity境界前後の30 caseを全11層で測定し、各caseでtrim→replay→再trimを実行しました。60 roundの全stepでpool keys/indices/valid、selected index、attention output、replay後stateがoracleとbyte-identicalです。17-token trimは全11層でfail closedし、stateを変更しません。

tail/journal appendは全contextで114,620 bytesと一定で、phase retentionは0.991です。一方、contiguous pool末尾のMLX functional scatterは256kで208,348,481 bytes、0.949 ms残ります。このprobeで状態表現を確定し、次節のopt-in production runtimeへ統合しました。

### Opt-in production compact NoPE DSA cache

production moduleはprobe scriptへ依存せず、`SingleNoPELatentCache`と`CompactIndexPoolCache`を実装します。前者はlatent 512を一つだけ保持し、後者はcontiguous pool keys/int64 indices/valid、最大3-token active tail、16-token rollback journalだけをauthoritative stateにします。decode/append/score/trim hot pathにNumPy、`.item()`、明示的`mx.eval`、CPU同期はありません。v4では容量を追加headroomではなくcache incarnation先頭からの絶対token位置として固定します。serverは最大prompt 256＋最大generation 4,096から両compact childへ4,352を渡し、追加容量は`reserve_until(absolute_token_capacity)`でのみ拡張します。v4 stateは絶対capacity、小さなcompress APE tensor、純粋pooling演算を保持するため、RAM APC clone直後もIndexer参照なしでtrimできます。CacheList rollbackとlong sparse prefill拒否は全cache mutation前にpreflightします。

prompt 1/16/128/256からの16-token decodeは全stepのfull-vocab logitsがDirectとbyte-identicalです。synthetic 2k sparse cacheでも5 measured stepのfull-model logits hashがDirectと一致しました。RAM APC restore位置mod 4 = 0/1/2/3は各16-token continuationがbyte-identicalで、prefill後のpacked Indexer full historyは存在せず、raw stateは最大19 tokenです。256-token caseのdecode中active memoryは増加せず、正のgrowth最大値はprompt 1の13.8 MBでした。

full-model synthetic-cache frontierは2 warmup＋5 samplesで、2k / 8k / 16k / 32k / 128k / 256kを測定しました。decodeは11.005から10.633 tok/s、retention 0.966です。2k Direct 10.920 tok/sに対してcompactは1.008×で、5% non-regression gateを通過しました。全11 DSA同期合計は13.444→19.877 ms、IndexPool updateは7.169→10.175 ms、pool carryは2.549→2.406 msです。256k active/peakは324.396/324.585 GBでOOMはありません。opt-in serverも`cache_backend=compact-nope-dsa`で起動し、`/health` HTTP 200を確認しました。

compactとDirectはAPC descriptorの`cache_backend`とcache ABIで分離します。RAM APCだけを許可し、compact disk APCはfail closedです。このruntimeはnon-speculativeです。KDA recurrent stateにrollback journalはないため、MTP/DFlash2やtarget verification対応済みとは扱いません。既定backendはDirect、prompt上限256、総context上限16,384のままです。

### Recurrent state materialization frontier

compact NoPE DSA cacheとDirect MoEを使い、cache容量を8,208 tokenへ事前予約した同一processで、materialization interval 0 / 50 / 128 / 256 / 512を各8,192 decode step測定しました。interval 50のgreedy token列を他armへteacher-forced replayし、EOSでは停止していません。materialization操作は固定mlx-vlmと同じ`mx.eval([entry.state for entry in cache]); mx.clear_cache()`です。runtime、server、cache ABI、admissionは変更していません。

interval 50 / 128 / 256 / 512の全checkpointでfull-vocab logits hashが完全一致し、cache state leaf数は全armで167のまま、NaNとMetal errorは0でした。materialization回数は163 / 64 / 32 / 16で、それぞれ`floor(8192 / interval)`と一致します。interval 256のwarm decode medianは92.163 msで、interval 50の92.044 msに対する回帰は0.129%です。active-memory正方向driftは4,789,621 bytesで64 MiB gate内、materialization medianは1.640 msでした。よってinterval 256を100k-token soak候補に選定します。512は観測armであり、このscreenだけではproduction候補にしません。

materializationなしのinterval 0も8,192 stepを完走し、logits hashとleaf数は一致しました。ただし最終cache memoryは342,683,480 bytesで、256の2,393,502 bytesより大きく、resource回収を省略する根拠にはなりません。Metal buffer object数を得る公開APIはないため、artifactは`metal_buffer_count_api_available=false`を明示し、byte数からbuffer数を推測していません。

### Recurrent state 100k-token soak

interval 256、compact NoPE DSA cache、Direct MoEを使い、100,016 tokenを予約した同一processで100,000 decode stepを実行しました。25k stop gateを通過し、25k / 50k / 75k / 100kの全milestoneへatomicにartifactを保存しました。scheduled materializationは予定どおり390回、別枠のfinal evidence materialization後もcountは390です。state leafは全boundaryで167、NaNとMetal errorは0、最初の8,192 tokenと全指定full-vocab logits hashはinterval-256 referenceと一致しました。

100k自体は完走し、90k–100kのdecode median 93.923 msは初期warm 10kの92.381 msに対してretention 0.984です。materializationは合計673.19 ms、median 1.647 ms、p95 2.102 ms、償却0.0067 ms/tokenでした。peakは321.182 GBで、final evidence materialization後も直前boundary帯へ戻りました。

ただしpost-materialization active-memory driftは79,275,215 bytesで64 MiB gateを超えたため、総合判定は`accepted=false`です。authoritative cache増分79,144,384 bytesと99.835%一致します。原因は`SingleNoPELatentCache`が初回だけ総容量を予約する一方、`CompactIndexPoolCache`が`total_tokens + reserve_tokens`を毎更新で再計算し、100,016-token headroomを前方へ移動させてpool bufferを256-token境界ごとに拡張することです。これはresource graph leakや数値破綻の証拠ではありませんが、「capacity growthを完全に排除したsoak」にはなっていません。

この失敗証拠を受け、compact cache ABIを`glm53-nope-dsa-v4-...-compact-indexpool-v4-fixed-absolute-capacity`へ更新しました。capacity 4,352ではtoken 1–4,352のlatent/IndexPool物理bufferが不変で、4,353で一度だけ4,608境界へ拡張します。`reserve_until(8192)`、RAM APC clone/restore、trim→replay、rejected updateも絶対capacityとstateを維持します。既定Direct backend、prompt/context admission、disk APC fail-closedは変更していません。

fixed-capacity v2 artifactでは、100,016 token設定をlatent 100,096 token、IndexPool 25,024 rowsへ一度だけalignmentし、全11 DSA層・全390 boundaryで同じ物理capacityを維持しました。authoritative cacheは全runで1,354,772,633 bytesのままdrift 0、post-materialization active driftは23,712 bytesです。100,000 token、390 scheduled materialization、leaf 167、NaN/Metal error 0、final evidence materialization、peak 321.099 GBをすべて通過しました。late retentionは0.985、materialization median/p95は1.543/1.657 ms、旧`79e1a60` runの全token/logits checkpoint hashとも一致し、総合`accepted=true`です。

opt-in serverは絶対capacity 4,352で起動し、`/health` HTTP 200を確認しました。この結果によりinterval 256はproduction固定のcorrectness/resource gateを通過しました。

### Bounded recurrent-state production policy

serverは固定mlx-vlmのgenerator import・生成より前に`MLX_VLM_BATCH_CACHE_EVAL_INTERVAL=256`を強制し、Directとcompactの両backendへ適用します。policy identityは`nested-cache-eval-clear-v1`です。既存のnested cache state評価と`mx.clear_cache()`の演算は変更せず、実generation batchが進んだstepだけを数えます。prefill、startup warmup、idle loop、空batchはcounterやcache stateを進めません。これは実行policyであり、cache表現ではないためcompact cache ABI v4とdisk APC namespaceは変更していません。

既定Direct backendでinterval 50と256を同一token列により各4,096 step連続実行しました。両armとも完走し、step 255/256/257、511/512、4095/4096を含む全記録点のfull-vocab logits hashと全生成token IDが一致しました。production armはstep 256から4,096まで正確に16回materializeし、NaNとMetal errorは0です。end-to-endは11.045から11.001 tok/sで回帰0.400%、2% gate内です。16/128-token既存oracleも全step一致しました。

Direct/compact serverはいずれも`/health` HTTP 200で、startup logにpolicyとintervalを出力します。request前のmetricsは`completed_materializations=0`、`decode_steps_since_materialization=0`であり、startup・idleによるstate mutationがないことを確認しました。compactのfixed-capacity 100k artifactも同じinterval 256でacceptedです。

### NoPE IndexPool safety gate

GLM-5.3-FlashのDSA cacheを`qk_rope_head_dim=0`、`mla_use_nope=true`、`kv_lora_rank=512`の独立したNoPE ABIとして監査します。Indexerの最終returnとattention gatherの両方で、indexが`-1`または`0 <= index < Kv`であることをGPU上で検査します。無効な正数をclipして実KVへ変換しません。

公式checkpointのlayer 3では、`index_topk=2048`に対してT=2047/2048のshort bypassとT=2049の実IndexPool経路を確認しました。T=2049の出力は`[1,1,2049,2051]`、unused slotは2,102,274、valid indexは0–2048、範囲外indexとNaNは0です。31/32/33、511/512/513、zero-valid、left-padding、partial final pool、one-shot/chunked/incrementalの有効index集合一致、反復hash一致にも合格しました。これはstate/sentinel correctness gateであり、sparse DSA kernelやprompt上限は変更していません。

### kpool4 expansion / KV dtype separation

`CompactIndexPoolCache`のpool selection後処理をpureな`expand_selected_pools()`へ抽出しました。入力はselected pool row、BF16のIndexPool token indices、独立bool validity、partial tailだけで、latent KV tensor・dtype・scaleを参照しません。返すtoken indexは常に`-1`または`[0, kv_len)`で、独立valid maskを併記します。productionのslot順、最終sanitize、attention gather側のrange再検査は維持しています。GLM-5.3の最大幅は2,048＋3 = 2,051で、16k–256kまで一定です。

fixtureはvalid pool数7/512/513、tail mod 0/1/2/3、2,047/2,048/2,049 bypass境界、4,351/4,352 server境界、16k/64k/128k/256k、正のOOB、invalid KV poison、repeat/restoreを通過しました。BF16、E4M3＋per-token/per-head FP32 scale、E4M3＋group64 FP32 scaleの全armでselected pool・token index・valid mask hashがbyte-identicalです。invalid slotをpoisonしてもlayer 3 attention出力は全armで不変です。

256k synthetic latent storageはBF16 268,435,456 bytes、per-token FP8 135,266,304 bytes、group64 FP8 142,606,336 bytesでした。gather＋dequantize medianは0.255 / 0.264 / 0.267 msです。FP8 outputのBF16比はper-tokenがrelative L2 0.00516、KL 8.70e-6、group64がL2 0.00564、KL 1.00e-5で、いずれもargmax一致・top-10 overlap 10/10でした。これは品質昇格gateではなくdtypeからの選択意味論分離probeです。IndexPool stateはBF16/int64/boolのまま、FP8 latent backendは未登録で、runtime/server/APC/cache ABIは変更していません。

### Long-context first-decode state ABI

Tier 1ではDirect/compactの両production cacheについてprompt 16/128/255/256を実prefillし、RAM APC snapshotからresident、restore、interval-256 materialization境界を通るfirst decodeを比較しました。全8 caseでfirst tokenとfull-vocab logits、resident/restore post-stateがbyte-identicalです。snapshotはdecode後も不変で、全11 DSA層のlatent/IndexPool offsetは正確に1進み、NaN、Metal error、予期しないcapacity growthはありません。materializationの非境界0回・境界1回も期待値と一致しました。

Tier 2ではcold prefillを実行せず、16k/64k/128kと262143〜262147 tokenのcanonical synthetic stateを全45層へ構築しました。34 KDA層はcontext非依存shape/dtypeとcanonical row-major layoutを持つ非ゼロconv/recurrent state、11 DSA層はlatent、contiguous IndexPool、最大19-token raw rollback stateです。Directは112 state leaves、compact/restoreは167で全context一定でした。全stateをmaterializeし、`context + 1`を256境界へ事前予約してからfirst decodeしています。

全8 context × 11 DSA層でDirect/compactのselected indexとattention outputがbyte-identicalです。全indexは`-1`または`[0, Kv)`、selected幅は最大2051で、Direct/compact full-model first logits、compact resident/restore post-stateも一致しました。262145/262144 first-decode latency比はDirect 1.010、compact resident 1.006、restore 0.968で1.5 gate内です。256k peakは331.456 GB、OOM・NaN・Metal errorはありません。

この結果が検証するのは「256k resident/RAM restore stateからfirst decodeへ遷移できること」です。256k cold prefillはunsupportedかつunvalidatedです。16k以上のcold promptはproduction admissionでfail closedし、拒否前後のcache hashも不変でした。prompt上限256、総context上限16,384、runtime/server/APC/cache ABIは変更していません。

## 検証

```bash
uv run pytest
uv run glm53 inspect /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash
```

`inspect`は既知metadata hashに加え、62 shardのsafetensors headerを読み、76,108 tensorの名前・shape・dtype・offset・file size、37,338 FP8/scale pair、総byte数、公式layout digest、NoPE/latent512 cache schemaを照合します。`attest`とserver起動はさらに全weight payloadを読み、既知checkpoint content digestへ照合します。

実機oracleを再照合するには次を実行します。

```bash
uv run python scripts/oracle_trace.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --tokens 16 --expect oracles/glm53-official-greedy-16.json

uv run python scripts/oracle_trace.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --tokens 128 --expect oracles/glm53-official-greedy-128.json

uv run python scripts/probe_long_context_first_decode_boundary.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --output bench-results/m3ultra512-long-context-first-decode-boundary-20260831.json

uv run python scripts/probe_resident_tensor_ownership.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash

uv run python scripts/probe_cache_state_lifecycle.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash

uv run python scripts/probe_kda_state_index_guards.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash

uv run python scripts/soak_layerwise_kda_state_digests.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --steps 4096 \
  --output bench-results/m3ultra512-layerwise-kda-state-digest-screen-20260902.json
```

実機gateではdense FP8 primitive、selected top-8 routing/score/clamp/down、固定promptの16/128-token full-vocab regression trace、256-token prefill、公式checkpoint attestation、OpenAI HTTP completion、`index_topk=2048`境界以降のIndexPool sentinel/rangeとchunked/incremental集合parityを確認します。golden traceは同じruntime由来の回帰検査であり、独立correctness oracleではありません。公式Transformers teacher-forced logits、KDA/DSA/IndexPool/mHCの層別intermediate parityとselected-KV sparse DSA性能はまだ追加gateです。

## Provenance

GLM-5.3 numerical fixesとstreaming converterはApache-2.0の[PipeNetwork/glm53-flash-mlx](https://github.com/PipeNetwork/glm53-flash-mlx) revision `b6665e8126c3b937031493e0580ef1e1c24f06cf`を基にしています。Server/APIとMetal primitiveはMITの`mlx-vlm` revision `e82d557d9f4b804cb1fc3eaaebc25488ba778a98`およびApple MLXを使用します。
