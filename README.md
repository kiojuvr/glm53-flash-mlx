# GLM-5.3-Flash MLX runtime for M3 Ultra 512 GB

`zai-org/GLM-5.3-Flash`をApple M3 Ultra 512 GBで動かすための、text-only・single-node・decode-first vertical sliceです。OpenCodeなどから利用できるOpenAI互換APIを提供しますが、GPU expert bucketingとsparse DSA prefillは未実装です。

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

## 3. OpenAI互換serverを起動する

```bash
uv run glm53 serve \
  --model /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --host 127.0.0.1 \
  --port 8080
```

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
# RAM APCを有効化
uv run glm53 serve --apc --apc-blocks 256

# SSD tierも使う（experimental。attested content identityを使用）
uv run glm53 serve --apc --apc-blocks 512 \
  --apc-disk-path /Volumes/SDXC-512/glm53-apc \
  --experimental-disk-apc
```

disk namespaceはcheckpoint全shard、index、tokenizer/chat template、KV codec設定、固定mlx-vlm revision、custom Metal kernel ABIから生成します。同じpathのweight payloadが置換された場合も古いstateを復元しません。RAM/disk APCはいずれもhybrid-state parity gateが未完了なので既定offです。

`/v1/metrics`と`/health`でqueue、prefill/decode速度、APC状態を確認できます。

## 現在の境界

- 対象はM3 Ultra 512 GB、batch 1、text target stackです。
- MTPはcorrectnessと追加weight trafficのgateが未完了のため既定offです。
- text-only runtimeではvision towerをloadしません。
- 1Mはmodel-native上限にすぎません。server既定はprompt 256、総context 16,384です。OpenCode exampleは安全な総context 4,352（prompt 256 + output 4,096）を広告します。
- batch-1 decodeはtop-8 expertを3個のMetal kernelへ融合しています。GPU上のexpert sort/bucket、grouped GEMM、request横断expert coalescingは次の性能段階です。
- 現在のprefill projectionは8 token rowでFP8 weightを共有するtiled-row Metal kernelです。ただしCPU expert bucketとDSA full-KV SDPAが残り、256-token probeでも旧経路から有意な改善はありません。大きなprompt向けruntimeではありません。
- 設計レポートの15 tok/s gateに対し、現在の常駐後実測は11.4 tok/sです。API/runtimeとして利用可能ですが、このperformance gateは未達です。

## M3 Ultra 512 GB実測

2026-08-28、このリポジトリの公式checkpointで測定した値です。

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
| layer 3 grouped FP8 MoE, 256 tokens | 111.46 → 22.08 ms / 5.05× / working peak +319.7 MB |

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
```

### Packed expert bank feasibility

GPU MoE prefillの前提確認として、実checkpointのlayer 3だけを4個の連続bufferへin-memory packingしました。`gate_up_weight`と`down_weight`はuint8 E4M3、scaleはFP32のままで、BF16展開はありません。全288 expert × 6 tensor = 1,728 sliceが元tensorとbyte-identicalで、既存selected top-8出力もbit-identicalでした。

全model常駐状態のactive memoryは319.706 GB、pack中peakは327.023 GBでした。module参照の切替、旧expert解放、`mx.clear_cache()`後は319.706 GBへ戻り、baselineとの差は4 bytesです。したがって層単位移行で元mmap-backed tensorを解放でき、定常的なexpert bank二重化を避けられることを確認しました。これはfeasibility probeであり、serverとloaderの既定経路はまだ変更していません。

0.202秒は既にresidentなtensorから1層をpackする時間です。attestation、checkpoint load、全weight residencyを含むserver-ready時間とは分離して扱います。

### Sorted grouped FP8 MoE feasibility

layer 3のpacked bankを直接読むprefill専用Metal kernelを実装しました。route planはGPU上のargsort、histogram、prefix sumで構築し、expert境界に揃えた32-route descriptorごとに32×32×32 `simdgroup_matrix` GEMMを実行します。FP8 weightはthreadgroup tileへFP32 decodeするだけで、永続的なBF16 weight展開はありません。hot pathにはNumPy変換、`.item()`、明示的`mx.eval`を含みません。

32/64/128/256/512 tokenの全点でDirectFP8MoEとの`rtol=0.02, atol=0.02` parityに合格し、speedupは2.17× / 3.45× / 4.42× / 5.05× / 5.24×でした。256 tokenのmax/mean/RMS errorは0.01953 / 0.001077 / 0.001972、追加working peakは319.7 MBです。実行tileはexpert境界を跨ぎません。測定済みの損益分岐点は256 routes（32 tokens）なので、それ未満は既存packed pathへfallbackします。公式checkpointの16-token oracleも全stepのlogits hashが不変で、batch-1 decodeはselected top-8経路を維持しています。

開発中のscalar grouped variantは256 tokenで0.48–0.71×に留まり棄却しました。性能向上はdispatch統合だけではなく、MMA GEMMによるweight reuseで得られています。このprobeもlayer 3限定で、serverとloaderの既定経路は変更していません。

## 検証

```bash
uv run pytest
uv run glm53 inspect /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash
```

`inspect`は既知metadata hashに加え、62 shardのsafetensors headerを読み、76,108 tensorの名前・shape・dtype・offset・file size、37,338 FP8/scale pair、総byte数、公式layout digestを照合します。`attest`とserver起動はさらに全weight payloadを読み、既知checkpoint content digestへ照合します。

実機oracleを再照合するには次を実行します。

```bash
uv run python scripts/oracle_trace.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --tokens 16 --expect oracles/glm53-official-greedy-16.json

uv run python scripts/oracle_trace.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --tokens 128 --expect oracles/glm53-official-greedy-128.json
```

実機gateではdense FP8 primitive、selected top-8 routing/score/clamp/down、固定promptの16/128-token full-vocab regression trace、256-token prefill、公式checkpoint attestation、OpenAI HTTP completionを確認します。golden traceは同じruntime由来の回帰検査であり、独立correctness oracleではありません。公式Transformers teacher-forced logits、KDA/DSA/IndexPool/mHCの層別intermediate parity、`index_topk=2048`以降のsparse IndexPool、chunked prefill parityはまだ追加gateです。

## Provenance

GLM-5.3 numerical fixesとstreaming converterはApache-2.0の[PipeNetwork/glm53-flash-mlx](https://github.com/PipeNetwork/glm53-flash-mlx) revision `b6665e8126c3b937031493e0580ef1e1c24f06cf`を基にしています。Server/APIとMetal primitiveはMITの`mlx-vlm` revision `e82d557d9f4b804cb1fc3eaaebc25488ba778a98`およびApple MLXを使用します。
