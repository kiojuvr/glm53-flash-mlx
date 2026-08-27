# GLM-5.3-Flash MLX runtime for M3 Ultra 512 GB

`zai-org/GLM-5.3-Flash`をApple M3 Ultra 512 GBで動かすための、text-only・single-node特化runtimeです。OpenCodeなどから利用できるOpenAI互換APIを提供します。

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

通常運転では同時sequenceを1に固定し、interactive decodeを優先します。既定memory設定はwired 440 GB、MLX cache 32 GB、prefill 2048 tokensです。

起動時にtext target全体（実測319.706 GB、297.75 GiB）をunified memoryへmaterializeします。この処理はM3 Ultra実測で約30.6秒です。保存dtypeはFP8/BF16のままであり、BF16 model copyは作りません。一時的なsmoke testだけ常駐化を省く場合は`--no-warm-residency`を指定できます。

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

[examples/opencode.json](examples/opencode.json)をプロジェクトの`opencode.json`へコピーするか、既存設定へprovider部分を統合します。serverは安定したmodel alias `glm-5.3-flash`を受け付けます。

```bash
opencode --model glm53/glm-5.3-flash
```

API keyを有効にした場合は、exampleの`options.apiKey`を`"{env:GLM53_API_KEY}"`へ変更します。

## Prefix cache（opt-in）

既定ではAPCを無効化しています。通常のBF16 hybrid state/KV cache容量は512 GB構成で十分であり、まず単一sessionの正しさを優先するためです。明示的に検証する場合だけ有効化してください。

```bash
# RAM APCを有効化
uv run glm53 serve --apc --apc-blocks 256

# SSD tierも使う（cacheは再計算可能なデータとして扱う）
uv run glm53 serve --apc --apc-blocks 512 \
  --apc-disk-path /Volumes/SDXC-512/glm53-apc
```

`/v1/metrics`と`/health`でqueue、prefill/decode速度、APC状態を確認できます。

## 現在の境界

- 対象はM3 Ultra 512 GB、batch 1、text target stackです。
- MTPはcorrectnessと追加weight trafficのgateが未完了のため既定offです。
- text-only runtimeではvision towerをloadしません。
- 1M contextはmodel上の上限です。実運用上限はcache容量、TTFT、Metal working setを計測して決めてください。
- batch-1 decodeはtop-8 expertを3個のMetal kernelへ融合しています。prefill用tiled grouped GEMMとrequest横断expert coalescingは次の性能段階です。
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

再測定コマンド:

```bash
uv run python scripts/bench_fp8.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --warm-residency --tokens 16
```

## 検証

```bash
uv run pytest
uv run glm53 inspect /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash
```

実機gateではMetal FP8 primitiveのdense reference一致、全45層のgreedy 16-token、checkpoint dtype/byte audit、OpenAI HTTP completionを確認しています。teacher forcing、router/IndexPoolの層別oracle、長文chunked prefillは追加gateです。巨大modelの実機試験は通常のmacOS sessionで行ってください。

## Provenance

GLM-5.3 numerical fixesとstreaming converterはApache-2.0の[PipeNetwork/glm53-flash-mlx](https://github.com/PipeNetwork/glm53-flash-mlx) revision `b6665e8126c3b937031493e0580ef1e1c24f06cf`を基にしています。Server/APIとMetal primitiveはMITの`mlx-vlm` revision `e82d557d9f4b804cb1fc3eaaebc25488ba778a98`およびApple MLXを使用します。
