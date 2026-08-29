# GLM-5.3-Flash MLX runtime for M3 Ultra 512 GB

`zai-org/GLM-5.3-Flash`をApple M3 Ultra 512 GBで動かすための、text-only・single-node・decode-first runtimeです。OpenCodeなどから利用できるOpenAI互換APIを提供します。既定は公式tensor layoutを直接読む経路で、全42 MoE層のpacked grouped FP8 prefillを実験的にopt-inできます。sparse DSA prefillは未実装です。

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

## 3. OpenAI互換serverを起動する

```bash
uv run glm53 serve \
  --model /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --host 127.0.0.1 \
  --port 8080
```

全42 routed MoE層を層単位で連続FP8 bankへ移行し、GPU sorted grouped prefillを使う場合だけ次を指定します。公式FP8＋FP32 scaleのままで、BF16 weight copyは作りません。

```bash
uv run glm53 serve \
  --model /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --experimental-packed-grouped-moe
```

このflagは既定offです。kernel単体の実測損益分岐は16 routesですが、17-token oracleの全logits hashを維持するためruntimeは256 routes未満をpacked Direct-compatible pathへfallbackします。したがって256-token prefillはgrouped、batch-1 decodeと短いpromptはbit-identicalなselected top-8経路です。

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

disk namespaceはcheckpoint全shard、index、tokenizer/chat template、KV codec設定、固定mlx-vlm revision、custom Metal kernel ABI、NoPE DSA cache ABIから生成します。NoPE ABIはlatent 512と`-1` sentinelを明示します。さらにMoE backendを分離し、packed-grouped時はgrouped kernel ABI、256-route runtime threshold、packed bank ABI、packed decode ABIを含めます。同じpathのweight payloadが置換された場合やDirectで生成したstateをgrouped backendが起動した場合も古いstateを復元しません。RAM/disk APCはいずれもhybrid-state parity gateが未完了なので既定offです。

`/v1/metrics`と`/health`でqueue、prefill/decode速度、APC状態を確認できます。

## 現在の境界

- 対象はM3 Ultra 512 GB、batch 1、text target stackです。
- MTPはcorrectnessと追加weight trafficのgateが未完了のため既定offです。
- text-only runtimeではvision towerをloadしません。
- 1Mはmodel-native上限にすぎません。server既定はprompt 256、総context 16,384です。OpenCode exampleは安全な総context 4,352（prompt 256 + output 4,096）を広告します。
- 既定のbatch-1 decodeはtop-8 expertを3個のMetal kernelへ融合しています。opt-in packed backendもbankを直接読む3 kernelを使い、実測13.26 tok/sです。
- 既定prefillはCPU expert bucketです。opt-in時はGPU route sortとgrouped MMAを全42 MoE層へ適用しますが、DSA full-KV SDPAは残ります。prompt上限256は変更していません。
- 設計レポートの15 tok/s gateに対し、現在の常駐後実測は11.4 tok/sです。API/runtimeとして利用可能ですが、このperformance gateは未達です。

## M3 Ultra 512 GB実測

2026-08-28〜29、このリポジトリの公式checkpointで測定した値です。

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
| layer 3 NoPE IndexPool, T=2049 | shape 1×1×2049×2051 / unused 2,102,274 / out-of-range 0 |
| layer 3 grouped route amplification | local 0.00408 → final 0.18605 / route固定 0.02034（9.15×縮小） |
| layer 5 grouped route amplification | local 0.00426 → final 0.20199 / route固定 0.01830（11.04×縮小） |
| suffix sweep最速screening通過 c=29 | warm median 4.573 s / Direct比1.241× / L2 0.01958 / KL 3.37e-4 |
| Direct-order BM8 parity anchor | 10/10 stage・full logits・全router hash一致 / warm median 6.554 s / Direct比0.866× |
| layer 3 DSA steady decode, 2049 → 256k | 2.331 → 2.614 ms / retention 0.892 / selected幅2051固定 |
| layer 3 DSA pool rebuild, 2049 → 256k | 2.558 → 5.604 ms / retention 0.456 / 256k pool update 3.137 ms |

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

uv run python scripts/localize_grouped_fp8_divergence.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --tokens 256 --warmups 2 --repeats 5 \
  --output bench-results/m3ultra512-grouped-fp8-divergence-20260828.json

uv run python scripts/probe_nope_indexpool_safety.py \
  /Volumes/KIOXIA-PRO-2/models/zai-org/GLM-5.3-Flash \
  --layer 3 \
  --output bench-results/m3ultra512-nope-indexpool-safety-20260828.json

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
```

### Packed expert bank feasibility

GPU MoE prefillの前提確認として、実checkpointのlayer 3だけを4個の連続bufferへin-memory packingしました。`gate_up_weight`と`down_weight`はuint8 E4M3、scaleはFP32のままで、BF16展開はありません。全288 expert × 6 tensor = 1,728 sliceが元tensorとbyte-identicalで、既存selected top-8出力もbit-identicalでした。

全model常駐状態のactive memoryは319.706 GB、pack中peakは327.023 GBでした。module参照の切替、旧expert解放、`mx.clear_cache()`後は319.706 GBへ戻り、baselineとの差は4 bytesです。したがって層単位移行で元mmap-backed tensorを解放でき、定常的なexpert bank二重化を避けられることを確認しました。これはfeasibility probeであり、serverとloaderの既定経路はまだ変更していません。

0.202秒は既にresidentなtensorから1層をpackする時間です。attestation、checkpoint load、全weight residencyを含むserver-ready時間とは分離して扱います。

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

### NoPE IndexPool safety gate

GLM-5.3-FlashのDSA cacheを`qk_rope_head_dim=0`、`mla_use_nope=true`、`kv_lora_rank=512`の独立したNoPE ABIとして監査します。Indexerの最終returnとattention gatherの両方で、indexが`-1`または`0 <= index < Kv`であることをGPU上で検査します。無効な正数をclipして実KVへ変換しません。

公式checkpointのlayer 3では、`index_topk=2048`に対してT=2047/2048のshort bypassとT=2049の実IndexPool経路を確認しました。T=2049の出力は`[1,1,2049,2051]`、unused slotは2,102,274、valid indexは0–2048、範囲外indexとNaNは0です。31/32/33、511/512/513、zero-valid、left-padding、partial final pool、one-shot/chunked/incrementalの有効index集合一致、反復hash一致にも合格しました。これはstate/sentinel correctness gateであり、sparse DSA kernelやprompt上限は変更していません。

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
```

実機gateではdense FP8 primitive、selected top-8 routing/score/clamp/down、固定promptの16/128-token full-vocab regression trace、256-token prefill、公式checkpoint attestation、OpenAI HTTP completion、`index_topk=2048`境界以降のIndexPool sentinel/rangeとchunked/incremental集合parityを確認します。golden traceは同じruntime由来の回帰検査であり、独立correctness oracleではありません。公式Transformers teacher-forced logits、KDA/DSA/IndexPool/mHCの層別intermediate parityとselected-KV sparse DSA性能はまだ追加gateです。

## Provenance

GLM-5.3 numerical fixesとstreaming converterはApache-2.0の[PipeNetwork/glm53-flash-mlx](https://github.com/PipeNetwork/glm53-flash-mlx) revision `b6665e8126c3b937031493e0580ef1e1c24f06cf`を基にしています。Server/APIとMetal primitiveはMITの`mlx-vlm` revision `e82d557d9f4b804cb1fc3eaaebc25488ba778a98`およびApple MLXを使用します。
