# Orbit Wars 改善計画（2P / 4P）— 戦略レベルの構造改善

最終更新: 2026-06-23
対象ベース: `sample130_4p_threat_reserve_from110`（現行の確定4Pベース。2Pは sample110 と同一挙動）

このドキュメントは、別のAIエージェントがそのまま着手できることを目的に、
**根拠（実コード・実ログ）／狙い／具体的な実装箇所／検証手順／禁止事項** を明記する。
数値（パラメータ）チューニングではなく、**意思決定アルゴリズムそのものの構造改善**を狙う。

---

## 0. 前提と厳守ルール（実験運用）

- **既存 sample は変更しない。** 新ディレクトリを `sampleNNN_<topic>_from<parent>/` 形式でコピーして作業する。
  - コピー例（PowerShell）:
    ```powershell
    cd C:\Users\yuu98\Desktop\kaggle\orbit-wars
    Copy-Item -Recurse sample130_4p_threat_reserve_from110 sampleNNN_<topic>_from130
    Get-ChildItem -Recurse sampleNNN_<topic>_from130 -Filter __pycache__ -Directory | Remove-Item -Recurse -Force
    ```
- **4P と 2P のコードパスを混ぜない。** 機能は `enable_*_4p` / `enable_*_2p` の独立フラグ＋`player_count` 分岐でガードし、
  片方の変更が他方をバイト単位で変えないこと。`orbit_lite/` を変更する場合は特に、4P 挙動の不変を別途確認する。
- **編集後は必ず機械的検証**を行う（推測で結果を書かない）:
  - `python -m py_compile main.py`
  - `python -c "import main; print(main.CONFIG_2P.<field>, main.CONFIG_4P.<field>)"` で設定値の実反映を確認
  - 変更前後の `diff`（params.json / main.py）を表示
- **評価は重いので、実行コマンドを提示してユーザーが回す。** エージェントは生出力だけを根拠に判断する。

### 評価プロトコル（実測）

- 2P: `evaluate.py --players 2 --agent <dir>/main.py --opponent sample8/main.py --seed-list <seeds> --workers 4`
- 判断順: **Wins/Losses → Crash rate → Average placement → Per-seed → Average score diff**。
- 2P は「確実に勝ち切る」優先（diff悪化でも勝ち数増なら前進）。
- 段階: まず 8〜20 seed、採用候補は random40。
- **注意（実測で確認済み）**: 2P の対 sample8 戦は同一 seed で seat 対称・ほぼ決定論的。draw（500ターン同点）が出る。
  勝敗カウント時は `reward` を使い、diff=0 の draw を勝ちに数えない。
- 4P: 固定30-seed プール（`EXPERIMENT_LOG_RECENT.md` の sample130 エントリに seed 列挙あり）＋ random20、seat0。

---

## 1. これまでに「実データで」確定している事実（必読）

### 1.1 4P: bolt-on 戦術は局所最適（sample131–137 の7連続失敗）

実ログ（`EXPERIMENT_LOG_RECENT.md` 2262–2326行）の要約:

| sample | 内容 | 結果 | 教訓 |
|---|---|---|---|
| 131 | 攻撃前に陥落予測の自惑星へ防御増援を**強制** | 30seed 16→15W, random20 7→5W | **行動の強制は害**。過剰防御で受動化、勝ち筋 seed を落とす |
| 132 | reserve 脅威を sum→(top1−top2 敵) に | WASH（勝ち数同じ） | より正確だが無効。棚上げ |
| 133 | 第三者が攻撃中の敵惑星への攻撃を禁止 | ほぼ NO-OP | 仮説（接触惑星を狙う）が誤り |
| 134 | reserve に latent enemy pressure を加算 | NO-OP | — |
| 135 | turn60前の敵攻撃を**全面禁止** | 30seed 16→5W（−11W） | **敵攻撃はネットでプラス**。全面禁止は勝ち筋 vulture capture も捨てる |
| 136 | 保持不能な敵captureだけ禁止（target側filter） | 却下、早期死直らず | フィルタは標的を見て**source を見ていない** |
| 137 | reserve に latent reachable-enemy を MAX 結合（Idea B） | 30seed 16→12W | 過剰 reserve→受動化（131/135 と同型の失敗） |

**4P早期死の真因（trace確定 / `tools/trace_4p.py`, seed 1399615834）:**
- t≤32: 4者対称・互角（むしろ自分が艦数リード 112 vs 89）。
- t32→t40: **自分から約37艦を launch し 3→1 惑星に減少**、一方2敵は 3→5 にきれいに拡張。t52 全滅。
- = 防御失敗ではない（総艦数は平坦なのに惑星を失う）。**SOURCE-STRIPPING**: 自宅を空にして（保持可能な）敵を攻撃し、
  **別の敵が空いた自宅を取る**。sample130 の reserve は「飛行中の敵艦隊」からしか source を守らず、
  「自分が空にした後に launch してくる潜在的な近隣敵 garrison」から守れない。

**確立した一般則（この エンジンに対して）:**
- **効く**: 精密・脅威トリガー・savable-gated な「悪いコミットを防ぐ」**ハード制約（選択肢を減らす）**。
- **効かない/害**: 行動を強制する／スコアを足す／選択肢を広範に禁止する／過剰に reserve する（すべて受動化を招く）。

### 1.2 2P: ほぼ未着手。コアエンジンの近視眼 greedy のみ

実コード（`sample130/main.py`, `orbit_lite/`）で確認した 2P の意思決定経路:

```
run_turn (main.py:2457)
 └ plan_lite_waves (main.py:2126)
    ├ build_target_shortlist          (orbit_lite/planner_core.py:279)  狙う候補（近接の敵/中立 ∪ 陥落予測の自惑星）
    ├ _tier_candidates × size_mult    (main.py:1982)                    各(src,tgt)を独立スコア
    │   └ score_candidates            (orbit_lite/planner_core.py:97)
    │       └ sparse_launch_flow_delta(orbit_lite/garrison_launch.py:311) horizon先までのnet投影（敵は無応手＝do-nothing）
    │       └ competitive_score       (orbit_lite/planner_core.py:83)   = 自分のnet − 相手のnet
    ├ _greedy_select                  (orbit_lite/planner_core.py:371)  波を貪欲に1つずつ選ぶ
    └ _plan_regroup                   (orbit_lite/planner_core.py:445)  余剰艦の再集結
```

- **4P専用の高度機能（true_one_ply / coord_followup / mini_rollout / threat_reserve / influence 等）は
  すべて `if player_count < 4: return` で 2P では無効。** 2P は素の近視眼 greedy のみ。
- **2P の弱点（実測トレース sample130 / `tools/trace_2p.py` ※下記注）:**
  - 序盤は対称で互角。**中盤の競合・拡張フェーズで主導権を失い、生産複利で押し切られる。**
  - 例: 「優勢な兵力を抱えたまま攻めに転じず（兵力死蔵）、相手の兵力集中に逆転される」。
  - baseline 実測（固定random20, seat0, vs sample8）: **8W / 4D / 8L**（draw=500ターン同点）。

### 1.3 効かないと実測で確認した方向（やり直さない）

- **horizon 延長（2P, `config.horizon` 18→28）= 完全無効。** CONFIG_2P.horizon=28 を import で確認した上で
  random20 が baseline と1ゲームも変わらなかった。理由: 2P は惑星が近接し有効攻撃の eta が 18 未満に収まるため、
  評価ホライズンを広げても候補も投影も増えない。さらに competitive(自分−相手)は対称局面で生産複利が相殺される。
  → **2P を horizon で変えようとしない。**

> 注: `tools/trace_2p.py` は本計画時点で再作成が必要な場合がある（過去セッションで作成有無が未確定）。
> 使う前に `Test-Path` で存在確認し、無ければ「seat別の (惑星数 / 総艦数 / 総生産) を毎ターン出す」最小スクリプトを作る。
> trace は軽い（1ゲーム数十秒）ので、対策設計の前に必ず負け筋を生出力で確認すること。

---

## 2. 4P 改善計画 — Idea A: greedy を global allocation に置換（構造改革）

### 2.1 なぜ（根拠）

- 真因 SOURCE-STRIPPING は「個々の launch は良いが、合算すると source を裸にする」**割り当ての大域的失敗**。
- 現状 `_greedy_select`（planner_core.py:371）は波を1つずつ貪欲選択し、source_budget を逐次引くだけ。
  各波は「この source をこれ以上引くと、別の敵に対して脆弱になる」という**他の選択との相互作用を見ていない**。
- bolt-on の reserve（sample130/134/137）は「事前に source から引いて隠す」ため、効きすぎると受動化する（実証済み）。
  **reserve（事前隔離）ではなく、割り当て問題として「source 脆弱性コスト」を選択と同時に最適化する**のが筋。

### 2.2 何を作るか（設計）

`_greedy_select` を、**source 容量制約と source 脆弱性コストを同時に扱う大域割り当て**に置き換える。
scipy は提出環境に無い前提なので **torch/numpy で自作**する。

- 決定変数: 各候補 launch（src→tgt, size tier）を採用するか（0/1 近似でよい）。
- 目的: Σ(競合スコア) − Σ(source脆弱性ペナルティ)。
  - source脆弱性ペナルティ = 「その source の残garrisonが、到達可能な敵force（in-flight＋latent）に対して不足する量」を、
    **採用した全 launch の合算ドレイン後**に評価する（greedy の逐次ではなく最終配分に対して）。
- 制約: 1 source の総送出 ≤ safe_drain、1 target に1波、role mutex（既存 `_greedy_select` の規則を踏襲）。

実装アプローチ（段階）:
1. **まず「貪欲＋ローカル改善（swap/drop）」で近似する**（min-cost-flow フル実装の前段。リスク低）。
   - `_greedy_select` の出力を初期解とし、「source が裸になっている波を drop すると全体スコアが上がるか」を
     数回スイープして局所改善。これは sample130 の reserve と違い、**個別 source を隠さず、配分結果を見て削る**。
2. 効果が出たら **min-cost-flow / 線形割り当てへ拡張**（torch で Sinkhorn 近似 or ハンガリアン自作）。

### 2.3 変更箇所

- 新規 2P/4P 共通の関数を `orbit_lite/planner_core.py` に追加（例 `_global_select`）。
  **`orbit_lite` を変更するため、4P 既存挙動の不変を別途確認すること**（§4 回帰チェック）。
- `plan_lite_waves`（main.py:2425 付近の `_greedy_select` 呼び出し）に
  `enable_global_select_4p` フラグで分岐を追加。デフォルト無効、CONFIG_4P で有効化。
- 4P 専用フラグなので 2P は従来の `_greedy_select` のまま（混線禁止）。

### 2.4 検証

- まず seed 1399615834（trace済みの早期死 seed）で `tools/trace_4p.py` を回し、
  **t32→40 の source-stripping 全滅が改善するか**を生出力で確認（heavy eval の前）。
- 固定30-seed ＋ random20 で sample130 比 **勝ち数非後退 ＋ crash 0%**。
- **禁止**: 受動化（早期 survival が伸びるのに勝ち数が減る）は sample131/135/137 と同じ失敗。
  「survival は伸びたが勝ち数減」なら即棄却。

---

## 3. 2P 改善計画 — 中盤の主導権喪失（生産複利負け）への構造対策

### 3.1 なぜ（根拠）

- 2P の負け筋は「序盤互角 → 中盤の競合で主導権を失い生産複利で負け」。
- 近視眼 greedy ＋ competitive(自分−相手) は **対称局面で過度に保守的**になり、お見合い（両者大兵力で対峙）で動かず、
  その間に相手が拡張して生産で上回る。horizon 延長では直らない（§1.3 実証済み）。
- 4P で確立した一般則（§1.1）が 2P にも効くはず: **「悪いコミットを防ぐハード制約」は効き、「強制・スコア加算」は害**。

### 3.2 候補レバー（着手順。各々 1 機能ずつ、独立に検証）

**レバー2A（最優先・低リスク）: source-stripping を 2P でも防ぐ「選択肢を減らす」制約**
- 4P の真因 source-stripping は 2P でも起こりうる（自陣を空にして攻めて取り返される＝兵力死蔵の裏返し）。
- sample130 の `_threat_reserve_source_budget`（main.py:2064）を **2P 専用フラグで有効化**するのではなく
  （※ reserve は受動化リスク。実際2Pでの無差別/精密化 reserve は過去セッションで baseline 超えできなかった疑い）、
  **Idea A の global-select（§2）を 2P にも適用**して「合算で source を裸にする攻撃を削る」方が筋が良い。
  - 2P は敵1人で割り当て問題が 4P より小さく、global-select の効果と安全性を**先に2Pで確かめられる**利点がある。
  - つまり **Idea A はまず 2P で試作 → 効けば 4P へ展開**、という順序を推奨（2P は低リスク・検証が速い）。

**レバー2B: 敵応手読み（true_one_ply）の 2P 適正版**
- 4P の `_apply_true_one_ply_rescore`（main.py:1843）＋ `_build_enemy_best_response_launches`（main.py:1660）を
  2P 専用フラグで有効化。**ただし「スコアを足す/攻めを促す」方向は §1.1 で害と確認済み**なので、
  **`hard_bad` による除外（取った瞬間取り返される手・自陣を裸にする手を消す）だけに絞る**こと。
  bonus 付与・base_weight 混合は入れない（行動誘導は害）。
- 重みは 2P なら competitive と整合する等重み（相手 net をそのまま引く、leader 分岐なし）。
- これは「選択肢を減らす」制約に該当し、一般則に合う。

**レバー2C: お見合い打開（最後・要注意）**
- お見合いで動かない問題に対し「中立惑星の継続奪取を促す」等の**行動誘導は §1.1 で害**。安易にやらない。
- やるなら「相手も動かない局面で、自分だけが安全に拡張できる中立 target を global-select が選べているか」を
  trace で確認し、選べていないなら build_target_shortlist の候補不足（中立が候補から漏れている）を疑う。
  = 候補生成の問題として扱い、スコア加算では対処しない。

### 3.3 変更箇所と検証

- レバー2A（Idea A の2P適用）: §2.3 と同じ関数を 2P 分岐で呼ぶ（`enable_global_select_2p`）。
- レバー2B: 上記2関数のガードを 2P 対応にし、CONFIG_2P で有効化。`hard_bad` 系のみ作用させる。
- 検証: 固定random20（seat0, vs sample8）で baseline **8W4D8L** 比、勝ち数非後退＋負けseed救済＋crash0。
  - まず負け seed を `tools/trace_2p.py` で見て、対策後に同 seed の崩れが直るかを生出力で確認してから heavy eval。

---

## 4. 4P 回帰チェック（`orbit_lite` を触る場合は必須）

- `orbit_lite` を変更したら、**2P/4P 専用フラグを OFF にした状態で sample130 と挙動一致**を確認する。
- 最小確認: 同一 seed で sample130 と新ディレクトリの 1ゲーム trace（4P）が一致するか、
  または固定30-seed の結果が一致するか。**4Pをバイト単位で変えていないことを実測で示す。**

---

## 5. 着手順サマリ（推奨）

1. **2P で Idea A（global-select の局所改善版, §2.2 段階1）を試作** — 低リスク・検証が速い・source-stripping/兵力死蔵に直接効く。
   - baseline 8W4D8L に対し random20 で勝ち数増を確認。
2. 効けば **同じ global-select を 4P へ展開**（§2）。trace で seed 1399615834 の早期死改善を確認後、30seed＋random20。
3. 補助として **2P の true_one_ply を `hard_bad` 除外のみ**で追加（§3.2 レバー2B）。
4. 各段階で §0 の機械的検証と禁止事項（受動化＝survival↑なのに勝ち減 は即棄却）を厳守。

## 6. やらないこと（実測で棄却済み・繰り返さない）

- 2P の horizon 変更（§1.3、完全無効）。
- 防御の強制増援・全面攻撃禁止・過剰 reserve（§1.1、すべて受動化で勝ち減）。
- 標的側だけの holdability フィルタ（§1.1 sample133/136、真因 source 側を外す）。
- スコアに bonus を足して行動を誘導する系（この エンジンでは一貫して害）。
