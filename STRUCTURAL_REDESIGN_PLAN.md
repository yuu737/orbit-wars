# Orbit Wars — 構造改革計画: orbit_lite の 2P/4P 分割とプランナ改変

最終更新: 2026-06-24
対象: 別のAIエージェントが**この文書だけで単独着手**できることを目的とする。
前提ベース: `sample157_4p_thirdparty_guard_from110`（確定4P）/ `sample8`（確定2P, = sample110/130 の2Pと挙動同一）。

---

## 0. この文書の位置づけ（必読・コールドスタート用の確立事実）

数値チューニング（config）は **2P/4P とも頭打ちが実証済み**。本計画は **orbit_lite のプランナそのものを 2P/4P で分割し、各々を構造改変**して「configでは届かない壁」を壊す。**高リスク**（orbit_lite は検証済み中核）なので、**バイト等価確認 → 計装で主因確定 → 1点ずつ改変 → 重回帰**の順を厳守する。

### 0.1 実行・評価環境（そのまま使える）
- venv python: `C:\tmp\ow\Scripts\python.exe`（torch 2.12 CPU, kaggle_environments 同梱）。
- 2P eval: `evaluate.py --players 2 --agent <dir>/main.py --opponent sample8/main.py --seed-list <csv> --workers 5`
- 4P eval: `evaluate.py --players 4 --agent <dir>/main.py --opponent sample7/main.py --opponent sample8/main.py --opponent bots/hairate5.py --seed-list <csv> --workers 5`
- **2Pは“弱い合成相手”では真因が出ない**。判定は `sample8` 直接 + **ミラーbaseline(sample8 vs sample8)比**、かつ**複数 random40**（カオスなミラーでpool分散が大きい）。
- **4Pの合成相手(sample7/8/hairate5)は弱すぎて本戦の真因を再現しない**。**自分の派生(sample157/165等)を相手プールに混ぜたA/B**で評価せよ（弱い相手だと“弱い相手には強いが本戦で負ける”罠に嵌まる）。
- 解析ツール: `tools/replay_analyze.py`（2P/4P・脱落順・WHO TOOK・SOURCE-STRIPPING・3rd-party判定。CJK名で藤田佑seatを自動選択）、`tools/shadow_compare.py`（上位者リプレイ盤面に自agentを走らせ手を差分）、`tools/trace_2p.py`。

### 0.2 致命的な落とし穴（必ず守る）
- **params.json は eval/trace で無視される**。`env.run([path])` は `__file__` を立てず `_HERE=cwd(project root)` にフォールバック → params.json 未発見 → dataclass デフォルトで動く。**config は必ずコード（CONFIG_2P/CONFIG_4P 構築 or dataclass）に焼く**。
  - 罠: `import main`（`__file__`あり）は params.json を読むので、**config 反映確認を import でやると eval と食い違う**（例: max_waves が6に化ける）。**活動量等の確認は必ず env.run 経路で**。
- **shadow_compare は agent を import する**ので同じく params が効く。2P系のconfig検証には使えない（4Pは params が roi/waves を上書きしないので有効）。
- orbit_lite を変えたら、**変更フラグ OFF で現行とバイト等価**を必ず先に示す（壊していない証明）。

### 0.3 これまでに棄却済み（やり直さない）
- 4P: 防御強制(sample131)/全面攻撃禁止(135)/過剰reserve(137)/guard強化(158)/guard target-aware(160)/全編活動↑(165)/開幕活動↑+guard off(167) — **全て受動化 or 過伸びで失敗**。
- 2P: global-select / one-ply hardbad / durable-capture(153-155) / 活動量config(161/162/166) — **全て中立 or カオスミラーのノイズ or 逆効果**。
- → **config 空間は尽きた。残るは planner 構造のみ。**

---

## 1. 構造的な壁（実データで確定した真因）

### 1.1 2P: 手数・変換の構造キャップ
- 本戦リプレイ `replays_my_2p_163`（提出 sample163 の負け）を `shadow_compare` で勝者と比較:
  - 勝者は **launch ~2倍（138 vs 76, 179 vs ~80）**、**高prod中立の確保 2-3倍**。
  - eval経路の手数計測: sample8=86, sample161=84, sample166(roi0.5/waves18/sources18)=**66** → **config をどう緩めても ~80 手で頭打ち、むしろ減る**。
- **真因仮説**: プランナは `capture_floor`（確実に取れる兵数）を満たす波しか撃たない（floor-gating）。勝者は **floor 未満の小launch（牽制/増援/連鎖）を多発**して盤面を支配。我々は「確実な捕獲」だけ撃つので手数が出ず、生産レースで負ける。

### 1.2 4P: 拡張スケールの構造プラトー
- 本戦リプレイ `replays_my_4p_163`（6局）:
  - 2局: 開幕1惑星死蔵（拡張パッシブ）。
  - 4局: 中盤まで競る(5-8惑星)が、**1人だけ16-23惑星に雪だるま**→全員解体。**我々は 5-8 で頭打ち**。
- config で活動↑すると過伸びして source-stripping で死ぬ（sample165/167）。**「拡張は増やすが過伸びはしない」balance が config では作れない**。

### 1.3 共通根（planner の3点）
1. **近視眼の目的関数**: `score_candidates`（planner_core）→ `competitive_score = 自net − 敵net`、敵は `sparse_launch_flow_delta` で **do-nothing 投影**。→ (a) 生産複利の将来価値を過小評価、(b) 「良い攻め」と「過伸び」を区別できない。
2. **floor-gating**: `capture_floor` を満たす波しか採用しない → 小launch(牽制/連鎖)を打てない → 2P手数キャップ。
3. **逐次greedy**: `_greedy_select` は1ターン内で波を貪欲選択するだけ。**複数ターンにまたがる拡張キャンペーン**（取る→そこから次を取る連鎖、リード時の継続拡張）を計画できない。

---

## 2. 全体設計（フェーズ構成・各フェーズで回帰ゲート）

```
Phase 0: orbit_lite を 2P/4P に分割（挙動は変えない＝バイト等価を証明）
Phase 1: プランナ計装トレースで「何が launch を止めているか」を生データで確定
Phase 2: 2P プランナ改変（floor-gating 緩和＝小launch許可 / 生産加重）
Phase 3: 4P プランナ改変（非近視眼の目的＝拡張継続 と 過伸び抑制 の両立）
各フェーズ: 機械検証 + 回帰(現行比 非後退) + 本戦リプレイ再解析
```
**原則**: 一度に1点だけ変える。各点で「変更フラグ OFF=現行バイト等価」「ON=改善 or 非後退」を実測。**4P guard の検証(sample157)を絶対に壊さない**。

---

## 3. Phase 0 — orbit_lite の分割（低リスク・最初に完了）

### 3.1 手順
1. `sampleNNN_split_from157/` を sample157 から複製。
2. `orbit_lite/` を `orbit_lite_2p/` と `orbit_lite_4p/` に複製（中身は当面同一）。`__init__.py` のパッケージ名整合に注意。
3. `main.py` の import を**遅延・分岐**にする:
   - 現状 `from orbit_lite.planner_core import ...` をトップレベルでやっている。これを **player_count 確定後にどちらのパッケージから読むか切替**える構造に変更。
   - 最小実装案: `main.py` 冒頭で両方を別名 import（`import orbit_lite_2p.planner_core as pc2`, `import orbit_lite_4p.planner_core as pc4`）し、`run_turn`/`plan_lite_waves` 内で `pc = pc4 if player_count>=4 else pc2` のように**ディスパッチ**。関数呼び出しを `pc.xxx(...)` に置換。
   - **注意（EXPERIMENT_LOG警告）**: 動的フォルダ wrapper は過去に import/cache 効果で挙動が壊れた。**movement の cache（PlanetMovement / garrison_status）が2P/4Pで混ざらないよう**、memory に持つ cache も分離するか、分割後に1局トレースで現行一致を確認。
4. 提出zip は両 orbit_lite を同梱（フラット: main.py + orbit_lite_2p/ + orbit_lite_4p/）。

### 3.2 Phase 0 完了ゲート（必須）
- **2P**: split版 vs sample8 と、現行(sample8)版 vs sample8 を**同一seedで全500ターン trace 一致**（`tools/trace_2p.py`）。
- **4P**: split版 と sample157 を固定15+random20 で**結果完全一致**（per-seed score まで）。
- 一致しなければ分割が挙動を変えている＝バグ。修正してから Phase1 へ。

---

## 4. Phase 1 — 計装トレース（orbit_lite はまだ“読むだけ”、ロジック不変）

目的: §1.3 の仮説（floor-gating が2P手数キャップの主因）を**生データで確定**してから改変する（推測実装の連敗を繰り返さない）。

### 4.1 計装対象（`orbit_lite_*/planner_core.py`）
- `_greedy_select`: 各ターン、各波選択で `fired/not` と、**not の理由分布**（`~taken`/`can_fund`(budget)/`tgt_used_as_src`/`contrib_defended`/`score<=roi`）を集計。
- `_tier_candidates` 直後（main側でも可）: **候補総数 / valid数 / floorで落ちた数(`clears_floor==False`) / reachableで落ちた数 / scoreが-infでない数** を集計。
- 出力は env変数で file 追記（`OW_PLAN_DBG`）。**stderr は trace/eval で suppress されるので必ずファイル**。Windows path で渡す（`/tmp`不可）。

### 4.2 走らせる盤面
- 2P: 本戦負け `replays_my_2p_163` の seed を `evaluate`/`trace_2p` で再現（相手 sample8）。「手数が勝者の半分」の局で**何が launch を止めているか**を見る。
- 4P: `replays_my_4p_163` のパッシブ局(81524856, 81525524) と プラトー局(81526045 等)。

### 4.3 Phase 1 アウトプット
- 「2Pは毎ターン候補の X% が **floor 未達**で落ち、残りも roi で Y% 落ちている」等、**主ボトルネックを1つに特定**した短いレポート。これで Phase2 の設計が一点に定まる。
- もし floor-gating でなく別要因（候補生成・budget・目的関数）なら、Phase2/3 の設計を**その要因に差し替える**（本文書の改変案は仮説。計装結果が優先）。

---

## 5. Phase 2 — 2P プランナ改変（手数・変換キャップを壊す）

**前提**: Phase1 で floor-gating が主因と確証された場合の設計（他要因なら差し替え）。`orbit_lite_2p` のみ変更、`orbit_lite_4p` は不変。

### 5.1 改変案 2A: floor 未満の「支援/牽制/連鎖」launch を許可
- 現状 `_tier_candidates` は `clears_floor = sizes >= floor_at_arr` を valid 条件に入れ、満たさない候補を捨てる。
- 変更: **floor 未満でも、(a) 標的が自軍に隣接して将来 floor 到達を早める“前進配備”、(b) 既に飛行中の友軍と合流して合算で floor を超える“連鎖/協調”** を候補として残す。
  - 安全策: floor未満 launch は **`min_ships_to_launch` 以上 & source の safe_drain 内**に限る（自陣を裸にしない）。
  - スコア: 合算到達（味方の inbound と合わせた capture）を `sparse_launch_flow_delta` の launch 集合に**味方 inbound も含めて**投影し直す（単独 do-nothing でなく協調投影）。
- 期待: 手数が増え（小launch多発）、高prod中立の確保が増える＝勝者の挙動に接近。
- 検証: eval経路で**手数が ~80→ より多く**なるか（env.run 計測）。対sample8 を **複数 random40 + ミラーbaseline比**で非後退〜改善。**カオスミラーなので1pool では判断しない**。

### 5.2 改変案 2B（2Aで不足なら）: 生産加重の目的関数
- `competitive_score`（自net−敵net）に **将来生産（保持できる prod の複利近似）** を加える。ただし**加点で行動誘導すると過去全敗**なので、**「同等netの中で durable production 最大を選ぶ再ランク（タイブレーク）」**として実装（加点でなく順序）。
- 高prod中立を勝者並みに取りに行けるか（shadow / replay_analyze の高prod確保量で確認）。

### 5.3 2P 完了ゲート
- 対sample8: **複数 random40 平均でミラーbaseline を明確に上回る**（pool分散を超える差）。crash 0%。
- 本戦リプレイ再取得（提出して）で**手数・高prod確保が勝者に接近**していること。

---

## 6. Phase 3 — 4P プランナ改変（プラトーを壊す: 拡張継続 ⊕ 過伸び抑制）

**前提**: `orbit_lite_4p` のみ変更。**sample157 の guard 検証を壊さない**（guard は main 側 `_global_select_4p` にあり、orbit_lite ではない。だが score/candidate を変えると guard の入力が変わる点に注意）。

### 6.1 核心: 「拡張するが過伸びしない」を planner で両立
- config では不可能だった理由＝**目的関数が近視眼（敵do-nothing）で、リード時に拡張を継続する誘因も、過伸びを抑える先読みも無い**。
- 改変案 3A: **状態依存の拡張ドライブ**。`score_candidates` に「**自分が劣勢/同等なら中立拡張を相対的に上げ、優勢なら確実な確保を維持**」する**再ランク**（加点でなく、局面の planet/prod 比でtarget優先度を並べ替え）。本戦の「中盤頭打ち」は“リードを広げ続ける”誘因が無いことが原因。
- 改変案 3B: **2手先の過伸び先読み**を score に内包（guardの“事後drop”でなく“事前評価”）。`sparse_launch_flow_delta` の投影に **「この launch 後、最強の第三者がこの source/新領域に届くか」** を1plyだけ織り込み、過伸びを**スコアで**沈める。guard(事後)と二重にならないよう、guard は最終安全網として残す。

### 6.2 4P 完了ゲート
- **派生混在A/B**（sample157/165/167 を相手プールに）で sample157 を**明確に上回る**（今回 sample165/167 は下回った＝この基準で判定）。
- 本戦リプレイで「5-8惑星頭打ち」が改善（最大惑星数が増え、雪だるま敗が減る）。crash 0%、**受動化なし**（survival↑だけで勝ち減は棄却）。

---

## 7. 検証・回帰プロトコル（全フェーズ共通）

1. **機械検証**: `py_compile`。変更フラグ OFF で現行とバイト等価（config asdict 差分 / orbit_lite diff / 1局trace一致）。
2. **退行ゲート**: 各変更は「変更前ベース比で**勝ち数・placement 非後退 + crash 0%**」。
3. **本物の相手**: 合成弱bot単独で判断しない。4Pは派生混在A/B、2Pは複数random40+ミラーbaseline。
4. **本戦ループ**: 候補ができたら**提出して負けリプレイを再取得→`replay_analyze`/`shadow_compare`で真因が薄まったか**を確認（合成evalより上位の判定基準）。
5. **受動化の即棄却**: survival↑なのに勝ち減は §1.1 の失敗型。即棄却。

---

## 8. リスクと撤退

- orbit_lite 改変は**回帰リスク最大**。各フェーズの「OFFでバイト等価」ゲートを通れない変更は即 revert。
- **4P guard(sample157) は現行の確定改善**。Phase3 で planner を変えて guard が劣化したら、その変更を捨てて sample157 に戻す（提出は常に“検証済みの最良”を維持）。
- 効果が出ない/リスクが見合わないと判明したら、**2P/4Pとも baseline + sample157 guard で確定**し、構造改革は中止する判断もあり（configが尽きた＝planner改革しか道は無いが、planner改革のROIが負なら撤退が正解）。

---

## 9. 参照マップ（コードの当たり所）

| 関心 | ファイル/関数 |
|---|---|
| 波の貪欲選択（手数・budget・mutex） | `orbit_lite/planner_core.py: _greedy_select` |
| 候補生成（floor/reachable/score） | `main.py: _tier_candidates` + `planner_core: capture_floor, reachable_mask` |
| 標的shortlist（近接で選抜） | `planner_core: build_target_shortlist` |
| スコア（自net−敵net, do-nothing投影） | `planner_core: score_candidates, competitive_score` + `garrison_launch: sparse_launch_flow_delta` |
| safe_drain（自陣を残す上限） | `planner_core: safe_drain` |
| 4P third-party guard（事後drop） | `main.py: _global_select_4p, _thirdparty_threat_4p`（orbit_liteではない） |
| 2P/4P config | `main.py: CONFIG_2P / CONFIG_4P`（**コードに焼く**） |

---

## 10. 着手順サマリ（次のAIエージェント向け）
1. **Phase 0**: 分割 + バイト等価を証明（壊していないこと）。ここで止めて報告。
2. **Phase 1**: 計装トレースで2P手数キャップの主因を1つ確定。報告。
3. 主因が floor-gating なら **Phase 2A**（小launch許可）を最小実装→複数pool+本戦で検証。
4. 並行/後続で **Phase 3A/3B**（4P拡張継続+過伸び先読み）を派生混在A/Bで検証。
5. 各段階で §7 の回帰ゲートと「受動化即棄却」を厳守。**推測実装禁止・必ず計装/trace の生データを根拠に**。
