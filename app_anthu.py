import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import gc
import time
from pathlib import Path

# ─────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MALDI-TOF MS 跨仪器统一处理平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.4rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.6rem;
        text-align: center;
    }
    .phase-header {
        background: linear-gradient(90deg, #1f77b4 0%, #4a9eff 100%);
        color: white;
        padding: 0.7rem 1rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        font-size: 1.15rem;
        font-weight: 600;
    }
    .info-box {
        background: #f0f7ff;
        border-left: 4px solid #1f77b4;
        padding: 0.8rem 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.6rem 0;
        font-size: 0.92rem;
    }
    .warn-box {
        background: #fff8e1;
        border-left: 4px solid #ff9800;
        padding: 0.8rem 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.6rem 0;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# session state 初始化
# ─────────────────────────────────────────────
for key, default in {
    'template_cols': None,
    'template_mz_values': None,
    'template_ready': False,
    'snr_threshold': 5,
    'align_tolerance': 5,
    'strain_threshold_pct': 90,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ═════════════════════════════════════════════════════════════
# 核心函数
# ═════════════════════════════════════════════════════════════

def extract_zip_txt(uploaded_zip) -> dict:
    """解析 ZIP → {filename: raw_bytes}，只提取 .txt"""
    result = {}
    with zipfile.ZipFile(uploaded_zip, 'r') as z:
        for name in z.namelist():
            bn = Path(name).name
            if bn.lower().endswith('.txt') and not name.startswith('__MACOSX') and bn:
                result[bn] = z.read(name)
    return result


def read_anthu_txt(raw_bytes: bytes) -> pd.DataFrame:
    """
    读安图单个 txt → DataFrame(mz, peak_height, peak_area, SNR, resolution)
    编码 gbk，第1行路径，第2行表头，第3行起数据
    """
    text = raw_bytes.decode('gbk', errors='replace')
    lines = text.splitlines()
    rows = []
    for line in lines[2:]:
        parts = [p.strip() for p in line.split('\t') if p.strip()]
        if len(parts) >= 5:
            try:
                rows.append([float(parts[0]), float(parts[1]),
                             float(parts[2]), float(parts[3]), float(parts[4])])
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=['mz', 'peak_height', 'peak_area', 'SNR', 'resolution'])


def filter_anthu_peaks(df: pd.DataFrame, snr_threshold: int = 5) -> pd.DataFrame:
    """安图峰表筛选: SNR >= threshold 且 peak_area > 0"""
    mask = (df['SNR'] >= snr_threshold) & (df['peak_area'] > 0)
    return df[mask].reset_index(drop=True)


def extract_strain_id(filename: str) -> str:
    """从安图文件名提取菌株ID。  D1_F250905209902_spectrum-1.txt → D1"""
    return filename.split('_')[0]


def anthu_to_feature_vector(peaks_df: pd.DataFrame,
                            template_mz: np.ndarray,
                            tolerance: float) -> np.ndarray:
    """
    单个安图样本峰表 → 按模板对齐的特征向量（TIC归一化）
    每个模板 mz 找距离 <= tolerance 的峰中 peak_height 最大值；未命中填 0
    """
    n = len(template_mz)
    vec = np.zeros(n)
    if len(peaks_df) == 0:
        return vec

    atu_mz = peaks_df['mz'].values
    atu_height = peaks_df['peak_height'].values.astype(float)

    for i, tmz in enumerate(template_mz):
        diffs = np.abs(atu_mz - tmz)
        mask = diffs <= tolerance
        if mask.any():
            vec[i] = atu_height[mask].max()

    s = vec.sum()
    if s > 0:
        vec = vec / s
    return vec


def cluster_mz(mz_array: np.ndarray, tolerance: float) -> np.ndarray:
    """距离 <= tolerance 的 mz 聚为一簇，取中位数。输入需已排序或会内部排序。"""
    if len(mz_array) == 0:
        return np.array([])
    sorted_mz = np.sort(mz_array)
    clusters, cur = [], [sorted_mz[0]]
    for mz in sorted_mz[1:]:
        if mz - cur[-1] <= tolerance:
            cur.append(mz)
        else:
            clusters.append(cur)
            cur = [mz]
    clusters.append(cur)
    return np.array([np.median(c) for c in clusters])


def find_anthu_unique_candidates(bruker_csv_df: pd.DataFrame,
                                 all_anthu_peaks: list,
                                 tolerance: float) -> np.ndarray:
    """
    找出所有安图峰中对齐不到布鲁克任何特征的候选 mz，聚类去重后返回中位数数组。
    """
    brk_cols = [c for c in bruker_csv_df.columns if c.startswith('mz_')]
    brk_mz = np.array([float(c.replace('mz_', '')) for c in brk_cols])

    all_atu_mz = []
    for fn, peaks_df in all_anthu_peaks:
        all_atu_mz.extend(peaks_df['mz'].values.tolist())
    if len(all_atu_mz) == 0:
        return np.array([])

    all_atu_mz = np.array(all_atu_mz)
    # 筛选：对齐不到布鲁克任何特征的峰
    candidates = []
    for am in all_atu_mz:
        if len(brk_mz) == 0 or np.min(np.abs(brk_mz - am)) > tolerance:
            candidates.append(am)
    if len(candidates) == 0:
        return np.array([])
    return cluster_mz(np.array(candidates), tolerance)


def compute_strain_detection(all_anthu_peaks: list,
                             candidate_mz: np.ndarray,
                             tolerance: float) -> pd.DataFrame:
    """
    对每个候选特征 mz，统计在多少菌株中至少有1个样本检测到。
    返回 DataFrame: mz, col_name, n_strains_detected, total_strains, detection_pct,
                    n_samples_detected, total_samples
    """
    if len(candidate_mz) == 0:
        return pd.DataFrame(columns=['mz', 'col_name', 'n_strains_detected',
                                     'total_strains', 'detection_pct',
                                     'n_samples_detected', 'total_samples'])

    # 按菌株分组
    strain_samples: dict[str, list] = {}
    for fn, peaks_df in all_anthu_peaks:
        sid = extract_strain_id(fn)
        strain_samples.setdefault(sid, []).append(peaks_df)

    total_strains = len(strain_samples)
    total_samples = len(all_anthu_peaks)

    records = []
    for cmz in candidate_mz:
        strains_hit = 0
        samples_hit = 0
        for sid, sample_list in strain_samples.items():
            strain_has = False
            for pdf in sample_list:
                if len(pdf) > 0 and np.any(np.abs(pdf['mz'].values - cmz) <= tolerance):
                    strain_has = True
                    samples_hit += 1
            if strain_has:
                strains_hit += 1

        records.append({
            'mz': round(float(cmz), 1),
            'col_name': f"mz_{int(round(cmz))}",
            'n_strains_detected': strains_hit,
            'total_strains': total_strains,
            'detection_pct': round(strains_hit / total_strains * 100, 1) if total_strains > 0 else 0.0,
            'n_samples_detected': samples_hit,
            'total_samples': total_samples,
        })
    return pd.DataFrame(records)


def build_unified_template(bruker_csv_df: pd.DataFrame,
                           candidate_mz_kept: np.ndarray):
    """布鲁克特征 + 经筛选保留的安图特征 → 统一模板（升序）"""
    brk_cols = [c for c in bruker_csv_df.columns if c.startswith('mz_')]
    brk_mz = np.array([float(c.replace('mz_', '')) for c in brk_cols])

    if len(candidate_mz_kept) == 0:
        return brk_cols, brk_mz

    all_mz = np.concatenate([brk_mz, candidate_mz_kept])
    sort_idx = np.argsort(all_mz)
    unified_mz = all_mz[sort_idx]
    unified_cols = [f"mz_{int(round(m))}" for m in unified_mz]
    return unified_cols, unified_mz


def bruker_csv_to_unified(bruker_csv_df: pd.DataFrame,
                          unified_mz: np.ndarray,
                          tolerance: float) -> np.ndarray:
    """布鲁克 CSV 每行映射到统一模板，新增安图列填 0，数值保持原样。"""
    brk_cols = [c for c in bruker_csv_df.columns if c.startswith('mz_')]
    brk_mz = np.array([float(c.replace('mz_', '')) for c in brk_cols])
    brk_values = bruker_csv_df[brk_cols].values.astype(float)

    out = np.zeros((len(bruker_csv_df), len(unified_mz)))
    for j, bmz in enumerate(brk_mz):
        diffs = np.abs(unified_mz - bmz)
        nearest = np.argmin(diffs)
        if diffs[nearest] <= tolerance:
            out[:, nearest] = brk_values[:, j]
    return out


# ═════════════════════════════════════════════════════════════
# 侧边栏
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 处理参数")
    st.markdown('<div class="info-box">参数调整放在<b>阶段1主区域</b>下方，方便随时修改。</div>', unsafe_allow_html=True)

    st.divider()
    st.header("💾 内存管理")
    if st.button("🧹 清理缓存（保留模板）", use_container_width=True):
        keys_to_keep = {'template_cols', 'template_mz_values', 'template_ready',
                        'snr_threshold', 'align_tolerance', 'strain_threshold_pct'}
        for k in list(st.session_state.keys()):
            if k not in keys_to_keep:
                del st.session_state[k]
        gc.collect()
        st.success("已清理")
        st.rerun()
    if st.button("🗑️ 完全清空", use_container_width=True):
        st.session_state.clear()
        gc.collect()
        st.success("已清空")
        st.rerun()

    st.divider()
    st.header("📖 流程说明")
    st.markdown("""
    **阶段1（训练集）：**
    - 上传布鲁克 CSV + 安图 ZIP
    - 布鲁克 91 列作为模板核心，保持不变
    - 安图独有峰按**菌株检测率**筛选：
      只有在足够多菌株中都能检测到的峰才追加为新特征，砍掉稀疏噪声列
    - 输出：筛选后的统一特征矩阵 + 模板

    **阶段2（验证集）：**
    - 只需上传安图 ZIP
    - 用阶段1的模板对齐，维度一致
    - 模板之外的峰自动忽略

    **菌株检测率逻辑：**
    每个安图独有特征，检查在多少菌株中至少有 1 个重复样本检测到。
    低于阈值的特征砍掉，避免模型噪声。
    """)


# ═════════════════════════════════════════════════════════════
# 主界面
# ═════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">🔬 MALDI-TOF MS 跨仪器统一处理平台</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">布鲁克 CSV + 安图 TXT → 菌株交集筛选 → 统一特征矩阵</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎯 阶段1: 训练集 → 建立模板", "🔄 阶段2: 验证集 → 应用模板"])


# ═════════════════════════════════════════════════════════════
# 阶段1
# ═════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="phase-header">📊 阶段1: 训练集处理，建立统一特征模板</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    <b>布鲁克</b>：上传之前用 R 处理好的 CSV（需含 <code>group</code> 列和 <code>mz_xxxx</code> 特征列）<br>
    <b>安图</b>：上传包含所有 txt 的 ZIP 压缩包<br>
    系统以布鲁克特征为核心模板，安图独有峰经<b>菌株检测率筛选</b>后才追加为新特征，大幅减少噪声列。
    </div>""", unsafe_allow_html=True)

    # ── 上传区 ──
    col_brk, col_atu = st.columns(2)
    with col_brk:
        brk_csv_upload = st.file_uploader("📁 布鲁克 CSV", type=['csv'], key='brk_csv',
                                          help="R 处理后的峰强度矩阵")
    with col_atu:
        atu_zip_upload = st.file_uploader("📁 安图 ZIP", type=['zip'], key='atu_zip',
                                          help="包含所有安图 txt 的压缩包")

    # ── 预检查 ──
    brk_df = None
    atu_files = {}

    if brk_csv_upload:
        brk_df = pd.read_csv(brk_csv_upload)
        brk_mz_cols = [c for c in brk_df.columns if c.startswith('mz_')]
        has_group = 'group' in brk_df.columns
        st.success("✅ 布鲁克 CSV 读入成功")
        c1, c2, c3 = st.columns(3)
        c1.metric("样本数", len(brk_df))
        c2.metric("特征数", len(brk_mz_cols))
        c3.metric("含 group 列", "✅ 是" if has_group else "❌ 否")

    if atu_zip_upload:
        atu_files = extract_zip_txt(atu_zip_upload)
        st.success(f"✅ 安图 ZIP 读入成功：{len(atu_files)} 个 txt")
        strains_detected = set(extract_strain_id(fn) for fn in atu_files.keys())
        st.info(f"🧬 检测到 {len(strains_detected)} 个菌株，每株约 {len(atu_files) // max(len(strains_detected), 1)} 个重复")

    # ── 参数控件 ──
    st.divider()
    p1, p2, p3 = st.columns(3)
    with p1:
        phase1_snr = st.slider(
            "🔍 安图 SNR 阈值", min_value=3, max_value=15,
            value=st.session_state.snr_threshold,
            help="安图峰表中 SNR < 此值的峰会被丢弃"
        )
        st.session_state.snr_threshold = phase1_snr
    with p2:
        phase1_tol = st.slider(
            "📏 对齐容差 (Da)", min_value=1, max_value=15,
            value=st.session_state.align_tolerance,
            help="安图峰与布鲁克特征 mz 的最大允许偏差"
        )
        st.session_state.align_tolerance = phase1_tol
    with p3:
        phase1_strain_pct = st.slider(
            "🧬 菌株检测率阈值 (%)", min_value=50, max_value=100,
            value=st.session_state.strain_threshold_pct, step=5,
            help="安图独有特征必须在此比例以上的菌株中检测到才能保留。越高越严格，噪声越少；越低保留特征越多。"
        )
        st.session_state.strain_threshold_pct = phase1_strain_pct

    # ── 处理按钮 ──
    can_run = (brk_df is not None) and (len(atu_files) > 0)
    if not can_run:
        st.markdown('<div class="warn-box">⚠️ 请同时上传布鲁克 CSV 和安图 ZIP 才能开始处理。</div>', unsafe_allow_html=True)

    if can_run and st.button("🎯 开始处理训练集", type="primary", use_container_width=True):

        progress = st.progress(0)
        status = st.empty()
        tolerance    = st.session_state.align_tolerance
        snr_thresh   = st.session_state.snr_threshold
        strain_pct   = st.session_state.strain_threshold_pct

        # ── Step 1: 读取并筛选所有安图峰 ──
        status.text("📖 Step 1/5: 读取安图 txt 并筛选峰...")
        progress.progress(5)

        all_anthu_peaks = []
        atu_read_info   = []
        for fn, raw in atu_files.items():
            raw_df  = read_anthu_txt(raw)
            filt_df = filter_anthu_peaks(raw_df, snr_thresh)
            all_anthu_peaks.append((fn, filt_df))
            atu_read_info.append((fn, len(raw_df), len(filt_df)))

        progress.progress(12)

        # ── Step 2: 找安图独有候选特征 ──
        status.text("🔍 Step 2/5: 识别安图独有候选特征（聚类去重）...")
        candidate_mz_all = find_anthu_unique_candidates(brk_df, all_anthu_peaks, tolerance)
        n_candidates_raw = len(candidate_mz_all)
        progress.progress(25)

        # ── Step 3: 菌株检测率计算 + 筛选 ──
        status.text("🧬 Step 3/5: 按菌株检测率筛选候选特征...")
        detection_df = compute_strain_detection(all_anthu_peaks, candidate_mz_all, tolerance)

        if len(detection_df) > 0:
            mask_keep       = detection_df['detection_pct'] >= strain_pct
            kept_df         = detection_df[mask_keep].copy()
            dropped_df      = detection_df[~mask_keep].copy()
            candidate_mz_kept = kept_df['mz'].values.astype(float)
        else:
            kept_df           = detection_df.copy()
            dropped_df        = detection_df.copy()
            candidate_mz_kept = np.array([])

        progress.progress(45)

        # ── Step 4: 构建统一模板 ──
        status.text("📐 Step 4/5: 构建统一特征模板...")
        unified_cols, unified_mz = build_unified_template(brk_df, candidate_mz_kept)
        n_brk_features  = len([c for c in brk_df.columns if c.startswith('mz_')])
        n_new_features  = len(unified_cols) - n_brk_features
        progress.progress(55)

        # ── Step 5: 映射所有样本 ──
        status.text("📊 Step 5/5: 映射样本到统一特征矩阵...")

        # 布鲁克映射
        brk_matrix = bruker_csv_to_unified(brk_df, unified_mz, tolerance)
        brk_out = pd.DataFrame(brk_matrix, columns=unified_cols)

        # ── TIC 归一化：每行 / 行总和 → 和安图一致（比例含义） ──
        brk_rowsum = brk_out[unified_cols].sum(axis=1)
        brk_out[unified_cols] = brk_out[unified_cols].div(brk_rowsum.replace(0, np.nan), axis=0).fillna(0)

        brk_out.insert(0, 'sample',
                       brk_df['group'].astype(str).values if 'group' in brk_df.columns
                       else [f"bruker_{i}" for i in range(len(brk_df))])
        brk_out.insert(1, 'instrument', 'bruker')
        progress.progress(70)

        # 安图映射
        atu_rows = []
        for fn, filt_df in all_anthu_peaks:
            atu_rows.append(anthu_to_feature_vector(filt_df, unified_mz, tolerance))

        atu_out = pd.DataFrame(np.vstack(atu_rows), columns=unified_cols)
        atu_out.insert(0, 'sample', [r[0] for r in all_anthu_peaks])
        atu_out.insert(1, 'instrument', 'anthu')
        progress.progress(85)

        # ── 合并 & 保存模板 ──
        combined = pd.concat([brk_out, atu_out], ignore_index=True)
        st.session_state.template_cols      = unified_cols
        st.session_state.template_mz_values = unified_mz
        st.session_state.template_ready     = True

        progress.progress(100)
        status.text("✅ 完成！")
        time.sleep(0.4)
        progress.empty()
        status.empty()

        # ═══════════════════════════════════════════
        # 结果展示
        # ═══════════════════════════════════════════
        st.success("✅ 训练集处理完成，统一特征模板已建立！")

        # ── 核心指标 ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总样本数",   len(combined))
        c2.metric("最终特征数", len(unified_cols),
                  delta=f"{n_new_features:+d} (安图新增)")
        c3.metric("布鲁克核心特征", n_brk_features)
        c4.metric("安图筛选后新增", n_new_features)

        c1, c2, c3 = st.columns(3)
        c1.metric("布鲁克样本数", len(brk_out))
        c2.metric("安图样本数",  len(atu_out))
        c3.metric("安图菌株数",  len(set(extract_strain_id(fn) for fn, _ in all_anthu_peaks)))

        # ── TIC 归一化验证 ──
        st.divider()
        st.markdown("### 📊 TIC 归一化验证（两种仪器强度已对齐）")
        brk_final_rowsum = brk_out[unified_cols].sum(axis=1)
        atu_final_rowsum = atu_out[unified_cols].sum(axis=1)
        nc1, nc2 = st.columns(2)
        nc1.metric("布鲁克 行总和", f"{brk_final_rowsum.mean():.4f}",
                   help="归一化后每行和=1，表示各峰占本样本总信号的比例")
        nc2.metric("安图   行总和", f"{atu_final_rowsum.mean():.4f}",
                   help="和布鲁克同一基准，直接可比")

        # ── 菌株筛选报告 ──
        st.divider()
        st.markdown("### 🧬 菌株检测率筛选报告")

        total_strains  = int(detection_df['total_strains'].iloc[0]) if len(detection_df) > 0 else 0
        strain_thresh_n = int(np.ceil(strain_pct / 100.0 * total_strains))

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("候选特征总数（去重后）", n_candidates_raw)
        rc2.metric(f"保留（检测率 ≥ {strain_pct}%，即 ≥ {strain_thresh_n} 菌株）",
                   f"{len(kept_df)} 个",
                   delta=f"{len(kept_df) - n_candidates_raw}")
        rc3.metric("砍掉（太稀疏）", f"{len(dropped_df)} 个")

        # 保留特征明细
        if len(kept_df) > 0:
            with st.expander(f"✅ 保留的 {len(kept_df)} 个安图特征", expanded=True):
                show = kept_df[['col_name','mz','n_strains_detected','total_strains',
                                'detection_pct','n_samples_detected','total_samples']].copy()
                show.columns = ['特征名','mz值','检测菌株数','总菌株数','检测率(%)','检测样本数','总样本数']
                st.dataframe(show.sort_values('检测率(%)', ascending=False).reset_index(drop=True),
                             use_container_width=True, hide_index=True)

        # 砍掉特征明细
        if len(dropped_df) > 0:
            with st.expander(f"❌ 砍掉的 {len(dropped_df)} 个特征（检测率太低）"):
                show = dropped_df[['col_name','mz','n_strains_detected','total_strains',
                                   'detection_pct','n_samples_detected','total_samples']].copy()
                show.columns = ['特征名','mz值','检测菌株数','总菌株数','检测率(%)','检测样本数','总样本数']
                st.dataframe(show.sort_values('检测率(%)', ascending=False).reset_index(drop=True),
                             use_container_width=True, hide_index=True)

        # ── 安图文件明细 ──
        st.divider()
        st.subheader("📋 安图文件明细")
        st.dataframe(
            pd.DataFrame(atu_read_info, columns=['文件名','原始峰数', f'筛选后峰数(SNR≥{snr_thresh})']),
            use_container_width=True, hide_index=True
        )

        # ── 特征矩阵预览 ──
        with st.expander("📊 合并特征矩阵预览（前10列）"):
            st.dataframe(combined[['sample','instrument'] + unified_cols[:10]].round(6),
                         use_container_width=True, hide_index=True)

        # ── 下载 ──
        st.divider()
        st.subheader("📥 下载")
        dl1, dl2 = st.columns(2)
        export_cols = ['sample'] + unified_cols

        dl1.download_button(
            "📊 训练集统一特征矩阵 CSV",
            data=combined[export_cols].to_csv(index=False),
            file_name="train_feature_matrix_unified.csv",
            mime="text/csv", use_container_width=True
        )
        dl2.download_button(
            "🎯 特征模板 CSV",
            data=pd.DataFrame({'feature_name': unified_cols,
                               'mz_value': unified_mz.round(1)}).to_csv(index=False),
            file_name="feature_template.csv",
            mime="text/csv", use_container_width=True
        )

        del combined, brk_out, atu_out, brk_matrix
        gc.collect()


# ═════════════════════════════════════════════════════════════
# 阶段2
# ═════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="phase-header">🔄 阶段2: 验证集处理（使用训练集模板）</div>', unsafe_allow_html=True)

    if not st.session_state.template_ready:
        st.markdown('<div class="warn-box">⚠️ 请先完成阶段1，建立特征模板后才能处理验证集。</div>', unsafe_allow_html=True)
    else:
        unified_cols = st.session_state.template_cols
        unified_mz   = st.session_state.template_mz_values

        st.success(f"✅ 特征模板就绪：{len(unified_cols)} 个特征")
        st.markdown("""<div class="info-box">
        上传安图 txt 的 ZIP，系统按训练集模板对齐，输出与训练集<b>维度完全一致</b>的特征矩阵。<br>
        模板之外的峰自动忽略；布鲁克独有的高 mz 特征在安图样本中填 0。
        </div>""", unsafe_allow_html=True)

        valid_zip_upload = st.file_uploader("📁 安图验证集 ZIP", type=['zip'], key='valid_zip')

        if valid_zip_upload:
            valid_files = extract_zip_txt(valid_zip_upload)
            if not valid_files:
                st.error("ZIP 中没有找到 .txt 文件")
            else:
                st.success(f"✅ 找到 {len(valid_files)} 个 txt 文件")

                # 参数控件
                p1, p2 = st.columns(2)
                with p1:
                    phase2_snr = st.slider(
                        "🔍 安图 SNR 阈值", min_value=3, max_value=15,
                        value=st.session_state.snr_threshold, key='valid_snr',
                        help="安图峰表中 SNR < 此值的峰会被丢弃"
                    )
                    st.session_state.snr_threshold = phase2_snr
                with p2:
                    phase2_tol = st.slider(
                        "📏 对齐容差 (Da)", min_value=1, max_value=15,
                        value=st.session_state.align_tolerance, key='valid_tol',
                        help="安图峰与模板特征 mz 的最大允许偏差。超出部分自动忽略"
                    )
                    st.session_state.align_tolerance = phase2_tol

                if st.button("🔄 开始处理验证集", type="primary", use_container_width=True):
                    progress = st.progress(0)
                    status   = st.empty()
                    tolerance  = st.session_state.align_tolerance
                    snr_thresh = st.session_state.snr_threshold

                    status.text("📖 读取并处理安图 txt...")
                    progress.progress(10)

                    valid_rows, valid_info = [], []
                    total = len(valid_files)
                    for i, (fn, raw) in enumerate(valid_files.items()):
                        raw_df  = read_anthu_txt(raw)
                        filt_df = filter_anthu_peaks(raw_df, snr_thresh)
                        valid_rows.append(anthu_to_feature_vector(filt_df, unified_mz, tolerance))
                        valid_info.append((fn, len(raw_df), len(filt_df)))
                        progress.progress(10 + int(75 * (i + 1) / total))

                    status.text("📦 构建输出矩阵...")
                    valid_df = pd.DataFrame(np.vstack(valid_rows), columns=unified_cols)
                    valid_df.insert(0, 'sample',     [r[0] for r in valid_info])
                    valid_df.insert(1, 'instrument', 'anthu')

                    progress.progress(100)
                    status.text("✅ 完成！")
                    time.sleep(0.4)
                    progress.empty()
                    status.empty()

                    # ═══ 结果展示 ═══
                    st.success("✅ 验证集处理完成！特征维度与训练集一致！")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("样本数",   len(valid_df))
                    c2.metric("特征数",   len(unified_cols))
                    c3.metric("特征一致性", "✅ 与训练集一致")

                    st.subheader("📋 文件明细")
                    st.dataframe(
                        pd.DataFrame(valid_info, columns=['文件名','原始峰数', f'筛选后峰数(SNR≥{snr_thresh})']),
                        use_container_width=True, hide_index=True
                    )

                    # 非零特征统计
                    feat_data      = valid_df[unified_cols].astype(float)
                    nonzero_counts = (feat_data > 0).sum(axis=1)
                    with st.expander("📊 各样本非零特征数与覆盖率"):
                        st.dataframe(pd.DataFrame({
                            '文件名':   valid_df['sample'].values,
                            '非零特征数': nonzero_counts.values,
                            '覆盖率':   (nonzero_counts / len(unified_cols) * 100).round(1).astype(str) + '%'
                        }), use_container_width=True, hide_index=True)

                    with st.expander("📊 特征矩阵预览（前10列）"):
                        st.dataframe(valid_df[['sample'] + unified_cols[:10]].round(6),
                                     use_container_width=True, hide_index=True)

                    # ── 下载 ──
                    st.divider()
                    st.download_button(
                        "📊 下载验证集特征矩阵 CSV",
                        data=valid_df[['sample'] + unified_cols].to_csv(index=False),
                        file_name="valid_feature_matrix_unified.csv",
                        mime="text/csv", use_container_width=True
                    )

                    del valid_df, feat_data
                    gc.collect()


# ── 底部 ──
st.divider()
st.markdown("<div style='text-align:center;color:#aaa;font-size:0.85rem;'>MALDI-TOF MS 跨仪器统一处理平台 · 布鲁克核心 + 安图菌株交集筛选</div>", unsafe_allow_html=True)
