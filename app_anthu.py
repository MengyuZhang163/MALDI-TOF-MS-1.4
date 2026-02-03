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
    'template_cols': None,            # 最终统一特征列名 list[str]
    'template_mz_values': None,       # 对应的 mz 数值 np.array
    'template_ready': False,
    'snr_threshold': 5,
    'align_tolerance': 5,
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


def anthu_to_feature_vector(peaks_df: pd.DataFrame,
                            template_mz: np.ndarray,
                            tolerance: float) -> np.ndarray:
    """
    将单个安图样本的峰表，按模板 mz 列表和容差对齐 → 特征向量
    - 每个模板 mz 在安图峰中找距离 <= tolerance 的最近峰，取其 peak_height
    - 未命中则为 0
    - 最后 TIC 归一化（/ sum）
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
            # 命中的峰中取 peak_height 最大的
            vec[i] = atu_height[mask].max()

    # TIC 归一化
    s = vec.sum()
    if s > 0:
        vec = vec / s
    return vec


def build_unified_template(bruker_csv_df: pd.DataFrame,
                           all_anthu_peaks: list,
                           tolerance: float):
    """
    以布鲁克 CSV 的特征列为基础，扫描所有安图样本的峰，
    找出无法对齐到布鲁克任何特征的峰 → 去重后追加为新特征。
    返回 (unified_cols: list[str], unified_mz: np.ndarray)
    """
    # 布鲁克已有的 mz
    brk_cols = [c for c in bruker_csv_df.columns if c.startswith('mz_')]
    brk_mz = np.array([float(c.replace('mz_', '')) for c in brk_cols])

    # 收集所有安图峰的 mz
    all_atu_mz = []
    for fn, peaks_df in all_anthu_peaks:
        all_atu_mz.extend(peaks_df['mz'].values.tolist())

    if len(all_atu_mz) == 0:
        return brk_cols, brk_mz

    all_atu_mz = np.array(all_atu_mz)

    # 筛选：哪些安图峰对齐不到布鲁克的任何特征
    new_mz_candidates = []
    for am in all_atu_mz:
        if len(brk_mz) == 0 or np.min(np.abs(brk_mz - am)) > tolerance:
            new_mz_candidates.append(am)

    if len(new_mz_candidates) == 0:
        return brk_cols, brk_mz

    # 对新候选峰去重（距离 <= tolerance 的聚为一簇，取中位数）
    new_mz_candidates = np.sort(new_mz_candidates)
    clusters = []
    current_cluster = [new_mz_candidates[0]]
    for mz in new_mz_candidates[1:]:
        if mz - current_cluster[-1] <= tolerance:
            current_cluster.append(mz)
        else:
            clusters.append(current_cluster)
            current_cluster = [mz]
    clusters.append(current_cluster)

    new_feature_mz = np.array([np.median(c) for c in clusters])

    # 合并并按 mz 升序排列
    all_mz = np.concatenate([brk_mz, new_feature_mz])
    sort_idx = np.argsort(all_mz)
    unified_mz = all_mz[sort_idx]
    unified_cols = [f"mz_{int(round(m))}" for m in unified_mz]

    return unified_cols, unified_mz


def bruker_csv_to_unified(bruker_csv_df: pd.DataFrame,
                          unified_mz: np.ndarray,
                          tolerance: float) -> np.ndarray:
    """
    将布鲁克 CSV 每行映射到统一模板。
    布鲁克原有列按最近邻对齐，新增的安图列填 0。
    数值保持原样（R 已处理好）。
    """
    brk_cols = [c for c in bruker_csv_df.columns if c.startswith('mz_')]
    brk_mz = np.array([float(c.replace('mz_', '')) for c in brk_cols])
    brk_values = bruker_csv_df[brk_cols].values.astype(float)

    n_samples = len(bruker_csv_df)
    n_unified = len(unified_mz)
    out = np.zeros((n_samples, n_unified))

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
    st.markdown('<div class="info-box">参数对<b>安图数据处理</b>生效；<br>布鲁克数据已是处理好的 CSV，直接读入。</div>', unsafe_allow_html=True)

    snr_threshold = st.slider(
        "安图 SNR 阈值", min_value=3, max_value=15, value=5,
        help="安图峰表中 SNR < 此值的峰会被丢弃"
    )
    align_tolerance = st.slider(
        "跨仪器对齐容差 (Da)", min_value=1, max_value=15, value=5,
        help="安图峰与模板特征 mz 之间的最大允许偏差。超出此范围的峰会作为新特征追加（阶段1）或直接忽略（阶段2）"
    )

    st.session_state.snr_threshold = snr_threshold
    st.session_state.align_tolerance = align_tolerance

    st.divider()
    st.header("💾 内存管理")
    if st.button("🧹 清理缓存（保留模板）", use_container_width=True):
        keys_to_keep = {'template_cols', 'template_mz_values', 'template_ready',
                        'snr_threshold', 'align_tolerance'}
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
    - 上传布鲁克已处理的 CSV（含 group 列 + mz 特征列）
    - 上传安图 txt 的 ZIP
    - 以布鲁克特征为基础，安图峰按容差对齐；
      对齐不到的安图峰追加为新特征
    - 输出：统一特征矩阵 + 特征模板

    **阶段2（验证集）：**
    - 只需上传安图 txt 的 ZIP
    - 用阶段1的模板对齐，输出维度与训练集一致
    - 模板之外的峰自动忽略
    """)


# ═════════════════════════════════════════════════════════════
# 主界面
# ═════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">🔬 MALDI-TOF MS 跨仪器统一处理平台</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">布鲁克 CSV + 安图 TXT → 统一特征矩阵</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎯 阶段1: 训练集 → 建立模板", "🔄 阶段2: 验证集 → 应用模板"])


# ═════════════════════════════════════════════════════════════
# 阶段1
# ═════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="phase-header">📊 阶段1: 训练集处理，建立统一特征模板</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    <b>布鲁克</b>：上传之前用 R 处理好的 CSV（需含 <code>group</code> 列和 <code>mz_xxxx</code> 特征列）<br>
    <b>安图</b>：上传包含所有 txt 的 ZIP 压缩包<br>
    系统会自动将两者对齐到统一的特征空间。
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

        st.success(f"✅ 布鲁克 CSV 读入成功")
        c1, c2, c3 = st.columns(3)
        c1.metric("样本数", len(brk_df))
        c2.metric("特征数", len(brk_mz_cols))
        c3.metric("含 group 列", "✅ 是" if has_group else "❌ 否")

    if atu_zip_upload:
        atu_files = extract_zip_txt(atu_zip_upload)
        st.success(f"✅ 安图 ZIP 读入成功：{len(atu_files)} 个 txt")

    # ── 处理按钮 ──
    can_run = (brk_df is not None) and (len(atu_files) > 0)
    if not can_run:
        st.markdown('<div class="warn-box">⚠️ 请同时上传布鲁克 CSV 和安图 ZIP 才能开始处理。</div>', unsafe_allow_html=True)

    if can_run and st.button("🎯 开始处理训练集", type="primary", use_container_width=True):

        progress = st.progress(0)
        status = st.empty()
        tolerance = st.session_state.align_tolerance
        snr_thresh = st.session_state.snr_threshold

        # ── Step 1: 读取并筛选所有安图峰 ──
        status.text("📖 Step 1/4: 读取安图 txt 并筛选峰...")
        progress.progress(5)

        all_anthu_peaks = []
        atu_read_info = []
        for fn, raw in atu_files.items():
            raw_df = read_anthu_txt(raw)
            filt_df = filter_anthu_peaks(raw_df, snr_thresh)
            all_anthu_peaks.append((fn, filt_df))
            atu_read_info.append((fn, len(raw_df), len(filt_df)))

        progress.progress(15)

        # ── Step 2: 构建统一模板 ──
        status.text("🔗 Step 2/4: 构建统一特征模板...")

        unified_cols, unified_mz = build_unified_template(brk_df, all_anthu_peaks, tolerance)
        n_brk_features = len([c for c in brk_df.columns if c.startswith('mz_')])
        n_new_features = len(unified_cols) - n_brk_features

        progress.progress(30)

        # ── Step 3: 布鲁克 → 统一矩阵 ──
        status.text("📊 Step 3/4: 映射布鲁克和安图到统一特征矩阵...")

        brk_matrix = bruker_csv_to_unified(brk_df, unified_mz, tolerance)
        brk_out = pd.DataFrame(brk_matrix, columns=unified_cols)
        brk_out.insert(0, 'sample',
                       brk_df['group'].astype(str).values if 'group' in brk_df.columns
                       else [f"bruker_{i}" for i in range(len(brk_df))])
        brk_out.insert(1, 'instrument', 'bruker')

        progress.progress(55)

        # ── Step 4: 安图 → 统一矩阵 ──
        atu_rows = []
        for fn, filt_df in all_anthu_peaks:
            vec = anthu_to_feature_vector(filt_df, unified_mz, tolerance)
            atu_rows.append(vec)

        atu_out = pd.DataFrame(np.vstack(atu_rows), columns=unified_cols)
        atu_out.insert(0, 'sample', [r[0] for r in all_anthu_peaks])
        atu_out.insert(1, 'instrument', 'anthu')

        progress.progress(75)

        # ── 合并 ──
        status.text("📦 Step 4/4: 合并输出...")
        combined = pd.concat([brk_out, atu_out], ignore_index=True)

        # 保存模板到 session
        st.session_state.template_cols = unified_cols
        st.session_state.template_mz_values = unified_mz
        st.session_state.template_ready = True

        progress.progress(100)
        status.text("✅ 完成！")
        time.sleep(0.4)
        progress.empty()
        status.empty()

        # ═══ 结果展示 ═══
        st.success("✅ 训练集处理完成，统一特征模板已建立！")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总样本数", len(combined))
        c2.metric("统一特征数", len(unified_cols))
        c3.metric("布鲁克原有特征", n_brk_features)
        c4.metric("安图新增特征", n_new_features)

        c1, c2 = st.columns(2)
        c1.metric("布鲁克样本数", len(brk_out))
        c2.metric("安图样本数", len(atu_out))

        # 安图文件明细
        st.subheader("📋 安图文件明细")
        info_df = pd.DataFrame(atu_read_info,
                               columns=['文件名', '原始峰数', f'筛选后峰数(SNR≥{snr_thresh})'])
        st.dataframe(info_df, use_container_width=True, hide_index=True)

        # 对齐详情
        with st.expander("🔗 对齐详情（新增特征列表）"):
            brk_mz_arr = np.array([float(c.replace('mz_', '')) for c in brk_df.columns if c.startswith('mz_')])
            new_features_info = []
            for col, mz in zip(unified_cols, unified_mz):
                if len(brk_mz_arr) == 0 or np.min(np.abs(brk_mz_arr - mz)) > tolerance:
                    new_features_info.append({'新增特征': col, 'mz值': f"{mz:.1f}", '来源': '安图独有'})
            if new_features_info:
                st.dataframe(pd.DataFrame(new_features_info), use_container_width=True, hide_index=True)
            else:
                st.info("所有安图峰均已对齐到布鲁克特征，无新增列。")

        # 特征矩阵预览
        with st.expander("📊 合并特征矩阵预览（前12列）"):
            preview_cols = ['sample', 'instrument'] + unified_cols[:10]
            st.dataframe(combined[preview_cols].round(6), use_container_width=True, hide_index=True)

        # ── 下载 ──
        st.divider()
        st.subheader("📥 下载")
        c1, c2 = st.columns(2)

        export_cols = ['sample'] + unified_cols
        c1.download_button(
            "📊 训练集统一特征矩阵 CSV",
            data=combined[export_cols].to_csv(index=False),
            file_name="train_feature_matrix_unified.csv",
            mime="text/csv",
            use_container_width=True
        )

        template_export = pd.DataFrame({
            'feature_name': unified_cols,
            'mz_value': unified_mz.round(1)
        })
        c2.download_button(
            "🎯 特征模板 CSV",
            data=template_export.to_csv(index=False),
            file_name="feature_template.csv",
            mime="text/csv",
            use_container_width=True
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
        unified_mz = st.session_state.template_mz_values

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

                if st.button("🔄 开始处理验证集", type="primary", use_container_width=True):
                    progress = st.progress(0)
                    status = st.empty()
                    tolerance = st.session_state.align_tolerance
                    snr_thresh = st.session_state.snr_threshold

                    status.text("📖 读取并处理安图 txt...")
                    progress.progress(10)

                    valid_rows = []
                    valid_info = []
                    total = len(valid_files)

                    for i, (fn, raw) in enumerate(valid_files.items()):
                        raw_df = read_anthu_txt(raw)
                        filt_df = filter_anthu_peaks(raw_df, snr_thresh)
                        vec = anthu_to_feature_vector(filt_df, unified_mz, tolerance)
                        valid_rows.append(vec)
                        valid_info.append((fn, len(raw_df), len(filt_df)))
                        progress.progress(10 + int(75 * (i + 1) / total))

                    status.text("📦 构建输出矩阵...")

                    valid_df = pd.DataFrame(np.vstack(valid_rows), columns=unified_cols)
                    valid_df.insert(0, 'sample', [r[0] for r in valid_info])
                    valid_df.insert(1, 'instrument', 'anthu')

                    progress.progress(100)
                    status.text("✅ 完成！")
                    time.sleep(0.4)
                    progress.empty()
                    status.empty()

                    # ═══ 结果展示 ═══
                    st.success("✅ 验证集处理完成！特征维度与训练集一致！")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("样本数", len(valid_df))
                    c2.metric("特征数", len(unified_cols))
                    c3.metric("特征一致性", "✅ 与训练集一致")

                    # 文件明细
                    st.subheader("📋 文件明细")
                    info_df = pd.DataFrame(valid_info,
                                           columns=['文件名', '原始峰数', f'筛选后峰数(SNR≥{snr_thresh})'])
                    st.dataframe(info_df, use_container_width=True, hide_index=True)

                    # 非零特征统计
                    feat_data = valid_df[unified_cols].astype(float)
                    nonzero_counts = (feat_data > 0).sum(axis=1)
                    with st.expander("📊 各样本非零特征数与覆盖率"):
                        nz_df = pd.DataFrame({
                            '文件名': valid_df['sample'].values,
                            '非零特征数': nonzero_counts.values,
                            '覆盖率': (nonzero_counts / len(unified_cols) * 100).round(1).astype(str) + '%'
                        })
                        st.dataframe(nz_df, use_container_width=True, hide_index=True)

                    # 预览
                    with st.expander("📊 特征矩阵预览（前10列）"):
                        preview_cols = ['sample'] + unified_cols[:10]
                        st.dataframe(valid_df[preview_cols].round(6), use_container_width=True, hide_index=True)

                    # ── 下载 ──
                    st.divider()
                    export_cols = ['sample'] + unified_cols
                    st.download_button(
                        "📊 下载验证集特征矩阵 CSV",
                        data=valid_df[export_cols].to_csv(index=False),
                        file_name="valid_feature_matrix_unified.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    del valid_df, feat_data
                    gc.collect()


# ── 底部 ──
st.divider()
st.markdown("<div style='text-align:center;color:#aaa;font-size:0.85rem;'>MALDI-TOF MS 跨仪器统一处理平台 · 布鲁克 CSV + 安图 TXT</div>", unsafe_allow_html=True)
