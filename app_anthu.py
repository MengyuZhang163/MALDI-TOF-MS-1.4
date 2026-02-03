import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import tempfile
import shutil
import os
import gc
import io
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation

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
    'template_mz': None,          # 训练集归出的分箱模板 (array)
    'bin_edges': None,            # 分箱边界
    'bin_size': 2,                # 分箱粒度
    'mz_min': 2000,
    'mz_max': 20500,
    'snr_threshold': 5,
    'prominence_factor': 1.0,
    'train_matrix': None,         # 训练集特征矩阵 DataFrame
    'template_ready': False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# 核心读取 / 处理函数
# ─────────────────────────────────────────────

def detect_file_type(raw_bytes: bytes) -> str:
    """
    判断 txt 是布鲁克(原始光谱)还是安图(峰表).
    逻辑: 尝试 gbk 解码，看第二行是否以 m/z 开头 → 安图；否则第一行直接是数字 → 布鲁克
    """
    try:
        text = raw_bytes.decode('gbk')
    except:
        text = raw_bytes.decode('utf-8', errors='replace')
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return 'unknown'
    # 安图第二行表头含 m/z
    if 'm/z' in lines[1].lower() or 'm/z' in lines[0].lower():
        return 'anthu'
    # 布鲁克第一行直接是 "数字 数字"
    parts = lines[0].split()
    if len(parts) == 2:
        try:
            float(parts[0])
            float(parts[1])
            return 'bruker'
        except:
            pass
    return 'unknown'


def read_bruker(raw_bytes: bytes) -> pd.DataFrame:
    """读布鲁克原始光谱 → DataFrame(mz, intensity)"""
    text = raw_bytes.decode('utf-8', errors='replace')
    rows = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=['mz', 'intensity'])


def read_anthu(raw_bytes: bytes) -> pd.DataFrame:
    """读安图峰表 → DataFrame(mz, peak_height, peak_area, SNR, resolution)"""
    text = raw_bytes.decode('gbk', errors='replace')
    lines = text.splitlines()
    # 跳过第1行(路径)和第2行(表头)
    data_lines = lines[2:]
    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split('\t') if p.strip()]
        if len(parts) >= 5:
            try:
                rows.append([float(parts[0]), float(parts[1]),
                             float(parts[2]), float(parts[3]), float(parts[4])])
            except ValueError:
                continue
    df = pd.DataFrame(rows, columns=['mz', 'peak_height', 'peak_area', 'SNR', 'resolution'])
    return df


def bruker_to_peaks(df: pd.DataFrame, prominence_factor: float = 1.0) -> pd.DataFrame:
    """
    布鲁克原始光谱 → 峰检测 → (mz, intensity_sqrt)
    intensity 做 sqrt 变换，用 MAD 估噪，prominence = MAD * prominence_factor
    """
    mz = df['mz'].values
    intensity = df['intensity'].values.astype(float)
    intensity_sqrt = np.sqrt(intensity)

    noise = median_abs_deviation(intensity_sqrt)
    if noise < 1e-10:
        noise = 1.0

    peaks_idx, _ = find_peaks(
        intensity_sqrt,
        height=noise * 1.5,
        distance=3,
        prominence=noise * prominence_factor
    )
    return pd.DataFrame({
        'mz': mz[peaks_idx],
        'intensity': intensity_sqrt[peaks_idx]
    })


def anthu_to_peaks(df: pd.DataFrame, snr_threshold: int = 5) -> pd.DataFrame:
    """
    安图峰表 → 筛选(SNR >= threshold & area > 0) → (mz, intensity_sqrt)
    peak_height 同样做 sqrt，保持跨仪器一致
    """
    mask = (df['SNR'] >= snr_threshold) & (df['peak_area'] > 0)
    filt = df[mask].copy()
    return pd.DataFrame({
        'mz': filt['mz'].values,
        'intensity': np.sqrt(filt['peak_height'].values.astype(float))
    })


def peaks_to_bin_vector(peaks_df: pd.DataFrame,
                        bin_edges: np.ndarray,
                        n_bins: int) -> np.ndarray:
    """
    将峰表映射到固定分箱网格 → 长度 n_bins 的向量（同 bin 取最大值）
    """
    vec = np.zeros(n_bins)
    mz_arr = peaks_df['mz'].values
    int_arr = peaks_df['intensity'].values
    # vectorized bin assignment
    indices = np.searchsorted(bin_edges, mz_arr, side='right') - 1
    valid = (indices >= 0) & (indices < n_bins)
    for idx, val in zip(indices[valid], int_arr[valid]):
        if val > vec[idx]:
            vec[idx] = val
    return vec


def tic_normalize(vec: np.ndarray) -> np.ndarray:
    """TIC归一化: 每个值 / 总和"""
    s = vec.sum()
    return vec / s if s > 0 else vec


def process_single_file(raw_bytes: bytes,
                        bin_edges: np.ndarray,
                        n_bins: int,
                        snr_threshold: int,
                        prominence_factor: float) -> tuple:
    """
    统一入口: 自动判断类型 → 峰检测/读取 → 分箱 → TIC归一化
    返回 (file_type, bin_vector_normalized, n_peaks_before_bin)
    """
    ftype = detect_file_type(raw_bytes)
    if ftype == 'bruker':
        raw_df = read_bruker(raw_bytes)
        peaks = bruker_to_peaks(raw_df, prominence_factor)
    elif ftype == 'anthu':
        raw_df = read_anthu(raw_bytes)
        peaks = anthu_to_peaks(raw_df, snr_threshold)
    else:
        return 'unknown', None, 0

    n_peaks = len(peaks)
    vec = peaks_to_bin_vector(peaks, bin_edges, n_bins)
    vec_norm = tic_normalize(vec)
    return ftype, vec_norm, n_peaks


def extract_zip(uploaded_zip) -> dict:
    """解析 ZIP → {filename: raw_bytes}，只提取 .txt"""
    result = {}
    with zipfile.ZipFile(uploaded_zip, 'r') as z:
        for name in z.namelist():
            bn = Path(name).name
            if bn.lower().endswith('.txt') and not name.startswith('__MACOSX'):
                result[bn] = z.read(name)
    return result


def build_bin_edges(mz_min: int, mz_max: int, bin_size: int):
    edges = np.arange(mz_min, mz_max + bin_size, bin_size)
    centers = (edges[:-1] + edges[1:]) / 2
    n_bins = len(centers)
    return edges, centers, n_bins


# ─────────────────────────────────────────────
# 主界面
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🔬 MALDI-TOF MS 跨仪器统一处理平台</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">布鲁克 & 安图 混合数据 → 统一分箱特征矩阵</div>', unsafe_allow_html=True)

# ── 侧边栏：参数 ──
with st.sidebar:
    st.header("⚙️ 处理参数")

    st.markdown('<div class="info-box">这些参数对<b>布鲁克和安图</b>同时生效，建议先用默认值试一次。</div>', unsafe_allow_html=True)

    bin_size = st.selectbox(
        "分箱粒度 (Da)", [1, 2, 5], index=1,
        help="2 Da 是 MALDI-TOF 领域主流选择，与 MALDIquant binPeaks tolerance=2 一致"
    )
    mz_min = st.number_input("m/z 下限", value=2000, step=100)
    mz_max = st.number_input("m/z 上限", value=20500, step=100)

    st.divider()
    st.subheader("🔍 峰筛选参数")
    snr_threshold = st.slider(
        "安图 SNR 阈值", min_value=3, max_value=15, value=5,
        help="安图峰表中 SNR < 此值的峰会被丢弃"
    )
    prominence_factor = st.slider(
        "布鲁克 prominence 倍数", min_value=0.5, max_value=3.0, value=1.0, step=0.1,
        help="布鲁克峰检测灵敏度调节。值越小检测到的峰越多"
    )

    # 同步到 session
    st.session_state.bin_size = bin_size
    st.session_state.mz_min = mz_min
    st.session_state.mz_max = mz_max
    st.session_state.snr_threshold = snr_threshold
    st.session_state.prominence_factor = prominence_factor

    st.divider()
    st.header("💾 内存管理")
    if st.button("🧹 清理缓存", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k not in ('bin_size','mz_min','mz_max','snr_threshold','prominence_factor'):
                st.session_state[k] = None
        st.session_state.template_ready = False
        gc.collect()
        st.success("已清理")
        st.rerun()

    st.divider()
    st.header("📖 流程说明")
    st.markdown("""
    1. **阶段1** — 上传训练集 ZIP（可混合布鲁克/安图 txt）
       - 自动识别每个文件的仪器类型
       - 统一分箱 → 生成特征模板
    2. **阶段2** — 上传验证集 ZIP
       - 使用训练集的特征模板
       - 输出与训练集**维度完全一致**的特征矩阵
    3. 最终两个矩阵可直接合并进机器学习模型
    """)


# ── 主内容：两个 Tab ──
tab1, tab2 = st.tabs(["🎯 阶段1: 训练集 → 建立模板", "🔄 阶段2: 验证集 → 应用模板"])

# ══════════════════════════════════════════════
# 阶段1
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="phase-header">📊 阶段1: 处理训练集，建立分箱特征模板</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    上传的 ZIP 中可以<b>同时包含布鲁克和安图的 txt</b>，系统会自动识别并统一处理。<br>
    如果你有标签信息（哪个文件对应哪个样本/组别），可以在下面附带一个 Excel。
    </div>""", unsafe_allow_html=True)

    train_zip_upload = st.file_uploader("上传训练集 ZIP", type=['zip'], key='train_zip')

    # 可选 Excel（样本标签）
    label_excel_upload = st.file_uploader(
        "（可选）上传标签 Excel（列: file, group）",
        type=['xlsx', 'xls'], key='label_excel'
    )

    if train_zip_upload:
        files_dict = extract_zip(train_zip_upload)
        if not files_dict:
            st.error("ZIP 中没有找到 .txt 文件")
        else:
            st.success(f"✅ 找到 {len(files_dict)} 个 txt 文件")

            # 预检查：识别类型
            type_counts = {'bruker': 0, 'anthu': 0, 'unknown': 0}
            for fn, raw in files_dict.items():
                t = detect_file_type(raw)
                type_counts[t] += 1
            col1, col2, col3 = st.columns(3)
            col1.metric("布鲁克", type_counts['bruker'])
            col2.metric("安图", type_counts['anthu'])
            col3.metric("未识别", type_counts['unknown'])

            if st.button("🎯 开始处理训练集", type="primary", use_container_width=True):
                progress = st.progress(0)
                status = st.empty()

                # 读取 label
                label_df = None
                if label_excel_upload:
                    try:
                        label_df = pd.read_excel(label_excel_upload)
                        # 标准化列名
                        label_df.columns = [c.strip().lower() for c in label_df.columns]
                        if 'file' in label_df.columns and 'group' in label_df.columns:
                            label_df['file'] = label_df['file'].astype(str).str.strip()
                        else:
                            st.warning("⚠️ Excel 列名需包含 'file' 和 'group'，将忽略标签信息")
                            label_df = None
                    except Exception as e:
                        st.warning(f"⚠️ 读取 Excel 失败: {e}，将忽略标签信息")

                # 构建分箱网格
                edges, centers, n_bins = build_bin_edges(
                    st.session_state.mz_min,
                    st.session_state.mz_max,
                    st.session_state.bin_size
                )
                st.session_state.bin_edges = edges

                status.text("🔄 正在逐文件处理...")
                progress.progress(10)

                records = []      # (filename, type, n_peaks, vec)
                total = len(files_dict)

                for i, (fn, raw) in enumerate(files_dict.items()):
                    ftype, vec_norm, n_peaks = process_single_file(
                        raw, edges, n_bins,
                        st.session_state.snr_threshold,
                        st.session_state.prominence_factor
                    )
                    if vec_norm is not None:
                        records.append((fn, ftype, n_peaks, vec_norm))
                    progress.progress(10 + int(70 * (i + 1) / total))

                status.text("📦 构建特征矩阵...")

                # 确定非零列（训练集中至少一个样本有峰的 bin）
                mat = np.vstack([r[3] for r in records])
                nonzero_mask = mat.sum(axis=0) > 0
                nonzero_indices = np.where(nonzero_mask)[0]
                active_centers = centers[nonzero_indices]

                # 裁剪到非零列
                mat_trimmed = mat[:, nonzero_indices]

                # 列名
                col_names = [f"mz_{c:.1f}" for c in active_centers]

                # 构建 DataFrame
                train_df = pd.DataFrame(mat_trimmed, columns=col_names)
                train_df.insert(0, 'sample', [r[0] for r in records])
                train_df.insert(1, 'instrument', [r[1] for r in records])
                train_df.insert(2, 'n_peaks_detected', [r[2] for r in records])

                # 加 group 标签
                if label_df is not None:
                    label_map = dict(zip(label_df['file'], label_df['group']))
                    train_df.insert(3, 'group',
                                    train_df['sample'].map(label_map).fillna('unknown'))

                # 保存到 session
                st.session_state.train_matrix = train_df
                st.session_state.template_mz = active_centers  # 模板 m/z
                st.session_state.template_ready = True

                progress.progress(100)
                status.text("✅ 完成！")

                import time; time.sleep(0.4)
                progress.empty()
                status.empty()

                # ── 结果展示 ──
                st.success("✅ 训练集处理完成，特征模板已建立！")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("样本数", len(train_df))
                c2.metric("活跃特征数", len(active_centers))
                c3.metric("m/z 范围", f"{active_centers.min():.0f}~{active_centers.max():.0f}")
                c4.metric("分箱粒度", f"{st.session_state.bin_size} Da")

                # 仪器分布
                st.subheader("📋 样本明细")
                st.dataframe(
                    train_df[['sample','instrument','n_peaks_detected'] +
                             (['group'] if 'group' in train_df.columns else [])],
                    use_container_width=True, hide_index=True
                )

                # 特征矩阵预览
                with st.expander("📊 特征矩阵预览（前10列）"):
                    preview_cols = ['sample','instrument'] + col_names[:10]
                    st.dataframe(train_df[preview_cols].round(6), use_container_width=True, hide_index=True)

                # 下载
                st.divider()
                st.subheader("📥 下载")
                c1, c2 = st.columns(2)

                # 下载特征矩阵（去掉 instrument 和 n_peaks 辅助列）
                export_cols = ['sample'] + (['group'] if 'group' in train_df.columns else []) + col_names
                c1.download_button(
                    "📊 训练集特征矩阵 CSV",
                    data=train_df[export_cols].to_csv(index=False),
                    file_name="train_feature_matrix.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # 下载模板（m/z列表）
                template_export = pd.DataFrame({
                    'bin_center': active_centers,
                    'feature_name': col_names
                })
                c2.download_button(
                    "🎯 特征模板 CSV",
                    data=template_export.to_csv(index=False),
                    file_name="feature_template.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# ══════════════════════════════════════════════
# 阶段2
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="phase-header">🔄 阶段2: 处理验证集（使用训练集模板）</div>', unsafe_allow_html=True)

    if not st.session_state.template_ready:
        st.markdown('<div class="warn-box">⚠️ 请先完成阶段1，建立特征模板后才能处理验证集。</div>', unsafe_allow_html=True)
    else:
        active_centers = st.session_state.template_mz
        col_names = [f"mz_{c:.1f}" for c in active_centers]

        st.success(f"✅ 特征模板就绪：{len(active_centers)} 个特征")
        st.markdown("""<div class="info-box">
        验证集同样可以混合布鲁克和安图 txt。输出的特征维度会与训练集<b>完全一致</b>。
        </div>""", unsafe_allow_html=True)

        valid_zip_upload = st.file_uploader("上传验证集 ZIP", type=['zip'], key='valid_zip')

        if valid_zip_upload:
            files_dict = extract_zip(valid_zip_upload)
            if not files_dict:
                st.error("ZIP 中没有找到 .txt 文件")
            else:
                st.success(f"✅ 找到 {len(files_dict)} 个 txt 文件")

                type_counts = {'bruker': 0, 'anthu': 0, 'unknown': 0}
                for fn, raw in files_dict.items():
                    type_counts[detect_file_type(raw)] += 1
                c1, c2, c3 = st.columns(3)
                c1.metric("布鲁克", type_counts['bruker'])
                c2.metric("安图", type_counts['anthu'])
                c3.metric("未识别", type_counts['unknown'])

                # 可选标签
                valid_label_upload = st.file_uploader(
                    "（可选）验证集标签 Excel（列: file, group）",
                    type=['xlsx','xls'], key='valid_label'
                )

                if st.button("🔄 开始处理验证集", type="primary", use_container_width=True):
                    progress = st.progress(0)
                    status = st.empty()

                    # 重建和训练集一致的 bin_edges（用全范围，后面映射到模板列）
                    edges, centers, n_bins = build_bin_edges(
                        st.session_state.mz_min,
                        st.session_state.mz_max,
                        st.session_state.bin_size
                    )

                    # 模板 m/z → 在 centers 中对应的 index
                    template_indices = []
                    for tc in active_centers:
                        idx = np.argmin(np.abs(centers - tc))
                        template_indices.append(idx)
                    template_indices = np.array(template_indices)

                    status.text("🔄 逐文件处理...")
                    progress.progress(10)

                    records = []
                    total = len(files_dict)

                    for i, (fn, raw) in enumerate(files_dict.items()):
                        ftype, vec_norm, n_peaks = process_single_file(
                            raw, edges, n_bins,
                            st.session_state.snr_threshold,
                            st.session_state.prominence_factor
                        )
                        if vec_norm is not None:
                            # 只取模板对应的列
                            vec_template = vec_norm[template_indices]
                            records.append((fn, ftype, n_peaks, vec_template))
                        progress.progress(10 + int(70 * (i + 1) / total))

                    status.text("📦 构建特征矩阵...")

                    mat = np.vstack([r[3] for r in records])
                    valid_df = pd.DataFrame(mat, columns=col_names)
                    valid_df.insert(0, 'sample', [r[0] for r in records])
                    valid_df.insert(1, 'instrument', [r[1] for r in records])
                    valid_df.insert(2, 'n_peaks_detected', [r[2] for r in records])

                    # 加标签
                    if valid_label_upload:
                        try:
                            ldf = pd.read_excel(valid_label_upload)
                            ldf.columns = [c.strip().lower() for c in ldf.columns]
                            if 'file' in ldf.columns and 'group' in ldf.columns:
                                ldf['file'] = ldf['file'].astype(str).str.strip()
                                lmap = dict(zip(ldf['file'], ldf['group']))
                                valid_df.insert(3, 'group',
                                                valid_df['sample'].map(lmap).fillna('unknown'))
                        except:
                            pass

                    progress.progress(100)
                    status.text("✅ 完成！")

                    import time; time.sleep(0.4)
                    progress.empty()
                    status.empty()

                    # ── 结果 ──
                    st.success("✅ 验证集处理完成！特征维度与训练集一致！")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("样本数", len(valid_df))
                    c2.metric("特征数", len(col_names))
                    c3.metric("特征一致性", "✅ 与训练集一致")

                    st.subheader("📋 样本明细")
                    st.dataframe(
                        valid_df[['sample','instrument','n_peaks_detected'] +
                                 (['group'] if 'group' in valid_df.columns else [])],
                        use_container_width=True, hide_index=True
                    )

                    with st.expander("📊 特征矩阵预览（前10列）"):
                        preview_cols = ['sample','instrument'] + col_names[:10]
                        st.dataframe(valid_df[preview_cols].round(6), use_container_width=True, hide_index=True)

                    # 下载
                    st.divider()
                    export_cols = ['sample'] + (['group'] if 'group' in valid_df.columns else []) + col_names
                    st.download_button(
                        "📊 下载验证集特征矩阵 CSV",
                        data=valid_df[export_cols].to_csv(index=False),
                        file_name="valid_feature_matrix.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    del valid_df
                    gc.collect()

# ── 底部 ──
st.divider()
st.markdown("<div style='text-align:center;color:#aaa;font-size:0.85rem;'>MALDI-TOF MS 跨仪器统一处理平台 · 布鲁克 & 安图</div>", unsafe_allow_html=True)
