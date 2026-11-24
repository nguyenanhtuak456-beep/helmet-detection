import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="AI Traffic Vision | Demo", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")

# -------------------- THEME --------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --bg:#07111f; --panel:#0d1b2b; --panel2:#102338; --line:#1d3850; --cyan:#22d3ee; --muted:#8ba6bb; }
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: radial-gradient(circle at 80% 0%, #102d43 0, #07111f 42%); color:#eaf8ff; }
[data-testid="stSidebar"] { background:#081522; border-right:1px solid #1d3850; }
[data-testid="stHeader"] { background:rgba(7,17,31,.85); }
.block-container { padding-top:1.5rem; max-width:1500px; }
section[data-testid="stSidebar"] .block-container { padding-top:1.5rem; }
h1,h2,h3 { color:#eaf8ff !important; letter-spacing:-.03em; }
.small-muted { color:#8ba6bb; font-size:.84rem; }
.brand { display:flex; gap:12px; align-items:center; padding:8px 4px 22px; }
.brand-icon { width:42px;height:42px;border-radius:13px;display:grid;place-items:center;background:linear-gradient(135deg,#0e7490,#22d3ee);font-size:22px;box-shadow:0 0 28px #22d3ee30; }
.brand-title { font-weight:800; letter-spacing:.06em; color:#eaf8ff; }
.brand-sub { font-size:10px; letter-spacing:.14em; color:#7fa4bb; margin-top:3px; }
.hero { padding:22px 24px; border:1px solid #1c4058; border-radius:18px; background:linear-gradient(110deg,#102b40dd,#0b1a2bdd); margin-bottom:20px; }
.eyebrow { color:#22d3ee; font-size:11px; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
.hero h1 { margin:7px 0 4px; font-size:30px; }
.hero p { color:#9bb5c8; margin:0; }
.pill { display:inline-block; border:1px solid #155e75; background:#083344; color:#67e8f9; border-radius:30px; padding:5px 10px; font-size:11px; font-weight:700; }
.panel { background:linear-gradient(145deg,#102338,#0b1b2b); border:1px solid #1d3850; border-radius:16px; padding:18px; margin-bottom:14px; }
.panel-title { font-size:13px; color:#c8e6f5; font-weight:700; letter-spacing:.06em; margin-bottom:13px; }
.kpi { background:linear-gradient(145deg,#10283c,#0c1c2d); border:1px solid #1d4057; border-radius:15px; padding:17px 18px; min-height:112px; }
.kpi-label { color:#91adbf; font-size:12px; }
.kpi-value { color:#effbff; font-size:29px; font-weight:800; margin:5px 0 2px; }
.kpi-note { color:#6f91a8; font-size:11px; }
.status-dot { display:inline-block; width:7px;height:7px;border-radius:50%;background:#34d399;margin-right:7px;box-shadow:0 0 10px #34d399; }
div.stButton > button { border-radius:10px; border:1px solid #23516a; font-weight:650; }
div.stButton > button[kind="primary"] { background:linear-gradient(100deg,#0891b2,#06b6d4); border:0; color:#03131e; }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { background:#102338; border-radius:9px 9px 0 0; padding:9px 16px; }
.stTabs [aria-selected="true"] { color:#67e8f9 !important; }
[data-testid="stFileUploader"] { background:#0a1b2b; border:1px dashed #28617a; border-radius:13px; padding:10px; }
[data-testid="stMetric"] { background:#102338; border:1px solid #1d4057; border-radius:12px; padding:12px; }
hr { border-color:#1d3850; }
</style>
""", unsafe_allow_html=True)

# -------------------- STATE --------------------
if "page" not in st.session_state: st.session_state.page = "Tổng quan"
if "demo_runs" not in st.session_state: st.session_state.demo_runs = []
if "demo_result" not in st.session_state: st.session_state.demo_result = None

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">🚦</div><div><div class="brand-title">AI TRAFFIC VISION</div><div class="brand-sub">INTELLIGENT TRAFFIC MONITORING</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted" style="letter-spacing:.13em;font-weight:700;margin:8px 0">KHÔNG GIAN LÀM VIỆC</div>', unsafe_allow_html=True)
    pages = ["Tổng quan", "Phân tích video", "Phân tích hình ảnh", "Lịch sử", "Cài đặt"]
    icons = {"Tổng quan":"▦", "Phân tích video":"▷", "Phân tích hình ảnh":"▧", "Lịch sử":"◷", "Cài đặt":"⚙"}
    for p in pages:
        if st.button(f"{icons[p]}   {p}", key=f"nav_{p}", use_container_width=True, type="primary" if st.session_state.page == p else "secondary"):
            st.session_state.page = p
            st.rerun()
    st.markdown("---")
    st.markdown('<div class="small-muted">TRẠNG THÁI HỆ THỐNG</div>', unsafe_allow_html=True)
    st.markdown('<p style="margin:8px 0 2px"><span class="status-dot"></span><b style="color:#6ee7b7">Giao diện đang hoạt động</b></p>', unsafe_allow_html=True)
    st.caption("Demo Mode · Không kết nối mô hình AI")
    st.markdown('<span class="pill">● PUBLIC DEMO</span>', unsafe_allow_html=True)

# -------------------- HELPERS --------------------
def hero(kicker, title, desc):
    st.markdown(f'<div class="hero"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{desc}</p></div>', unsafe_allow_html=True)

def kpi(label, value, note, color="#67e8f9"):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value" style="color:{color}">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

def demo_stats():
    # Fixed illustrative values: never presented as actual model inference.
    return {"vehicles":128,"helmet":105,"no_helmet":23,"rate":82.0,"confidence":91.4}

# -------------------- PAGES --------------------
if st.session_state.page == "Tổng quan":
    hero("TRAFFIC INTELLIGENCE PLATFORM", "Trung tâm giám sát giao thông", "Không gian trải nghiệm giao diện AI Traffic Vision — các số liệu bên dưới chỉ mang tính minh họa.")
    s = st.session_state.demo_result or demo_stats()
    cols = st.columns(4)
    with cols[0]: kpi("TỔNG PHƯƠNG TIỆN", s["vehicles"], "Dữ liệu minh họa")
    with cols[1]: kpi("ĐỘI MŨ BẢO HIỂM", s["helmet"], "Dữ liệu minh họa", "#34d399")
    with cols[2]: kpi("KHÔNG ĐỘI MŨ", s["no_helmet"], "Dữ liệu minh họa", "#fb7185")
    with cols[3]: kpi("TỶ LỆ ĐỘI MŨ", f'{s["rate"]:.1f}%', "Dữ liệu minh họa", "#fbbf24")
    left, right = st.columns([1.55,1], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="panel-title">▥  TỔNG QUAN PHÂN LOẠI</div>', unsafe_allow_html=True)
        chart = pd.DataFrame({"Phân loại":["Đội mũ","Không đội mũ"],"Số lượng":[s["helmet"],s["no_helmet"]]})
        st.bar_chart(chart.set_index("Phân loại"), color="#22d3ee", height=260)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">◉  TRẠNG THÁI NỀN TẢNG</div><p><span class="status-dot"></span>Giao diện hoạt động bình thường</p><p style="color:#8ba6bb">Chế độ: Demo tương tác</p><p style="color:#8ba6bb">AI inference: Đã tắt</p><p style="color:#8ba6bb">GPU: Không yêu cầu</p></div>', unsafe_allow_html=True)
        st.info("Bạn có thể mở các mục ở thanh bên để thử tải tệp, điều chỉnh tham số và xem trước giao diện.")
    st.markdown('<div class="small-muted">Lưu ý: Không có nhận diện YOLOv8s hoặc tracking ByteTrack trong bản website demo.</div>', unsafe_allow_html=True)

elif st.session_state.page == "Phân tích video":
    hero("VIDEO WORKSPACE", "Phân tích video giao thông", "Tải video lên để trải nghiệm khu vực xem trước và các công cụ điều khiển.")
    a,b = st.columns([1.35,.85], gap="large")
    with a:
        st.markdown('<div class="panel"><div class="panel-title">▣  VIDEO ĐẦU VÀO</div>', unsafe_allow_html=True)
        video = st.file_uploader("Chọn video từ thiết bị", type=["mp4","mov","avi","m4v"], key="demo_video")
        if video:
            st.video(video)
            st.success(f"Đã tải lên: {video.name} · {video.size/1024/1024:.2f} MB")
        else:
            st.markdown('<div style="padding:35px 12px;text-align:center;color:#8ba6bb">⬆<br><b>Chưa có video</b><br><small>Tải video để xem trước ngay tại đây</small></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="panel-title">☷  ĐIỀU KHIỂN HIỂN THỊ</div>', unsafe_allow_html=True)
        conf_h = st.slider("Ngưỡng tin cậy — Đội mũ", 0.10, 1.00, 0.35, 0.05)
        conf_n = st.slider("Ngưỡng tin cậy — Không đội mũ", 0.10, 1.00, 0.40, 0.05)
        conf_b = st.slider("Ngưỡng tin cậy — Xe máy", 0.10, 1.00, 0.35, 0.05)
        st.markdown("**Lớp hiển thị**")
        sh = st.toggle("Đội mũ", True)
        sn = st.toggle("Không đội mũ", True)
        sb = st.toggle("Xe máy", True)
        tid = st.toggle("Hiển thị Track ID", True)
        st.caption("Các điều khiển đang hoạt động ở mức giao diện; không tác động đến mô hình AI.")
        st.markdown('</div>', unsafe_allow_html=True)
    c1,c2 = st.columns([1,2])
    with c1:
        if st.button("▶  Trải nghiệm phân tích demo", type="primary", use_container_width=True, disabled=video is None):
            result = demo_stats()
            st.session_state.demo_result = result
            st.session_state.demo_runs.insert(0,{"time":datetime.now().strftime("%d/%m/%Y %H:%M:%S"),"file":video.name,"vehicles":result["vehicles"],"helmet":result["helmet"],"no_helmet":result["no_helmet"],"mode":"Mô phỏng"})
            st.success("Đã tạo phiên kết quả minh họa. Không có AI inference được thực hiện.")
    with c2:
        st.caption(f"Thiết lập hiện tại · Đội mũ {conf_h:.2f} · Không đội mũ {conf_n:.2f} · Xe máy {conf_b:.2f} · Track ID {'Bật' if tid else 'Tắt'}")
    if st.session_state.demo_result:
        st.markdown("#### Kết quả minh họa gần nhất")
        r=st.session_state.demo_result
        cols=st.columns(3)
        with cols[0]: kpi("PHƯƠNG TIỆN",r["vehicles"],"Số liệu giả lập")
        with cols[1]: kpi("ĐỘI MŨ",r["helmet"],"Số liệu giả lập","#34d399")
        with cols[2]: kpi("KHÔNG ĐỘI MŨ",r["no_helmet"],"Số liệu giả lập","#fb7185")

elif st.session_state.page == "Phân tích hình ảnh":
    hero("IMAGE WORKSPACE", "Phân tích hình ảnh", "Tải ảnh lên, xem trước và thử các tùy chọn hiển thị.")
    files=st.file_uploader("Chọn một hoặc nhiều ảnh", type=["jpg","jpeg","png","webp","bmp"], accept_multiple_files=True, key="demo_images")
    if files:
        st.success(f"Đã tải {len(files)} ảnh lên giao diện.")
        cols=st.columns(min(3,len(files)))
        for i,f in enumerate(files):
            with cols[i%len(cols)]:
                st.image(f, caption=f.name, use_container_width=True)
        st.info("Ảnh được hiển thị để trải nghiệm giao diện. Bản Demo không tạo bounding box hoặc nhãn nhận diện thật.")
    else:
        st.markdown('<div class="panel" style="text-align:center;padding:45px;color:#8ba6bb">▧<br><b>Khu vực tải ảnh</b><br><small>JPG · PNG · WEBP · BMP</small></div>', unsafe_allow_html=True)

elif st.session_state.page == "Lịch sử":
    hero("ACTIVITY LOG", "Lịch sử phiên trải nghiệm", "Các phiên mô phỏng đã được tạo trong lần truy cập hiện tại.")
    if st.session_state.demo_runs:
        st.dataframe(pd.DataFrame(st.session_state.demo_runs), use_container_width=True, hide_index=True)
        if st.button("Xóa lịch sử phiên", type="secondary"):
            st.session_state.demo_runs=[]
            st.rerun()
    else:
        st.info("Chưa có phiên trải nghiệm. Hãy vào mục Phân tích video và bấm Trải nghiệm phân tích demo.")

elif st.session_state.page == "Cài đặt":
    hero("PREFERENCES", "Cài đặt giao diện", "Tùy chỉnh trải nghiệm hiển thị trên website demo.")
    left,right=st.columns(2)
    with left:
        st.markdown('<div class="panel"><div class="panel-title">GIAO DIỆN</div>', unsafe_allow_html=True)
        st.selectbox("Màu nhấn",["Cyan Technology","Electric Blue","Emerald"])
        st.selectbox("Mật độ hiển thị",["Tiêu chuẩn","Thoáng","Thu gọn"])
        st.toggle("Hiệu ứng chuyển động",True)
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">THÔNG TIN HỆ THỐNG</div><p>Phiên bản: UI Demo 1.0</p><p>Inference: Disabled</p><p>Detector: Not connected</p><p>Tracker: Not connected</p></div>',unsafe_allow_html=True)

st.markdown('<div style="border-top:1px solid #1d3850;margin-top:30px;padding:15px 0;color:#6f91a8;font-size:11px;display:flex;justify-content:space-between"><span>AI TRAFFIC VISION · INTERACTIVE DEMO</span><span>DEMO DATA ONLY · NO AI INFERENCE</span></div>',unsafe_allow_html=True)
