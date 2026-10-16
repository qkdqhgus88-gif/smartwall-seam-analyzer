"""Smart Wall Seam Visibility Analyzer — Streamlit demo.
Run: streamlit run app.py
All bezel values and visualizations are illustrative design scenarios, not measured OLED characteristics.
"""
from pathlib import Path
import io
import math
import base64
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='Smart Wall | Seam Analyzer', page_icon='📺', layout='wide')

st.markdown('''<style>
:root{
  --bg:#0b0e14; --panel:#141925; --panel2:#1b2130; --line:#273142;
  --text:#e6edf3; --muted:#9aa7b8; --accent:#38bdf8; --accent2:#7c5cff;
}
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.1rem;max-width:1260px;}
.hero{
  background:linear-gradient(135deg,#131a2a 0%,#0e1320 55%,#0b0e14 100%);
  border:1px solid var(--line);border-radius:18px;padding:24px 30px;margin-bottom:18px;
  position:relative;overflow:hidden;
}
.hero:before{content:"";position:absolute;right:-70px;top:-70px;width:240px;height:240px;
  background:radial-gradient(circle,rgba(56,189,248,.22),transparent 70%);}
.hero:after{content:"";position:absolute;left:35%;bottom:-90px;width:260px;height:180px;
  background:radial-gradient(circle,rgba(124,92,255,.16),transparent 70%);}
.hero .badge{display:inline-block;font-size:.70rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);border:1px solid rgba(56,189,248,.4);border-radius:999px;padding:3px 12px;margin-bottom:12px;}
.hero h1{margin:0;font-size:2.15rem;font-weight:800;letter-spacing:-.01em;color:#fff;line-height:1.08;}
.hero h1 span{background:linear-gradient(90deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p{margin:.55rem 0 0;color:var(--muted);max-width:760px;font-size:.96rem;line-height:1.5;}
.hero .chips{margin-top:14px;display:flex;gap:8px;flex-wrap:wrap;}
.hero .chips span{font-size:.76rem;color:var(--text);background:var(--panel2);
  border:1px solid var(--line);border-radius:8px;padding:4px 11px;}
.sec{font-size:1.02rem;font-weight:700;color:#fff;margin:2px 0 12px;
  padding-left:12px;border-left:3px solid var(--accent);}
[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--line);
  border-radius:14px;padding:13px 16px;}
[data-testid="stMetricValue"]{font-size:1.5rem;color:#fff;}
[data-testid="stMetricLabel"]{color:var(--muted);}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:6px 16px;margin-top:14px;}
.panel .row{display:flex;justify-content:space-between;gap:12px;padding:9px 0;
  border-bottom:1px dashed var(--line);font-size:.88rem;color:var(--muted);}
.panel .row:last-child{border-bottom:none;}
.panel .row b{color:var(--text);font-weight:600;text-align:right;}
.status{border-radius:12px;padding:12px 16px;margin-top:14px;font-size:.9rem;font-weight:600;border:1px solid;}
.status.ok{background:rgba(34,197,94,.10);border-color:rgba(34,197,94,.45);color:#8ef0b0;}
.status.off{background:rgba(245,158,11,.10);border-color:rgba(245,158,11,.45);color:#f4c87a;}
[data-testid="stImage"] img{border-radius:12px;border:1px solid var(--line);
  box-shadow:0 14px 34px rgba(0,0,0,.5);}
[data-testid="stSidebar"]{background:#0d121c;border-right:1px solid var(--line);}
.stDownloadButton button,.stButton button{border-radius:10px;}
.cap{color:var(--muted);font-size:.82rem;line-height:1.5;}
</style>''', unsafe_allow_html=True)

st.markdown('''<div class="hero">
  <span class="badge">OLED · 2×2 Modular Wall</span>
  <h1>Smart Wall <span>Seam Analyzer</span></h1>
  <p>16:9 OLED 모듈 4장을 이어 붙인 월 디스플레이의 이음새 시인성을 설계 파라미터(베젤·기구 간격·패널 크기·시청 거리)로 시뮬레이션합니다.</p>
  <div class="chips"><span>모듈별 4K UHD · 3840×2160</span><span>16:9 · 2×2</span><span>설계 검토용 시뮬레이션</span></div>
</div>''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sec">설계 입력</div>', unsafe_allow_html=True)
    diag=st.radio('개별 OLED 크기', [55,65,77], index=1, horizontal=True, format_func=lambda x:f'{x}형')
    bezel=st.slider('각 패널의 내부 베젤 폭 (mm)',0.0,3.0,0.3,0.05,
                    help='네 패널의 가로·세로 내부 경계에 동일한 베젤 폭을 가정합니다.')
    gap=st.slider('패널 사이 기구적 간격 (mm)',0.0,3.0,0.2,0.05)
    dist=st.slider('시청 거리 (m)',1.0,6.0,3.0,0.5)
    st.divider()
    pattern=st.radio('화면 패턴',['흰색','컬러(연속 이미지)','동영상(연속 영상)'],index=1)
    subject=None; video_key=None; video_speed=0.5
    if pattern=='컬러(연속 이미지)':
        subject=st.selectbox('연속 이미지 종류',['풍경','거실 인테리어','도시 야경','추상 패턴'])
    elif pattern=='동영상(연속 영상)':
        video_key=st.selectbox('연속 영상 종류',['산 풍경 타임랩스','밤하늘·호수'])
        video_speed=st.select_slider('재생 속도',options=[0.25,0.5,0.75,1.0],value=0.5,
                                     help='타임랩스가 빠르게 느껴지면 배속을 낮추세요.')
    show_seam=st.checkbox('이음새 표시',value=True,help='끄면 경계 없는 원본 상태와 비교합니다.')
    st.caption('해상도 선택 옵션 없음 — 개별 패널 4K 고정')

w=diag*25.4*16/math.sqrt(16**2+9**2)
h=diag*25.4*9/math.sqrt(16**2+9**2)
seam=bezel*2+gap
pitch=w/3840
angle=math.degrees(2*math.atan(seam/(2*dist*1000)))*60
arc_one=2*dist*1000*math.tan(math.radians(1/60)/2)
wall_frac=seam/(2*w+seam)  # seam as a fraction of the full 2-panel wall width (what gets drawn)


@st.cache_data(show_spinner=False)
def load_scene(pattern, subject):
    if pattern!='컬러(연속 이미지)':
        return Image.new('RGB',(1600,900),(245,245,245))
    name={'풍경':'landscape.jpg','거실 인테리어':'interior.jpg','도시 야경':'city.jpg','추상 패턴':'abstract.jpg'}[subject]
    return Image.open(BASE/'assets'/name).convert('RGB')


def render_wall(image, panel_w, s, visible, out_w=1600):
    """One source image shown across the full 2-panel wall, with a true-scale 2×2 seam overlaid.
    The seam covers a strip of width s/(2*panel_w+s) of the wall; fractional coverage keeps
    sub-pixel seams physically faithful after downscale (no artificial thickening).
    """
    size=2400
    line_px=(s/(2*panel_w+s)*size) if visible else 0.0
    out_h=round(size*9/16)
    scene=image.resize((size,out_h),Image.Resampling.LANCZOS)
    if visible and line_px>0:
        arr=np.asarray(scene).astype(np.float32).copy()
        cx,cy=size/2,out_h/2
        coverage_x=np.clip(np.minimum(np.arange(size)+1,cx+line_px/2)-np.maximum(np.arange(size),cx-line_px/2),0,1)
        coverage_y=np.clip(np.minimum(np.arange(out_h)+1,cy+line_px/2)-np.maximum(np.arange(out_h),cy-line_px/2),0,1)
        alpha=1-(1-coverage_y[:,None])*(1-coverage_x[None,:])
        arr=arr*(1-alpha[:,:,None])+np.array([10,12,15],dtype=np.float32)*alpha[:,:,None]
        scene=Image.fromarray(np.uint8(np.clip(arr,0,255)))
    return scene.resize((out_w,round(out_w*9/16)),Image.Resampling.LANCZOS)


VIDEO_FILES={'산 풍경 타임랩스':'vid_mountain.mp4','밤하늘·호수':'vid_nightsky.mp4'}


@st.cache_data(show_spinner=False)
def video_data_uri(filename):
    data=(BASE/'assets'/filename).read_bytes()
    return 'data:video/mp4;base64,'+base64.b64encode(data).decode('ascii')


@st.cache_data(show_spinner=False)
def image_data_uri(subject):
    img=load_scene('컬러(연속 이미지)',subject)
    buf=io.BytesIO(); img.save(buf,format='JPEG',quality=86)
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')


def retina_html(kind, uri, color, zoom, seam_frac, blur_frac, visible, speed=0.5):
    """Close-up of the panel junction at retinal scale. Distance is modeled as the eye's
    acuity blur (a ~1 arcmin spot projected onto the wall grows with distance), so the seam
    softens and melts away as viewing distance increases — no camera move, no perspective shift.
    """
    vis='true' if (visible and seam_frac>0) else 'false'
    if kind=='video':
        media=f'<video id="m" src="{uri}" autoplay muted loop playsinline></video>'
    elif kind=='image':
        media=f'<img id="m" src="{uri}">'
    else:
        media=''
    return (
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        'html,body{margin:0;background:transparent}'
        '.wrap{position:relative;width:100%;max-width:760px;margin:0 auto;aspect-ratio:16/9;'
        f'background:{color};border-radius:12px;overflow:hidden;border:1px solid #273142;'
        'box-shadow:0 14px 34px rgba(0,0,0,.5)}'
        '.view{position:absolute;inset:0;will-change:filter}'
        '#m{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform-origin:center center}'
        '.seam{position:absolute;background:#0a0c0f;display:none}'
        '</style></head><body><div class="wrap"><div class="view" id="view">'
        f'{media}'
        '<div id="vseam" class="seam" style="left:50%;top:0;height:100%;transform:translateX(-50%)"></div>'
        '<div id="hseam" class="seam" style="top:50%;left:0;width:100%;transform:translateY(-50%)"></div>'
        '</div></div>'
        '<script>(function(){'
        'var view=document.getElementById("view"),wrap=document.querySelector(".wrap"),m=document.getElementById("m");'
        'var vs=document.getElementById("vseam"),hs=document.getElementById("hseam");'
        f'var zoom={zoom:.4f},sfrac={seam_frac:.6f},bfrac={blur_frac:.6f},vis={vis},speed={speed};'
        'if(m&&m.tagName==="VIDEO"){var sp=function(){try{m.playbackRate=speed;}catch(e){}};'
        'm.addEventListener("loadedmetadata",sp);m.addEventListener("play",sp);sp();}'
        'if(m){m.style.transform="scale("+zoom+")";}'
        'function layout(){var W=wrap.clientWidth;view.style.filter="blur("+(bfrac*W).toFixed(2)+"px)";'
        'if(!vis){vs.style.display="none";hs.style.display="none";return;}'
        'var t=Math.max(1,sfrac*W);vs.style.display="block";hs.style.display="block";'
        'vs.style.width=t+"px";hs.style.height=t+"px";}'
        'window.addEventListener("resize",layout);layout();setTimeout(layout,200);'
        '})();</script>'
        '</body></html>')


def seam_wall_html(filename, frac, visible, speed=0.5):
    """Play one continuous clip with a true-scale 2×2 seam overlaid (no re-encoding).
    JS measures the real player width and draws the seam at exactly `frac` of it, matching the
    image path: a sub-pixel seam becomes a 1px line at reduced opacity (same faithful appearance).
    """
    uri=video_data_uri(filename)
    vis='true' if (visible and frac>0) else 'false'
    return (
        '<!doctype html><html><head><meta charset="utf-8"><style>'
        'html,body{margin:0;background:transparent}'
        '.wrap{position:relative;width:100%;max-width:820px;margin:0 auto;aspect-ratio:16/9;'
        'background:#0a0c0f;border-radius:12px;overflow:hidden;border:1px solid #273142;'
        'box-shadow:0 14px 34px rgba(0,0,0,.5)}'
        'video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block}'
        '.seam{position:absolute;background:#0a0c0f;display:none}'
        '</style></head><body><div class="wrap">'
        f'<video src="{uri}" autoplay muted loop playsinline controls></video>'
        '<div id="vseam" class="seam" style="left:50%;top:0;height:100%;transform:translateX(-50%)"></div>'
        '<div id="hseam" class="seam" style="top:50%;left:0;width:100%;transform:translateY(-50%)"></div>'
        '<script>(function(){'
        'var v=document.querySelector("video"),wrap=document.querySelector(".wrap");'
        'var vs=document.getElementById("vseam"),hs=document.getElementById("hseam");'
        f'var frac={frac:.8f},vis={vis},speed={speed};'
        'function spd(){try{v.playbackRate=speed;}catch(e){}}'
        'function layout(){if(!vis){return;}var W=wrap.clientWidth,t=frac*W,op=1;'
        'if(t<1){op=t;t=1;}vs.style.display="block";hs.style.display="block";'
        'vs.style.width=t+"px";vs.style.opacity=op;hs.style.height=t+"px";hs.style.opacity=op;}'
        'v.addEventListener("loadedmetadata",spd);v.addEventListener("play",spd);spd();'
        'window.addEventListener("resize",layout);layout();setTimeout(layout,200);'
        '})();</script>'
        '</div></body></html>')


is_video=(pattern=='동영상(연속 영상)')
image=None
if not is_video:
    scene=load_scene(pattern,subject)
    image=render_wall(scene,w,seam,show_seam)

main,details=st.columns([1.85,1],gap='large')
with main:
    st.markdown('<div class="sec">4모듈 OLED 월 미리보기</div>', unsafe_allow_html=True)
    if is_video:
        components.html(seam_wall_html(VIDEO_FILES[video_key],wall_frac,show_seam,video_speed),height=480)
        st.markdown('<p class="cap">단일 연속 영상을 2×2로 분할 · 음소거 자동 반복 재생 · 이음새는 실제 치수 비율(소리 없음)</p>', unsafe_allow_html=True)
    else:
        st.image(image,use_container_width=True)
        st.markdown('<p class="cap">4개 영역 모두 단일 원본을 분할한 연속 콘텐츠 · 이음새는 실제 치수 비율</p>', unsafe_allow_html=True)
        mem=io.BytesIO();image.save(mem,format='PNG')
        st.download_button('현재 화면 PNG 저장',data=mem.getvalue(),file_name='smartwall_seam_preview.png',mime='image/png')
    st.markdown('<p class="cap">실제 치수 비율에서 이음새는 전체 벽 폭의 약 0.03% 수준이라 서브픽셀로 거의 보이지 않는 것이 정상입니다. 베젤·기구 간격을 키우거나 이음새 표시를 끄고 켜며 비교해 보세요. 흰 배경·사진·동영상 모두 동일한 비율로 그려집니다.</p>', unsafe_allow_html=True)

with details:
    st.markdown('<div class="sec">계산 결과</div>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    c1.metric('전체 이음새 폭',f'{seam:.2f} mm',help='베젤 두 개 + 기구적 간격')
    c2.metric('시야각',f'{angle:.2f}′',help=f'{dist:g}m 시청 거리 기준')
    c3,c4=st.columns(2)
    c3.metric('픽셀 환산 폭',f'{seam/pitch:.2f} px')
    c4.metric('4K 픽셀 피치',f'{pitch:.3f} mm')
    st.markdown(f'''<div class="panel">
<div class="row"><span>패널 1장 활성 화면</span><b>{w:,.1f} × {h:,.1f} mm</b></div>
<div class="row"><span>단일 패널 폭 대비 이음새</span><b>{100*seam/w:.4f}%</b></div>
<div class="row"><span>전체 벽 폭 대비 (화면 렌더 비율)</span><b>{100*wall_frac:.4f}%</b></div>
<div class="row"><span>화면 1000px당 이음새</span><b>약 {wall_frac*1000:.2f} px</b></div>
<div class="row"><span>{dist:g}m에서 1분각 길이</span><b>{arc_one:.2f} mm</b></div>
<div class="row"><span>이음새 구성 (좌+간격+우)</span><b>{bezel:.2f} + {gap:.2f} + {bezel:.2f} = {seam:.2f} mm</b></div>
</div>''', unsafe_allow_html=True)
    if 1.5 <= seam <= 2.5:
        st.markdown('<div class="status ok">✓ 팀 PPT의 잠정 목표 2.0 ± 0.5 mm 범위 내</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status off">! 팀 PPT의 잠정 목표 2.0 ± 0.5 mm 범위 밖</div>', unsafe_allow_html=True)
    st.markdown('<p class="cap">사전 설계 목표와의 비교이며 실제 품질 합격 판정이 아닙니다. 시청 거리는 시야각(분각)에만 영향을 주고, 화면에 그려지는 이음새 비율은 바꾸지 않습니다.</p>', unsafe_allow_html=True)

st.divider()
st.markdown('<div class="sec">거리 체감 · 눈이 지각하는 이음새 (접합부 확대)</div>', unsafe_allow_html=True)
ACUITY=1.0  # arcmin, 20/20 기준
P_MM=200.0  # 확대해 보여줄 접합부 패치의 실제 폭(mm)
W_wall=2*w+seam
zoom=W_wall/P_MM
seam_frac_patch=seam/P_MM
sigma_mm=dist*1000.0*(ACUITY/3438.0)   # 시력 1′ 스폿이 벽에 투영된 폭 — 거리에 비례해 커짐
blur_frac=sigma_mm/P_MM
theta=(seam/(dist*1000.0))*3438.0      # 이음새 각크기(분각)
if theta < ACUITY:
    verdict,vclass='✓ 사실상 안 보임 — 이음새 없는 한 화면처럼 지각','ok'
elif theta < 2*ACUITY:
    verdict,vclass='◐ 경계 — 희미하게 인지되는 수준','off'
else:
    verdict,vclass='● 뚜렷 — 가까워서 이음새가 인지됨','off'
if pattern=='흰색':
    rkind,ruri,rcolor='white','','#f0f0f0'
elif is_video:
    rkind,ruri,rcolor='video',video_data_uri(VIDEO_FILES[video_key]),'#0a0c0f'
else:
    rkind,ruri,rcolor='image',image_data_uri(subject),'#0a0c0f'
rc1,rc2=st.columns([1.5,1],gap='large')
with rc1:
    components.html(retina_html(rkind,ruri,rcolor,zoom,seam_frac_patch,blur_frac,show_seam,video_speed),height=445)
    st.markdown('<p class="cap">접합부를 확대한 정면 뷰입니다. 사이드바의 <b>시청 거리</b>를 움직이면 눈의 분해능(≈1′)만큼 화면이 흐려져, 멀어질수록 이음새가 지각 한계 이하로 녹아 사라집니다. 원근·시점 변화 없음.</p>', unsafe_allow_html=True)
with rc2:
    st.markdown(f'''<div class="panel">
<div class="row"><span>이음새 각크기 ({dist:g} m)</span><b>{theta:.2f}′</b></div>
<div class="row"><span>눈의 분해능 한계</span><b>{ACUITY:.1f}′ · 20/20</b></div>
<div class="row"><span>접합부 확대 배율</span><b>약 {zoom:.0f}×</b></div>
<div class="row"><span>지각 블러 폭 (벽 기준)</span><b>{sigma_mm:.2f} mm</b></div>
</div>''', unsafe_allow_html=True)
    st.markdown(f'<div class="status {vclass}">{verdict}</div>', unsafe_allow_html=True)
    st.markdown('<p class="cap">각크기가 분해능(1′)보다 작아지면 사람 눈은 이음새를 더 이상 구분하지 못합니다.</p>', unsafe_allow_html=True)

st.divider()
st.markdown('<div class="sec">설계안 비교 · 현재 조건과 1mm 이음새</div>', unsafe_allow_html=True)
comparison=pd.DataFrame([
    {'설계안':'현재 입력','이음새 폭(mm)':round(seam,2),'시야각(분각)':round(angle,3),'픽셀 환산(px)':round(seam/pitch,2)},
    {'설계안':'이음새 1mm 가정','이음새 폭(mm)':1.0,'시야각(분각)':round(math.degrees(2*math.atan(1/(2*dist*1000)))*60,3),'픽셀 환산(px)':round(1/pitch,2)},
])
st.dataframe(comparison,hide_index=True,use_container_width=True)
st.download_button('비교 결과 CSV 다운로드',data=comparison.to_csv(index=False).encode('utf-8-sig'),file_name='smartwall_seam_metrics.csv',mime='text/csv')

with st.expander('계산식 · 적용 범위 · 해석상 주의사항'):
    st.latex(r'W = D\times 25.4\times \frac{16}{\sqrt{16^2+9^2}}')
    st.latex(r'S = 2B + G')
    st.latex(r'\theta_{arcmin} = 2\arctan\left(\frac{S}{2L}\right)\frac{180\times60}{\pi}')
    st.latex(r'\mathrm{pixel\ equivalent}= S/(W/3840)')
    st.write('D: 대각선(inch), W: 활성 화면 너비(mm), B: 각 내부 베젤(mm), G: 기구적 간격(mm), S: 전체 이음새(mm), L: 시청 거리(mm).')
    st.warning('실제 비율 화면은 모니터 해상도에서 서브픽셀 두께가 되어 거의 보이지 않을 수 있습니다. 이것이 실제 시청 환경에서 이음새가 보이지 않는다는 뜻은 아닙니다. 1분각 역시 시인성 합격/불합격의 절대 기준이 아닙니다.')
    st.write('모듈의 베젤은 내부 비발광 가장자리를 뜻하며, 네 모듈에 동일한 값으로 모델링했습니다. 모든 화면 크기는 활성 화면(16:9) 기준입니다. 봉지 여백, 반사율, 밝기, 색 편차, 단차, 관측각과 실제 롤러블 구조는 반영되지 않았습니다. 55/65/77형 4K 롤러블 모듈의 상용 공급 가능성을 뜻하지 않습니다.')
    st.write('모든 베젤/간격 입력값은 사용자가 정하는 설계 가정입니다. 실제 측정값이 아닙니다.')
    st.caption('연속 이미지 4종은 Unsplash 무료 라이선스 사진, 연속 영상 2종은 Mixkit 무료 라이선스 영상입니다.')
