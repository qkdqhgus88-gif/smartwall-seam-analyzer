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
    dist=st.slider('시청 거리 (m)',1.0,6.0,5.0,0.5)
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
wall_frac=seam/(2*w+seam)          # seam as a fraction of the full 2-panel wall width (what gets drawn)
wall_w_m=(2*w+seam)/1000.0
wall_h_m=(2*h+seam)/1000.0

IMAGE_FILES={'풍경':'landscape.jpg','거실 인테리어':'interior.jpg','도시 야경':'city.jpg','추상 패턴':'abstract.jpg'}
VIDEO_FILES={'산 풍경 타임랩스':'vid_mountain.mp4','밤하늘·호수':'vid_nightsky.mp4'}


@st.cache_data(show_spinner=False)
def video_data_uri(filename):
    data=(BASE/'assets'/filename).read_bytes()
    return 'data:video/mp4;base64,'+base64.b64encode(data).decode('ascii')


@st.cache_data(show_spinner=False)
def image_data_uri(filename):
    img=Image.open(BASE/'assets'/filename).convert('RGB')
    buf=io.BytesIO(); img.save(buf,format='JPEG',quality=86)
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')


def room_data_uri():
    # Not cached: the backdrop file may be swapped, and an arg-less cache would pin a stale image.
    data=(BASE/'assets'/'room.webp').read_bytes()
    return 'data:image/webp;base64,'+base64.b64encode(data).decode('ascii')

# Display (active 2×2 screen) corners in the frontal room render, as fractions of the image (TL,TR,BR,BL).
ROOM_AR_W=1024.0; ROOM_AR_H=683.0
SCREEN_CORNERS=[[0.140,0.070],[0.886,0.070],[0.886,0.717],[0.140,0.717]]


def room_html(kind, uri, color, frac, visible, speed=0.5):
    """Photo-real showroom: the shared living-room render is the backdrop, and the live content
    (white/photo/video + true-scale 2×2 seam) is perspective-warped (CSS matrix3d homography)
    onto the display's quad so it sits exactly where the panel is in the photo."""
    vis='true' if (visible and frac>0) else 'false'
    if kind=='video':
        media=f'<video id="m" src="{uri}" autoplay muted loop playsinline></video>'
    elif kind=='image':
        media=f'<img id="m" src="{uri}">'
    else:
        media=''
    bg=room_data_uri()
    css=('html,body{margin:0;background:transparent}*{box-sizing:border-box}'
         '.scene{position:relative;width:100%;max-width:1020px;margin:0 auto;aspect-ratio:'+f'{ROOM_AR_W}/{ROOM_AR_H}'+';'
         'border-radius:12px;overflow:hidden;border:1px solid #2a2622;'
         "background-image:url('"+bg+"');background-size:cover;background-position:center;background-repeat:no-repeat}"
         '#warp{position:absolute;left:0;top:0;width:1000px;height:562px;transform-origin:0 0;background:'+color+'}'
         '#m{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block}'
         '.seam{position:absolute;background:#0a0c0f;display:none}')
    return (
        '<!doctype html><html><head><meta charset="utf-8"><style>'+css+'</style></head><body>'
        '<div class="scene"><div id="warp">'+media+
        '<div id="vseam" class="seam" style="left:50%;top:0;height:100%;transform:translateX(-50%)"></div>'
        '<div id="hseam" class="seam" style="top:50%;left:0;width:100%;transform:translateY(-50%)"></div>'
        '</div></div>'
        '<script>(function(){'
        'var scene=document.querySelector(".scene"),warp=document.getElementById("warp"),m=document.getElementById("m");'
        'var vs=document.getElementById("vseam"),hs=document.getElementById("hseam");'
        f'var frac={frac:.8f},vis={vis},speed={speed},Wc=1000,Hc=562;'
        'var C='+repr(SCREEN_CORNERS).replace(' ','')+';'
        'function adj(x){return[x[4]*x[8]-x[5]*x[7],x[2]*x[7]-x[1]*x[8],x[1]*x[5]-x[2]*x[4],x[5]*x[6]-x[3]*x[8],x[0]*x[8]-x[2]*x[6],x[2]*x[3]-x[0]*x[5],x[3]*x[7]-x[4]*x[6],x[1]*x[6]-x[0]*x[7],x[0]*x[4]-x[1]*x[3]];}'
        'function mm(a,b){var c=[],i,j,k,s;for(i=0;i<3;i++)for(j=0;j<3;j++){s=0;for(k=0;k<3;k++)s+=a[3*i+k]*b[3*k+j];c[3*i+j]=s;}return c;}'
        'function mv(x,v){return[x[0]*v[0]+x[1]*v[1]+x[2]*v[2],x[3]*v[0]+x[4]*v[1]+x[5]*v[2],x[6]*v[0]+x[7]*v[1]+x[8]*v[2]];}'
        'function basis(p){var m=[p[0][0],p[1][0],p[2][0],p[0][1],p[1][1],p[2][1],1,1,1];var v=mv(adj(m),[p[3][0],p[3][1],1]);return mm(m,[v[0],0,0,0,v[1],0,0,0,v[2]]);}'
        'if(m&&m.tagName==="VIDEO"){var sp=function(){try{m.playbackRate=speed;}catch(e){}};m.addEventListener("loadedmetadata",sp);m.addEventListener("play",sp);sp();}'
        'function layout(){var W=scene.clientWidth,H=scene.clientHeight;'
        'var d=[[C[0][0]*W,C[0][1]*H],[C[1][0]*W,C[1][1]*H],[C[2][0]*W,C[2][1]*H],[C[3][0]*W,C[3][1]*H]];'
        'var s=[[0,0],[Wc,0],[Wc,Hc],[0,Hc]];var t=mm(basis(d),adj(basis(s)));'
        'for(var i=0;i<9;i++)t[i]=t[i]/t[8];'
        'warp.style.transform="matrix3d("+[t[0],t[3],0,t[6],t[1],t[4],0,t[7],0,0,1,0,t[2],t[5],0,t[8]].join(",")+")";'
        'if(!vis){vs.style.display="none";hs.style.display="none";return;}'
        'var tk=frac*Wc,op=1;if(tk<1){op=tk;tk=1;}'
        'vs.style.display="block";hs.style.display="block";vs.style.width=tk+"px";vs.style.opacity=op;hs.style.height=tk+"px";hs.style.opacity=op;}'
        'window.addEventListener("resize",layout);layout();setTimeout(layout,150);setTimeout(layout,500);'
        '})();</script></body></html>')


if pattern=='흰색':
    disp_kind,disp_uri,disp_color='white','','#f4f4f4'
elif pattern=='동영상(연속 영상)':
    disp_kind,disp_uri,disp_color='video',video_data_uri(VIDEO_FILES[video_key]),'#0a0c0f'
else:
    disp_kind,disp_uri,disp_color='image',image_data_uri(IMAGE_FILES[subject]),'#0a0c0f'

st.markdown('<div class="sec">실제 공간 미리보기 · 쇼룸</div>', unsafe_allow_html=True)
components.html(room_html(disp_kind,disp_uri,disp_color,wall_frac,show_seam,video_speed),height=700)
st.markdown('<p class="cap">거실에 4모듈 월 디스플레이를 설치한 모습입니다. 디스플레이 영역에 현재 패턴(흰색·사진·동영상)과 2×2 이음새가 실제 치수 비율로 합성됩니다. 실제 치수 비율에서 이음새는 전체 벽 폭의 약 0.03% 수준이라 거의 보이지 않는 것이 정상입니다.</p>', unsafe_allow_html=True)

m1,m2,m3,m4=st.columns(4)
m1.metric('이음새 폭',f'{seam:.2f} mm',help='베젤 두 개 + 기구적 간격')
m2.metric('픽셀 환산',f'{seam/pitch:.2f} px',help='4K 픽셀 피치 기준')
m3.metric('벽 대비 이음새',f'{100*wall_frac:.4f}%',help='전체 벽(2패널) 폭 대비 — 화면에 그려지는 비율')
m4.metric('시야각',f'{angle:.2f}′',help=f'{dist:g}m 시청 거리 기준 (거리 슬라이더에 따라 변함)')

st.divider()
left,right=st.columns([1,1],gap='large')
with left:
    st.markdown('<div class="sec">세부 수치</div>', unsafe_allow_html=True)
    st.markdown(f'''<div class="panel">
<div class="row"><span>패널 1장 활성 화면</span><b>{w:,.1f} × {h:,.1f} mm</b></div>
<div class="row"><span>전체 벽(2×2) 크기</span><b>{wall_w_m:.2f} × {wall_h_m:.2f} m</b></div>
<div class="row"><span>단일 패널 폭 대비 이음새</span><b>{100*seam/w:.4f}%</b></div>
<div class="row"><span>화면 1000px당 이음새</span><b>약 {wall_frac*1000:.2f} px</b></div>
<div class="row"><span>4K 픽셀 피치</span><b>{pitch:.3f} mm</b></div>
<div class="row"><span>{dist:g}m에서 1분각 길이</span><b>{arc_one:.2f} mm</b></div>
<div class="row"><span>이음새 구성 (좌+간격+우)</span><b>{bezel:.2f} + {gap:.2f} + {bezel:.2f} = {seam:.2f} mm</b></div>
</div>''', unsafe_allow_html=True)
    if 1.5 <= seam <= 2.5:
        st.markdown('<div class="status ok">✓ 팀 PPT의 잠정 목표 2.0 ± 0.5 mm 범위 내</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status off">! 팀 PPT의 잠정 목표 2.0 ± 0.5 mm 범위 밖</div>', unsafe_allow_html=True)
    st.markdown('<p class="cap">사전 설계 목표와의 비교이며 실제 품질 합격 판정이 아닙니다. 시청 거리는 시야각(분각)에만 영향을 주고, 화면에 그려지는 이음새 비율은 바꾸지 않습니다.</p>', unsafe_allow_html=True)
with right:
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
    st.caption('연속 이미지 4종은 Unsplash 무료 라이선스 사진, 연속 영상 2종은 Mixkit 무료 라이선스 영상입니다. 쇼룸 배경은 제공된 콘셉트 렌더 이미지이며 실제 설치 사진이 아닙니다.')
