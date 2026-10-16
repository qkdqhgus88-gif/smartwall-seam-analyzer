"""Smart Wall Seam Visibility Analyzer — Streamlit demo.
Run: streamlit run app.py
All bezel values and visualizations are illustrative design scenarios, not measured OLED characteristics.
"""
from pathlib import Path
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='Smart Wall | Seam Analyzer', page_icon='📺', layout='wide')
st.markdown('''<style>
.block-container {padding-top: 1.8rem; max-width: 1220px;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
.subtle {color:#768493;font-size:0.85rem;}
.note {padding: 0.8rem 1rem; border: 1px solid #c9d7e4; border-radius: .65rem;}
</style>''', unsafe_allow_html=True)
st.title('Smart Wall · Seam Visibility Analyzer')
st.caption('4개 16:9 OLED 모듈 (2×2) | 모듈별 4K UHD 3840×2160 고정 | 설계 검토용 시뮬레이션')

with st.sidebar:
    st.header('설계 입력')
    diag=st.radio('개별 OLED 크기', [55,65,77], index=1, horizontal=True, format_func=lambda x:f'{x}형')
    bezel=st.slider('각 패널의 내부 베젤 폭 (mm)',0.0,3.0,0.3,0.05,
                    help='네 패널의 가로·세로 내부 경계에 동일한 베젤 폭을 가정합니다.')
    gap=st.slider('패널 사이 기구적 간격 (mm)',0.0,3.0,0.2,0.05)
    dist=st.slider('시청 거리 (m)',1.0,6.0,3.0,0.5)
    pattern=st.radio('화면 패턴',['흰색','회색','검정','컬러(연속 이미지)'],horizontal=False,index=3)
    subject=None
    if pattern=='컬러(연속 이미지)':
        subject=st.selectbox('연속 이미지 종류',['풍경','거실 인테리어','도시 야경','추상 패턴'])
    display_mode=st.radio('이음새 표시 배율',['실제 치수 비율','이음새 확대'],horizontal=True)
    show_seam=st.checkbox('이음새 표시',value=True,help='끄면 원본 연속 영상의 경계 없는 상태를 비교합니다.')
    st.caption('해상도 선택 옵션 없음 — 개별 패널 4K 고정')

w=diag*25.4*16/math.sqrt(16**2+9**2)
h=diag*25.4*9/math.sqrt(16**2+9**2)
seam=bezel*2+gap
pitch=w/3840
angle=math.degrees(2*math.atan(seam/(2*dist*1000)))*60
arc_one=2*dist*1000*math.tan(math.radians(1/60)/2)

@st.cache_data(show_spinner=False)
def load_scene(pattern, subject):
    if pattern!='컬러(연속 이미지)':
        rgb={'흰색':(245,245,245),'회색':(147,147,147),'검정':(12,14,20)}[pattern]
        return Image.new('RGB',(1600,900),rgb)
    name={'풍경':'landscape.jpg','거실 인테리어':'interior.jpg','도시 야경':'city.jpg','추상 패턴':'abstract.jpg'}[subject]
    return Image.open(BASE/'assets'/name).convert('RGB')


def render_wall(image, panel_w, panel_h, s, mode, visible, out_w=1600):
    """Render four crops from ONE source image; do not restart image coordinates per quadrant.
    Real: geometry is proportional to physical dimensions. Super-sampling helps subpixel seams.
    Zoom: only the seam is exaggerated; not to scale.
    """
    # Use a wider high-resolution raster for true-scale antialiasing.
    size=out_w
    true_px=s/(2*panel_w+s)*size
    line_px=true_px if mode=='실제 치수 비율' else max(4.0,s*12.0)
    if not visible: line_px=0.0
    if mode=='실제 치수 비율':
        # limit huge memory footprint: raster is 2400x1350 and output is 1600x900
        size=2400
        line_px=(s/(2*panel_w+s)*size) if visible else 0.0
    out_h=round(size*9/16)
    # Continuous base: each quadrant comes from one image. Preserve content continuity when overlay toggles.
    scene=image.resize((size,out_h),Image.Resampling.LANCZOS)
    # For zoom mode, emulate splitting and separating the source quadrants while keeping continuity.
    if mode=='이음새 확대' and visible and line_px>0:
        px=int(round(line_px)); px=min(px,80)
        left=(size-px)//2; right=size-px-left
        top=(out_h-px)//2; bottom=out_h-px-top
        srcw,srch=image.size
        result=Image.new('RGB',(size,out_h),(10,12,15))
        quadrants=[((0,0,srcw//2,srch//2),(0,0,left,top)),
                   ((srcw//2,0,srcw,srch//2),(left+px,0,size,top)),
                   ((0,srch//2,srcw//2,srch),(0,top+px,left,out_h)),
                   ((srcw//2,srch//2,srcw,srch),(left+px,top+px,size,out_h))]
        for src,dst in quadrants:
            tile=image.crop(src).resize((dst[2]-dst[0],dst[3]-dst[1]),Image.Resampling.LANCZOS)
            result.paste(tile,dst[:2])
        return result
    if visible and line_px>0:
        # fractional seam width: cover pixels by area proportion for a physically faithful downsample.
        arr=np.asarray(scene).astype(np.float32).copy()
        cx,cy=size/2,out_h/2
        coverage_x=np.clip(np.minimum(np.arange(size)+1,cx+line_px/2)-np.maximum(np.arange(size),cx-line_px/2),0,1)
        coverage_y=np.clip(np.minimum(np.arange(out_h)+1,cy+line_px/2)-np.maximum(np.arange(out_h),cy-line_px/2),0,1)
        alpha=1-(1-coverage_y[:,None])*(1-coverage_x[None,:])
        arr=arr*(1-alpha[:,:,None])+np.array([10,12,15],dtype=np.float32)*alpha[:,:,None]
        scene=Image.fromarray(np.uint8(np.clip(arr,0,255)))
    return scene.resize((out_w,round(out_w*9/16)),Image.Resampling.LANCZOS)

scene=load_scene(pattern,subject)
image=render_wall(scene,w,h,seam,display_mode,show_seam)
main,details=st.columns([1.85,1],gap='large')
with main:
    st.subheader('4모듈 OLED 화면 구성')
    st.image(image,caption=('실제 패널·이음새 치수 비율' if display_mode=='실제 치수 비율' else '이음새를 확대해 강조한 비교 화면') + ' · 4개 영역 모두 단일 원본 이미지 기반',use_container_width=True)
    st.info('컬러 패턴은 하나의 원본 이미지를 2×2로 분할한 연속 콘텐츠입니다. 이음새 OFF와 ON을 비교해 보세요.')
    mem=io.BytesIO();image.save(mem,format='PNG')
    st.download_button('현재 화면 PNG 저장',data=mem.getvalue(),file_name='smartwall_seam_preview.png',mime='image/png')
with details:
    st.subheader('계산 결과')
    st.metric('전체 이음새 폭',f'{seam:.2f} mm',help='베젤 두 개 + 기구적 간격')
    st.metric('시야각 (해당 시청 거리)',f'{angle:.2f}′')
    c1,c2=st.columns(2)
    c1.metric('픽셀 환산 폭',f'{seam/pitch:.2f} px')
    c2.metric('4K 픽셀 피치',f'{pitch:.3f} mm')
    st.write(f'**패널 1장 활성 화면:** {w:,.1f} × {h:,.1f} mm')
    st.write(f'**화면 너비 대비 이음새:** {100*seam/w:.4f}%')
    st.write(f'**{dist:g}m에서 1분각 길이:** {arc_one:.2f} mm')
    st.markdown('**이음새 구성**')
    st.write(f'{bezel:.2f} (좌 베젤) + {gap:.2f} (기구 간격) + {bezel:.2f} (우 베젤) = **{seam:.2f} mm**')
    if 1.5 <= seam <= 2.5:
        st.success('팀 PPT의 잠정 목표 2.0 ± 0.5 mm 범위 내')
    else:
        st.warning('팀 PPT의 잠정 목표 2.0 ± 0.5 mm 범위 밖')
    st.caption('이 비교는 사전 설계 목표와의 비교이며, 실제 품질 합격 판정이 아닙니다.')

st.divider()
st.subheader('설계안 비교: 현재 조건과 1mm 이음새')
comparison=pd.DataFrame([
    {'설계안':'현재 입력','이음새 폭(mm)':round(seam,2),'시야각(분각)':round(angle,3),'픽셀 환산(px)':round(seam/pitch,2)},
    {'설계안':'이음새 1mm 가정','이음새 폭(mm)':1.0,'시야각(분각)':round(math.degrees(2*math.atan(1/(2*dist*1000)))*60,3),'픽셀 환산(px)':round(1/pitch,2)},
])
st.dataframe(comparison,hide_index=True,use_container_width=True)
st.download_button('비교 결과 CSV 다운로드',data=comparison.to_csv(index=False).encode('utf-8-sig'),file_name='smartwall_seam_metrics.csv',mime='text/csv')
with st.expander('계산식·적용 범위·해석상 주의사항'):
    st.latex(r'W = D\times 25.4\times \frac{16}{\sqrt{16^2+9^2}}')
    st.latex(r'S = 2B + G')
    st.latex(r'\theta_{arcmin} = 2\arctan\left(\frac{S}{2L}\right)\frac{180\times60}{\pi}')
    st.latex(r'\mathrm{pixel\ equivalent}= S/(W/3840)')
    st.write('D: 대각선(inch), W: 활성 화면 너비(mm), B: 각 내부 베젤(mm), G: 기구적 간격(mm), S: 전체 이음새(mm), L: 시청 거리(mm).')
    st.warning('실제 비율 화면은 모니터 해상도에서 서브픽셀 두께가 되어 거의 보이지 않을 수 있습니다. 이것이 실제 시청 환경에서 이음새가 보이지 않는다는 뜻은 아닙니다. 1분각 역시 시인성 합격/불합격의 절대 기준이 아닙니다.')
    st.write('모듈의 베젤은 내부 비발광 가장자리를 뜻하며, 네 모듈에 동일한 값으로 모델링했습니다. 모든 화면 크기는 활성 화면(16:9) 기준입니다. 봉지 여백, 반사율, 밝기, 색 편차, 단차, 관측각과 실제 롤러블 구조는 반영되지 않았습니다. 55/65/77형 4K 롤러블 모듈의 상용 공급 가능성을 뜻하지 않습니다.')
    st.write('모든 베젤/간격 입력값은 사용자가 정하는 설계 가정입니다. 실제 측정값이 아닙니다.')
