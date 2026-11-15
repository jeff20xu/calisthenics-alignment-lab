from pathlib import Path
import hashlib
import json
import math
import os
import tempfile
import plotly.express as px
import streamlit as st
from analyzer import analyze, ensure_model
from biomechanics import METRICS, observations
from live import LiveProcessor, ice_configuration
from streamlit_webrtc import WebRtcMode, webrtc_streamer


def revoke_live_camera():
    # Non-widget state is authoritative across component and fragment reruns.
    st.session_state['_camera_consent'] = False
    st.session_state['live_consent'] = False
    if not st.session_state.get('_live_stopping'):
        st.session_state['_live_generation'] = st.session_state.get('_live_generation', 0) + 1
    st.session_state['_live_stopping'] = True
    context = st.session_state.get('_live_context')
    processor = context.video_processor if context is not None else None
    if processor is not None:
        processor.close()


def change_input_mode():
    st.session_state['_input_mode'] = st.session_state['input_mode']
    if st.session_state['_input_mode'] != 'Live camera':
        revoke_live_camera()


def change_camera_consent():
    st.session_state['_camera_consent'] = st.session_state['live_consent']
    if not st.session_state['_camera_consent']:
        revoke_live_camera()


def stop_live_camera():
    context = st.session_state.get('_live_context')
    if context is None:
        st.session_state['_live_stopping'] = False
        return
    processor = context.video_processor
    if processor is not None:
        processor.close()
    if context.state.playing or context.state.signalling:
        st.session_state['_live_context'] = webrtc_streamer(
            key=st.session_state['_live_component_key'], mode=WebRtcMode.SENDRECV,
            rtc_configuration=st.session_state.get('_live_rtc'),
            media_stream_constraints={'video': True, 'audio': False},
            desired_playing_state=False, sendback_audio=False,
            media_toggle_controls=False,
        )
    else:
        st.session_state.pop('_live_context', None)
        st.session_state['_live_stopping'] = False


@st.fragment(run_every=0.5)
def live_measurements(context, selected_skill):
    if (st.session_state.get('_input_mode') != 'Live camera'
            or not st.session_state.get('_camera_consent')
            or st.session_state.get('_live_stopping')
            or st.session_state.get('_live_context') is not context):
        return
    processor = context.video_processor
    if processor is None:
        st.info('Waiting for browser video. Press START and allow camera access; the first frame initializes the detector.')
        return
    snapshot = processor.snapshot()
    if snapshot['error']:
        st.session_state['_live_error'] = snapshot['error']
        st.rerun(scope='app')
    if snapshot['closed']:
        st.info('Camera stopped. Press START for a new analysis.')
        return
    row, summary = snapshot['row'], snapshot['summary']
    if row is None:
        st.info('Waiting for the first analyzed frame…')
        return
    a, b, c = st.columns(3)
    a.metric('Measured frames', f"{summary['measured_fraction']:.0%}")
    b.metric('Live elapsed time', f"{summary['duration_s']:.1f} s")
    if selected_skill == 'Handstand push-up':
        c.metric('Full repetitions', summary['full_reps'])
    else:
        c.metric('Longest tracked posture', f"{summary['longest_tracked_posture_s']:.1f} s")
    st.caption(f"Tracking status: {row['status']}. Latest processed frame: {row['time_s']:.2f} s.")
    values = []
    for key, label in METRICS.items():
        value = row.get(key)
        values.append({'Measurement': label, 'Latest value': f'{value:.2f}' if value is not None and math.isfinite(value) else '—'})
    st.dataframe(values, hide_index=True, width='stretch')


def live_camera(config):
    st.subheader('Live browser camera')
    st.warning('On a hosted app, camera frames are sent to and processed on the app server. '
               'Live video is not recorded or saved by this app. Microphone access is disabled. '
               'Only use a camera view you consent to share with the server.')
    st.caption('Use HTTPS (or localhost), then consent below, press START, and allow camera access. '
               'STOP releases the detector; START begins a fresh analysis. Changing analysis settings '
               'resets calibration, tracked time, and repetitions, even while the camera is running.')
    with st.expander('Camera and connection troubleshooting'):
        st.write('If permission is denied, enable camera access in your browser site settings and try again. '
                 'Close other apps using the camera. Remote HTTP pages cannot access the camera: use HTTPS. '
                 'If connection negotiation stalls or video never arrives, try a different network. '
                 'The default STUN server is not sufficient on every network; the host may need TURN '
                 'configured through ICE_SERVERS in environment variables or Streamlit secrets. '
                 'ICE_SERVERS must be a JSON array of objects with urls and, for authenticated TURN, '
                 'username and credential. TURN credentials are necessarily delivered to the browser; '
                 'use restricted, short-lived credentials, not account API keys.')
    st.session_state['live_consent'] = st.session_state.get('_camera_consent', False)
    st.checkbox('I consent to send my camera frames to this app server.',
                key='live_consent', on_change=change_camera_consent)
    if not st.session_state.get('_camera_consent'):
        stop_live_camera()
        st.info('Camera access stays off until you consent and press START.')
        return
    if st.session_state.get('_live_stopping'):
        stop_live_camera()
        if st.session_state.get('_live_stopping'):
            st.info('Stopping the previous camera connection. Wait for it to close before pressing START again.')
            return
    try:
        raw_ice = os.environ.get('ICE_SERVERS')
        if raw_ice is None:
            try:
                raw_ice = st.secrets.get('ICE_SERVERS')
            except FileNotFoundError:
                raw_ice = None
        rtc = ice_configuration(raw_ice)
    except (ValueError, TypeError) as exc:
        stop_live_camera()
        st.error(f'Live camera configuration error: {exc}')
        return
    st.session_state['_live_rtc'] = rtc
    if st.session_state.get('_live_error'):
        stop_live_camera()
        st.error(st.session_state['_live_error'])
        previous = st.session_state.get('_live_context')
        stopping = previous is not None and (previous.state.playing or previous.state.signalling)
        if st.button('Reset live analysis', disabled=stopping):
            st.session_state.pop('_live_error', None)
            st.rerun()
        st.caption('After stopping, reset and press START to try again. Video upload remains available.')
        return
    try:
        if not st.session_state.get('_live_model_ready'):
            with st.spinner('Preparing the official MediaPipe pose model…'):
                ensure_model()
            st.session_state['_live_model_ready'] = True
        previous = st.session_state.get('_live_context')
        processor = previous.video_processor if previous is not None else None
        if processor is not None:
            processor.configure(config)
        component_key = f"live-camera-{st.session_state.get('_live_generation', 0)}"
        st.session_state['_live_component_key'] = component_key
        context = webrtc_streamer(
            key=component_key, mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc,
            video_processor_factory=lambda: LiveProcessor(config),
            media_stream_constraints={
                'video': {'width': {'ideal': 640}, 'height': {'ideal': 480},
                          'frameRate': {'ideal': 15, 'max': 20}},
                'audio': False,
            },
            async_processing=True, sendback_audio=False, media_toggle_controls=False,
            video_html_attrs={'autoPlay': True, 'controls': False, 'muted': True},
        )
        st.session_state['_live_context'] = context
    except Exception as exc:
        stop_live_camera()
        st.error(f'Could not start live camera: {exc}')
        st.info('Check camera permission, HTTPS, and the connection guidance above, then try START again.')
        return
    st.caption('The returned video is annotated on the server. Inspect the overlay: confident landmarks '
               'can still be wrong. Only the latest measurements are retained; there is no live recording or audio.')
    live_measurements(context, config['skill'])

st.set_page_config(page_title='Calisthenics Alignment Lab', layout='wide')
st.title('Calisthenics Alignment Lab')
st.caption('Upload or live-camera joint stacking, body alignment, and mechanical leverage analysis')
st.info('Film one athlete from the side with a fixed, level camera. Keep wrists through ankles visible. '
        'Measurements are 2-D estimates, not medical advice or measured joint torque.')
with st.sidebar:
    st.header('Analysis settings')
    skill = st.radio('Skill', ['Handstand','Handstand push-up','Planche'])
    side = st.radio('Camera-facing body side', ['Left','Right'], horizontal=True)
    variant = st.radio('Planche variation', ['Full','Straddle','Advanced tuck','Tuck'], disabled=skill!='Planche')
    confidence = st.slider('Minimum landmark confidence', .3,.9,.6,.05)
    inverted = st.checkbox('Rotate image 180° for the pose detector', value=skill!='Planche', key=f'detector_{skill}',
                           help='Useful for inverted poses. The output and geometry stay in the original orientation. Turn off for planches or if tracking fails.')
    st.caption('Analyze the original upright video. Do not upload footage rotated sideways. Try the other detector orientation if tracking fails.')
    with st.expander('Optional illustrative gravity model'):
        st.warning('Not net shoulder torque. Static trunk + leg model only; disabled for push-ups.')
        enabled = st.checkbox('Enable partial gravitational-moment estimate', disabled=skill=='Handstand push-up')
        mass = st.number_input('Body mass (kg)',20.0,200.0,70.0,1.0)
        torso = st.number_input('Measured shoulder-to-hip length (m)',.2,.9,.5,.01)
        st.caption('Assumes trunk = 50%, paired thighs = 20%, paired shanks = 9% of body mass, '
                   'each concentrated at its segment midpoint. Ignores arms, head, feet, contact forces, '
                   'and acceleration. These approximate assumptions are illustrative, not individualized anatomy. '
                   'Requires a side-on, unforeshortened torso during the initial calibration frames.')

with st.expander('How to read the measurements'):
    st.markdown('''- **180° elbow angle:** straight-arm geometry. Push-ups intentionally bend the elbows.
- **Stack offset:** horizontal shoulder-to-wrist distance divided by the calibrated torso length; not a balance score.
- **Body-line deviation:** hip distance from the shoulder–ankle line, divided by torso length.
- **Shoulder opening:** elbow–shoulder–hip angle.
- **Planche leverage:** shoulder lean and hip–ankle horizontal distance. More leverage is not inherently better or worse.
- **Tracked posture time:** a geometric posture match, not proof of an unsupported hold.
- **Calibration:** first 10 reliable frames establish torso scale; keep the torso visible and side-on.
- **HSPU repetitions:** sustained straight arms → elbow bend ≤100° → straight arms ≥160°. This is a configurable training definition in the source, not a universal competition standard.
- Missing or low-confidence joints produce gaps rather than invented measurements. Charts never join those gaps.''')

config = {'skill':skill,'side':side,'variant':variant,'confidence':confidence,'detector_inverted':inverted,
          'mass_kg':mass if enabled and skill!='Handstand push-up' else 0,
          'torso_m':torso if enabled and skill!='Handstand push-up' else 0}
st.session_state['input_mode'] = st.session_state.get('_input_mode', 'Video upload')
st.session_state.setdefault('_input_mode', 'Video upload')
st.radio('Input mode', ['Video upload', 'Live camera'], horizontal=True,
         key='input_mode', on_change=change_input_mode)
if st.session_state['_input_mode'] == 'Live camera':
    live_camera(config)
else:
    stop_live_camera()
    st.caption('Uploaded clips are processed on the app server, which is remote when this app is hosted. '
               'Temporary processing files are deleted after analysis; results remain in this browser session’s server memory '
               'for preview and download. Upload only footage you consent to share with the host.')

    uploaded = st.file_uploader('Upload a clip (MP4, MOV, AVI, or WebM; maximum 3 minutes / 200 MB)',type=['mp4','mov','avi','webm'])
    if uploaded is not None:
        data = uploaded.getvalue()
        signature = hashlib.sha256(data+json.dumps(config,sort_keys=True).encode()).hexdigest()
        if st.button('Analyze video',type='primary'):
            st.session_state.pop('result',None)
            progress = st.progress(0.0,text='Loading pose model and analyzing frames…')
            try:
                with tempfile.TemporaryDirectory(prefix='calisthenics-') as work:
                    source = Path(work)/('input'+Path(uploaded.name).suffix.lower())
                    target = Path(work)/'annotated.mp4'
                    source.write_bytes(data)
                    df, summary = analyze(source,target,progress=lambda v: progress.progress(v,text='Analyzing video…'),**config)
                    st.session_state['result'] = {'signature':signature,'df':df,'summary':summary,'video':target.read_bytes()}
                progress.progress(1.0,text='Analysis complete')
            except Exception as exc:
                st.error(f'Analysis failed: {exc}')
            finally:
                progress.empty()
        result = st.session_state.get('result')
        if result and result['signature'] != signature:
            st.warning('The video or settings changed. Analyze again to update the results.')
        elif result:
            df, summary = result['df'],result['summary']
            a,b,c = st.columns(3)
            a.metric('Measured frames',f"{summary['measured_fraction']:.0%}")
            b.metric('Clip duration',f"{summary['duration_s']:.1f} s")
            if skill=='Handstand push-up':
                c.metric('Full repetitions',summary['full_reps'])
            else:
                c.metric('Longest tracked posture',f"{summary['longest_tracked_posture_s']:.1f} s")
            if summary['measured_fraction'] < .5:
                st.warning('Less than half the analyzed frames matched the selected posture with reliable landmarks. '
                           'Inspect the overlay before interpreting results; try the other detector orientation or camera-facing side.')
            video_tab, chart_tab, data_tab = st.tabs(['Annotated video','Measurements','Data and downloads'])
            with video_tab:
                st.video(result['video'])
                st.caption('The export has no audio. Green segments are tracked joints; the vertical line passes through the wrist. '
                           'Landmarks may still be wrong despite high model confidence—inspect the overlay.')
                for note in observations(df,skill,variant):
                    st.write(note)
            with chart_tab:
                options = [k for k in METRICS if df[k].notna().any()]
                if options:
                    metric = st.selectbox('Measurement',options,format_func=lambda k: METRICS[k])
                    fig = px.line(df,x='time_s',y=metric,labels={'time_s':'Time (seconds)',metric:METRICS[metric]})
                    fig.update_traces(connectgaps=False)
                    st.plotly_chart(fig,width='stretch')
                    st.dataframe(df[options].describe().loc[['min','50%','max']].rename(index={'50%':'median'}),width='stretch')
                else:
                    st.info('No reliable measurements available. Review the video and settings.')
            with data_tab:
                st.dataframe(df,width='stretch')
                st.download_button('Download annotated MP4',result['video'],'calisthenics-annotated.mp4','video/mp4')
                st.download_button('Download measurements CSV',df.to_csv(index=False).encode(),'calisthenics-measurements.csv','text/csv')
                st.download_button('Download analysis settings and summary',json.dumps(summary,indent=2),'calisthenics-summary.json','application/json')
    else:
        st.markdown('### Start with a short side-view clip\nChoose a skill, select the visible body side, and upload your video. '
                    'Processing runs on the app server, not necessarily on your device. The first analysis downloads '
                    'the official MediaPipe pose model; inference does not use a separate remote AI service.')
