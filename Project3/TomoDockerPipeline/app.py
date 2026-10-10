from __future__ import annotations
import os, re, shutil, subprocess, sys, threading, zipfile
from datetime import datetime
from pathlib import Path
from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
ROOT = Path(os.getenv('PROJECT_ROOT', '/workspace')).resolve()
SCRIPTS = ROOT / 'Scripts'
LOGS = ROOT / 'Logs'
EXPERIMENTS = ROOT / 'Experiments'
MODELS = ROOT / 'Models'
LOCK = threading.Lock()

def safe_name(value):
    name = re.sub(r'[^A-Za-z0-9_-]+', '_', str(value).strip()).strip('_')
    if not name or len(name) > 80:
        raise ValueError('Invalid experiment name')
    return name

def exp_path(name):
    path = (EXPERIMENTS / safe_name(name)).resolve()
    if path.parent != EXPERIMENTS.resolve():
        raise ValueError('Unsafe experiment path')
    return path

def initialise(path):
    # Data only. Scripts and Logs are never copied into experiments.
    for relative in ('Video', 'Images', 'Reports'):
        (path / relative).mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

def list_experiments():
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted((p for p in EXPERIMENTS.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        videos = list((path / 'Video').glob('*.avi')) if (path / 'Video').exists() else []
        items.append({'name': path.name, 'video': videos[0].name if videos else None})
    return items

def run_command(command, working_directory, output):
    output.append('$ ' + ' '.join(map(str, command)))
    # Every script treats the experiment folder as its project root, even if the
    # container itself sets PROJECT_ROOT=/workspace. Logs stay in the shared Logs folder.
    env = {**os.environ,
           'PROJECT_ROOT': str(working_directory),
           'LOG_DIR': str(LOGS),
           'LOGS_DIR': str(LOGS),
           'PYTHONUNBUFFERED': '1'}
    result = subprocess.run(command, cwd=working_directory, env=env, capture_output=True,
                            text=True, errors='replace', timeout=28800)
    text = ((result.stdout or '') + ('\n' + result.stderr if result.stderr else '')).strip()
    if text:
        output.append(text)
    if result.returncode:
        raise RuntimeError(f'Process exited with code {result.returncode}')

@app.get('/')
def index():
    return render_template('index.html')

@app.get('/api/experiments')
def experiments():
    return jsonify(list_experiments())

@app.post('/api/experiments')
def create_experiment():
    path = exp_path((request.get_json(silent=True) or {}).get('name', ''))
    if path.exists():
        return jsonify(ok=False, error='Experiment already exists'), 409
    initialise(path)
    return jsonify(ok=True, name=path.name)

@app.delete('/api/experiments/<name>')
def delete_experiment(name):
    try:
        path = exp_path(name)
    except ValueError as error:
        return jsonify(ok=False, error=str(error)), 400
    if not path.is_dir():
        return jsonify(ok=False, error='Experiment not found'), 404
    # Never delete while a pipeline run is in progress.
    if not LOCK.acquire(False):
        return jsonify(ok=False, error='A pipeline is running - wait for it to finish before deleting'), 409
    try:
        shutil.rmtree(path)
    except OSError as error:
        return jsonify(ok=False, error=f'Could not delete experiment: {error}'), 500
    finally:
        LOCK.release()
    return jsonify(ok=True, name=path.name)

@app.post('/api/experiments/<name>/video')
def upload_video(name):
    path = exp_path(name); initialise(path)
    upload = request.files.get('video')
    if not upload or not upload.filename:
        return jsonify(ok=False, error='Select an AVI video'), 400
    filename = secure_filename(upload.filename)
    if Path(filename).suffix.lower() != '.avi':
        return jsonify(ok=False, error='Only .avi video input is accepted'), 400
    for old in (path / 'Video').glob('*.avi'):
        old.unlink()
    upload.save(path / 'Video' / filename)
    return jsonify(ok=True, filename=filename)

@app.post('/api/experiments/<name>/run')
def run_pipeline(name):
    if not LOCK.acquire(False):
        return jsonify(ok=False, error='Another experiment is running'), 409
    output = []
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = LOGS / f'{stamp}_{safe_name(name)}_pipeline.log'
    try:
        path = exp_path(name); initialise(path)
        videos = list((path / 'Video').glob('*.avi'))
        if len(videos) != 1:
            raise ValueError('Upload exactly one AVI video before running')
        source_video = videos[0]
        options = request.get_json(silent=True) or {}
        use_lama = bool(options.get('lama'))
        use_esrgan = bool(options.get('esrgan'))

        # Regenerate outputs but retain the video and central trace logs.
        for relative in ('Images', 'Reports'):
            target = path / relative
            if target.exists(): shutil.rmtree(target)
            target.mkdir(parents=True)

        temporary_active = None
        if source_video.name != 'active.avi':
            temporary_active = path / 'Video' / 'active.avi'
            shutil.copy2(source_video, temporary_active)
        try:
            run_command([sys.executable, '-u', str(SCRIPTS / '02_Get_Avi_SHA256_Hash.py')], path, output)
        finally:
            if temporary_active: temporary_active.unlink(missing_ok=True)

        video_name = source_video.name
        run_command([sys.executable, '-u', str(SCRIPTS / '03_Get_Video_Metadata.py'), '--video-file-name', video_name], path, output)
        run_command([sys.executable, '-u', str(SCRIPTS / '04_Get_Video_Frames.py'), '--video-file-name', video_name], path, output)
        run_command([sys.executable, '-u', str(SCRIPTS / '05_Run_TinyUNet_Inference.py'),
                     '--model-path', str(MODELS / 'TinyUnet_Model.pt'),
                     '--input-folder', 'Images/02_Frames',
                     '--output-folder', 'Images/03_TinyUNet_Inference_Output'], path, output)
        if use_lama:
            run_command([sys.executable, '-u', str(SCRIPTS / '06_Run_LaMa_Inference.py'),
                         '--model-path', str(MODELS / 'big-lama.pt'), '--stage', 'all'], path, output)
        if use_esrgan:
            source = '06_LaMa_Inference_Output' if use_lama else '03_TinyUNet_Inference_Output'
            run_command([sys.executable, '-u', str(SCRIPTS / '07_Real-ESRGAN_Inference_Output.py'),
                         '--source-folder', source,
                         '--output-folder', '07_Real-ESRGAN_Inference_Output',
                         '--model-path', str(MODELS / 'RealESRGAN_x4plus.pth')], path, output)

        trace = f'Experiment: {name}\nTimestamp: {stamp}\nLaMa: {use_lama}\nReal-ESRGAN: {use_esrgan}\n\n' + '\n\n'.join(output)
        log_file.write_text(trace, encoding='utf-8')
        return jsonify(ok=True, output='\n\n'.join(output)[-50000:], log=str(log_file.relative_to(ROOT)))
    except Exception as error:
        failure = '\n\n'.join(output) + f'\n\nERROR: {error}'
        LOGS.mkdir(parents=True, exist_ok=True)
        failed_log = LOGS / f'{stamp}_{safe_name(name)}_pipeline_FAILED.log'
        failed_log.write_text(failure, encoding='utf-8')
        return jsonify(ok=False, error=str(error), output=failure[-50000:], log=str(failed_log.relative_to(ROOT))), 400
    finally:
        LOCK.release()

# Static pages from the Scripts folder (Digital Twin, Image Gallery).
# Only web file types are served, so .py, model and log files are never exposed.
SCRIPT_WEB_TYPES = {'.html', '.css', '.js', '.json', '.png', '.jpg', '.jpeg', '.svg', '.ico', '.woff', '.woff2'}

@app.get('/scripts/<path:filename>')
def scripts_files(filename):
    if Path(filename).suffix.lower() not in SCRIPT_WEB_TYPES:
        abort(404)
    return send_from_directory(SCRIPTS, filename)

@app.get('/api/experiments/<name>/download')
def download(name):
    path = exp_path(name)
    if not path.is_dir():
        return jsonify(ok=False, error='Experiment not found'), 404
    archive = Path('/tmp') / f'{path.name}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for item in path.rglob('*'):
            if item.is_file(): bundle.write(item, item.relative_to(path.parent))
    return send_file(archive, as_attachment=True, download_name=archive.name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
