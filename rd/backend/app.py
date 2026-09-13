from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from time import perf_counter
import os, tempfile
from track_analyzer import TrackConditionTracker, analyze_image
from model_service import WeatherModel
from safety_engine import (
    VehicleMonitor, TyreMonitor, decide,
    VEHICLE_TYPES, DRIVING_CONDITIONS, TYRE_CONDITIONS,
)
from video_analyzer import analyze_video

app=Flask(__name__)
CORS(app)
tracker=TrackConditionTracker()
model=WeatherModel()

@app.get('/api/health')
def health():
    return jsonify({'status':'ok','model_loaded':model.available,'engine':'trained-resnet18' if model.available else 'heuristic-cv-fallback'})

@app.get('/api/model')
def model_info():
    return jsonify({'loaded':model.available,'device':str(model.device),'classes':model.classes or ['Dry','Damp','Drying','Wet']})

@app.get('/api/options')
def options():
    """Vehicle / tyre / driving-condition enums for building frontend selects."""
    return jsonify({
        'vehicle_types': VEHICLE_TYPES,
        'driving_conditions': DRIVING_CONDITIONS,
        'tyre_conditions': TYRE_CONDITIONS,
    })

@app.post('/api/reset')
def reset():
    session=request.form.get('session','default'); tracker.reset(session); return jsonify({'status':'reset','session':session})

@app.post('/api/analyze')
def analyze():
    if 'image' not in request.files: return jsonify({'error':'missing image field'}),400
    try:
        image=Image.open(request.files['image'].stream).convert('RGB')
    except Exception as e:
        return jsonify({'error':f'invalid image: {e}'}),400
    session=request.form.get('session','default')
    t0=perf_counter()
    model_result=model.predict(image)
    if model_result is None:
        model_result=analyze_image(image)
    road_result=tracker.analyze(image,session,model_result)
    road_result['inference_ms']=round((perf_counter()-t0)*1000,1)

    # --- ROADDOC extension: vehicle + tyre monitoring + safety decision engine ---
    form=request.form
    vehicle=VehicleMonitor(
        vehicle_type=form.get('vehicle_type'),
        speed=form.get('speed'),
        location=form.get('location'),
        driving_condition=form.get('driving_condition'),
    )
    tyre=TyreMonitor(
        pressure=form.get('tyre_pressure'),
        recommended_pressure=form.get('tyre_recommended_pressure'),
        condition=form.get('tyre_condition'),
        temperature=form.get('tyre_temperature'),
    )
    safety=decide(road_result, vehicle, tyre)
    road_result['safety']=safety
    return jsonify(road_result)

@app.post('/api/analyze-video')
def analyze_video_endpoint():
    """Automatic video analysis: road condition + pothole/crack detection +
    speed estimated from the footage itself (optical flow, auto-scaled via
    detected lane markings) + an auto-computed tyre pressure recommendation
    per segment. No manual speed or target-pressure entry needed — only
    vehicle type / driving condition (and, optionally, the tyre's *current*
    reported pressure/condition, since no camera can see actual PSI —
    that still needs a real TPMS sensor if you want it compared against).
    """
    if 'video' not in request.files:
        return jsonify({'error': 'missing video field'}), 400
    video_file = request.files['video']
    form = request.form

    suffix = os.path.splitext(video_file.filename or '')[1] or '.mp4'
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            video_file.save(tmp.name)
            tmp_path = tmp.name
        t0 = perf_counter()
        result = analyze_video(
            tmp_path,
            vehicle_type=form.get('vehicle_type', 'car'),
            driving_condition=form.get('driving_condition', 'normal'),
            tyre_condition=form.get('tyre_condition') or None,
            tyre_pressure=form.get('tyre_pressure') or None,
            tyre_temperature=form.get('tyre_temperature') or None,
        )
        result['processing_ms'] = round((perf_counter() - t0) * 1000, 1)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'video analysis failed: {e}'}), 400
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=False)
