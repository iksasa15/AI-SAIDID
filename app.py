from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sklearn.cluster import KMeans
import numpy as np
import random
import math
from typing import List

# إنشاء تطبيق FastAPI
app = FastAPI()

@app.get("/ambulance", response_class=HTMLResponse)
async def ambulance_dashboard():
    with open("ambulance.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# قائمة WebSocket للمسعفين
active_websockets: List[WebSocket] = []

# قاعدة بيانات بسيطة للأحياء مع الإحداثيات
neighborhoods = {
    "الملز": (24.686952, 46.720654),
    "العليا": (24.711666, 46.674444),
    "النسيم": (24.744377, 46.809402),
    "الصحافة": (24.774265, 46.655974),
    "الروضة": (24.760539, 46.784900),
    # يمكن إضافة المزيد من الأحياء هنا
}

# قائمة لتخزين أسماء الأحياء التي تم إرسال بلاغات عنها
reported_neighborhoods = set()

# حساب المسافة الجغرافية باستخدام صيغة Haversine
def haversine_distance(coord1, coord2):
    R = 6371  # نصف قطر الأرض بالكيلومترات
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# توليد مواقع الحوادث مع تصنيف الإصابات عشوائيًا
def generate_random_accidents(city_lat, city_lon, radius, num_accidents):
    accidents = []
    severities = ["critical", "moderate", "minor"]
    for i in range(num_accidents):
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, radius)
        lat = city_lat + (distance / 111) * math.cos(angle)
        lon = city_lon + (distance / (111 * math.cos(math.radians(city_lat)))) * math.sin(angle)
        severity = random.choice(severities)  # تصنيف الإصابة عشوائيًا
        accidents.append({
            "id": i + 1,
            "name": f"Person {i + 1}",
            "phone": f"+96650{random.randint(1000000, 9999999)}",
            "location": [lat, lon],
            "severity": severity
        })
    return accidents

# توليد أسماء عشوائية للمستشفيات
def generate_hospital_names(num_hospitals):
    return [f"Hospital {chr(65 + i)}" for i in range(num_hospitals)]

# تحديد مواقع المستشفيات أو مراكز الإسعاف باستخدام KMeans
def determine_locations(accidents, num_centers):
    accident_locations = [accident["location"] for accident in accidents]
    kmeans = KMeans(n_clusters=num_centers, random_state=42).fit(accident_locations)
    return kmeans.cluster_centers_.tolist()

# إيجاد أقرب مستشفى أو مركز إسعاف لحادث معين
def find_nearest_center(accident, centers, center_names, speed_kmh):
    nearest_center = None
    nearest_name = None
    shortest_distance = float("inf")

    for center, name in zip(centers, center_names):
        distance = haversine_distance(accident["location"], center)
        if distance < shortest_distance:
            shortest_distance = distance
            nearest_center = center
            nearest_name = name

    # حساب وقت الوصول بالدقائق
    speed_km_min = speed_kmh / 60
    time_to_reach = shortest_distance / speed_km_min

    return {
        "center_name": nearest_name,
        "center_location": nearest_center,
        "distance_km": round(shortest_distance, 2),
        "time_minutes": round(time_to_reach, 2)
    }

# مسار لتحليل النص واكتشاف الحي
@app.post("/detect_location")
async def detect_location(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()  # إزالة المسافات الإضافية من النص

    # البحث عن الحي في النص
    detected_neighborhood = None
    for neighborhood in neighborhoods.keys():
        # التحقق من أن اسم الحي موجود كنص كامل وليس كجزء من كلمة
        if neighborhood in text.split():
            detected_neighborhood = neighborhood
            break  # التوقف عند العثور على أول حي

    if detected_neighborhood:
        # التحقق مما إذا تم إرسال بلاغ لهذا الحي من قبل
        if detected_neighborhood in reported_neighborhoods:
            return JSONResponse(content={
                "message": f"تم إرسال البلاغ لهذا الحي بالفعل: {detected_neighborhood}"
            }, status_code=400)

        # إضافة الحي إلى قائمة البلاغات المرسلة
        reported_neighborhoods.add(detected_neighborhood)

        # جلب إحداثيات الحي
        coordinates = neighborhoods[detected_neighborhood]

        # إنشاء حادثة واحدة فقط في هذا الحي
        accident = generate_random_accidents(
            city_lat=coordinates[0],
            city_lon=coordinates[1],
            radius=0.1,  # نصف قطر صغير جدًا لضمان بقاء الحادثة في نطاق الحي
            num_accidents=1
        )[0]

        # إرسال البيانات إلى واجهة المسعف عبر WebSocket
        for websocket in active_websockets:
            await websocket.send_json({
                "type": "new_accidents",
                "data": [accident]
            })

        return JSONResponse(content={
            "location": detected_neighborhood,
            "coordinates": {"lat": coordinates[0], "lon": coordinates[1]},
            "accident": accident
        })
    else:
        return JSONResponse(content={"error": "لم يتم العثور على الحي."}, status_code=404)

# مسار لإنشاء الحوادث
@app.post("/generate_accidents")
async def generate_accidents(request: Request):
    body = await request.json()
    num_accidents = body.get("num_accidents", 10)
    city_center = (24.7136, 46.6753)  # مركز المدينة (الرياض)
    radius = 10  # نصف قطر المنطقة بالكيلومترات
    accidents = generate_random_accidents(city_center[0], city_center[1], radius, num_accidents)

    num_hospitals = max(1, num_accidents // 10)
    num_ambulance_centers = max(1, num_accidents // 5)

    hospital_locations = determine_locations(accidents, num_hospitals)
    ambulance_locations = determine_locations(accidents, num_ambulance_centers)

    hospital_names = generate_hospital_names(num_hospitals)

    for accident in accidents:
        accident["nearest_hospital"] = find_nearest_center(accident, hospital_locations, hospital_names, speed_kmh=80)
        accident["nearest_ambulance"] = find_nearest_center(accident, ambulance_locations, [f"Ambulance Center {i+1}" for i in range(num_ambulance_centers)], speed_kmh=100)

    for websocket in active_websockets:
        await websocket.send_json({"type": "new_accidents", "data": accidents})
        await websocket.send_json({"type": "hospital_data", "hospitals": [{"name": name, "location": location} for name, location in zip(hospital_names, hospital_locations)]})
        await websocket.send_json({"type": "ambulance_data", "ambulance_centers": [{"name": f"Ambulance Center {i+1}", "location": location} for i, location in enumerate(ambulance_locations)]})

    return JSONResponse(content={
        "accidents": sorted(accidents, key=lambda x: x["severity"] == "critical", reverse=True),
        "hospitals": [{"name": name, "location": location} for name, location in zip(hospital_names, hospital_locations)],
        "ambulance_centers": [{"name": f"Ambulance Center {i+1}", "location": location} for i, location in enumerate(ambulance_locations)]
    })

# مسار WebSocket للمسعفين
@app.websocket("/ws/ambulance")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        active_websockets.remove(websocket)

# مسار لعرض صفحة البلاغات الصوتية
@app.get("/voice_report", response_class=HTMLResponse)
async def voice_report_page():
    with open("voice_report.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# مسار لعرض الصفحة الرئيسية
@app.get("/", response_class=HTMLResponse)
async def homepage():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
@app.post("/detect_location")
async def detect_location(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    print(f"Received text: {text}")  # لتتبع النص المستلم
    ...    
@app.post("/send_report")
async def send_report(request: Request):
    body = await request.json()
    location = body.get("location")
    severity = body.get("severity")

    # Generate a single accident at the reported location
    coordinates = neighborhoods.get(location)
    if coordinates:
        accident = generate_random_accidents(
            city_lat=coordinates[0],
            city_lon=coordinates[1],
            radius=0.1,
            num_accidents=1
        )[0]
        accident["severity"] = severity

        # Send the new accident to all connected ambulance dispatchers
        for websocket in active_websockets:
            await websocket.send_json({
                "type": "new_accidents",
                "data": [accident]
            })

        return JSONResponse(content={"message": "Report sent successfully"})
    else:
        return JSONResponse(content={"error": "Invalid location"}, status_code=400)        