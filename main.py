import os
import io
import json
import base64
from datetime import datetime, timedelta

import numpy as np
import cv2

from flask import Flask, request, jsonify, send_file, send_from_directory, render_template
from flask_cors import CORS

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt,
)
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# -------------------------
# ENV
# -------------------------
load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# -------------------------
# CONFIG
# -------------------------
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=7)

db = SQLAlchemy(app)
jwt = JWTManager(app)

PDF_FOLDER = "pdfs"
os.makedirs(PDF_FOLDER, exist_ok=True)

# ✅ Nuevo sistema (mismo ratio que 600x400)
X_MAX = 60  # eje X 0..60
Y_MAX = 40  # eje Y 0..40

# -------------------------
# MODELS
# -------------------------
class Club(db.Model):
    __tablename__ = "clubs"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    plan = db.Column(db.String(50), default="founder")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    club_id = db.Column(db.Integer, db.ForeignKey("clubs.id"), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(300), nullable=False)
    role = db.Column(db.String(50), default="club_admin")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Green(db.Model):
    __tablename__ = "greens"
    id = db.Column(db.Integer, primary_key=True)
    club_id = db.Column(db.Integer, db.ForeignKey("clubs.id"), nullable=False)
    hole_number = db.Column(db.Integer, nullable=False)
    # puntos guardados en sistema 60x40: [[x,y],...]
    points_json = db.Column(db.Text, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("club_id", "hole_number", name="uq_green_club_hole"),
    )


with app.app_context():
    db.create_all()

# -------------------------
# STATIC ROUTES (para /manifest.json y /sw.js)
# -------------------------
@app.route("/manifest.json")
def manifest():
    return send_from_directory(app.static_folder, "manifest.json")


@app.route("/sw.js")
def service_worker():
    return send_from_directory(app.static_folder, "sw.js")


@app.route("/icon.svg")
def icon_svg():
    return send_from_directory(app.static_folder, "icon.svg")

# -------------------------
# HOME
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------
# AUTH
# -------------------------
@app.route("/auth/bootstrap", methods=["POST"])
def bootstrap():
    """
    Crea 1 club + 1 usuario admin SOLO si la BD está vacía.
    Ideal para demo rápida.
    """
    if User.query.first() is not None:
        return jsonify({"ok": False, "error": "Bootstrap already done"}), 400

    data = request.json or {}
    club_name = (data.get("club_name") or "Club Demo").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or len(password) < 6:
        return jsonify({"ok": False, "error": "Email y password (>=6) requeridos"}), 400

    club = Club(name=club_name, plan="founder")
    db.session.add(club)
    db.session.flush()

    user = User(
        club_id=club.id,
        email=email,
        password_hash=generate_password_hash(password),
        role="club_admin",
    )
    db.session.add(user)
    db.session.commit()

    return jsonify({"ok": True, "club_id": club.id, "email": user.email})


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = User.query.filter_by(email=email).first()
    if user is None or not check_password_hash(user.password_hash, password):
        return jsonify({"ok": False, "error": "Credenciales inválidas"}), 401

    claims = {"club_id": user.club_id, "role": user.role, "email": user.email}
    token = create_access_token(identity=str(user.id), additional_claims=claims)

    return jsonify({"ok": True, "access_token": token, "club_id": user.club_id, "role": user.role})


def _club_id_from_jwt() -> int:
    claims = get_jwt()
    return int(claims.get("club_id"))

# -------------------------
# UTILIDADES
# -------------------------
def decode_base64_image(data: str):
    """base64 -> OpenCV BGR"""
    if not data:
        return None
    if "," in data:
        data = data.split(",")[1]
    img_bytes = base64.b64decode(data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def normalize_points(points):
    """Acepta [{x,y}] o [[x,y]] -> [[float,float]]"""
    normalized = []
    for p in points or []:
        if isinstance(p, dict):
            normalized.append([float(p.get("x", 0)), float(p.get("y", 0))])
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            normalized.append([float(p[0]), float(p[1])])
    return normalized

# -------------------------
# DETECCIÓN GREEN
# -------------------------
def detect_green_contour(img_bgr):
    """Detecta contorno verde principal"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([30, 40, 40])
    upper = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    if cv2.contourArea(contour) > img_area * 0.90:
        return None

    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    pts = approx.reshape(-1, 2)
    return pts.tolist()


@app.route("/detect_green", methods=["POST"])
@jwt_required()
def detect_green():
    data = request.json or {}
    img = decode_base64_image(data.get("image"))
    if img is None:
        return jsonify({"points": []})

    target_w = int(data.get("target_w", 600))
    target_h = int(data.get("target_h", 400))

    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    pts = detect_green_contour(img_resized)
    if not pts:
        return jsonify({"points": []})

    return jsonify({"points": [{"x": int(p[0]), "y": int(p[1])} for p in pts]})

# -------------------------
# GREENS (persistencia por club/hoyo) en sistema 60x40
# -------------------------
@app.route("/greens/<int:hole_number>", methods=["GET"])
@jwt_required()
def get_green(hole_number):
    club_id = _club_id_from_jwt()
    g = Green.query.filter_by(club_id=club_id, hole_number=hole_number).first()
    if not g:
        return jsonify({"ok": True, "points": []})
    return jsonify({"ok": True, "points": json.loads(g.points_json)})


@app.route("/greens/<int:hole_number>", methods=["PUT"])
@jwt_required()
def put_green(hole_number):
    club_id = _club_id_from_jwt()
    data = request.json or {}
    points = normalize_points(data.get("points", []))

    if len(points) < 3:
        return jsonify({"ok": False, "error": "Se requieren >=3 puntos"}), 400

    # clamp a rangos 60x40 por seguridad
    clamped = []
    for x, y in points:
        x = max(0.0, min(float(X_MAX), float(x)))
        y = max(0.0, min(float(Y_MAX), float(y)))
        clamped.append([x, y])

    existing = Green.query.filter_by(club_id=club_id, hole_number=hole_number).first()
    if existing:
        existing.points_json = json.dumps(clamped)
        existing.updated_at = datetime.utcnow()
    else:
        existing = Green(
            club_id=club_id,
            hole_number=hole_number,
            points_json=json.dumps(clamped)
        )
        db.session.add(existing)

    db.session.commit()
    return jsonify({"ok": True})

# -------------------------
# PDF GRID: 60x40
# -------------------------
def draw_grid_axes(c, x0, y0, w, h, scale):
    """
    Dibuja grid tipo pin sheet para sistema:
    X: 0..60 (arriba)
    Y: 0..40 (izquierda)
    """
    # verticales (X)
    for vx in range(0, X_MAX + 1, 5):
        x = x0 + vx * scale
        if vx % 10 == 0:
            c.setStrokeColor(colors.lightgrey)
            c.setLineWidth(0.7)
        else:
            c.setStrokeColor(colors.whitesmoke)
            c.setLineWidth(0.4)
        c.line(x, y0, x, y0 + h)

    # horizontales (Y)
    for vy in range(0, Y_MAX + 1, 5):
        y = y0 + vy * scale
        if vy % 10 == 0:
            c.setStrokeColor(colors.lightgrey)
            c.setLineWidth(0.7)
        else:
            c.setStrokeColor(colors.whitesmoke)
            c.setLineWidth(0.4)
        c.line(x0, y, x0 + w, y)

    # marco
    c.setStrokeColor(colors.grey)
    c.setLineWidth(0.8)
    c.rect(x0, y0, w, h)

    # labels
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.grey)

    # X arriba
    for vx in range(0, X_MAX + 1, 10):
        x = x0 + vx * scale
        c.setStrokeColor(colors.grey)
        c.setLineWidth(0.6)
        c.line(x, y0 + h, x, y0 + h - 4)
        c.drawCentredString(x, y0 + h + 2, str(vx))

    # Y izquierda
    for vy in range(0, Y_MAX + 1, 10):
        y = y0 + vy * scale
        c.setStrokeColor(colors.grey)
        c.setLineWidth(0.6)
        c.line(x0, y, x0 + 4, y)
        c.drawRightString(x0 - 2, y - 2, str(vy))

# -------------------------
# PIN PGA (adaptado a 60x40)
# vertical: 0..40 desde borde inferior real hacia arriba
# horizontal: 0..60 desde borde izq/der real sobre la línea yLine
# -------------------------
def compute_pin_pga(points_60_40, vertical, horizontal, side):
    pts = np.array(points_60_40, dtype=np.float32)
    if len(pts) < 3:
        return None

    vertical = float(max(0, min(Y_MAX, vertical)))
    horizontal = float(max(0, min(X_MAX, horizontal)))
    side = side if side in ("left", "right") else "left"

    bottom_y = float(np.min(pts[:, 1]))
    top_y = float(np.max(pts[:, 1]))

    yLine = bottom_y + vertical
    if yLine > top_y:
        yLine = top_y

    intersections = []
    n = len(pts)

    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]

        if (y1 <= yLine <= y2) or (y2 <= yLine <= y1):
            if abs(y2 - y1) > 1e-9:
                t = (yLine - y1) / (y2 - y1)
                x_int = x1 + t * (x2 - x1)
                intersections.append(float(x_int))

    if len(intersections) < 2:
        cx = float(np.mean(pts[:, 0]))
        return {
            "pin": (cx, yLine),
            "leftEdge": cx,
            "rightEdge": cx,
            "yLine": yLine,
            "bottom": bottom_y
        }

    intersections.sort()
    leftEdge = intersections[0]
    rightEdge = intersections[-1]
    seg_w = rightEdge - leftEdge

    if seg_w <= 1e-6:
        cx = (leftEdge + rightEdge) / 2
        return {
            "pin": (cx, yLine),
            "leftEdge": leftEdge,
            "rightEdge": rightEdge,
            "yLine": yLine,
            "bottom": bottom_y
        }

    if side == "left":
        pin_x = leftEdge + horizontal
    else:
        pin_x = rightEdge - horizontal

    pin_x = max(leftEdge, min(rightEdge, pin_x))
    return {
        "pin": (pin_x, yLine),
        "leftEdge": leftEdge,
        "rightEdge": rightEdge,
        "yLine": yLine,
        "bottom": bottom_y
    }

# -------------------------
# GENERAR PDF (JWT + persistencia)
# puntos vienen en sistema 60x40
# -------------------------
@app.route("/generate_pdf", methods=["POST"])
@jwt_required()
def generate_pdf():
    club_id = _club_id_from_jwt()

    data = request.json or {}
    greens = data.get("greens", [])
    club_name = data.get("club", "Club")
    campo = data.get("campo", "Campo")
    layout = int(data.get("layout", 1))
    fecha = data.get("fecha") or datetime.now().strftime("%d/%m/%Y")

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.black)
    c.drawString(40, height - 35, f"Reporte Green — {club_name}")

    c.setFont("Helvetica", 11)
    c.setFillColor(colors.grey)
    c.drawString(40, height - 52, campo)

    c.setFont("Helvetica", 10)
    c.drawRightString(width - 40, height - 35, f"{len(greens)} hoyos")
    c.drawRightString(width - 40, height - 52, fecha)

    c.setStrokeColor(colors.lightgrey)
    c.setLineWidth(0.6)
    c.line(30, height - 60, width - 30, height - 60)

    layout_map = {1: (1, 1), 3: (1, 3), 6: (2, 3), 9: (3, 3), 18: (3, 6)}
    cols, rows = layout_map.get(layout, (1, 1))

    box_w = width / cols
    box_h = (height - 85) / rows

    if layout == 18:
        green_stroke = 0.8
        pin_radius = 2.6
    elif layout >= 9:
        green_stroke = 1.0
        pin_radius = 3.0
    else:
        green_stroke = 1.2
        pin_radius = 3.8

    for index, g in enumerate(greens):
        if index > 0 and index % layout == 0:
            c.showPage()

        page_index = index % layout
        col = page_index % cols
        row = page_index // cols

        x_box = col * box_w
        y_box = height - 70 - (row + 1) * box_h

        margin = 14 if layout >= 9 else 18
        inner_x = x_box + margin
        inner_y = y_box + margin
        inner_w = box_w - margin * 2
        inner_h = box_h - margin * 2

        c.setStrokeColor(colors.lightgrey)
        c.setLineWidth(0.6)
        c.rect(inner_x, inner_y, inner_w, inner_h)

        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(colors.black)

        hole_number = int(g.get("hole_number") or (index + 1))
        c.drawString(inner_x + 6, inner_y + inner_h - 16, f"Hoyo {hole_number}")

        points = normalize_points(g.get("points", []))

        # Si no vienen puntos, intenta cargar desde BD
        if len(points) < 3:
            saved = Green.query.filter_by(club_id=club_id, hole_number=hole_number).first()
            if saved:
                points = json.loads(saved.points_json)

        # Guardar/actualizar si vienen puntos válidos
        if len(points) >= 3:
            clamped = []
            for x, y in points:
                x = max(0.0, min(float(X_MAX), float(x)))
                y = max(0.0, min(float(Y_MAX), float(y)))
                clamped.append([x, y])
            points = clamped

            saved = Green.query.filter_by(club_id=club_id, hole_number=hole_number).first()
            if saved:
                saved.points_json = json.dumps(points)
                saved.updated_at = datetime.utcnow()
            else:
                db.session.add(
                    Green(club_id=club_id, hole_number=hole_number, points_json=json.dumps(points))
                )
            

        if len(points) < 3:
            continue

        vertical = float(g.get("vertical", 0))
        horizontal = float(g.get("horizontal", 0))
        side = g.get("side", "left")

        # Área de plot
        plot_top_pad = 22
        plot_x = inner_x + 10
        plot_y = inner_y + 18
        plot_w = inner_w - 20
        plot_h = inner_h - (plot_top_pad + 30)

        # escala consistente con 60x40 (ratio 3:2)
        scale = min(plot_w / float(X_MAX), plot_h / float(Y_MAX))
        plot_w = X_MAX * scale
        plot_h = Y_MAX * scale

        plot_x = inner_x + (inner_w - plot_w) / 2
        plot_y = inner_y + (inner_h - plot_h) / 2 - 8

        draw_grid_axes(c, plot_x, plot_y, plot_w, plot_h, scale)

        # dibujar green (relleno + borde)
        scaled = [(plot_x + p[0] * scale, plot_y + p[1] * scale) for p in points]

        c.setStrokeColor(colors.Color(0.20, 0.45, 0.20))
        c.setFillColor(colors.Color(0.86, 0.93, 0.86))
        c.setLineWidth(green_stroke)

        path = c.beginPath()
        path.moveTo(scaled[0][0], scaled[0][1])
        for x, y in scaled[1:]:
            path.lineTo(x, y)
        path.close()
        c.drawPath(path, stroke=1, fill=1)

        # Calcular pin
        pin_info = compute_pin_pga(points, vertical, horizontal, side)
        if pin_info is None:
            continue

        (pin_x, pin_y) = pin_info["pin"]
        leftEdge = pin_info["leftEdge"]
        rightEdge = pin_info["rightEdge"]
        yLine = pin_info["yLine"]
        bottom_y = pin_info["bottom"]

        # ---- Ancho disponible en esa altura ----
        availableWidth = rightEdge - leftEdge
        leftSpace = pin_x - leftEdge
        rightSpace = rightEdge - pin_x

        # ---- Texto ancho disponible ----
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.black)

        c.drawString(
            plot_x,
            plot_y + plot_h + 10,
            f"Ancho disponible a {vertical:.1f} del fondo: {availableWidth:.1f}   |   Izq: {leftSpace:.1f}   Der: {rightSpace:.1f}"
        )

        # Línea base inferior (desde borde inferior real)
        c.setStrokeColor(colors.Color(0.72, 0.55, 0.08))
        c.setLineWidth(1.2 if layout < 18 else 0.9)
        c.line(
            plot_x,
            plot_y + bottom_y * scale,
            plot_x + plot_w,
            plot_y + bottom_y * scale
        )

        gold = colors.Color(0.72, 0.55, 0.08)

        # Línea horizontal pin
        c.setStrokeColor(gold)
        c.setLineWidth(1.2 if layout < 18 else 0.9)
        c.line(plot_x, plot_y + yLine * scale, plot_x + plot_w, plot_y + yLine * scale)

        # ---- Marcas de borde izquierdo/derecho en la altura yLine ----
        c.setStrokeColor(colors.Color(0.85, 0.15, 0.15))  # rojo suave
        c.setLineWidth(1)

        tick = 5  # alto del "tick" en puntos PDF (visual)
        xL = plot_x + leftEdge * scale
        xR = plot_x + rightEdge * scale
        yY = plot_y + yLine * scale

        # tick izquierdo
        c.line(xL, yY - tick, xL, yY + tick)

        # tick derecho
        c.line(xR, yY - tick, xR, yY + tick)

        # ---- Línea de ancho disponible (entre bordes) ----
        c.setStrokeColor(colors.Color(0.15, 0.25, 0.85))  # azul
        c.setLineWidth(1)
        c.line(xL, yY, xR, yY)

        # mini flechas (triangulitos simples)
        arrow = 4
        # izquierda
        c.line(xL, yY, xL + arrow, yY + arrow)
        c.line(xL, yY, xL + arrow, yY - arrow)
        # derecha
        c.line(xR, yY, xR - arrow, yY + arrow)
        c.line(xR, yY, xR - arrow, yY - arrow)

        # Segmento útil
        c.setStrokeColor(colors.Color(0.20, 0.55, 0.85))
        c.setLineWidth(1.0 if layout < 18 else 0.7)
        c.line(
            plot_x + leftEdge * scale, plot_y + yLine * scale,
            plot_x + rightEdge * scale, plot_y + yLine * scale
        )

        # Línea vertical guía
        c.setStrokeColor(gold)
        c.setLineWidth(1.2 if layout < 18 else 0.9)
        c.line(plot_x + pin_x * scale, plot_y, plot_x + pin_x * scale, plot_y + plot_h)

        # Pin
        cx = plot_x + pin_x * scale
        cy = plot_y + pin_y * scale

        c.setFillColor(colors.white)
        c.circle(cx, cy, pin_radius + 1.2, fill=1, stroke=0)
        c.setFillColor(colors.red)
        c.circle(cx, cy, pin_radius, fill=1, stroke=0)

        # Texto mediciones
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.black)
        lado_txt = "izquierda" if side == "left" else "derecha"
        c.drawRightString(
            inner_x + inner_w - 6,
            inner_y + inner_h - 16,
            f"{int(vertical)} fondo | {int(horizontal)} {lado_txt}"
        )

        # Escala texto abajo
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.grey)
        c.drawCentredString(inner_x + inner_w / 2, inner_y + 6, f"Escala: {X_MAX} × {Y_MAX}")

    db.session.commit()

    c.save()
    buffer.seek(0)

    resp = send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=False,
        download_name="green_report.pdf"
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp

# -------------------------
# SERVIR PDF (si lo usas)
# -------------------------
@app.route("/pdfs/<filename>")
def serve_pdf(filename):
    return send_from_directory(PDF_FOLDER, filename)

# -------------------------
# RUN (solo local)
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
