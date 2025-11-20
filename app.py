import datetime
import io
import json
import mysql.connector
from mysql.connector import pooling
import functools
import time
import hashlib
import boto3
from botocore.client import Config
from flask import Flask, request, jsonify, render_template, url_for, redirect, flash, g, Response
from PIL import Image, ImageDraw, ExifTags
import requests
import os
import uuid
from werkzeug.middleware.proxy_fix import ProxyFix
from celery import Celery
from celery.result import AsyncResult
from celery.schedules import crontab
from itertools import cycle
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
import secrets
from datetime import timedelta, date
from collections import defaultdict
import redis
from astral.moon import phase as moon_phase_calc
from itsdangerous import URLSafeTimedSerializer
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

log = logging.getLogger(__name__)

# ###--- Flask App Initialization & Config ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
if not app.config['SECRET_KEY']:
    raise ValueError("No SECRET_KEY set for Flask application. Please set it in your .env file.")

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=os.environ.get('REDIS_URL'),
    storage_options={"socket_connect_timeout": 30}
)

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    app.logger.info('GameCam AI app is starting...')

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.config['SERVER_NAME'] = os.environ.get('SERVER_NAME', 'localhost')

# --- Global Config ---
MAX_CAMERA_LIMIT = 5  # Hardcoded limit for all users

REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
S3_ENDPOINT_URL = os.environ.get('S3_ENDPOINT_URL')
S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY')
S3_SECRET_KEY = os.environ.get('S3_SECRET_KEY')
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')

db_config = {
    'host': os.environ.get('MYSQL_HOST', 'mariadb'),
    'user': os.environ.get('MYSQL_USER'),
    'password': os.environ.get('MYSQL_PASSWORD'),
    'database': os.environ.get('MYSQL_DATABASE'),
    'port': int(os.environ.get('MYSQL_PORT', 3306)),
    'pool_name': 'gamecam_pool',
    'pool_size': 10,
    'autocommit': True
}

CODEPROJECT_HOSTS_STR = os.environ.get('CODEPROJECT_HOSTS',
                                       'http://localhost:32168,http://localhost:32169,http://localhost:32170')
CODEPROJECT_HOSTS = [host.strip() for host in CODEPROJECT_HOSTS_STR.split(',')]
CODEPROJECT_API_URL_TEMPLATE = "/v1/vision/custom/best"
codeproject_host_cycler = cycle(CODEPROJECT_HOSTS)
MIN_CONFIDENCE = 0.4
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY')
TRAINING_DATA_PATH = "/app/training_data"
ANIMAL_CLASS_MAP = {
    'deer': 0, 'hog': 1, 'pig': 2, 'boar': 3, 'raccoon': 4, 'coyote': 5,
    'person': 6, 'cat': 7, 'dog': 8, 'bear': 9, 'fox': 10, 'bobcat': 11, 'elk': 12,
    'turkey': 13, 'mountain_lion': 14, 'badger': 15, 'porcupine': 16, 'opossum': 17,
}
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# --- Database Connection Pool (Fork-Safe) ---
_db_pool = None
_db_pool_pid = 0


def get_db_pool():
    global _db_pool, _db_pool_pid
    current_pid = os.getpid()

    if _db_pool is None or _db_pool_pid != current_pid:
        try:
            log.info(f"Process {current_pid}: Creating new DB connection pool.")
            _db_pool = mysql.connector.pooling.MySQLConnectionPool(**db_config)
            _db_pool_pid = current_pid
            log.info(f"Process {current_pid}: DB pool created.")
        except mysql.connector.Error as err:
            log.error(f"Process {current_pid}: Error creating connection pool: {err}")
            return None

    return _db_pool


# --- Email, S3, Redis, Celery, User Login Config & Schema ---
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
app.config['MAIL_USE_SSL'] = os.environ.get('MAIL_USE_SSL', 'false').lower() in ['true', 'on', '1']
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', os.environ.get('MAIL_USERNAME'))
mail = Mail(app)

if S3_ENDPOINT_URL:
    s3_client = boto3.client('s3', endpoint_url=S3_ENDPOINT_URL, aws_access_key_id=S3_ACCESS_KEY,
                             aws_secret_access_key=S3_SECRET_KEY, config=Config(signature_version='s3v4'))
redis_client = redis.from_url(REDIS_URL)

celery = Celery(app.name)
app.config.from_mapping(CELERY=dict(
    broker_url=REDIS_URL,
    result_backend=REDIS_URL,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_heartbeat=30,
    broker_transport_options={
        'visibility_timeout': 43200,
        'socket_keepalive': True,
        'fanout_prefix': True,
        'socket_timeout': 30
    }
))
celery.config_from_object(app.config["CELERY"])

celery.conf.beat_schedule = {
    'cleanup-images-every-day': {
        'task': 'app.cleanup_old_images_task',
        'schedule': crontab(hour=2, minute=0),
    },
}

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin):
    def __init__(self, id, username, email, confirmed=False, role='user'):
        self.id = id
        self.username = username
        self.email = email
        self.confirmed = confirmed
        self.role = role


@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    user_data = None
    try:
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user_data = cursor.fetchone()
    finally:
        cursor.close()

    if user_data:
        return User(
            id=user_data['id'],
            username=user_data['username'],
            email=user_data['email'],
            confirmed=user_data.get('confirmed'),
            role=user_data.get('role', 'user')
        )
    return None


def get_db_connection():
    if 'db' not in g:
        pool = get_db_pool()
        if not pool:
            raise mysql.connector.Error("Database pool is not available.")
        g.db = pool.get_connection()
    return g.db


@app.teardown_appcontext
def teardown_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def initialize_database():
    pool = get_db_pool()
    if not pool:
        log.error("Could not get DB pool for initialization.")
        return
    conn = pool.get_connection()
    cursor = conn.cursor()
    try:
        log.info("Initializing database schema...")
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS users
                       (
                           id
                           INT
                           AUTO_INCREMENT
                           PRIMARY
                           KEY,
                           username
                           VARCHAR
                       (
                           150
                       ) UNIQUE NOT NULL,
                           email VARCHAR
                       (
                           150
                       ) UNIQUE NOT NULL,
                           password_hash VARCHAR
                       (
                           255
                       ) NOT NULL,
                           reset_token VARCHAR
                       (
                           100
                       ),
                           reset_token_expiration DATETIME,
                           confirmed BOOLEAN DEFAULT FALSE,
                           confirmed_on DATETIME,
                           role VARCHAR
                       (
                           50
                       ) NOT NULL DEFAULT 'user'
                           )
                       ''')
        # Safe alters to ensure existing DBs work with new code
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN confirmed BOOLEAN DEFAULT FALSE")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN confirmed_on DATETIME")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN role VARCHAR(50) NOT NULL DEFAULT 'user'")
        except mysql.connector.Error as err:
            if err.errno != 1060: raise
            pass

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS cameras
                       (
                           id
                           INT
                           AUTO_INCREMENT
                           PRIMARY
                           KEY,
                           name
                           VARCHAR
                       (
                           150
                       ) NOT NULL,
                           location_zip VARCHAR
                       (
                           10
                       ) NOT NULL,
                           user_id INT,
                           is_active TINYINT
                       (
                           1
                       ) DEFAULT 1,
                           FOREIGN KEY
                       (
                           user_id
                       ) REFERENCES users
                       (
                           id
                       ) ON DELETE CASCADE
                           )
                       ''')
        try:
            cursor.execute("ALTER TABLE cameras ADD COLUMN is_active TINYINT(1) DEFAULT 1")
        except mysql.connector.Error as err:
            if err.errno != 1060: raise
            pass

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS sightings
                       (
                           id
                           INT
                           AUTO_INCREMENT
                           PRIMARY
                           KEY,
                           timestamp
                           DATETIME
                           NOT
                           NULL,
                           image_path
                           VARCHAR
                       (
                           255
                       ),
                           original_image_path VARCHAR
                       (
                           255
                       ),
                           thumbnail_path VARCHAR
                       (
                           255
                       ),
                           file_hash VARCHAR
                       (
                           64
                       ),
                           user_id INT,
                           camera_id INT,
                           weather_data JSON,
                           retained BOOLEAN DEFAULT FALSE,
                           feedback_status VARCHAR
                       (
                           20
                       ) DEFAULT 'unverified',
                           uploaded_at DATETIME,
                           predictions JSON,
                           FOREIGN KEY
                       (
                           user_id
                       ) REFERENCES users
                       (
                           id
                       ) ON DELETE CASCADE,
                           FOREIGN KEY
                       (
                           camera_id
                       ) REFERENCES cameras
                       (
                           id
                       )
                         ON DELETE SET NULL
                           )
                       ''')
        # Add missing columns safely if table existed
        for col_sql in [
            "ALTER TABLE sightings ADD COLUMN uploaded_at DATETIME",
            "ALTER TABLE sightings ADD COLUMN thumbnail_path VARCHAR(255)",
            "ALTER TABLE sightings ADD COLUMN predictions JSON"
        ]:
            try:
                cursor.execute(col_sql)
            except:
                pass

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS animals
                       (
                           id
                           INT
                           AUTO_INCREMENT
                           PRIMARY
                           KEY,
                           name
                           VARCHAR
                       (
                           50
                       ) UNIQUE NOT NULL
                           )
                       ''')
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS sighting_animals
                       (
                           sighting_id
                           INT,
                           animal_id
                           INT,
                           PRIMARY
                           KEY
                       (
                           sighting_id,
                           animal_id
                       ),
                           FOREIGN KEY
                       (
                           sighting_id
                       ) REFERENCES sightings
                       (
                           id
                       ) ON DELETE CASCADE,
                           FOREIGN KEY
                       (
                           animal_id
                       ) REFERENCES animals
                       (
                           id
                       )
                         ON DELETE CASCADE
                           )
                       ''')
        try:
            cursor.execute("CREATE UNIQUE INDEX idx_user_file_hash ON sightings(user_id, file_hash)")
        except mysql.connector.Error as err:
            if err.errno != 1061: raise

        log.info("Checking for and creating database indexes...")
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sightings_retained ON sightings(retained)")
        except:
            pass

        try:
            cursor.execute("DROP INDEX idx_sightings_user_camera_time ON sightings")
        except:
            pass

        try:
            cursor.execute(
                "CREATE INDEX idx_sightings_user_camera_time ON sightings(user_id, camera_id, timestamp ASC)")
        except:
            pass

        conn.commit()
        log.info("Database schema initialized.")
    finally:
        cursor.close()
        conn.close()


@app.cli.command('init-db')
def init_db_command():
    initialize_database()
    log.info('Database schema has been initialized.')


# --- Helper Functions ---
def send_confirmation_email(user_email):
    token = s.dumps(user_email, salt='email-confirm-salt')
    confirm_url = url_for('confirm_email', token=token, _external=True)
    msg = Message('Confirm Your Email Address', recipients=[user_email],
                  sender=('GameCam AI', app.config['MAIL_DEFAULT_SENDER']))
    msg.body = f'Thanks for signing up for GameCam AI! Please click the link to confirm your email address:\n\n{confirm_url}\n\nIf you did not make this request, please ignore this email.'
    mail.send(msg)


def send_welcome_email(user):
    msg = Message('Welcome to GameCam AI!', recipients=[user.email],
                  sender=('GameCam AI', app.config['MAIL_DEFAULT_SENDER']))
    msg.body = f"Hi {user.username},\n\nWelcome to GameCam AI! We're excited to have you on board."
    mail.send(msg)


def send_password_change_notification(user):
    msg = Message('Your GameCam AI Password Has Been Changed', recipients=[user.email],
                  sender=('GameCam AI', app.config['MAIL_DEFAULT_SENDER']))
    msg.body = f"Hi {user.username},\n\nThis is a confirmation that the password for your account has been changed."
    mail.send(msg)


def send_email_change_notification(user, old_email):
    msg = Message('Your GameCam AI Email Address Has Been Updated', recipients=[user.email],
                  sender=('GameCam AI', app.config['MAIL_DEFAULT_SENDER']))
    msg.body = f"Hi {user.username},\n\nThis confirms your email has been updated to {user.email}."
    mail.send(msg)
    if old_email:
        msg_old = Message('Security Alert: Email Change for Your Account', recipients=[old_email],
                          sender=('GameCam AI', app.config['MAIL_DEFAULT_SENDER']))
        msg_old.body = f"Hi,\n\nThis is an alert that the email for your account was changed to {user.email}."
        mail.send(msg_old)


def send_reset_email(user, token):
    msg = Message('Password Reset Request', recipients=[user.email],
                  sender=('GameCam AI', app.config['MAIL_DEFAULT_SENDER']))
    reset_url = url_for('reset_password', token=token, _external=True)
    msg.body = f'To reset your password, visit the following link:\n{reset_url}'
    mail.send(msg)


def get_image_timestamp(image_file):
    try:
        img = Image.open(image_file)
        exif_data = img._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == 'DateTimeOriginal':
                    return datetime.datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
    except Exception:
        pass
    return datetime.datetime.now()


def analyze_image_with_codeproject(image_bytes, server_url):
    try:
        api_url = f"{server_url}{CODEPROJECT_API_URL_TEMPLATE}"
        response = requests.post(
            api_url, files={"image": image_bytes},
            data={"min_confidence": MIN_CONFIDENCE}, timeout=60
        )
        response.raise_for_status()
        response_data = response.json()
        if not response_data.get("success"):
            return {"status": "error", "message": response_data.get("error", "Unknown AI error")}
        animals_found = []
        predictions_to_draw = []
        target_animals = {'deer', 'hog', 'pig', 'boar', 'raccoon', 'coyote', 'person', 'cat', 'dog', 'bear', 'fox',
                          'bobcat', 'elk'}
        for prediction in response_data.get("predictions", []):
            label = prediction.get("label", "").lower()
            if label in target_animals:
                animal_type = 'hog' if label in ['hog', 'pig', 'boar'] else label
                animals_found.append(animal_type)
                predictions_to_draw.append(prediction)
        result = {"status": "success", "data": list(set(animals_found)), "image_bytes": None,
                  "predictions": predictions_to_draw}
        if predictions_to_draw:
            img = Image.open(io.BytesIO(image_bytes))
            draw = ImageDraw.Draw(img)
            for pred in predictions_to_draw:
                top, left, bottom, right = pred['y_min'], pred['x_min'], pred['y_max'], pred['x_max']
                draw.rectangle([left, top, right, bottom], outline="red", width=5)
            img_buffer = io.BytesIO()
            img.save(img_buffer, format=img.format or 'JPEG')
            result["image_bytes"] = img_buffer.getvalue()
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_lat_lon_from_zip(zip_code):
    cache_key = f"zip::{zip_code}"
    cached_coords = redis_client.get(cache_key)
    if cached_coords:
        lat, lon = json.loads(cached_coords)
        return lat, lon
    url = f"https://api.zippopotam.us/us/{zip_code}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        place = data['places'][0]
        lat = float(place['latitude'])
        lon = float(place['longitude'])
        if lat and lon:
            redis_client.set(cache_key, json.dumps([lat, lon]), ex=86400)
        return lat, lon
    except (requests.RequestException, KeyError, IndexError) as e:
        log.warning(f"Error geocoding ZIP {zip_code} with Zippopotam: {e}")
        return None, None


def get_historical_weather(lat, lon, timestamp, api_key):
    if not lat or not lon: return None
    rounded_minute = (timestamp.minute // 30) * 30
    rounded_timestamp = timestamp.replace(minute=rounded_minute, second=0, microsecond=0)
    cache_key = f"weather::{lat}:{lon}:{rounded_timestamp.isoformat()}"
    cached_weather = redis_client.get(cache_key)
    if cached_weather: return json.loads(cached_weather)
    unix_timestamp = int(time.mktime(timestamp.timetuple()))
    url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lon}&dt={unix_timestamp}&appid={api_key}&units=imperial"
    weather_data = {}
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        weather_data = data.get('data', [{}])[0]
    except requests.RequestException as e:
        log.warning(f"Error getting historical weather: {e}")
    try:
        moon_p = moon_phase_calc(timestamp.date()) / 28.0
    except Exception as e:
        log.warning(f"Error calculating moon phase: {e}")
        moon_p = None
    result = {
        "temperature": weather_data.get('temp'), "pressure": weather_data.get('pressure'),
        "wind_speed": weather_data.get('wind_speed'), "wind_direction": weather_data.get('wind_deg'),
        "weather_condition": weather_data.get('weather', [{}])[0].get('main'), "moon_phase": moon_p
    }
    redis_client.set(cache_key, json.dumps(result), ex=3600)
    return result


# --- Celery Tasks ---
@celery.task(bind=True, max_retries=3, default_retry_delay=60)
def process_image_task(self, file_path, original_filename, user_id, camera_id, metadata={}):
    logger = logging.getLogger(__name__)
    with app.app_context():
        try:
            if not os.path.exists(file_path):
                logger.warning(f"[PROCESS] File not found: {original_filename}, file_path={file_path}")
                return

            logger.debug(
                f"[PROCESS] Starting {original_filename}, file_path={file_path}, exists={os.path.exists(file_path)}")
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            file_hash = hashlib.sha256(image_bytes).hexdigest()

            pool = get_db_pool()
            if not pool:
                raise Exception("Worker process could not get DB pool.")
            conn_check = pool.get_connection()
            cursor_check = conn_check.cursor(dictionary=True)
            try:
                cursor_check.execute("SELECT id FROM sightings WHERE user_id = %s AND file_hash = %s",
                                     (user_id, file_hash))
                if cursor_check.fetchone():
                    logger.warning(f"Duplicate file detected in worker, skipping: {original_filename}...")
                    return {'status': 'duplicate'}
            finally:
                cursor_check.close()
                conn_check.close()

            timestamp = None
            if metadata and metadata.get('original_timestamp'):
                try:
                    timestamp = datetime.datetime.fromisoformat(metadata['original_timestamp'])
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse timestamp from metadata: {metadata.get('original_timestamp')}")

            if not timestamp:
                timestamp = get_image_timestamp(io.BytesIO(image_bytes))

            codeproject_server_url = next(codeproject_host_cycler)
            logger.debug(f"Starting AI analysis for: {original_filename}")
            analysis_result = analyze_image_with_codeproject(image_bytes, codeproject_server_url)
            logger.debug(f"Finished AI analysis for: {original_filename}")

            if analysis_result["status"] == "error":
                raise Exception(analysis_result['message'])

            animals_found = analysis_result["data"]
            annotated_image_bytes = analysis_result.get("image_bytes")
            predictions_json = json.dumps(analysis_result.get("predictions", []))

            base_filename = f"{uuid.uuid4()}{os.path.splitext(original_filename)[1] or '.jpg'}"
            original_image_path = f"originals/{base_filename}"
            s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=original_image_path, Body=io.BytesIO(image_bytes),
                                 ContentType='image/jpeg')

            display_image_path = original_image_path
            if annotated_image_bytes:
                annotated_image_path = f"annotated/{base_filename}"
                s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=annotated_image_path,
                                     Body=io.BytesIO(annotated_image_bytes), ContentType='image/jpeg')
                display_image_path = annotated_image_path

            thumbnail_path = None
            image_for_thumb_bytes = annotated_image_bytes if annotated_image_bytes else image_bytes
            if image_for_thumb_bytes:
                try:
                    img = Image.open(io.BytesIO(image_for_thumb_bytes))
                    img.thumbnail((480, 480))
                    thumb_buffer = io.BytesIO()
                    img_format = img.format or 'JPEG'
                    img.save(thumb_buffer, format=img_format, quality=85)
                    thumbnail_s3_key = f"thumbnails/{base_filename}"
                    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=thumbnail_s3_key,
                                         Body=thumb_buffer.getvalue(),
                                         ContentType=f'image/{img_format.lower()}')
                    thumbnail_path = thumbnail_s3_key
                except Exception as e:
                    logger.error(f"Error creating thumbnail for {original_filename}: {e}")

            uploaded_at_time = datetime.datetime.utcnow()

            conn = None
            cursor = None
            try:
                pool = get_db_pool()
                if not pool:
                    raise Exception("Worker process could not get DB pool.")
                conn = pool.get_connection()
                cursor = conn.cursor(dictionary=True)
                cursor.execute(
                    "INSERT INTO sightings (timestamp, image_path, original_image_path, thumbnail_path, file_hash, user_id, camera_id, uploaded_at, predictions) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (timestamp, display_image_path, original_image_path, thumbnail_path, file_hash, user_id, camera_id,
                     uploaded_at_time, predictions_json))
                sighting_id = cursor.lastrowid

                if OPENWEATHER_API_KEY:
                    cursor.execute("SELECT location_zip FROM cameras WHERE id = %s AND user_id = %s",
                                   (camera_id, user_id))
                    camera_data = cursor.fetchone()
                    if camera_data:
                        lat, lon = get_lat_lon_from_zip(camera_data['location_zip'])
                        weather_json = get_historical_weather(lat, lon, timestamp, OPENWEATHER_API_KEY)
                        if weather_json:
                            cursor.execute("UPDATE sightings SET weather_data = %s WHERE id = %s",
                                           (json.dumps(weather_json), sighting_id))

                if animals_found:
                    format_strings_select = ','.join(['%s'] * len(animals_found))
                    cursor.execute(f"SELECT id, name FROM animals WHERE name IN ({format_strings_select})",
                                   tuple(animals_found))
                    animal_id_map = {row['name']: row['id'] for row in cursor.fetchall()}

                    existing_animal_names = set(animal_id_map.keys())
                    new_animal_names = [name for name in animals_found if name not in existing_animal_names]
                    if new_animal_names:
                        new_animal_tuples = [(name,) for name in new_animal_names]
                        cursor.executemany("INSERT INTO animals (name) VALUES (%s)", new_animal_tuples)
                        format_strings_new = ','.join(['%s'] * len(new_animal_names))
                        cursor.execute(f"SELECT id, name FROM animals WHERE name IN ({format_strings_new})",
                                       tuple(new_animal_names))
                        for row in cursor.fetchall():
                            animal_id_map[row['name']] = row['id']

                    sighting_animal_tuples = []
                    if animal_id_map:
                        sighting_animal_tuples = [(sighting_id, animal_id_map[name]) for name in animals_found if
                                                  name in animal_id_map]
                    if sighting_animal_tuples:
                        cursor.executemany("INSERT INTO sighting_animals (sighting_id, animal_id) VALUES (%s, %s)",
                                           sighting_animal_tuples)
                conn.commit()

            except Exception as db_e:
                logger.exception(f"DATABASE ERROR for file {original_filename}: {db_e}")
                if conn: conn.rollback()
                raise
            finally:
                if cursor: cursor.close()
                if conn: conn.close()

            status = 'processed' if animals_found else 'no_animals'
            return {'status': status, 'image_key': display_image_path, 'animals': animals_found}

        except Exception as e:
            logger.exception(f"OVERALL TASK ERROR for file {original_filename}")
            if "could not get DB pool" in str(e) or "Too many connections" in str(e):
                logger.warning(f"DB connection error for {original_filename}, will retry")
                try:
                    self.retry(exc=e)
                except Exception as retry_error:
                    logger.error(f"Max retries exceeded: {retry_error}")
                    raise
            else:
                logger.error(f"Non-retryable error: {e}")
                raise
            raise
        finally:
            logger.debug(f"[CLEANUP] About to remove: {file_path}, exists before? {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    if os.path.exists(file_path + '.info'):
                        os.remove(file_path + '.info')
                    logger.debug(f"Cleaned up temp files for {original_filename}")
                except Exception as cleanup_error:
                    logger.debug(f"Cleanup error: {cleanup_error}")
                    pass


@celery.task
def cleanup_old_images_task():
    logger = logging.getLogger(__name__)
    conn = None
    cursor = None
    with app.app_context():
        try:
            logger.info("Starting daily cleanup of old images...")
            pool = get_db_pool()
            if not pool:
                logger.error("Cleanup task could not get DB pool.")
                return
            conn = pool.get_connection()
            cursor = conn.cursor(dictionary=True)

            cutoff_date = datetime.datetime.utcnow() - timedelta(days=60)

            cursor.execute(
                "SELECT id, image_path, original_image_path, thumbnail_path FROM sightings WHERE uploaded_at < %s AND retained = FALSE",
                (cutoff_date,))
            rows = cursor.fetchall()

            if not rows:
                logger.info("No images found uploaded more than 60 days ago to delete.")
                return

            logger.info(f"Found {len(rows)} sightings with images uploaded with more than 60 days ago to delete...")

            objects_to_delete = []
            sighting_ids_to_update = []

            for row in rows:
                sighting_ids_to_update.append(row['id'])
                if row['image_path']:
                    objects_to_delete.append({'Key': row['image_path']})
                if row['original_image_path']:
                    objects_to_delete.append({'Key': row['original_image_path']})
                if row['thumbnail_path']:
                    objects_to_delete.append({'Key': row['thumbnail_path']})

            if s3_client and objects_to_delete:
                for i in range(0, len(objects_to_delete), 1000):
                    chunk = objects_to_delete[i:i + 1000]
                    s3_client.delete_objects(Bucket=S3_BUCKET_NAME, Delete={'Objects': chunk})
                    logger.info(f"Deleted {len(chunk)} image files from S3.")

            if sighting_ids_to_update:
                format_strings = ','.join(['%s'] * len(sighting_ids_to_update))
                cursor.execute(
                    f"UPDATE sightings SET image_path = NULL, original_image_path = NULL, thumbnail_path = NULL WHERE id IN ({format_strings})",
                    tuple(sighting_ids_to_update))
                conn.commit()
                logger.info(f"Detached image paths from {len(sighting_ids_to_update)} database records.")

            logger.info("Daily cleanup task finished successfully.")

        except Exception as e:
            if conn and conn.is_connected():
                conn.rollback()
            logger.exception(f"An error occurred during the daily cleanup task: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()


# --- Auth and User Routes ---
@app.route('/')
def landing_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('landing.html')


@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember = request.form.get('remember') is not None
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user_data = cursor.fetchone()

            if user_data and check_password_hash(user_data['password_hash'], password):
                if not user_data.get('confirmed'):
                    flash('Please confirm your email address before logging in.', 'warning')
                    return redirect(url_for('login'))

                user = User(id=user_data['id'], username=user_data['username'], email=user_data['email'],
                            confirmed=user_data.get('confirmed'), role=user_data.get('role', 'user'))
                login_user(user, remember=remember)
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password')
        finally:
            cursor.close()
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        agree_terms = request.form.get('agree_terms')

        if not agree_terms:
            flash('You must agree to the Terms of Service to create an account.', 'danger')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match.')
            return render_template('register.html')

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
            if cursor.fetchone():
                flash('Username or email already exists.')
                return render_template('register.html')

            password_hash = generate_password_hash(password)

            cursor.execute(
                "INSERT INTO users (username, email, password_hash, confirmed) VALUES (%s, %s, %s, %s)",
                (username, email, password_hash, False))
            conn.commit()

            send_confirmation_email(email)
            flash('A confirmation email has been sent to your address. Please click the link to activate your account.',
                  'success')
            return redirect(url_for('login'))

        except Exception as e:
            conn.rollback()
            flash('An error occurred during registration. Please try again.', 'danger')
            app.logger.exception(f"Registration error: {e}")
        finally:
            cursor.close()

    return render_template('register.html')


@app.route('/confirm/<token>')
def confirm_email(token):
    try:
        email = s.loads(token, salt='email-confirm-salt', max_age=3600)
    except:
        flash('The confirmation link is invalid or has expired.', 'danger')
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user_data = cursor.fetchone()
        if user_data['confirmed']:
            flash('Account already confirmed. Please log in.', 'success')
        else:
            cursor.execute("UPDATE users SET confirmed = TRUE, confirmed_on = %s WHERE email = %s",
                           (datetime.datetime.utcnow(), email))
            conn.commit()
            flash('You have confirmed your account. Thanks!', 'success')
    except Exception as e:
        conn.rollback()
        flash('An error occurred during email confirmation.', 'danger')
        app.logger.exception(f"Email confirmation error: {e}")
    finally:
        cursor.close()

    return redirect(url_for('login'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        email = request.form['email']
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user_data = cursor.fetchone()
            if user_data:
                token = secrets.token_urlsafe(32)
                expiration = datetime.datetime.utcnow() + timedelta(hours=1)
                cursor.execute("UPDATE users SET reset_token = %s, reset_token_expiration = %s WHERE id = %s",
                               (token, expiration, user_data['id']))
                conn.commit()
                user = User(id=user_data['id'], username=user_data['username'], email=user_data['email'])
                send_reset_email(user, token)
        except Exception as e:
            conn.rollback()
            app.logger.exception(f"Forgot password error: {e}")
        finally:
            cursor.close()

        flash('If an account with that email exists, a password reset link has been sent.')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if current_user.is_authenticated: return redirect(url_for('index'))
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM users WHERE reset_token = %s AND reset_token_expiration > UTC_TIMESTAMP()",
                       (token,))
        user_data = cursor.fetchone()
        if not user_data:
            flash('That is an invalid or expired token.', 'warning')
            return redirect(url_for('forgot_password'))
        if request.method == 'POST':
            password = request.form['password']
            password_hash = generate_password_hash(password)
            cursor.execute(
                "UPDATE users SET password_hash = %s, reset_token = NULL, reset_token_expiration = NULL WHERE id = %s",
                (password_hash, user_data['id']))
            conn.commit()
            flash('Your password has been updated! You can now log in.', 'success')
            return redirect(url_for('login'))
    except Exception as e:
        conn.rollback()
        flash('An error occurred while resetting your password.', 'danger')
        app.logger.exception(f"Password reset error: {e}")
    finally:
        cursor.close()

    return render_template('reset_password.html', token=token)


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT password_hash, email, username FROM users WHERE id = %s", (current_user.id,))
            user_data = cursor.fetchone()

            if not user_data:
                flash('User not found.', 'danger')
                return redirect(url_for('profile'))

            current_password = request.form.get('current_password')
            old_email = user_data['email']
            old_username = user_data['username']

            if not check_password_hash(user_data['password_hash'], current_password):
                flash('Your current password was incorrect.', 'danger')
                return redirect(url_for('profile'))

            username_updated = False
            email_updated = False
            password_updated = False

            if 'new_username' in request.form:
                new_username = request.form.get('new_username')
                if new_username and new_username != old_username:
                    cursor.execute("SELECT id FROM users WHERE username = %s", (new_username,))
                    if cursor.fetchone():
                        flash('That username is already taken. Please choose another.', 'danger')
                    else:
                        cursor.execute("UPDATE users SET username = %s WHERE id = %s", (new_username, current_user.id))
                        username_updated = True
                        current_user.username = new_username

            elif 'new_email' in request.form:
                new_email = request.form.get('new_email')
                if new_email and new_email != old_email:
                    cursor.execute("UPDATE users SET email = %s WHERE id = %s", (new_email, current_user.id))
                    email_updated = True
                    current_user.email = new_email

            elif 'new_password' in request.form:
                new_password = request.form.get('new_password')
                if new_password:
                    new_password_hash = generate_password_hash(new_password)
                    cursor.execute("UPDATE users SET password_hash = %s WHERE id = %s",
                                   (new_password_hash, current_user.id))
                    password_updated = True

            conn.commit()

            if username_updated:
                flash('Your username has been updated.', 'success')
            if email_updated:
                send_email_change_notification(current_user, old_email)
                flash('Your email has been updated.', 'success')
            if password_updated:
                send_password_change_notification(current_user)
                flash('Your password has been updated.', 'success')

        except Exception as e:
            conn.rollback()
            app.logger.exception(f"Profile update error: {e}")
            flash('An error occurred while updating your profile.', 'danger')
        finally:
            cursor.close()

        return redirect(url_for('profile'))

    return render_template('profile.html')


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        subject_prefix = request.form.get('subject')
        message_body = request.form.get('message_body')

        if not subject_prefix or not message_body:
            flash('Please select a subject and write a message.', 'danger')
            return render_template('feedback.html')

        if current_user.is_authenticated:
            user_name = current_user.username
            user_email = current_user.email
            user_id_str = str(current_user.id)
        else:
            user_name = request.form.get('name')
            user_email = request.form.get('email')
            user_id_str = "Guest (Not Logged In)"

            if not user_name or not user_email:
                flash('Please provide your name and email address.', 'danger')
                return render_template('feedback.html')

        try:
            admin_email = app.config.get('MAIL_USERNAME')
            if not admin_email:
                flash('Feedback system is not configured. Please try again later.', 'danger')
                return redirect(url_for('index'))

            subject = f"Feedback from {user_name} ({subject_prefix})"
            body = f"""
            You have received feedback from a user:

            User: {user_name}
            Email: {user_email}
            User ID: {user_id_str}

            Subject: {subject_prefix}
            -------------------------------------

            Message:
            {message_body}
            """

            msg = Message(
                subject=subject,
                recipients=[admin_email],
                sender=('GameCam AI Feedback', app.config['MAIL_DEFAULT_SENDER']),
                reply_to=user_email,
                body=body
            )
            mail.send(msg)
            flash('Thank you for your feedback! We have received your message.', 'success')

            if current_user.is_authenticated:
                return redirect(url_for('index'))
            else:
                return redirect(url_for('landing_page'))

        except Exception as e:
            app.logger.exception(f"Error sending feedback.html email: {e}")
            flash('An error occurred while sending your feedback. Please try again later.', 'danger')

    return render_template('feedback.html')


@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    users = []
    try:
        query = """
                SELECT u.id, \
                       u.username, \
                       u.email, \
                       u.confirmed, \
                       u.role,
                       COUNT(s.id)         AS total_uploads,
                       COUNT(s.image_path) AS stored_images
                FROM users u
                         LEFT JOIN sightings s ON u.id = s.user_id
                GROUP BY u.id, u.username, u.email, u.confirmed, u.role
                ORDER BY u.id;
                """
        cursor.execute(query)
        users = cursor.fetchall()
    finally:
        cursor.close()

    total_users = len(users)
    return render_template('admin.html', users=users, total_users=total_users)


# --- Public Pages ---
@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/donate')
def donate():
    return render_template('donate.html')
@app.route('/tos')
def tos():
    return render_template('tos.html')


@app.route('/privacy')
def privacy():
    return render_template('privacy.html')


# --- Main App Routes & API Endpoints ---
@app.route('/index')
@login_required
def index():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cameras = []
    try:
        cursor.execute("SELECT id, name FROM cameras WHERE user_id = %s ORDER BY name", (current_user.id,))
        cameras = cursor.fetchall()
    finally:
        cursor.close()
    return render_template('index.html', cameras=cameras)


@app.route('/cameras')
@login_required
def manage_cameras():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cameras = []

    # Hardcode global limit
    camera_limit = MAX_CAMERA_LIMIT

    try:
        query = """
                SELECT c.id, \
                       c.name, \
                       c.location_zip, \
                       c.is_active, \
                       COUNT(DISTINCT CASE WHEN sa.sighting_id IS NOT NULL THEN s.id END) AS total_sightings,
                       COUNT(DISTINCT CASE WHEN s.retained = TRUE THEN s.id END)          AS retained_sightings,
                       COUNT(DISTINCT s.id)                                               AS image_uploads
                FROM cameras c
                         LEFT JOIN sightings s ON c.id = s.camera_id AND s.user_id = %s
                         LEFT JOIN sighting_animals sa ON s.id = sa.sighting_id
                WHERE c.user_id = %s
                GROUP BY c.id, c.name, c.location_zip, c.is_active
                ORDER BY c.name
                """
        cursor.execute(query, (current_user.id, current_user.id))
        cameras = cursor.fetchall()
    finally:
        cursor.close()

    active_camera_count = sum(1 for cam in cameras if cam['is_active'])

    return render_template('cameras.html',
                           cameras=cameras,
                           camera_limit=camera_limit,
                           active_camera_count=active_camera_count)


@app.route('/cameras/summary-data')
@login_required
def get_camera_summary_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cameras_data = []
    try:
        query = """
                SELECT c.id, \
                       c.name, \
                       c.location_zip, \
                       c.is_active, \
                       COUNT(DISTINCT CASE WHEN sa.sighting_id IS NOT NULL THEN s.id END) AS total_sightings,
                       COUNT(DISTINCT CASE WHEN s.retained = TRUE THEN s.id END)          AS retained_sightings,
                       COUNT(DISTINCT s.id)                                               AS image_uploads
                FROM cameras c
                         LEFT JOIN sightings s ON c.id = s.camera_id AND s.user_id = %s
                         LEFT JOIN sighting_animals sa ON s.id = sa.sighting_id
                WHERE c.user_id = %s
                GROUP BY c.id, c.name, c.location_zip, c.is_active
                ORDER BY c.name
                """
        cursor.execute(query, (current_user.id, current_user.id))
        cameras_data = cursor.fetchall()
    except Exception as e:
        app.logger.exception(f"Error getting camera summary data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        cursor.close()

    return jsonify({'success': True, 'cameras': cameras_data})


@app.route('/cameras/add', methods=['POST'])
@login_required
def add_camera():
    name, location_zip = request.form.get('name'), request.form.get('location_zip')
    if not name or not location_zip:
        flash('Camera name and ZIP code are required.', 'warning')
    else:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            # Enforce limit
            cursor.execute("SELECT COUNT(id) FROM cameras WHERE user_id = %s", (current_user.id,))
            current_camera_count = cursor.fetchone()[0]

            if current_camera_count >= MAX_CAMERA_LIMIT:
                flash(f'You have reached the limit of {MAX_CAMERA_LIMIT} cameras.', 'danger')
                return redirect(url_for('manage_cameras'))

            cursor.execute("INSERT INTO cameras (name, location_zip, user_id, is_active) VALUES (%s, %s, %s, %s)",
                           (name, location_zip, current_user.id, True))
            conn.commit()
            flash('Camera added successfully!', 'success')
        except Exception as e:
            conn.rollback()
            flash('An error occurred while adding the camera.', 'danger')
            app.logger.exception(f"Add camera error: {e}")
        finally:
            cursor.close()

    return redirect(url_for('manage_cameras'))


@app.route('/cameras/rename/<int:camera_id>', methods=['POST'])
@login_required
def rename_camera(camera_id):
    new_name = request.form.get('new_name')
    if not new_name or not new_name.strip():
        flash('New camera name cannot be empty.', 'warning')
        return redirect(url_for('manage_cameras'))

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE cameras SET name = %s WHERE id = %s AND user_id = %s",
                       (new_name.strip(), camera_id, current_user.id))
        conn.commit()

        if cursor.rowcount == 0:
            flash('Could not rename camera. It may not exist or you do not have permission.', 'danger')
        else:
            flash('Camera renamed successfully!', 'success')
    except Exception as e:
        conn.rollback()
        flash('An error occurred while renaming the camera.', 'danger')
        app.logger.exception(f"Rename camera error: {e}")
    finally:
        cursor.close()

    return redirect(url_for('manage_cameras'))


@app.route('/cameras/delete/<int:camera_id>', methods=['POST'])
@login_required
def delete_camera(camera_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT image_path, original_image_path, thumbnail_path FROM sightings WHERE camera_id = %s AND user_id = %s",
            (camera_id, current_user.id)
        )
        sightings = cursor.fetchall()

        objects_to_delete = []
        if sightings:
            for sighting in sightings:
                if sighting['image_path']:
                    objects_to_delete.append({'Key': sighting['image_path']})
                if sighting['original_image_path']:
                    objects_to_delete.append({'Key': sighting['original_image_path']})
                if sighting['thumbnail_path']:
                    objects_to_delete.append({'Key': sighting['thumbnail_path']})

        if s3_client and objects_to_delete:
            for i in range(0, len(objects_to_delete), 1000):
                chunk = objects_to_delete[i:i + 1000]
                s3_client.delete_objects(Bucket=S3_BUCKET_NAME, Delete={'Objects': chunk})

        cursor.execute("DELETE FROM sightings WHERE camera_id = %s AND user_id = %s", (camera_id, current_user.id))
        cursor.execute("DELETE FROM cameras WHERE id = %s AND user_id = %s", (camera_id, current_user.id))
        conn.commit()

        flash('Camera, all its sightings, and associated image files have been deleted.', 'success')
    except Exception as e:
        conn.rollback()
        flash('An error occurred while deleting the camera.', 'danger')
        app.logger.exception(f"Delete camera error: {e}")
    finally:
        cursor.close()

    return redirect(url_for('manage_cameras'))


@app.route('/cameras/delete-data/<int:camera_id>', methods=['POST'])
@login_required
def delete_camera_data(camera_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id FROM cameras WHERE id = %s AND user_id = %s", (camera_id, current_user.id))
        if not cursor.fetchone():
            flash('Invalid camera specified.', 'danger')
            return redirect(url_for('manage_cameras'))

        cursor.execute(
            "SELECT image_path, original_image_path FROM sightings WHERE user_id = %s AND camera_id = %s AND retained = FALSE",
            (current_user.id, camera_id))
        rows = cursor.fetchall()

        if s3_client and rows:
            objects_to_delete = [{'Key': k} for r in rows for k in (r['image_path'], r['original_image_path']) if k]
            if objects_to_delete:
                s3_client.delete_objects(Bucket=S3_BUCKET_NAME, Delete={'Objects': objects_to_delete})

        cursor.execute(
            "UPDATE sightings SET image_path = NULL, original_image_path = NULL WHERE user_id = %s AND camera_id = %s AND retained = FALSE",
            (current_user.id, camera_id))
        conn.commit()

        flash(f'Successfully deleted image files for non-retained sightings. The data has been kept.', 'success')
    except Exception as e:
        conn.rollback()
        flash('An error occurred while deleting camera data.', 'danger')
        app.logger.exception(f"Delete camera data error: {e}")
    finally:
        cursor.close()

    return redirect(url_for('manage_cameras'))


@app.route('/hooks/tusd', methods=['POST'])
def tusd_hook():
    app.logger.debug(f"[HOOK ENTRY] Received POST: {request.json}")
    if not request.json or request.json.get('Type') != 'post-finish':
        return jsonify(success=True)

    try:
        event_info = request.json.get('Event', {})
        upload_info = event_info.get('Upload', {})
        storage_info = upload_info.get('Storage', {})
        metadata = upload_info.get('MetaData', {})

        file_path = storage_info.get('Path')
        original_filename = metadata.get('filename')
        user_id = metadata.get('user_id')
        camera_id = metadata.get('camera_id')

        app.logger.debug(
            f"file_path={file_path}, original_filename={original_filename}, user_id={user_id}, camera_id={camera_id}")
        if not all([file_path, original_filename, user_id, camera_id]):
            app.logger.warning(f"Hook received with missing data: {request.json}")
            return jsonify({'error': 'Missing required metadata in hook'}), 400

        conn_check = None
        cursor_check = None
        try:
            pool = get_db_pool()
            if not pool:
                raise Exception("tusd_hook could not get DB pool.")

            conn_check = pool.get_connection()
            cursor_check = conn_check.cursor(dictionary=True)

            # Check if the camera they are uploading to is active (user manually paused it)
            cursor_check.execute("SELECT is_active FROM cameras WHERE id = %s AND user_id = %s", (camera_id, user_id))
            camera = cursor_check.fetchone()

            if not camera or not camera['is_active']:
                app.logger.warning(f"Upload blocked for inactive camera {camera_id} (user {user_id})")
                response = jsonify(success=False,
                                   error="This camera is currently set to inactive.")
                response.status_code = 403
                return response

        except Exception as e:
            app.logger.exception(f"Error checking camera active status in tusd_hook: {e}")
            response = jsonify(success=False, error="Could not verify camera status.")
            response.status_code = 500
            return response
        finally:
            if cursor_check: cursor_check.close()
            if conn_check: conn_check.close()

        app.logger.info(f"TUSD POST-FINISH: {original_filename}, user={user_id}, camera={camera_id}")

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                task = process_image_task.apply_async(args=[file_path, original_filename, user_id, camera_id, metadata],
                                                      queue='ai_queue')
                app.logger.info(f"TASK QUEUED: {task.id} for {original_filename}")
                return jsonify(success=True, task_id=task.id), 200
            except Exception as queue_error:
                app.logger.error(
                    f"Attempt {attempt + 1}/{max_attempts} failed to queue {original_filename}: {queue_error}")
                if attempt == max_attempts - 1:
                    raise
                time.sleep(0.1)

    except Exception as e:
        app.logger.exception("Error processing tusd hook")

        file_path = request.json.get('Event', {}).get('Upload', {}).get('Storage', {}).get('Path')
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/task-status/<task_id>')
@login_required
def task_status(task_id):
    task = process_image_task.AsyncResult(task_id)
    response = {'state': task.state}
    if task.state == 'SUCCESS':
        response['result'] = task.result
    return jsonify(response)


@app.route('/stream-task-status/<task_id>')
@login_required
def stream_task_status(task_id):
    def event_stream():
        task = process_image_task.AsyncResult(task_id)
        while task.state == 'PENDING' or task.state == 'PROGRESS' or task.state == 'STARTED':
            time.sleep(2)
            task = process_image_task.AsyncResult(task_id)

        response_data = {'state': task.state}
        if task.state == 'SUCCESS':
            response_data['result'] = task.result
        elif task.state == 'FAILURE':
            response_data['result'] = 'Processing failed.'

        yield f"data: {json.dumps(response_data)}\n\n"

    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/data')
@login_required
def get_data_route():
    camera_id = request.args.get('camera_id', 'all')
    animal_filter = request.args.get('animal', 'all')
    num_windows = int(request.args.get('prediction_windows', 2))
    window_width = int(request.args.get('window_width', 3))
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    if not start_date:
        start_date = (date.today() - timedelta(days=30)).strftime('%Y-%m-%d')

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    rows = []
    detected_animals = []

    try:
        cursor.execute("""
                       SELECT DISTINCT a.name
                       FROM animals a
                                JOIN sighting_animals sa ON a.id = sa.animal_id
                                JOIN sightings s ON sa.sighting_id = s.id
                       WHERE s.user_id = %s
                       ORDER BY a.name
                       """, (current_user.id,))
        detected_animals_rows = cursor.fetchall()
        detected_animals = [row['name'] for row in detected_animals_rows]
        params = [current_user.id]
        query = """
                SELECT a.name as animal, s.timestamp
                FROM sightings s
                         LEFT JOIN sighting_animals sa ON s.id = sa.sighting_id
                         LEFT JOIN animals a ON sa.animal_id = a.id
                WHERE s.user_id = %s
                """

        if camera_id != 'all':
            query += " AND s.camera_id = %s"
            params.append(camera_id)
        if start_date:
            query += " AND s.timestamp >= %s"
            params.append(start_date)
        if end_date:
            end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
            query += " AND s.timestamp < %s"
            params.append(end_date_obj.strftime('%Y-%m-%d'))
        query += " ORDER BY s.timestamp ASC"
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
    finally:
        cursor.close()

    if not rows:
        return jsonify({"dailyData": {}, "timelineData": {}, "uniqueAnimals": detected_animals, "primeTimeData": {},
                        "animalBreakdownData": {}})

    daily_activity = defaultdict(lambda: [0] * 7)
    timeline_activity = defaultdict(dict)
    day_sighting_counts = defaultdict(int)
    animal_counts = defaultdict(int)
    unique_days = set()
    for row in rows:
        animal, dt = row['animal'], row['timestamp']
        date_obj, day_of_week, date_key = dt.date(), int(dt.strftime('%w')), dt.date().isoformat()
        unique_days.add(date_obj)
        if animal:
            daily_activity[animal][day_of_week] += 1
            timeline_activity[animal][date_key] = timeline_activity[animal].get(date_key, 0) + 1
            animal_counts[animal] += 1

        if animal_filter == 'all':
            if animal:
                day_sighting_counts[dt.hour] += 1
        elif animal == animal_filter:
            day_sighting_counts[dt.hour] += 1

    num_days = len(unique_days) if unique_days else 1
    avg_hourly_sightings = [day_sighting_counts.get(h, 0) / num_days for h in range(24)]
    prime_time_labels = [f"{h:02d}:00" for h in range(24)]

    prime_time_title = f"Avg. {'All Animals' if animal_filter == 'all' else animal_filter.capitalize()} Sightings"
    prime_time_datasets = [
        {"label": prime_time_title, "data": avg_hourly_sightings, "borderColor": 'rgba(234, 179, 8, 1)',
         "backgroundColor": 'rgba(234, 179, 8, 0.5)', "tension": 0.4, "fill": True}]
    window_sums = [(sum((avg_hourly_sightings * 2)[i:i + window_width]), i) for i in range(24)]
    window_sums.sort(key=lambda x: x[0], reverse=True)
    peak_windows, used_hours = [], set()
    for _, start_hour in window_sums:
        if len(peak_windows) >= num_windows: break
        window_hours = {(start_hour + i) % 24 for i in range(window_width)}
        if not used_hours.intersection(window_hours):
            peak_windows.append(start_hour)
            used_hours.update(window_hours)
    annotations = {}
    for i, start_hour in enumerate(peak_windows):
        end_hour = (start_hour + window_width - 1) % 24
        if start_hour <= end_hour:
            annotations[f'box{i + 1}'] = {'type': 'box', 'xMin': start_hour - 0.5, 'xMax': end_hour + 0.5,
                                          'backgroundColor': 'rgba(59, 130, 246, 0.25)',
                                          'borderColor': 'rgba(59, 130, 246, 0.5)', 'borderWidth': 1,
                                          'label': {'content': "Prime Time", 'enabled': True, 'position': "start",
                                                    'color': 'white'}}
        else:
            annotations[f'box{i + 1}a'] = {'type': 'box', 'xMin': start_hour - 0.5, 'xMax': 23.5,
                                           'backgroundColor': 'rgba(59, 130, 246, 0.25)',
                                           'borderColor': 'rgba(59, 130, 246, 0.5)', 'borderWidth': 1}
            annotations[f'box{i + 1}b'] = {'type': 'box', 'xMin': -0.5, 'xMax': end_hour + 0.5,
                                           'backgroundColor': 'rgba(59, 130, 246, 0.25)',
                                           'borderColor': 'rgba(59, 130, 246, 0.5)', 'borderWidth': 1,
                                           'label': {'content': "Prime Time", 'enabled': True, 'position': "start",
                                                     'color': 'white'}}
    prime_time_data = {"labels": prime_time_labels, "datasets": prime_time_datasets, "annotations": annotations}

    color_map = {
        'deer': ('rgba(34, 197, 94, 0.7)', 'rgba(34, 197, 94, 1)'),
        'hog': ('rgba(249, 115, 22, 0.7)', 'rgba(249, 115, 22, 1)'),
        'turkey': ('rgba(161, 98, 7, 0.7)', 'rgba(161, 98, 7, 1)'),
        'mountain_lion': ('rgba(234, 179, 8, 0.7)', 'rgba(234, 179, 8, 1)'),
        'raccoon': ('rgba(107, 114, 128, 0.7)', 'rgba(107, 114, 128, 1)'),
        'coyote': ('rgba(202, 138, 4, 0.7)', 'rgba(202, 138, 4, 1)'),
        'person': ('rgba(239, 68, 68, 0.7)', 'rgba(239, 68, 68, 1)'),
        'cat': ('rgba(192, 132, 252, 0.7)', 'rgba(192, 132, 252, 1)'),
        'dog': ('rgba(59, 130, 246, 0.7)', 'rgba(59, 130, 246, 1)'),
        'bear': ('rgba(75, 51, 33, 0.7)', 'rgba(75, 51, 33, 1)'),
        'fox': ('rgba(245, 101, 5, 0.7)', 'rgba(245, 101, 5, 1)'),
        'bobcat': ('rgba(217, 119, 6, 0.7)', 'rgba(217, 119, 6, 1)'),
        'elk': ('rgba(13, 148, 136, 0.7)', 'rgba(13, 148, 136, 1)'),
        'badger': ('rgba(71, 85, 105, 0.7)', 'rgba(71, 85, 105, 1)'),
        'porcupine': ('rgba(163, 163, 163, 0.7)', 'rgba(163, 163, 163, 1)'),
        'opossum': ('rgba(226, 232, 240, 0.7)', 'rgba(226, 232, 240, 1)')
    }
    default_color = ('rgba(209, 213, 219, 0.7)', 'rgba(209, 213, 219, 1)')

    animal_breakdown_labels = list(animal_counts.keys())
    animal_breakdown_colors = [color_map.get(animal, default_color)[0] for animal in animal_breakdown_labels]
    animal_breakdown_data = {"labels": animal_breakdown_labels, "datasets": [{"data": list(animal_counts.values()),
                                                                              "backgroundColor": animal_breakdown_colors}]}

    daily_labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    daily_datasets = [{"label": animal.capitalize(), "data": daily_activity.get(animal, [0] * 7),
                       "backgroundColor": color_map.get(animal, default_color)[0]} for animal in detected_animals]
    daily_data = {"labels": daily_labels, "datasets": daily_datasets}

    all_dates = sorted(list(set(d for animal_dates in timeline_activity.values() for d in animal_dates.keys())))
    timeline_datasets = [
        {"label": animal.capitalize(), "data": [timeline_activity.get(animal, {}).get(date, 0) for date in all_dates],
         "borderColor": color_map.get(animal, default_color)[1], "tension": 0.1} for animal in detected_animals]
    timeline_data = {"labels": all_dates, "datasets": timeline_datasets}

    return jsonify({"dailyData": daily_data, "timelineData": timeline_data, "uniqueAnimals": detected_animals,
                    "primeTimeData": prime_time_data, "animalBreakdownData": animal_breakdown_data})


@app.route('/gallery-data')
@login_required
def gallery_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    page = request.args.get('page', 1, type=int)
    limit = 50
    offset = (page - 1) * limit

    params = [current_user.id]
    query_base = " FROM sightings s LEFT JOIN sighting_animals sa ON s.id = sa.sighting_id LEFT JOIN animals a ON sa.animal_id = a.id WHERE s.user_id = %s"

    camera_id = request.args.get('camera_id', 'all')
    if camera_id != 'all': query_base += " AND s.camera_id = %s"; params.append(camera_id)
    start_date = request.args.get('start_date')
    if start_date: query_base += " AND s.timestamp >= %s"; params.append(start_date)
    end_date = request.args.get('end_date')
    if end_date:
        end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        query_base += " AND s.timestamp < %s";
        params.append(end_date_obj.strftime('%Y-%m-%d'))

    retained_only = request.args.get('retained_only') == 'true'
    if retained_only:
        query_base += " AND s.retained = TRUE"

    sightings = []
    unique_animals = []
    try:
        cursor.execute(f"SELECT DISTINCT a.name {query_base} ORDER BY a.name", tuple(params))
        unique_animals = [row['name'] for row in cursor.fetchall() if row['name']]
        unique_animals.append("Empty")

        query = f"SELECT s.id, s.timestamp, s.image_path, s.original_image_path, s.thumbnail_path, s.retained, s.feedback_status, s.camera_id, GROUP_CONCAT(a.name) as animals {query_base}"
        animal_filter = request.args.get('animal', 'all')
        if animal_filter == 'empty':
            query += " GROUP BY s.id HAVING animals IS NULL"
        elif animal_filter != 'all':
            query += " GROUP BY s.id HAVING FIND_IN_SET(%s, animals)"
            params.append(animal_filter)
        else:
            query += " GROUP BY s.id"

        query += " ORDER BY s.timestamp DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        cursor.execute(query, tuple(params))
        sightings = cursor.fetchall()
    finally:
        cursor.close()

    for s in sightings:
        s['timestamp'] = s['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

    return jsonify({'sightings': sightings, 'uniqueAnimals': unique_animals})


@app.route('/toggle-retain/<int:sighting_id>', methods=['POST'])
@login_required
def toggle_retain(sighting_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    new_status = None
    try:
        cursor.execute("UPDATE sightings SET retained = NOT retained WHERE id = %s AND user_id = %s",
                       (sighting_id, current_user.id))
        conn.commit()
        cursor.execute("SELECT retained FROM sightings WHERE id = %s", (sighting_id,))
        new_status = cursor.fetchone()[0]
    except Exception as e:
        conn.rollback()
        app.logger.exception(f"Toggle retain error: {e}")
        return jsonify({'success': False, 'error': 'An error occurred'}), 500
    finally:
        cursor.close()

    return jsonify({'success': True, 'retained': bool(new_status)})


@app.route('/sighting/delete/<int:sighting_id>', methods=['POST'])
@login_required
def delete_sighting(sighting_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT image_path, original_image_path, thumbnail_path FROM sightings WHERE id = %s AND user_id = %s",
            (sighting_id, current_user.id)
        )
        sighting = cursor.fetchone()

        if not sighting:
            return jsonify({'success': False, 'error': 'Sighting not found or permission denied'}), 404

        objects_to_delete = []
        if sighting.get('image_path'):
            objects_to_delete.append({'Key': sighting['image_path']})
        if sighting.get('original_image_path'):
            objects_to_delete.append({'Key': sighting['original_image_path']})
        if sighting.get('thumbnail_path'):
            objects_to_delete.append({'Key': sighting['thumbnail_path']})

        if s3_client and objects_to_delete:
            s3_client.delete_objects(Bucket=S3_BUCKET_NAME, Delete={'Objects': objects_to_delete})

        cursor.execute("DELETE FROM sightings WHERE id = %s", (sighting_id,))
        conn.commit()

    except Exception as e:
        conn.rollback()
        app.logger.exception(f"Delete sighting error: {e}")
        return jsonify({'success': False, 'error': 'An internal server error occurred.'}), 500
    finally:
        cursor.close()

    return jsonify({'success': True})


@app.route('/sighting/move/<int:sighting_id>', methods=['POST'])
@login_required
def move_sighting(sighting_id):
    new_camera_id = request.form.get('new_camera_id')
    if not new_camera_id:
        return jsonify({'success': False, 'error': 'No new camera selected.'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SELECT id FROM sightings WHERE id = %s AND user_id = %s", (sighting_id, current_user.id))
        if not cursor.fetchone():
            return jsonify({'success': False, 'error': 'Sighting not found or permission denied.'}), 404

        cursor.execute("SELECT id FROM cameras WHERE id = %s AND user_id = %s", (new_camera_id, current_user.id))
        if not cursor.fetchone():
            return jsonify({'success': False, 'error': 'Destination camera not found or permission denied.'}), 404

        cursor.execute("UPDATE sightings SET camera_id = %s WHERE id = %s", (new_camera_id, sighting_id))
        conn.commit()

        return jsonify({'success': True, 'sighting_id': sighting_id})

    except Exception as e:
        conn.rollback()
        app.logger.exception(f"Error moving sighting: {e}")
        return jsonify({'success': False, 'error': 'An internal error occurred.'}), 500
    finally:
        cursor.close()


@app.route('/confirm-detection/<int:sighting_id>', methods=['POST'])
@login_required
def confirm_detection(sighting_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT original_image_path FROM sightings WHERE id = %s AND user_id = %s",
                       (sighting_id, current_user.id))
        sighting = cursor.fetchone()
        if sighting and sighting['original_image_path']:
            source_key = sighting['original_image_path']
            dest_folder = os.path.join(TRAINING_DATA_PATH, 'confirmed')
            os.makedirs(dest_folder, exist_ok=True)
            dest_path = os.path.join(dest_folder, os.path.basename(source_key))
            s3_client.download_file(S3_BUCKET_NAME, source_key, dest_path)

            cursor.execute("UPDATE sightings SET feedback_status = 'confirmed' WHERE id = %s", (sighting_id,))
            conn.commit()
    except Exception as e:
        conn.rollback()
        app.logger.exception(f"Confirm detection error: {e}")
        return jsonify({'success': False, 'error': 'An internal server error occurred.'}), 500
    finally:
        cursor.close()

    return jsonify({'success': True})


@app.route('/annotate-sighting/<int:sighting_id>', methods=['POST'])
@login_required
def annotate_sighting(sighting_id):
    correct_animal = request.form.get('animal')
    if not correct_animal or correct_animal not in ANIMAL_CLASS_MAP:
        return jsonify({'success': False, 'error': 'Invalid animal specified.'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(
            "SELECT image_path, original_image_path, predictions FROM sightings WHERE id = %s AND user_id = %s",
            (sighting_id, current_user.id))
        sighting = cursor.fetchone()

        image_to_save_path = sighting.get('image_path') or sighting.get('original_image_path')

        if not sighting or not image_to_save_path:
            return jsonify({'success': False, 'error': 'Sighting or image not found.'}), 404

        unique_folder_name = str(uuid.uuid4())
        source_key = image_to_save_path
        base_filename = os.path.basename(source_key)

        image_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=source_key)
        image_content = image_obj['Body'].read()

        image_s3_key = f"training_data/needs_review/{unique_folder_name}/{base_filename}"
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=image_s3_key, Body=image_content)

        img = Image.open(io.BytesIO(image_content))
        img_width, img_height = img.size

        predictions = json.loads(sighting['predictions']) if sighting.get('predictions') else []

        yolo_lines = []
        class_id = ANIMAL_CLASS_MAP.get(correct_animal)
        #
        if predictions and class_id is not None:
            for pred in predictions:
                x_min, y_min, x_max, y_max = pred['x_min'], pred['y_min'], pred['x_max'], pred['y_max']
                dw = 1.0 / img_width
                dh = 1.0 / img_height
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0
                w = x_max - x_min
                h = y_max - y_min
                x_norm = x_center * dw
                y_norm = y_center * dh
                w_norm = w * dw
                h_norm = h * dh
                yolo_lines.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        if yolo_lines:
            txt_filename = os.path.splitext(base_filename)[0] + '.txt'
            txt_s3_key = f"training_data/needs_review/{unique_folder_name}/{txt_filename}"
            yolo_bytes = "\n".join(yolo_lines).encode('utf-8')
            s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=txt_s3_key, Body=yolo_bytes)

        cursor.execute("INSERT IGNORE INTO animals (name) VALUES (%s)", (correct_animal,))
        cursor.execute("SELECT id FROM animals WHERE name = %s", (correct_animal,))
        animal_result = cursor.fetchone()

        if animal_result:
            animal_id = animal_result['id']
            cursor.execute("INSERT IGNORE INTO sighting_animals (sighting_id, animal_id) VALUES (%s, %s)",
                           (sighting_id, animal_id))

        cursor.execute("UPDATE sightings SET feedback_status = 'needs_review' WHERE id = %s", (sighting_id,))
        conn.commit()

        cursor.execute("""
                       SELECT GROUP_CONCAT(a.name) as animals
                       FROM sighting_animals sa
                                JOIN animals a ON sa.animal_id = a.id
                       WHERE sa.sighting_id = %s
                       """, (sighting_id,))
        result = cursor.fetchone()
        new_animals_str = result['animals'] if result and result['animals'] else ''

        return jsonify({
            'success': True,
            'message': 'Sighting annotated and moved for review.',
            'sighting_id': sighting_id,
            'animals': new_animals_str
        })

    except Exception as e:
        conn.rollback()
        app.logger.exception(f"Error during annotation: {e}")
        return jsonify({'success': False, 'error': 'An internal server error occurred.'}), 500
    finally:
        cursor.close()


@app.route('/remove-animal-ids/<int:sighting_id>', methods=['POST'])
@login_required
def remove_animal_ids(sighting_id):
    data = request.get_json()
    animal_names = data.get('animal_names')
    if not animal_names or not isinstance(animal_names, list):
        return jsonify({'success': False, 'error': 'Animal names not provided.'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id FROM sightings WHERE id = %s AND user_id = %s", (sighting_id, current_user.id))
        if not cursor.fetchone():
            return jsonify({'success': False, 'error': 'Permission denied.'}), 403

        format_strings = ','.join(['%s'] * len(animal_names))
        cursor.execute(f"SELECT id FROM animals WHERE name IN ({format_strings})", tuple(animal_names))
        animals = cursor.fetchall()

        if not animals:
            return jsonify({'success': True, 'message': 'Animals not found, no action taken.'})

        animal_ids = [a['id'] for a in animals]
        format_strings_ids = ','.join(['%s'] * len(animal_ids))
        cursor.execute(f"DELETE FROM sighting_animals WHERE sighting_id = %s AND animal_id IN ({format_strings_ids})",
                       (sighting_id, *animal_ids))
        conn.commit()

        cursor.execute("""
                       SELECT GROUP_CONCAT(a.name) as animals
                       FROM sighting_animals sa
                                JOIN animals a ON sa.animal_id = a.id
                       WHERE sa.sighting_id = %s
                       """, (sighting_id,))
        result = cursor.fetchone()
        new_animals = result['animals'] if result and result['animals'] else ''

        return jsonify({
            'success': True,
            'message': 'Animal IDs removed.',
            'sighting_id': sighting_id,
            'animals': new_animals
        })

    except Exception as e:
        conn.rollback()
        app.logger.exception(f"Error removing animal IDs: {e}")
        return jsonify({'success': False, 'error': 'An internal server error occurred.'}), 500
    finally:
        cursor.close()


@app.route('/sighting/change-timestamp/<int:sighting_id>', methods=['POST'])
@login_required
def change_sighting_timestamp(sighting_id):
    new_timestamp_str = request.form.get('new_timestamp')
    if not new_timestamp_str:
        return jsonify({'success': False, 'error': 'New timestamp was not provided.'}), 400

    try:
        new_timestamp = datetime.datetime.strptime(new_timestamp_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid timestamp format. Please use YYYY-MM-DD HH:MM:SS.'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id FROM sightings WHERE id = %s AND user_id = %s", (sighting_id, current_user.id))
        if not cursor.fetchone():
            return jsonify({'success': False, 'error': 'Permission denied or sighting not found.'}), 404

        cursor.execute("UPDATE sightings SET timestamp = %s WHERE id = %s", (new_timestamp, sighting_id))
        conn.commit()

        return jsonify({
            'success': True,
            'message': 'Timestamp updated successfully.',
            'new_timestamp': new_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'sighting_id': sighting_id
        })

    except Exception as e:
        conn.rollback()
        app.logger.exception(f"Error changing timestamp for sighting {sighting_id}: {e}")
        return jsonify({'success': False, 'error': 'An internal server error occurred while updating.'}), 500
    finally:
        cursor.close()


@app.route('/download/<path:file_key>')
@login_required
def download_file(file_key):
    try:
        s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        return Response(
            s3_object['Body'].iter_chunks(),
            mimetype=s3_object['ContentType'],
            headers={"Content-Disposition": f"inline;filename={os.path.basename(file_key)}"}
        )
    except Exception as e:
        app.logger.exception(f"Error fetching file from S3: {e}")
        return "File not found", 404


# --- Bulk Action Endpoints ---
def get_valid_sighting_ids(sighting_ids, user_id, cursor):
    if not sighting_ids or not isinstance(sighting_ids, list):
        return []
    placeholders = ','.join(['%s'] * len(sighting_ids))
    query = f"SELECT id FROM sightings WHERE user_id = %s AND id IN ({placeholders})"
    params = [user_id] + sighting_ids
    cursor.execute(query, tuple(params))
    return [row['id'] for row in cursor.fetchall()]


def get_updated_animal_lists(sighting_ids, cursor):
    if not sighting_ids:
        return {}
    placeholders = ','.join(['%s'] * len(sighting_ids))
    query = f"""
        SELECT s.id, GROUP_CONCAT(a.name) as animals
        FROM sightings s
        LEFT JOIN sighting_animals sa ON s.id = sa.sighting_id
        LEFT JOIN animals a ON sa.animal_id = a.id
        WHERE s.id IN ({placeholders})
        GROUP BY s.id
    """
    cursor.execute(query, tuple(sighting_ids))
    return {row['id']: (row['animals'] if row['animals'] else '') for row in cursor.fetchall()}


@app.route('/sightings/bulk-delete', methods=['POST'])
@login_required
def bulk_delete_sightings():
    data = request.get_json()
    sighting_ids = data.get('sighting_ids')

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        valid_ids = get_valid_sighting_ids(sighting_ids, current_user.id, cursor)
        if not valid_ids:
            return jsonify({'success': False, 'error': 'No valid sightings selected.'}), 400

        placeholders = ','.join(['%s'] * len(valid_ids))
        cursor.execute(
            f"SELECT image_path, original_image_path, thumbnail_path FROM sightings WHERE id IN ({placeholders})",
            tuple(valid_ids)
        )
        sightings = cursor.fetchall()

        objects_to_delete = []
        for s in sightings:
            if s.get('image_path'): objects_to_delete.append({'Key': s['image_path']})
            if s.get('original_image_path'): objects_to_delete.append({'Key': s['original_image_path']})
            if s.get('thumbnail_path'): objects_to_delete.append({'Key': s['thumbnail_path']})

        if s3_client and objects_to_delete:
            for i in range(0, len(objects_to_delete), 1000):
                chunk = objects_to_delete[i:i + 1000]
                s3_client.delete_objects(Bucket=S3_BUCKET_NAME, Delete={'Objects': chunk})

        cursor.execute(f"DELETE FROM sightings WHERE id IN ({placeholders})", tuple(valid_ids))
        conn.commit()

        return jsonify({'success': True, 'deleted_count': len(valid_ids)})

    except Exception as e:
        conn.rollback()
        app.logger.exception(f"Bulk delete error: {e}")
        return jsonify({'success': False, 'error': 'An internal server error occurred.'}), 500
    finally:
        cursor.close()


@app.route('/sightings/bulk-move', methods=['POST'])
@login_required
def bulk_move_sightings():
    data = request.get_json()
    sighting_ids = data.get('sighting_ids')
    new_camera_id = data.get('new_camera_id')

    if not new_camera_id:
        return jsonify({'success': False, 'error': 'No new camera selected.'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id FROM cameras WHERE id = %s AND user_id = %s", (new_camera_id, current_user.id))
        if not cursor.fetchone():
            return jsonify({'success': False, 'error': 'Destination camera not found.'}), 404

        valid_ids = get_valid_sighting_ids(sighting_ids, current_user.id, cursor)
        if not valid_ids:
            return jsonify({'success': False, 'error': 'No valid sightings selected.'}), 400

        placeholders = ','.join(['%s'] * len(valid_ids))
        params = [new_camera_id] + valid_ids
        cursor.execute(f"UPDATE sightings SET camera_id = %s WHERE id IN ({placeholders})", tuple(params))
        conn.commit()

        return jsonify({'success': True, 'moved_count': len(valid_ids)})

    except Exception as e:
        conn.rollback()
        app.logger.exception(f"Bulk move error: {e}")
        return jsonify({'success': False, 'error': 'An internal server error occurred.'}), 500
    finally:
        cursor.close()


@app.route('/sightings/bulk-add-id', methods=['POST'])
@login_required
def bulk_add_id():
    data = request.get_json()
    sighting_ids = data.get('sighting_ids')
    animal_name = data.get('animal')

    if not animal_name or animal_name not in ANIMAL_CLASS_MAP:
        return jsonify({'success': False, 'error': 'Invalid animal specified.'}), 400

    class_id = ANIMAL_CLASS_MAP.get(animal_name)

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        valid_ids = get_valid_sighting_ids(sighting_ids, current_user.id, cursor)
        if not valid_ids:
            return jsonify({'success': False, 'error': 'No valid sightings selected.'}), 400

        cursor.execute("INSERT IGNORE INTO animals (name) VALUES (%s)", (animal_name,))
        cursor.execute("SELECT id FROM animals WHERE name = %s", (animal_name,))
        animal_result = cursor.fetchone()
        animal_id = animal_result['id']

        insert_tuples = [(s_id, animal_id) for s_id in valid_ids]
        cursor.executemany("INSERT IGNORE INTO sighting_animals (sighting_id, animal_id) VALUES (%s, %s)",
                           insert_tuples)

        for sighting_id in valid_ids:
            try:
                cursor.execute(
                    "SELECT image_path, original_image_path, predictions FROM sightings WHERE id = %s",
                    (sighting_id,)
                )
                sighting = cursor.fetchone()
                image_to_save_path = sighting.get('image_path') or sighting.get('original_image_path')

                if not sighting or not image_to_save_path:
                    continue

                unique_folder_name = str(uuid.uuid4())
                source_key = image_to_save_path
                base_filename = os.path.basename(source_key)

                image_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=source_key)
                image_content = image_obj['Body'].read()

                image_s3_key = f"training_data/needs_review/{unique_folder_name}/{base_filename}"
                s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=image_s3_key, Body=image_content)

                img = Image.open(io.BytesIO(image_content))
                img_width, img_height = img.size

                predictions = json.loads(sighting['predictions']) if sighting.get('predictions') else []
                yolo_lines = []

                if predictions and class_id is not None:
                    for pred in predictions:
                        x_min, y_min, x_max, y_max = pred['x_min'], pred['y_min'], pred['x_max'], pred['y_max']
                        dw = 1.0 / img_width
                        dh = 1.0 / img_height
                        x_center = (x_min + x_max) / 2.0
                        y_center = (y_min + y_max) / 2.0
                        w = x_max - x_min
                        h = y_max - y_min
                        x_norm = x_center * dw
                        y_norm = y_center * dh
                        w_norm = w * dw
                        h_norm = h * dh
                        yolo_lines.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

                if yolo_lines:
                    txt_filename = os.path.splitext(base_filename)[0] + '.txt'
                    txt_s3_key = f"training_data/needs_review/{unique_folder_name}/{txt_filename}"
                    yolo_bytes = "\n".join(yolo_lines).encode('utf-8')
                    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=txt_s3_key, Body=yolo_bytes)

            except Exception:
                pass

        placeholders = ','.join(['%s'] * len(valid_ids))
        cursor.execute(f"UPDATE sightings SET feedback_status = 'needs_review' WHERE id IN ({placeholders})",
                       tuple(valid_ids))

        conn.commit()
        updated_sightings = get_updated_animal_lists(valid_ids, cursor)
        return jsonify({'success': True, 'updated_sightings': updated_sightings})

    except Exception as e:
        conn.rollback()
        app.logger.exception(f"Bulk add ID error: {e}")
        return jsonify({'success': False, 'error': 'An internal server error occurred.'}), 500
    finally:
        cursor.close()


@app.route('/sightings/bulk-remove-id', methods=['POST'])
@login_required
def bulk_remove_id():
    data = request.get_json()
    sighting_ids = data.get('sighting_ids')
    animal_name = data.get('animal_name')

    if not animal_name:
        return jsonify({'success': False, 'error': 'No animal specified.'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        valid_ids = get_valid_sighting_ids(sighting_ids, current_user.id, cursor)
        if not valid_ids:
            return jsonify({'success': False, 'error': 'No valid sightings selected.'}), 400

        cursor.execute("SELECT id FROM animals WHERE name = %s", (animal_name,))
        animal_result = cursor.fetchone()
        if not animal_result:
            return jsonify({'success': True, 'message': 'Animal not found, nothing to remove.'})
        animal_id = animal_result['id']

        placeholders = ','.join(['%s'] * len(valid_ids))
        params = [animal_id] + valid_ids
        cursor.execute(f"DELETE FROM sighting_animals WHERE animal_id = %s AND sighting_id IN ({placeholders})",
                       tuple(params))
        conn.commit()

        updated_sightings = get_updated_animal_lists(valid_ids, cursor)
        return jsonify({'success': True, 'updated_sightings': updated_sightings})

    except Exception as e:
        conn.rollback()
        app.logger.exception(f"Bulk remove ID error: {e}")
        return jsonify({'success': False, 'error': 'An internal server error occurred.'}), 500
    finally:
        cursor.close()


@app.route('/sightings/bulk-toggle-retain', methods=['POST'])
@login_required
def bulk_toggle_retain():
    data = request.get_json()
    sighting_ids = data.get('sighting_ids')

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        valid_ids = get_valid_sighting_ids(sighting_ids, current_user.id, cursor)
        if not valid_ids:
            return jsonify({'success': False, 'error': 'No valid sightings selected.'}), 400

        placeholders = ','.join(['%s'] * len(valid_ids))
        cursor.execute(f"UPDATE sightings SET retained = NOT retained WHERE id IN ({placeholders})", tuple(valid_ids))
        conn.commit()

        cursor.execute(f"SELECT id, retained FROM sightings WHERE id IN ({placeholders})", tuple(valid_ids))
        updated_statuses = {row['id']: bool(row['retained']) for row in cursor.fetchall()}

        return jsonify({'success': True, 'updated_statuses': updated_statuses})

    except Exception as e:
        conn.rollback()
        app.logger.exception(f"Bulk toggle retain error: {e}")
        return jsonify({'success': False, 'error': 'An internal server error occurred.'}), 500
    finally:
        cursor.close()


if __name__ == '__main__':
    pass