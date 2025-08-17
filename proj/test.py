from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, g
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import cv2
import json
import time
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip
from functools import wraps
import tempfile
from werkzeug.security import generate_password_hash, check_password_hash
import torch
from nomeroff_net.tools.ocr_tools import StrLabelConverter
from torchvision.models import efficientnet_b2


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = os.urandom(24).hex()

db = SQLAlchemy(app)

torch.serialization.add_safe_globals([efficientnet_b2])

class Car(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plates = db.Column(db.String(20), unique=True, nullable=False)
    fio = db.Column(db.String(100), nullable=False)
    room = db.Column(db.String(10), nullable=False)
    phone = db.Column(db.String(15), nullable=False)


class Log(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plates = db.Column(db.String(20), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)    



# Разрешить класс StrLabelConverter для безопасной загрузки
torch.serialization.add_safe_globals([StrLabelConverter])


number_plate_detection_and_reading = pipeline(
    "number_plate_detection_and_reading", 
    image_loader="opencv"
)


with app.app_context():
    db.create_all()


@app.route('/')
def index():
    return render_template('index.html')

from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Пожалуйста, войдите в систему')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        
        if User.query.filter_by(username=username).first():
            flash('Пользователь с таким именем уже существует')
            return redirect(url_for('register'))

        
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('Регистрация прошла успешно! Теперь вы можете войти.')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            
            session['user_id'] = user.id
            flash('Вы успешно вошли в систему!')
            return redirect(url_for('index'))
        else:
            flash('Неверное имя пользователя или пароль')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Вы вышли из системы')
    return redirect(url_for('index'))



@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Файл не выбран')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('Файл не выбран')
            return redirect(request.url)

        if file:
            
            upload_folder = app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            
            filepath = os.path.join(upload_folder, file.filename)
            file.save(filepath)
            print(f"Файл сохранен: {filepath}")

            
            try:
                (images, images_bboxs, 
                 images_points, images_zones, region_ids, 
                 region_names, count_lines, 
                 confidences, texts) = unzip(number_plate_detection_and_reading([filepath]))

                
                print(f"Распознанные номера: {texts}")

                
                if texts and texts[0]:  
                    plates = texts[0][0] 
                else:
                    flash('Номер не распознан')
                    print("Номер не распознан")
                    return redirect(request.url)

                
                car = Car.query.filter_by(plates=plates).first()
                print(f"Найденный автомобиль: {car}")

                if car:
                    
                    log_entry = Log(plates=plates, date=datetime.utcnow())
                    db.session.add(log_entry)
                    db.session.commit()

                    print(f"Автомобиль найден: {car.fio}, комната: {car.room}, телефон: {car.phone}")
                    return render_template('result.html', plates=plates, car=car)
                else:
                    flash('Автомобиль не найден в базе данных')
                    print("Автомобиль не найден в базе данных")
                    return redirect(request.url)
            except Exception as e:
                flash(f'Ошибка при обработке изображения: {str(e)}')
                print(f"Ошибка: {e}")
                return redirect(request.url)

    return render_template('upload.html')


last_detected_plate = None
last_detection_status = None

def detect_plate(frame):
    global last_detected_plate, last_detection_status
    try:
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_filename = temp_file.name
            cv2.imwrite(temp_filename, frame)  

        
        (images, images_bboxs, images_points, images_zones, region_ids, 
         region_names, count_lines, confidences, texts) = unzip(
            number_plate_detection_and_reading([temp_filename]) 
        )

        os.remove(temp_filename)

        if texts and texts[0]:  
            plate_number = texts[0][0]  
            last_detected_plate = plate_number

            
            with app.app_context():  
                car = Car.query.filter_by(plates=plate_number).first()
                if car:
                    last_detection_status = "Проезд разрешен"
                    
                    log_entry = Log(plates=plate_number, date=datetime.utcnow())
                    db.session.add(log_entry)
                    db.session.commit()
                else:
                    last_detection_status = "Проезд запрещен"
        else:
            last_detected_plate = None  
            last_detection_status = "Номер не распознан"
    except Exception as e:
        print(f"Ошибка при распознавании номера: {e}")
        last_detected_plate = None
        last_detection_status = "Ошибка при распознавании"

def generate_frames():
    cap = cv2.VideoCapture(0) 
    last_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        current_time = time.time()
        if current_time - last_time >= 10:
            detect_plate(frame)  
            last_time = current_time

        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/get_plate')
def get_plate():
    global last_detected_plate, last_detection_status
    return {
        'plate': last_detected_plate,
        'status': last_detection_status
    }

# Маршрут для потоковой передачи видео
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera')
@login_required
def camera():
    return render_template('camera.html')


@app.route('/add_car', methods=['GET', 'POST'])
@login_required
def add_car():
    if request.method == 'POST':
        plates = request.form.get('plates')
        fio = request.form.get('fio')
        room = request.form.get('room')
        phone = request.form.get('phone')

        try:
            new_car = Car(
                plates=plates,
                fio=fio,
                room=room,
                phone=phone
            )
            db.session.add(new_car)
            db.session.commit()
            flash('Машина успешно добавлена!')
        except Exception as e:
            db.session.rollback()
            flash(f'Ошибка: {str(e)}')
        finally:
            db.session.close()

        return redirect(url_for('index'))

    return render_template('add_car.html')

@app.route('/upload_json', methods=['GET', 'POST'])
@login_required
def upload_json():
    project_dir = os.path.dirname(os.path.abspath(__file__))  
    json_file_path = os.path.join(project_dir, 'car_data.json')  

    
    if not os.path.exists(json_file_path):
        flash(f'Файл {json_file_path} не найден.')
        return redirect(url_for('upload_json'))

    if request.method == 'POST':
        try:
            selected_entries = request.form.getlist('selected_entries')
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            for index in selected_entries:
                item = data[int(index)]
                
                if not Car.query.filter_by(plates=item['plates']).first():
                    new_car = Car(
                        plates=item['plates'],
                        fio=item['fio'],
                        room=item['room'],
                        phone=item['phone']
                    )
                    db.session.add(new_car)
            db.session.commit()
            flash('Выбранные заявки успешно добавлены в базу данных!')
        except Exception as e:
            db.session.rollback()
            flash(f'Ошибка при загрузке данных: {str(e)}')
        finally:
            db.session.close()
        return redirect(url_for('upload_json'))

    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        
        filtered_entries = []
        for idx, entry in enumerate(data):
            if not Car.query.filter_by(plates=entry['plates']).first():
                filtered_entries.append({"index": idx, **entry})
    except Exception as e:
        flash(f'Ошибка при чтении JSON-файла: {str(e)}')
        filtered_entries = []
    
    return render_template('upload_json.html', entries=filtered_entries)

@app.route('/whitelist')
@login_required
def whitelist():
    
    cars = Car.query.all()
    return render_template('whitelist.html', cars=cars)

@app.route('/delete_car/<int:id>', methods=['POST'])
@login_required
def delete_car(id):
    car = Car.query.get_or_404(id)  
    try:
        db.session.delete(car)  
        db.session.commit()
        flash('Машина успешно удалена!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Ошибка при удалении машины: {str(e)}', 'danger')
    return redirect(url_for('whitelist'))

@app.route('/edit_car/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_car(id):
    car = Car.query.get_or_404(id)  
    if request.method == 'POST':
        try:
           
            car.plates = request.form.get('plates')
            car.fio = request.form.get('fio')
            car.room = request.form.get('room')
            car.phone = request.form.get('phone')
            db.session.commit()
            flash('Данные машины успешно обновлены!', 'success')
            return redirect(url_for('whitelist'))
        except Exception as e:
            db.session.rollback()
            flash(f'Ошибка при обновлении данных: {str(e)}', 'danger')
    return render_template('edit_car.html', car=car)

@app.route('/logs')
@login_required
def view_logs():
    
    logs = Log.query.order_by(Log.date.desc()).all()  
    return render_template('logs.html', logs=logs)

if __name__ == '__main__':
    app.run(debug=True)