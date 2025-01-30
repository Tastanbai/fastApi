from fastapi import APIRouter, Depends, HTTPException, status, Form
import mysql.connector
from auth import get_current_user, hash_password

router = APIRouter()

DB_CONFIG = {
    "host": "face.tabet-kitap.kz",
    "user": "fastapi_user",
    "password": "secure_password",
    "database": "face_db",
    "port": 3306
}

@router.post("/register")
async def register_user(
    username: str = Form(...),
    password: str = Form(...),
    is_admin: bool = Form(False),  # По умолчанию обычный пользователь
    current_user: dict = Depends(get_current_user)
):
    # Только администратор может добавлять пользователей
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Только администраторы могут добавлять пользователей")

    hashed_password = hash_password(password)

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("INSERT INTO users (username, password, is_admin) VALUES (%s, %s, %s)", 
                       (username, hashed_password, is_admin))
        conn.commit()

        cursor.close()
        conn.close()

        return {"message": f"Пользователь {username} успешно зарегистрирован"}
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Ошибка базы данных: {e}")
