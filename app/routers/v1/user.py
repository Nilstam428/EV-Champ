from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.user import UserCreate, UserResponse
from app.crud import user as crud_user

router = APIRouter()

@router.post("/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user"""
    try:
        # Check if email already exists
        existing_email = crud_user.get_user_by_email(db, user.email)
        if existing_email:
            raise HTTPException(
                status_code=409,
                detail="Email already registered"
            )
        
        # Check if username already exists
        existing_username = crud_user.get_user_by_username(db, user.username)
        if existing_username:
            raise HTTPException(
                status_code=409,
                detail="Username already taken"
            )
        
        db_user = crud_user.create_user(db=db, user=user)
        return db_user
    except HTTPException as e:
        raise e
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while creating the user: {str(e)}"
        )