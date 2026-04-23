from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
import uuid
import shutil
from pathlib import Path
from src.api.db.session import get_db
from src.api.db.models import Coin, User
from src.api.auth.deps import require_role
from src.api.schemas import CoinCreate, CoinResponse
from src.agents.gatekeeper import get_gatekeeper

router = APIRouter(prefix="/api/admin/coins", tags=["Admin Coins"])
UPLOAD_DIR = Path("data/processed/admin_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.get("/", response_model=List[CoinResponse])
async def get_coins(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Coin))
    return result.scalars().all()

@router.post("/analyze-prefill")
async def analyze_prefill(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    name = uuid.uuid4().hex
    temp_path = UPLOAD_DIR / f"{name}.{ext}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    gk = get_gatekeeper()
    return {"suggested_cn_type_id": "1015", "duplicate_warning": None, "param": str(temp_path)}

@router.post("/", response_model=CoinResponse)
async def create_coin(
    coin_in: CoinCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin", "curator"))
):
    new_coin = Coin(**coin_in.model_dump(), user_id=current_user.id)
    db.add(new_coin)
    await db.commit()
    await db.refresh(new_coin)
    return new_coin

@router.delete("/{coin_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_coin(
    coin_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin"))
):
    result = await db.execute(select(Coin).where(Coin.id == coin_id))
    coin = result.scalar_one_or_none()
    if not coin:
        raise HTTPException(status_code=404, detail="Coin not found")
    await db.delete(coin)
    await db.commit()
