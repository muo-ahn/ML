from fastapi import APIRouter
from app.models.items import Item

router = APIRouter()

@router.get("/items/")
def read_items():
    return [{"name": "item1"}, {"name": "item2"}]

@router.post("/items/")
def create_item(item: Item):
    return {"name": item.name, "description": item.description}
