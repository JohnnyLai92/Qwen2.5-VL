from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Union
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import io
import base64
import uvicorn
import json
from torchvision import transforms

app = FastAPI(title="Qwen VL API", version="1.0")

# ========== 模型初始化 ==========
try:
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="mps"
    )
    model.to("mps")
    print("模型加載成功！")
except Exception as e:
    raise RuntimeError(f"模型加載失敗：{str(e)}")

# ========== 數據模型定義 ==========
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[dict]]

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 64
    temperature: Optional[float] = 0.7

# ========== 輔助函數 ==========
def process_image(image_data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),  # 将图像转换为张量
            # ... 其他必要的转换 ...
        ])
        return preprocess(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像处理失败：{str(e)}")


# ========== API 路由 ==========
@app.post("/v1/chat/completions")
async def chat_endpoint(
    model_name: str = Form(...),  # 修改變量名以避免衝突
    messages: str = Form(...),  # JSON 字串，需反序列化
    max_tokens: int = Form(64),
    temperature: float = Form(0.7),
    image_file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    try:
        # 反序列化 messages
        messages = json.loads(messages)

        # 驗證 model 名稱
        if model_name != "Qwen/Qwen2.5-VL-7B-Instruct":
            raise HTTPException(status_code=400, detail="不支援的模型")

        # 處理圖像輸入
        image = None
        if image_file:
            image = process_image(await image_file.read())
        elif image_base64:
            try:
                image = process_image(base64.b64decode(image_base64))
            except:
                raise HTTPException(status_code=400, detail="無效的 Base64 圖像")

        # 文字處理
        text_input = messages[-1]["content"]

        # 模型輸入準備
        if image:
            inputs = processor(
                text=text_input,
                images=image,
                return_tensors="pt",
                padding=True
            )
        else:
            inputs = processor(
                text=text_input,
                return_tensors="pt",
                padding=True
            )

        # 設備轉移
        inputs = {k: v.to("mps") for k, v in inputs.items()}

        # 模型推理
        with torch.no_grad():
            output_ids = model.generate(  # 確保使用全局 model 變量
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                #pad_token_id=processor.eos_token_id 
            )

        # 回應生成
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        response_text = processor.decode(generated_ids, skip_special_tokens=True)

        return {
            "id": f"chat-{hash(response_text)}",
            "object": "chat.completion",
            "created": 1620000000,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }]
        }

    except Exception as e:
        # 捕獲所有異常並返回錯誤信息
        return {"error": str(e)}

# ========== 啟動配置 ==========
if __name__ == "__main__":
    uvicorn.run(
        app="runapi:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )